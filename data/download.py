"""Download public datasets with resume + checksum verification; write MANIFEST.md.

Usage:
    python -m data.download --groups general business agentic --max-docs 2000000
    python -m data.download --datasets fineweb-edu wikipedia-en --max-files 2
    python -m data.download --sample          # offline: install bundled fixture corpus
    python -m data.download --manifest-only   # regenerate data/MANIFEST.md

Outputs `data/raw/<name>/docs.jsonl` with records {"id","text","source","meta"},
the downloaded files under `data/raw/<name>/files/`, and state in `data/raw/manifest.json`.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

import requests

from core.config import ROOT, load_config, resolve
from core.logging import get_logger
from data.registry import BY_NAME, REGISTRY, Dataset

log = get_logger("training")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# --------------------------------------------------------------------- checksums
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_or_record(path: Path, expected: str | None = None) -> str:
    """Verify `path` against `expected` (or its .sha256 sidecar); record the digest."""
    side = path.with_name(path.name + ".sha256")
    digest = sha256_file(path)
    want = expected or (side.read_text().strip() if side.exists() else None)
    if want and want != digest:
        raise ValueError(f"checksum mismatch for {path.name}: {digest} != {want}")
    side.write_text(digest)
    return digest


# --------------------------------------------------------------------- transfers
def download_url(url: str, dest: Path, retries: int = 4, timeout: int = 60,
                 expected_sha256: str | None = None, session: requests.Session | None = None) -> Path:
    """HTTP download that resumes from `<dest>.part` with Range requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.with_name(dest.name + ".sha256").exists():
        verify_or_record(dest, expected_sha256)
        return dest
    part = dest.with_name(dest.name + ".part")
    s = session or requests.Session()
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with s.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:      # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except (requests.RequestException, OSError) as e:
            wait = 2 ** (attempt + 1)
            log.event("download", "retry", tool="download", details=f"{url}: {e}; retry in {wait}s")
            if attempt == retries - 1:
                raise
            time.sleep(wait)
    part.replace(dest)
    verify_or_record(dest, expected_sha256)
    return dest


def download_hf(ds: Dataset, dest: Path, max_files: int = 0) -> list[Path]:
    """Download matching files of an HF dataset repo (hf_hub_download resumes)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(ds.repo, repo_type="dataset")
                   if any(fnmatch.fnmatch(f, p) for p in ds.patterns))
    limit = max_files or ds.max_files
    if limit:
        files = files[:limit]
    lfs = {}
    try:
        for info in api.get_paths_info(ds.repo, files, repo_type="dataset"):
            if getattr(info, "lfs", None):
                lfs[info.path] = info.lfs.sha256
    except Exception as e:  # metadata is best-effort; local digest still recorded
        log.event("download", "warn", tool="hf", details=f"no LFS checksums for {ds.repo}: {e}")
    out = []
    for f in files:
        p = Path(hf_hub_download(ds.repo, f, repo_type="dataset", local_dir=str(dest)))
        verify_or_record(p, lfs.get(f))
        out.append(p)
    return out


# --------------------------------------------------------------------- readers
def _join(rec: dict, fields: list[str]) -> str:
    parts = []
    for k in fields:
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(str(v).strip())
    return "\n\n".join(p for p in parts if p)


def iter_records(path: Path, ds: Dataset) -> Iterator[dict]:
    """Yield raw dict records from a downloaded file in any supported format."""
    name = path.name.lower()
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = [c for c in ds.text_fields if c in pf.schema_arrow.names] or None
        for batch in pf.iter_batches(batch_size=1024, columns=cols):
            yield from batch.to_pylist()
    elif name.endswith((".jsonl", ".jsonl.gz", ".json.gz")) or (name.endswith(".json") and ds.fmt == "jsonl"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif name.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        yield from (data if isinstance(data, list) else [data])
    elif name.endswith(".csv"):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            yield from csv.DictReader(f)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                low = member.lower()
                if low.endswith(".csv"):
                    with z.open(member) as fh:
                        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                        sample = text.read(4096)
                        text = io.TextIOWrapper(z.open(member), encoding="utf-8", errors="replace")
                        delim = ";" if sample.count(";") > sample.count(",") else ","
                        yield from csv.DictReader(text, delimiter=delim)
                elif low.endswith(".xml"):
                    try:
                        root = ElementTree.fromstring(z.read(member))
                    except ElementTree.ParseError:
                        continue
                    yield {el.tag: (el.text or "") for el in root.iter()}


def convert(ds: Dataset, files: list[Path], out: Path, max_docs: int = 0) -> int:
    """Convert downloaded files into docs.jsonl. Returns the number of documents."""
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for path in files:
            for i, rec in enumerate(iter_records(path, ds)):
                text = _join(rec, ds.text_fields)
                if not text:
                    continue
                f.write(json.dumps({"id": f"{ds.name}:{path.name}:{i}", "text": text,
                                    "source": ds.name, "meta": {"group": ds.group, "license": ds.license}},
                                   ensure_ascii=False) + "\n")
                n += 1
                if max_docs and n >= max_docs:
                    return n
    return n


# --------------------------------------------------------------------- manifest
def _state_path(raw: Path) -> Path:
    return raw / "manifest.json"


def load_state(raw: Path) -> dict:
    p = _state_path(raw)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(raw: Path, state: dict) -> None:
    p = _state_path(raw)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def write_manifest(raw: Path, path: Path | None = None) -> Path:
    """Render data/MANIFEST.md from the registry and the download state."""
    state = load_state(raw)
    path = path or ROOT / "data" / "MANIFEST.md"
    lines = ["# Data manifest", "",
             "Every dataset used for training, with source, license, and local status.",
             "Regenerate with `python -m data.download --manifest-only`.", "",
             "| Dataset | Group | License | Source | Status | Files | Size | Docs | SHA-256 (first file) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds in REGISTRY:
        st = state.get(ds.name, {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            ds.name, ds.group, ds.license, ds.source, st.get("status", "not downloaded"),
            len(st.get("files", [])), human_size(st.get("bytes", 0)), st.get("docs", 0),
            (st.get("files") or [{}])[0].get("sha256", "-")[:16]))
    lines += ["", "## Notes", ""]
    lines += [f"* **{ds.name}** — {ds.notes}" for ds in REGISTRY if ds.notes]
    if "sample" in state:
        lines += ["", "## Bundled sample (offline fixture)", "",
                  f"* `data/sample/` installed at {state['sample'].get('at')}: {state['sample'].get('docs')} docs. "
                  "Original text written for tests in this repo; used only for smoke runs, not for the real model."]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------- main
def fetch_dataset(ds: Dataset, raw: Path, max_docs: int = 0, max_files: int = 0) -> dict:
    dest = raw / ds.name / "files"
    t0 = time.time()
    if ds.kind == "hf":
        files = download_hf(ds, dest, max_files)
    elif ds.kind == "url":
        urls = ds.urls[:max_files] if max_files else ds.urls
        files = []
        for u in urls:
            fname = hashlib.sha1(u.encode()).hexdigest()[:10] + "_" + u.rstrip("/").split("/")[-1].split("?")[0]
            if "." not in fname[-6:]:
                fname += ".zip"
            files.append(download_url(u, dest / fname))
    else:
        return {"status": "generated locally"}
    n = convert(ds, files, raw / ds.name / "docs.jsonl", max_docs)
    return {"status": "ok", "docs": n, "seconds": round(time.time() - t0, 1),
            "bytes": sum(p.stat().st_size for p in files),
            "files": [{"path": str(p.relative_to(raw)), "sha256": p.with_name(p.name + ".sha256").read_text()}
                      for p in files], "license": ds.license,
            "at": dt.datetime.now(dt.timezone.utc).isoformat()}


def install_sample(raw: Path) -> int:
    """Copy the bundled offline fixture corpus into data/raw/sample/docs.jsonl."""
    from data.sample.build_sample import build_corpus
    docs = build_corpus()
    (raw / "sample").mkdir(parents=True, exist_ok=True)
    with open(raw / "sample" / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    st = load_state(raw)
    st["sample"] = {"status": "ok", "docs": len(docs), "at": dt.datetime.now(dt.timezone.utc).isoformat()}
    save_state(raw, st)
    return len(docs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    raw = resolve(cfg["paths"]["data_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    if a.sample:
        print(f"installed sample corpus: {install_sample(raw)} docs")
    if not a.manifest_only:
        chosen = [BY_NAME[n] for n in a.datasets] + [d for d in REGISTRY if d.group in a.groups]
        for ds in dict.fromkeys(chosen):
            if ds.kind == "local":
                continue
            print(f"==> {ds.name} ({ds.license})")
            try:
                res = fetch_dataset(ds, raw, a.max_docs, a.max_files)
            except Exception as e:  # keep going; record the failure in the manifest
                res = {"status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
                log.error("download", e, tool="download", details=ds.name)
            st = load_state(raw)
            st[ds.name] = res
            save_state(raw, st)
            print(f"    {res.get('status')} docs={res.get('docs', 0)}")
    print(f"manifest: {write_manifest(raw)}")


if __name__ == "__main__":
    main()
