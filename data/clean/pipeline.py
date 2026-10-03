"""Reproducible data-cleaning pipeline (stages 1–6; stage 7 sharding is `data/clean/shard.py`).

Usage:
    python -m data.clean.pipeline                      # all data/raw/*/docs.jsonl
    python -m data.clean.pipeline --workers 16 --offline-benchmarks

Each stage logs counts in/out and rejection reasons to `data/reports/`, plus 100
random kept and 100 random removed records for human review. A combined
before/after report is written to `data/reports/cleaning_report.md`.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable, Iterator

from core.config import load_config, resolve
from core.logging import get_logger
from data.clean import decontam, langid, normalize, pii, quality
from data.clean.dedup import Deduplicator
from data.registry import BY_NAME

log = get_logger("training")
StageFn = Callable[[dict, dict], "tuple[dict | None, str]"]


class StageStats:
    def __init__(self, idx: int, name: str, k: int, seed: int):
        self.idx, self.name, self.k = idx, name, k
        self.rng = random.Random(seed + idx)
        self.n_in = self.n_out = self.chars_in = self.chars_out = 0
        self.reasons: Counter = Counter()
        self.by_source_in: Counter = Counter()
        self.by_source_out: Counter = Counter()
        self.kept: list = []
        self.removed: list = []
        self.seen_kept = self.seen_removed = 0
        self.t0 = time.time()

    def _reservoir(self, bucket: list, seen: int, item: dict) -> None:
        if len(bucket) < self.k:
            bucket.append(item)
        else:
            j = self.rng.randint(0, seen)
            if j < self.k:
                bucket[j] = item

    def record(self, before: dict, after: dict | None, reason: str) -> None:
        self.n_in += 1
        self.chars_in += len(before["text"])
        self.by_source_in[before["source"]] += 1
        self.reasons[reason] += 1
        if after is None:
            self._reservoir(self.removed, self.seen_removed,
                            {"id": before["id"], "reason": reason, "text": before["text"][:1500]})
            self.seen_removed += 1
        else:
            self.n_out += 1
            self.chars_out += len(after["text"])
            self.by_source_out[after["source"]] += 1
            changed = after["text"] != before["text"]
            item = {"id": after["id"], "reason": reason, "text": after["text"][:1500]}
            if changed:
                item["before"] = before["text"][:1500]
            self._reservoir(self.kept, self.seen_kept, item)
            self.seen_kept += 1

    def summary(self) -> dict:
        return {"stage": self.idx, "name": self.name, "docs_in": self.n_in, "docs_out": self.n_out,
                "removed": self.n_in - self.n_out, "chars_in": self.chars_in, "chars_out": self.chars_out,
                "reasons": dict(self.reasons.most_common()), "by_source_in": dict(self.by_source_in),
                "by_source_out": dict(self.by_source_out), "seconds": round(time.time() - self.t0, 2)}

    def write(self, reports: Path) -> None:
        base = reports / f"stage{self.idx}_{self.name}"
        with open(f"{base}_samples.jsonl", "w", encoding="utf-8") as f:
            for item in self.kept:
                f.write(json.dumps({"kept": True, **item}, ensure_ascii=False) + "\n")
            for item in self.removed:
                f.write(json.dumps({"kept": False, **item}, ensure_ascii=False) + "\n")
        Path(f"{base}_stats.json").write_text(json.dumps(self.summary(), indent=2))


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _apply(fn: StageFn, cfg: dict, doc: dict) -> tuple[dict, dict | None, str]:
    out, reason = fn(doc, cfg)
    return doc, out, reason


def run_stage(idx: int, name: str, fn: StageFn, docs: Iterable[dict], out_path: Path, cfg: dict,
              reports: Path, workers: int = 1) -> dict:
    st = StageStats(idx, name, cfg.get("samples_per_stage", 100), cfg.get("seed", 1234))
    with open(out_path, "w", encoding="utf-8") as f:
        if workers > 1:
            with Pool(workers) as pool:
                results = pool.imap(functools.partial(_apply, fn, cfg), docs, chunksize=64)
                for before, after, reason in results:
                    st.record(before, after, reason)
                    if after is not None:
                        f.write(json.dumps(after, ensure_ascii=False) + "\n")
        else:
            for doc in docs:
                after, reason = fn(doc, cfg)
                st.record(doc, after, reason)
                if after is not None:
                    f.write(json.dumps(after, ensure_ascii=False) + "\n")
    st.write(reports)
    s = st.summary()
    log.event(f"clean.stage{idx}.{name}", "done", duration=s["seconds"],
              details=f"in={s['docs_in']} out={s['docs_out']} reasons={s['reasons']}")
    print(f"stage {idx} {name:<10} in={s['docs_in']:>8} out={s['docs_out']:>8}  {dict(list(s['reasons'].items())[:6])}")
    return s


def build_reference_lm(path_in: Path, cfg: dict, max_docs: int = 50000) -> quality.KneserNeyLM:
    """Train the perplexity reference model on high-quality sources that pass heuristics."""
    ref_sources = {n for n, d in BY_NAME.items() if d.quality_reference}
    texts, fallback = [], []
    for doc in read_jsonl(path_in):
        if quality.heuristics(doc["text"], cfg) is not None:
            continue
        if doc["source"] in ref_sources:
            texts.append(doc["text"])
        elif len(fallback) < max_docs:
            fallback.append(doc["text"])
        if len(texts) >= max_docs:
            break
    if not texts:  # offline sample run: no reference datasets present
        texts = fallback
    return quality.KneserNeyLM().fit(texts)


def write_report(stats: list[dict], reports: Path) -> Path:
    (reports / "cleaning_report.json").write_text(json.dumps(stats, indent=2))
    first, last = stats[0], stats[-1]
    lines = ["# Cleaning report", "",
             f"Input: **{first['docs_in']:,} docs / {first['chars_in']:,} chars** → "
             f"Output: **{last['docs_out']:,} docs / {last['chars_out']:,} chars** "
             f"({100 * last['docs_out'] / max(first['docs_in'], 1):.1f}% docs kept).", "",
             "| Stage | Docs in | Docs out | Removed | Chars in | Chars out | Top reasons | Seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        top = ", ".join(f"{k}: {v}" for k, v in list(s["reasons"].items())[:5])
        lines.append(f"| {s['stage']}. {s['name']} | {s['docs_in']:,} | {s['docs_out']:,} | {s['removed']:,} | "
                     f"{s['chars_in']:,} | {s['chars_out']:,} | {top} | {s['seconds']} |")
    lines += ["", "## Docs per source (before → after)", "", "| Source | Before | After |", "|---|---|---|"]
    for src, n in sorted(first["by_source_in"].items()):
        lines.append(f"| {src} | {n:,} | {last['by_source_out'].get(src, 0):,} |")
    lines += ["", "Review samples: `data/reports/stage<N>_<name>_samples.jsonl` "
              "(100 random kept + 100 random removed per stage; `before` shows the pre-stage text when modified)."]
    p = reports / "cleaning_report.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", help="docs.jsonl files (default: data/raw/*/docs.jsonl)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--offline-benchmarks", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reports-dir", default=None)
    a = ap.parse_args(argv)
    cfg_all = load_config()
    cfg = cfg_all["data"]
    raw = resolve(cfg_all["paths"]["data_raw"])
    out_dir = Path(a.out_dir) if a.out_dir else resolve(cfg_all["paths"]["data_cleaned"])
    reports = Path(a.reports_dir) if a.reports_dir else resolve(cfg_all["paths"]["data_reports"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p) for p in a.inputs] if a.inputs else sorted(raw.glob("*/docs.jsonl"))
    if not inputs:
        raise SystemExit("no inputs: run `python -m data.download ...` (or --sample) first")

    def all_docs() -> Iterator[dict]:
        for p in inputs:
            yield from read_jsonl(p)

    stats = []
    s1, s2, s3, s4, s5, s6 = (out_dir / f"stage{i}.jsonl" for i in range(1, 7))
    stats.append(run_stage(1, "normalize", normalize.stage, all_docs(), s1, cfg, reports, a.workers))
    stats.append(run_stage(2, "langid", langid.stage, read_jsonl(s1), s2, cfg, reports, a.workers))
    lm = build_reference_lm(s2, cfg)
    lm.save(out_dir / "reference_lm.pkl")
    q_fn = functools.partial(quality.stage, lm=lm)
    stats.append(run_stage(3, "quality", q_fn, read_jsonl(s2), s3, cfg, reports, 1))
    dd = Deduplicator(cfg)
    stats.append(run_stage(4, "dedup", dd.stage, read_jsonl(s3), s4, cfg, reports, 1))
    stats.append(run_stage(5, "pii_toxicity", pii.stage, read_jsonl(s4), s5, cfg, reports, a.workers))
    bench = decontam.load_benchmarks(out_dir / "benchmarks", offline=a.offline_benchmarks)
    dc = decontam.Decontaminator(bench, cfg.get("decontam_ngram", 13))
    stats.append(run_stage(6, "decontam", dc.stage, read_jsonl(s5), s6, cfg, reports, 1))
    s6.replace(out_dir / "cleaned.jsonl")
    for p in (s1, s2, s3, s4, s5):
        p.unlink(missing_ok=True)
    print(f"report: {write_report(stats, reports)}")
    return stats


if __name__ == "__main__":
    main()
