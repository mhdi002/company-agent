"""Stage 6: benchmark / eval-set decontamination by word n-gram overlap."""
from __future__ import annotations

import re
from pathlib import Path

from core.logging import get_logger
from data.registry import BENCHMARKS

log = get_logger("training")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngram_set(text: str, n: int) -> set[tuple]:
    w = _words(text)
    if len(w) < n:
        return {tuple(w)} if len(w) >= 8 else set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


class Decontaminator:
    def __init__(self, texts: list[str], n: int = 13):
        self.n = n
        self.index: set[tuple] = set()
        self.short: set[tuple] = set()
        for t in texts:
            w = _words(t)
            if len(w) >= n:
                self.index |= ngram_set(t, n)
            elif len(w) >= 8:
                self.short.add(tuple(w))

    def contaminated(self, text: str) -> bool:
        w = _words(text)
        for i in range(max(len(w) - self.n + 1, 0)):
            if tuple(w[i:i + self.n]) in self.index:
                return True
        if self.short:
            joined = " " + " ".join(w) + " "
            for s in self.short:
                if " " + " ".join(s) + " " in joined:
                    return True
        return False

    def stage(self, doc: dict, cfg: dict) -> tuple[dict | None, str]:
        if self.contaminated(doc["text"]):
            return None, "benchmark_overlap"
        return doc, "kept"


def load_benchmarks(cache_dir: Path, offline: bool = False, max_per: int = 20000) -> list[str]:
    """Load benchmark texts from HF (cached). Falls back to the bundled offline list."""
    from data.sample.build_sample import benchmark_texts
    texts = list(benchmark_texts())
    if offline:
        return texts
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
        api = HfApi()
        for name, (repo, _lic) in BENCHMARKS.items():
            files = [f for f in api.list_repo_files(repo, repo_type="dataset")
                     if f.endswith(".parquet") and ("test" in f or "validation" in f)][:4]
            for f in files:
                p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(cache_dir / name))
                tbl = pq.read_table(p)
                cols = [c for c in tbl.column_names if tbl.schema.field(c).type in ("string", "large_string")]
                for row in tbl.select(cols).to_pylist()[:max_per]:
                    texts.append(" ".join(str(v) for v in row.values() if v))
    except Exception as e:
        log.event("decontam", "warn", details=f"benchmark download failed, using offline list: {e}")
    return texts
