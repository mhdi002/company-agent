"""Stage 4: exact and near-duplicate removal (MinHash LSH), across all datasets."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX = np.uint64((1 << 32) - 1)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def exact_key(text: str) -> str:
    return hashlib.sha1(norm_text(text).encode()).hexdigest()


def shingles(text: str, k: int = 5) -> set[int]:
    w = norm_text(text).split()
    if len(w) < k:
        return {int.from_bytes(hashlib.blake2b(" ".join(w).encode(), digest_size=4).digest(), "little")}
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + k]).encode(), digest_size=4).digest(), "little")
            for i in range(len(w) - k + 1)}


class MinHashLSH:
    """MinHash signatures with banded LSH; query+insert in one call."""

    def __init__(self, num_perm: int = 128, bands: int = 32, threshold: float = 0.8, seed: int = 1):
        assert num_perm % bands == 0
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 2**31 - 1, num_perm).astype(np.uint64)
        self.b = rng.randint(0, 2**31 - 1, num_perm).astype(np.uint64)
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.threshold = threshold
        self.tables: list[dict] = [defaultdict(list) for _ in range(bands)]
        self.sigs: dict[str, np.ndarray] = {}

    def signature(self, sh: set[int]) -> np.ndarray:
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        h = (np.outer(x, self.a) + self.b) % _MERSENNE & _MAX
        return h.min(axis=0)

    @staticmethod
    def jaccard_est(s1: np.ndarray, s2: np.ndarray) -> float:
        return float(np.mean(s1 == s2))

    def add_or_match(self, key: str, text: str) -> str | None:
        """Return the key of a near-duplicate already indexed, else index `key` and return None."""
        sig = self.signature(shingles(text))
        cands = set()
        for b in range(self.bands):
            band = sig[b * self.rows:(b + 1) * self.rows].tobytes()
            cands.update(self.tables[b].get(band, ()))
        for c in cands:
            if self.jaccard_est(sig, self.sigs[c]) >= self.threshold:
                return c
        for b in range(self.bands):
            self.tables[b][sig[b * self.rows:(b + 1) * self.rows].tobytes()].append(key)
        self.sigs[key] = sig
        return None


class Deduplicator:
    def __init__(self, cfg: dict):
        self.exact: dict[str, str] = {}
        self.lsh = MinHashLSH(cfg.get("minhash_perm", 128), cfg.get("minhash_bands", 32),
                              cfg.get("near_dup_threshold", 0.8), cfg.get("seed", 1))

    def stage(self, doc: dict, cfg: dict) -> tuple[dict | None, str]:
        k = exact_key(doc["text"])
        if k in self.exact:
            src = self.exact[k].split(":")[0]
            return None, "exact_dup" + ("_cross" if src != doc["source"] else "")
        self.exact[k] = doc["id"]
        m = self.lsh.add_or_match(doc["id"], doc["text"])
        if m is not None:
            return None, "near_dup" + ("_cross" if m.split(":")[0] != doc["source"] else "")
        return doc, "kept"
