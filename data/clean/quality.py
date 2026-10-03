"""Stage 3: quality filtering — heuristics plus a perplexity filter.

The perplexity filter uses our own interpolated Kneser-Ney word trigram model
trained on a high-quality reference slice (datasets flagged
`quality_reference` in the registry, after heuristics).
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-z0-9']+|[^\sa-z0-9]", re.I)
_ALPHA = re.compile(r"[^\W\d_]", re.U)
_DIGIT = re.compile(r"\d")
_SYMBOL = re.compile(r"[#$%&*@^~<>{}\[\]|\\_=+]")
STOP = set("the and of to in is that for with are this on was by be as at from it an or have not".split())


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def heuristics(text: str, cfg: dict) -> str | None:
    """Return a rejection reason or None if the text passes."""
    n = len(text)
    if n < cfg.get("min_chars", 200):
        return "too_short"
    if n > cfg.get("max_chars", 100000):
        return "too_long"
    if len(_SYMBOL.findall(text)) / n > cfg.get("max_symbol_ratio", 0.1):
        return "symbol_ratio"
    if len(_DIGIT.findall(text)) / n > cfg.get("max_digit_ratio", 0.15):
        return "digit_ratio"
    if len(_ALPHA.findall(text)) / n < 0.6:
        return "low_alpha"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        dup = 1 - len(set(lines)) / len(lines)
        if dup > cfg.get("max_dup_line_frac", 0.3):
            return "dup_lines"
    words = text.lower().split()
    if len(words) < 30:
        return "too_few_words"
    for k in (2, 3):
        grams = Counter(zip(*[words[i:] for i in range(k)]))
        if grams:
            top_count = grams.most_common(1)[0][1]
            frac = top_count * k / len(words)
            if top_count > 2 and frac > cfg.get("max_top_ngram_frac", 0.2):
                return f"top_{k}gram"
    if sum(1 for w in words if w in STOP) / len(words) < 0.03:
        return "no_stopwords"
    mean_len = sum(len(w) for w in words) / len(words)
    if not 2.5 <= mean_len <= 12:
        return "word_length"
    return None


class KneserNeyLM:
    """Interpolated Kneser-Ney word trigram LM (discount D)."""

    def __init__(self, discount: float = 0.75, vocab_size: int = 50000):
        self.D = discount
        self.vocab_size = vocab_size

    def fit(self, texts: list[str]) -> "KneserNeyLM":
        wc = Counter()
        seqs = []
        for t in texts:
            toks = tokenize(t)
            wc.update(toks)
            seqs.append(toks)
        self.vocab = {w for w, _ in wc.most_common(self.vocab_size)}
        tri, bi_ctx = Counter(), Counter()
        for toks in seqs:
            s = ["<s>", "<s>"] + [w if w in self.vocab else "<unk>" for w in toks] + ["</s>"]
            for i in range(2, len(s)):
                tri[(s[i - 2], s[i - 1], s[i])] += 1
        for (u, v, w), c in tri.items():
            bi_ctx[(u, v)] += c
        self.tri = tri
        self.bi_ctx = bi_ctx
        self.tri_followers = Counter((u, v) for (u, v, _w) in tri)            # N1+(u v ·)
        # continuation counts
        cont_vw = Counter((v, w) for (_u, v, w) in tri)                     # N1+(· v w)
        self.cont_vw = cont_vw
        self.cont_v = Counter()                                              # N1+(· v ·)
        self.foll_v = Counter()                                              # N1+(v ·) over cont pairs
        for (v, w), c in cont_vw.items():
            self.cont_v[v] += c
            self.foll_v[v] += 1
        cont_w = Counter(w for (_v, w) in cont_vw)                           # N1+(· w)
        self.cont_w = cont_w
        self.cont_total = sum(cont_w.values())
        self.V = len(self.vocab) + 2
        return self

    def _p_uni(self, w: str) -> float:
        return 0.9 * self.cont_w.get(w, 0) / max(self.cont_total, 1) + 0.1 / self.V

    def _p_bi(self, v: str, w: str) -> float:
        cv = self.cont_v.get(v, 0)
        if cv == 0:
            return self._p_uni(w)
        a = max(self.cont_vw.get((v, w), 0) - self.D, 0) / cv
        lam = self.D * self.foll_v[v] / cv
        return a + lam * self._p_uni(w)

    def _p_tri(self, u: str, v: str, w: str) -> float:
        c = self.bi_ctx.get((u, v), 0)
        if c == 0:
            return self._p_bi(v, w)
        a = max(self.tri.get((u, v, w), 0) - self.D, 0) / c
        lam = self.D * self.tri_followers[(u, v)] / c
        return a + lam * self._p_bi(v, w)

    def perplexity(self, text: str, max_words: int = 1000) -> float:
        toks = [w if w in self.vocab else "<unk>" for w in tokenize(text)[:max_words]]
        s = ["<s>", "<s>"] + toks + ["</s>"]
        nll = 0.0
        for i in range(2, len(s)):
            nll -= math.log(max(self._p_tri(s[i - 2], s[i - 1], s[i]), 1e-12))
        return math.exp(nll / max(len(s) - 2, 1))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: Path) -> "KneserNeyLM":
        m = cls()
        with open(path, "rb") as f:
            m.__dict__.update(pickle.load(f))
        return m


def stage(doc: dict, cfg: dict, lm: KneserNeyLM | None = None) -> tuple[dict | None, str]:
    reason = heuristics(doc["text"], cfg)
    if reason:
        return None, reason
    if lm is not None:
        ppl = lm.perplexity(doc["text"])
        if ppl > cfg.get("perplexity_max", 3000.0):
            return None, "perplexity"
        doc = dict(doc, meta={**doc.get("meta", {}), "ppl": round(ppl, 1)})
    return doc, "kept"

