"""Output canonicalization and deterministic agreement tests for self-consistency.

Kinds:
  label     exact match after normalization (field label + country, project choice, JSON fields)
  set       items normalized to a set; clusters by Jaccard ≥ threshold (fact lists, fact sheets)
  free_text normalized key-fact (content-word) set overlap, or hashed TF-IDF embedding cosine
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

STOP = set("the a an and or of to in is are was for with on by as at from it its this that be have has our we "
           "their they which who into per also can will not".split())


def norm_str(s: Any) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"[^\w\s%.-]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def canon_json(v: Any) -> str:
    """Stable string for any JSON value with normalized strings."""
    def walk(x):
        if isinstance(x, dict):
            return {norm_str(k): walk(val) for k, val in sorted(x.items(), key=lambda kv: norm_str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [walk(i) for i in x]
        if isinstance(x, str):
            return norm_str(x)
        if isinstance(x, float):
            return round(x, 4)
        return x
    return json.dumps(walk(v), sort_keys=True, ensure_ascii=False)


def keyfacts(text: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-z0-9]+", norm_str(text)) if t not in STOP and len(t) > 1)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def embed(text: str, dim: int = 512) -> list[float]:
    """Hashed TF (log) embedding of word unigrams+bigrams, L2-normalised. Local and deterministic."""
    toks = [t for t in re.findall(r"[a-z0-9]+", norm_str(text)) if t not in STOP]
    feats = Counter(toks + [a + "_" + b for a, b in zip(toks, toks[1:])])
    v = [0.0] * dim
    for f, c in feats.items():
        h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
        v[h % dim] += (1 if (h >> 31) & 1 else -1) * (1 + math.log(c))
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def item_set(value: Any, item_key: str | None = None) -> frozenset[str]:
    """Normalize a list output into a set of item strings (using item_key for dict items)."""
    if isinstance(value, dict):
        for k in ("claims", "facts", "projects", "items"):
            if isinstance(value.get(k), list):
                value = value[k]
                break
    if not isinstance(value, list):
        return frozenset([canon_json(value)])
    out = set()
    for it in value:
        if isinstance(it, dict):
            key = item_key if item_key and item_key in it else next(
                (k for k in ("fact", "text", "title", "claim") if k in it), None)
            out.add(norm_str(it[key]) if key else canon_json(it))
        else:
            out.add(norm_str(it))
    return frozenset(out)


def set_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Soft Jaccard over items: items match when their key-fact sets overlap ≥ 0.6."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    bk = [keyfacts(x) for x in b]
    matched = 0
    used = set()
    for x in a:
        kx = keyfacts(x)
        for j, ky in enumerate(bk):
            if j not in used and jaccard(kx, ky) >= 0.6:
                used.add(j)
                matched += 1
                break
    return matched / (len(a) + len(b) - matched)


def cluster(outputs: list[Any], kind: str, threshold: float = 0.6, method: str = "keyfacts",
            label_fn=None) -> list[int | None]:
    """Assign each output a cluster id (index of its representative). None for invalid outputs.

    label: exact match of label_fn(output) (default canon_json).
    set / free_text: greedy clustering in index order against cluster representatives.
    """
    ids: list[int | None] = [None] * len(outputs)
    reps: list[tuple[int, Any]] = []
    for i, out in enumerate(outputs):
        if out is None:
            continue
        if kind == "label":
            key = (label_fn or canon_json)(out)
            for r, rkey in reps:
                if rkey == key:
                    ids[i] = r
                    break
            else:
                reps.append((i, key))
                ids[i] = i
            continue
        if kind == "set":
            rep_val = item_set(out)
            sim = set_similarity
        elif method == "embedding":
            rep_val = embed(out if isinstance(out, str) else canon_json(out))
            sim = cosine
        else:
            rep_val = keyfacts(out if isinstance(out, str) else canon_json(out))
            sim = jaccard
        for r, rv in reps:
            if sim(rep_val, rv) >= threshold:
                ids[i] = r
                break
        else:
            reps.append((i, rep_val))
            ids[i] = i
    return ids
