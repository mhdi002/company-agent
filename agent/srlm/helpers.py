"""Helper API exposed inside the SRLM REPL sandbox.

Pure Python, no network or file access. Functions take either a text or the
name of a sandbox variable; `_resolve` turns a variable into (key, text) pairs:
str → one pair; dict[str,str] → items; dict[str,dict] → flattened; list of
dicts with "text" → (id or url, text).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from tools.field_classifier import extract_fields as _extract_fields

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an and or of to in is are was for with on by as at from it its this that be have has our we "
           "their they you your which who more than into over per also can will".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def resolve_texts(value: Any, key: str = "context") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(key, value)]
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            out.extend(resolve_texts(v, f"{k}"))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for i, v in enumerate(value):
            if isinstance(v, dict) and "text" in v:
                out.append((str(v.get("id") or v.get("url") or i), str(v["text"])))
            elif isinstance(v, dict):
                out.append((str(v.get("id", i)), " ".join(str(x) for x in v.values())))
            else:
                out.append((f"{key}[{i}]", str(v)))
        return out
    return [(key, str(value))]


_BARRIER = re.compile(r"^(.*?)\s+(?:is|are)\s+(?:a|an|the)?\s*(?:major|key|main|significant)?\s*"
                      r"(?:bottleneck|barrier|obstacle)s?\s+(for|to|in|of)\s+(.*)$", re.I)
_HAVE = re.compile(r"^(?:few|many|most|some)\s+[\w\s-]*?\s+(?:have|use|still|are|do)\s+(?:not\s+)?(.*)$", re.I)
_FEW_VERB = re.compile(r"^(?:few|many|most|some)\s+((?:[\w-]+\s+){0,3}?[\w-]+s)\s+([a-z]+)\s+(.*)$", re.I)
_STILL = re.compile(r"\s+(?:remains|is still|are still)\s+(?:limited|rare|manual|done in spreadsheets|low).*$", re.I)
_TREND_VERB = re.compile(r"\s+(increases?|reduces?|improves?|requires?|helps?|combines?|shortens?|exposes?|provides?|"
                         r"lets?|makes?|simplif(?:y|ies)|supports?|saves?|identif(?:y|ies)|is|are|now)\b", re.I)


def _cap(s: str) -> str:
    s = s.strip(" ,.")
    return s[:1].upper() + s[1:]


def project_title(sentence: str) -> str:
    """Heuristic rewrite of an evidence sentence into a project title."""
    s = sentence.strip().rstrip(".")
    if _STILL.search(s):
        s = _STILL.sub("", s)
        if re.match(r"integration of", s, re.I):
            return _cap(re.sub(r"^integration of", "Integrate", s, flags=re.I))[:90]
        return _cap("Digital " + s[:1].lower() + s[1:])[:90]
    s = re.sub(r"\b(?:on|in)\s+(whiteboards|spreadsheets|paper)\b", "digitally", s, flags=re.I)
    m = _BARRIER.match(s)
    if m:
        return _cap(f"Address {m.group(1).lower()} {m.group(2)} {m.group(3)}")[:90]
    m = _HAVE.match(s)
    if m:
        return _cap(m.group(1))[:90]
    m = _FEW_VERB.match(s)
    if m and m.group(2).lower() not in {"of", "in", "and", "for", "with"}:
        return _cap(f"{m.group(2)} {m.group(3)}")[:90]
    m = _TREND_VERB.search(s)
    if m and m.start() > 8:
        subj = s[:m.start()].strip()
        if not (len(subj) > 1 and subj[1].isupper()):
            subj = subj[:1].lower() + subj[1:]
        return _cap("Pilot: " + subj)[:90]
    return _cap(s)[:90]


class Helpers:
    """Bound to the sandbox namespace so helpers can take variable names."""

    def __init__(self, ns: dict):
        self.ns = ns

    def _get(self, var: Any) -> list[tuple[str, str]]:
        if isinstance(var, str) and var in self.ns and not var.startswith("_"):
            return resolve_texts(self.ns[var], var)
        return resolve_texts(var)

    # ---- spec helpers ----
    def search_context(self, regex: str, var: Any = "context", window: int = 150, max_hits: int = 20) -> list[dict]:
        """Regex search over a variable; returns hits with source key and surrounding snippet."""
        pat = re.compile(regex)
        hits = []
        for key, text in self._get(var):
            for m in pat.finditer(text):
                a, b = max(m.start() - window, 0), min(m.end() + window, len(text))
                hits.append({"source": key, "start": m.start(), "match": m.group(0)[:200],
                             "snippet": text[a:b].replace("\n", " ")})
                if len(hits) >= max_hits:
                    return hits
        return hits

    def bm25(self, query: str, k: int = 5, var: Any = "context", chunk_chars: int = 600) -> list[dict]:
        """BM25 over paragraph chunks of a variable; returns top-k chunks with scores."""
        chunks = []
        for key, text in self._get(var):
            paras = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text) if p.strip()]
            buf = ""
            for p in paras:
                if len(buf) + len(p) > chunk_chars and buf:
                    chunks.append((key, buf))
                    buf = ""
                buf = (buf + " " + p).strip()
            if buf:
                chunks.append((key, buf))
        if not chunks:
            return []
        docs = [tokens(c[1]) for c in chunks]
        N = len(docs)
        avg = sum(len(d) for d in docs) / N or 1.0
        df = Counter(t for d in docs for t in set(d))
        q = tokens(query)
        scored = []
        for (key, text), d in zip(chunks, docs):
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
            scored.append({"source": key, "score": round(s, 3), "text": text})
        scored.sort(key=lambda x: -x["score"])
        return [x for x in scored[:k] if x["score"] > 0]

    def slice(self, var: Any, a: int = 0, b: int | None = None) -> str:
        """Substring [a:b] of a text variable (dicts/lists are joined first)."""
        text = "\n\n".join(t for _, t in self._get(var))
        return text[a:b]

    def extract_fields(self, text: Any) -> dict:
        """name, country, field, services, size (when stated) from text or a variable."""
        joined = "\n\n".join(t for _, t in self._get(text))
        out = _extract_fields(joined)
        out.pop("field_scores", None)
        return out

    # ---- extra helpers ----
    def describe(self, var: str) -> str:
        v = self.ns.get(var)
        if isinstance(v, str):
            return f"{var}: str, {len(v)} chars"
        if isinstance(v, dict):
            return f"{var}: dict, {len(v)} keys: {list(v)[:8]}"
        if isinstance(v, list):
            return f"{var}: list, {len(v)} items"
        return f"{var}: {type(v).__name__}"

    def sentences(self, var: Any) -> list[dict]:
        out = []
        for key, text in self._get(var):
            for s in re.split(r"(?<=[.!?])\s+|\n+", text):
                s = s.strip()
                if len(s) > 20:
                    out.append({"source": key, "text": s})
        return out

    def overlap(self, a: str, b: str) -> float:
        ta, tb = set(tokens(a)), set(tokens(b))
        return len(ta & tb) / max(len(ta | tb), 1)

    def project_from_gap(self, sentence: str) -> str:
        """Turn a gap or trend statement into a short project title (heuristic rewrite)."""
        return project_title(sentence)

    def namespace(self) -> dict:
        return {"search_context": self.search_context, "bm25": self.bm25, "slice": self.slice,
                "extract_fields": self.extract_fields, "describe": self.describe, "sentences": self.sentences,
                "overlap": self.overlap, "project_from_gap": self.project_from_gap}


HELPER_DOC = """search_context(regex, var="context", window=150, max_hits=20) -> [{source, start, match, snippet}]
bm25(query, k=5, var="context") -> [{source, score, text}]
slice(var, a, b) -> str
extract_fields(text_or_var) -> {name, country, field, services, size}
sentences(var) -> [{source, text}]   overlap(a, b) -> float   project_from_gap(sentence) -> str
describe(var) -> str   FINAL(value)  # sets the program output (JSON-serialisable) and ends the program"""
