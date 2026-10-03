"""Stage 5: PII scrubbing and toxicity/NSFW filtering."""
from __future__ import annotations

import re

from data.clean.lexicons import NSFW, TOXIC

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
PHONE = re.compile(r"(?<![\w])(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{3,4}(?![\w])")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
ADDRESS = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b\.?")


def _luhn(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    s = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            x -= 9 if x > 9 else 0
        s += x
    return s % 10 == 0


def scrub(text: str) -> tuple[str, dict]:
    """Replace PII with placeholders; return (text, counts)."""
    counts: dict[str, int] = {}

    def sub(pat: re.Pattern, tag: str, s: str, check=None) -> str:
        def r(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            counts[tag] = counts.get(tag, 0) + 1
            return f"<{tag}>"
        return pat.sub(r, s)

    text = sub(EMAIL, "EMAIL", text)
    text = sub(SSN, "ID", text)
    text = sub(IBAN, "IBAN", text)
    text = sub(CARD, "CARD", text, _luhn)
    text = sub(IPV4, "IP", text)
    text = sub(ADDRESS, "ADDRESS", text)
    # phones last: need at least 9 digits so years and amounts survive
    text = sub(PHONE, "PHONE", text, lambda s: sum(c.isdigit() for c in s) >= 9)
    return text, counts


def toxicity_score(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    bad = sum(1 for w in words if w in TOXIC or w in NSFW)
    return bad / len(words)


def stage(doc: dict, cfg: dict) -> tuple[dict | None, str]:
    tox = toxicity_score(doc["text"])
    if tox > cfg.get("max_toxic_frac", 0.01):
        return None, "toxic"
    text, counts = scrub(doc["text"])
    meta = dict(doc.get("meta", {}))
    if counts:
        meta["pii"] = counts
    return dict(doc, text=text, meta=meta), "kept_scrubbed" if counts else "kept"
