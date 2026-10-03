"""Stage 1: encoding repair, Unicode normalization, HTML and boilerplate removal."""
from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

MOJIBAKE = re.compile("[ÃÂâ][\u0080-¿‘-›€]|Ã|â€")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏  ﻿]")
HTML_HINT = re.compile(r"<\s*(html|body|div|p|br|span|a|script|style|nav|footer|table|li)\b", re.I)
BOILERPLATE = re.compile(
    r"(cookie|all rights reserved|privacy policy|terms of (use|service)|subscribe to our newsletter|"
    r"sign in|log in|skip to (main )?content|accept all|javascript is disabled|share on (facebook|twitter))", re.I)
DROP_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "form", "iframe", "svg", "aside"]


def fix_encoding(text: str) -> str:
    """Repair common UTF-8-decoded-as-latin-1/cp1252 mojibake, segment by segment."""
    if not MOJIBAKE.search(text):
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(codec).decode("utf-8")
            if len(MOJIBAKE.findall(fixed)) < len(MOJIBAKE.findall(text)):
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    # Mixed text: repair each mojibake run independently.
    def repl(m: re.Match) -> str:
        s = m.group(0)
        for codec in ("cp1252", "latin-1"):
            try:
                return s.encode(codec).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return s
    return re.sub(r"[ÃÂâ][\u0080-¿‘-›€]{1,2}", repl, text)


def strip_html(text: str) -> str:
    if not HTML_HINT.search(text):
        return text
    soup = BeautifulSoup(text, "lxml")
    for t in soup(DROP_TAGS):
        t.decompose()
    for t in soup.select("[class*=cookie], [id*=cookie], [class*=banner]"):
        t.decompose()
    return soup.get_text("\n")


def remove_boilerplate(text: str) -> str:
    """Drop short lines that look like navigation/legal boilerplate."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        if len(s) < 120 and BOILERPLATE.search(s):
            continue
        if s.count("|") >= 2 and len(s) / (s.count("|") + 1) < 20:   # "Home | About | Contact"
            continue
        out.append(s)
    return "\n".join(out)


def normalize_ws(text: str) -> str:
    text = CONTROL.sub("", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(text: str) -> str:
    """Full stage-1 transform."""
    text = fix_encoding(text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_html(text)
    text = remove_boilerplate(text)
    return normalize_ws(text)


def stage(doc: dict, cfg: dict) -> tuple[dict | None, str]:
    text = normalize(doc["text"])
    if not text:
        return None, "empty_after_normalize"
    doc = dict(doc, text=text)
    return doc, "kept"
