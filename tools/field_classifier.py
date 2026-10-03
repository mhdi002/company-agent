"""Rule-based extraction of company name, country, field, services, and size from page text.

Used directly by the agent and exposed inside the SRLM sandbox as
`extract_fields(text)`. Pure Python (no heavy imports) so it runs inside the
memory-limited REPL worker.
"""
from __future__ import annotations

import re
from collections import Counter

FIELDS: dict[str, list[str]] = {
    "renewable energy": ["solar", "photovoltaic", "pv", "wind", "renewable", "battery storage", "energy storage",
                         "inverter", "grid", "hydro", "geothermal", "clean energy"],
    "logistics": ["logistics", "freight", "shipping", "container", "warehouse", "warehousing", "drayage",
                  "forwarding", "customs", "intermodal", "last-mile", "delivery", "terminal", "supply chain"],
    "software": ["software", "saas", "web application", "mobile app", "cloud", "platform", "developer",
                 "api", "devops", "qa automation", "app development"],
    "agriculture": ["farm", "farmers", "agriculture", "crop", "irrigation", "soil", "agritech", "harvest",
                    "growers", "livestock", "agronomy"],
    "manufacturing": ["manufacturing", "manufacturer", "cnc", "machining", "fabrication", "assembly", "factory",
                      "production", "robot", "automation", "machine vision", "sheet metal"],
    "healthcare": ["health", "healthcare", "patient", "clinical", "hospital", "medical", "telehealth",
                   "clinician", "care homes", "medtech", "diagnostic"],
    "finance": ["finance", "financial", "credit", "lending", "loan", "bank", "banking", "insurance",
                "risk modelling", "regulatory reporting", "payments", "fintech"],
    "construction": ["construction", "contractor", "contracting", "building", "design-build", "timber framing",
                     "civil engineering", "prefabricated", "architect"],
    "education": ["education", "learning", "training", "courses", "learners", "students", "vocational",
                  "school", "e-learning", "curriculum"],
    "retail": ["retail", "grocery", "store", "stores", "shop", "e-commerce", "customers", "private label",
               "supermarket", "merchandise"],
}

COUNTRIES = ["Germany", "United Kingdom", "Canada", "Australia", "Netherlands", "Singapore", "United States",
             "France", "Spain", "Italy", "Sweden", "Norway", "Denmark", "Finland", "Ireland", "India", "Japan",
             "South Korea", "Brazil", "Mexico", "South Africa", "New Zealand", "Switzerland", "Austria",
             "Belgium", "Poland", "Portugal", "Israel", "United Arab Emirates", "China", "Indonesia", "Kenya",
             "Nigeria", "Egypt", "Turkey", "Chile", "Argentina", "Colombia", "Vietnam", "Malaysia", "Thailand"]
COUNTRY_ALIASES = {"UK": "United Kingdom", "U.K.": "United Kingdom", "England": "United Kingdom",
                   "USA": "United States", "U.S.": "United States", "The Netherlands": "Netherlands",
                   "Deutschland": "Germany", "UAE": "United Arab Emirates"}
LEGAL_SUFFIX = r"(GmbH|Ltd\.?|Limited|Inc\.?|LLC|B\.V\.|AG|Pty Ltd|Pte Ltd|Co\.|S\.A\.|SAS|plc|Corp\.?|AB|Oy|SpA)"
NAME_RE = re.compile(rf"\b([A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*){{0,4}}\s+{LEGAL_SUFFIX})")
SIZE_RE = re.compile(r"\b(\d[\d,]*)\s+(employees|staff|people|team members)\b", re.I)
SERVICE_RE = re.compile(r"(?<![/\w])(?:our services|services|we offer|we provide|the company offers|specialising in|specializing in)\s*[:\-]?\s*(.+?)(?:\.|\n|$)", re.I)


def classify_field(text: str) -> tuple[str, dict[str, int]]:
    """Return (best field label, keyword hit counts per field)."""
    low = text.lower()
    scores: Counter = Counter()
    for field, kws in FIELDS.items():
        for kw in kws:
            n = len(re.findall(rf"\b{re.escape(kw)}\b", low))
            if n:
                scores[field] += n * (2 if " " in kw else 1)
    if not scores:
        return "unknown", {}
    return scores.most_common(1)[0][0], dict(scores)


def find_country(text: str) -> str:
    counts: Counter = Counter()
    for c in COUNTRIES:
        counts[c] += len(re.findall(rf"\b{re.escape(c)}\b", text))
    for alias, c in COUNTRY_ALIASES.items():
        counts[c] += len(re.findall(rf"(?<!\w){re.escape(alias)}(?!\w)", text))
    best = counts.most_common(1)
    return best[0][0] if best and best[0][1] else "unknown"


def find_name(text: str) -> str:
    m = Counter(x[0].strip() for x in NAME_RE.findall(text))
    return m.most_common(1)[0][0] if m else "unknown"


def find_services(text: str, limit: int = 8) -> list[str]:
    out: list[str] = []
    for m in SERVICE_RE.finditer(text):
        for part in re.split(r";|,|\band\b", m.group(1)):
            s = part.strip(" .:-").lower()
            if 3 <= len(s) <= 80 and s not in out and not s.startswith("we work") and not re.search(r"https?:|\[|\]", s):
                out.append(s)
    return out[:limit]


def find_size(text: str) -> str:
    m = SIZE_RE.search(text)
    return f"{m.group(1)} {m.group(2).lower()}" if m else "not stated"


def extract_fields(text: str) -> dict:
    """Extract name, country, field, services, and size (only when stated)."""
    field, scores = classify_field(text)
    return {"name": find_name(text), "country": find_country(text), "field": field,
            "services": find_services(text), "size": find_size(text), "field_scores": scores}
