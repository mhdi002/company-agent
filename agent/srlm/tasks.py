"""Decision units (spec §5.6): task builders, output validators (grounding), and ground-truth scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.srlm.canonical import keyfacts, norm_str
from tools.field_classifier import FIELDS

PROPOSAL_SECTIONS = ["Cover", "Executive Summary", "Company Overview", "Field and Market Analysis",
                     "Problem/Opportunity Analysis", "Proposed Project(s)", "Objectives", "Scope", "Methodology",
                     "Timeline and Milestones", "Deliverables", "Team and Resources", "Risks and Mitigation",
                     "Budget Range", "Expected Impact", "Conclusion", "Sources"]
# Sections whose content is assembled mechanically (no fact sheet needed).
MECHANICAL_SECTIONS = {"Cover", "Sources"}
FACT_SHEET_SECTIONS = [s for s in PROPOSAL_SECTIONS if s not in MECHANICAL_SECTIONS]

SECTION_QUERIES: dict[str, str] = {
    "Executive Summary": "company offers services customers opportunity project",
    "Company Overview": "founded based company employees customers services products offers",
    "Field and Market Analysis": "trend market increasingly expected growth adoption standards demand increases reduces widely already",
    "Problem/Opportunity Analysis": "few limited still bottleneck barrier remains manual spreadsheets lack",
    "Proposed Project(s)": "project automated forecasting integration monitoring platform scheduling",
    "Objectives": "reduce improve increase savings efficiency downtime waste",
    "Scope": "services products operations sites customers systems data",
    "Methodology": "data monitoring sensors platform integration analytics standards",
    "Timeline and Milestones": "project since founded years operations",
    "Deliverables": "platform dashboard system portal app products",
    "Team and Resources": "employees team staff engineers experience",
    "Risks and Mitigation": "barrier bottleneck risk adoption fatigue regulation requirements",
    "Budget Range": "employees capacity customers projects scale",
    "Expected Impact": "reduces saves improves increases revenue savings downtime",
    "Conclusion": "company services opportunity project customers",
}

GAP_RE = re.compile(r"\b(few|still|limited|bottleneck|barrier|remains|lack|manual|spreadsheets|whiteboards)\b", re.I)


@dataclass
class DecisionTask:
    name: str                     # field_extraction | evidence_extraction | project_selection | fact_sheet
    unit_id: str                  # unique id, e.g. "nordwind-solar.example/field"
    query: str
    variables: dict
    kind: str                     # label | set | free_text
    validate: Callable[[Any], str | None]
    label_fn: Callable[[Any], str] | None = None
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- field extraction
def _field_label(o: Any) -> str:
    return f"{norm_str(o.get('field', ''))}|{norm_str(o.get('country', ''))}"


def _validate_field(o: Any) -> str | None:
    if not isinstance(o, dict):
        return "FINAL must be a dict with keys field, country, services"
    if not isinstance(o.get("field"), str) or not o["field"]:
        return "missing 'field'"
    if o["field"] != "unknown" and norm_str(o["field"]) not in FIELDS:
        return f"'field' must be one of {sorted(FIELDS)} or 'unknown'"
    if not isinstance(o.get("services", []), list):
        return "'services' must be a list"
    return None


def field_task(domain: str, pages: dict[str, str]) -> DecisionTask:
    context = "\n\n".join(f"[{u}]\n{t}" for u, t in pages.items())
    return DecisionTask(
        "field_extraction", f"{domain}/field",
        f"Identify the working field of the company at {domain}: return FINAL({{'field', 'country', 'services', "
        f"'name', 'size'}}). field must be one of: {', '.join(sorted(FIELDS))} or 'unknown'.",
        {"context": context, "pages": pages}, "label", _validate_field, _field_label, {"domain": domain})


# ----------------------------------------------------------------------------- evidence extraction
def _grounded(text: str, source: str) -> bool:
    a, b = norm_str(text), norm_str(source)
    if a and a in b:
        return True
    ka = keyfacts(text)
    return bool(ka) and len(ka & keyfacts(source)) / len(ka) >= 0.9


def make_evidence_validator(evidence: list[dict]) -> Callable[[Any], str | None]:
    by_id = {e["id"]: e for e in evidence}

    def v(o: Any) -> str | None:
        if not isinstance(o, list) or not o:
            return "FINAL must be a non-empty list of {'fact', 'source_id', 'url'}"
        for i, f in enumerate(o):
            if not isinstance(f, dict) or not {"fact", "source_id"} <= set(f):
                return f"item {i} must have 'fact' and 'source_id'"
            src = by_id.get(f["source_id"])
            if src is None:
                return f"item {i}: unknown source_id {f['source_id']!r}"
            if not _grounded(f["fact"], src["text"]):
                return f"item {i}: fact is not found in source {f['source_id']} (must be quoted from the text)"
        return None
    return v


def evidence_task(domain: str, profile: dict, evidence: list[dict]) -> DecisionTask:
    return DecisionTask(
        "evidence_extraction", f"{domain}/evidence",
        "Extract the facts relevant to this company's field and services from `evidence`: existing work, trends "
        "and gaps. FINAL(list of {'fact': quoted sentence, 'source_id', 'url'}). Facts must be quoted verbatim.",
        {"evidence": evidence, "profile": profile}, "set", make_evidence_validator(evidence), meta={"domain": domain})


# ----------------------------------------------------------------------------- project selection
def _project_label(o: Any) -> str:
    titles = sorted(norm_str(p.get("title", "")) for p in o[:3] if isinstance(p, dict))
    return "|".join(titles)


def make_project_validator(facts: list[dict]) -> Callable[[Any], str | None]:
    ids = {f["source_id"] for f in facts}

    def v(o: Any) -> str | None:
        if not isinstance(o, list) or not o:
            return "FINAL must be a non-empty ranked list of {'title', 'rationale', 'evidence_ids'}"
        for i, p in enumerate(o):
            if not isinstance(p, dict) or not p.get("title"):
                return f"item {i} needs a 'title'"
            ev = p.get("evidence_ids", [])
            if not isinstance(ev, list) or not ev or any(e not in ids for e in ev):
                return f"item {i}: evidence_ids must reference source_ids from facts"
        return None
    return v


def project_task(domain: str, profile: dict, facts: list[dict]) -> DecisionTask:
    return DecisionTask(
        "project_selection", f"{domain}/projects",
        "Select the projects this company most likely needs, ranked, from the gaps and trends in `facts`. "
        "FINAL(list of {'title', 'rationale', 'evidence_ids'}), best first, at most 3.",
        {"facts": facts, "profile": profile}, "label", make_project_validator(facts), _project_label,
        {"domain": domain})


# ----------------------------------------------------------------------------- fact sheets
def build_corpus(pages: dict[str, str], facts: list[dict], projects: list[dict]) -> list[dict]:
    corpus = [{"id": f"page:{u}", "url": u, "text": t} for u, t in pages.items()]
    seen = set()
    for f in facts:
        key = (f["source_id"], f["fact"])
        if key not in seen:
            seen.add(key)
            corpus.append({"id": f["source_id"], "url": f.get("url", ""), "text": f["fact"]})
    # Selected projects are not public sources: their rationale facts are already cited via `facts`.
    # merge texts of identical ids (several facts from the same source)
    merged: dict[str, dict] = {}
    for d in corpus:
        if d["id"] in merged:
            merged[d["id"]]["text"] += " " + d["text"]
        else:
            merged[d["id"]] = dict(d)
    return list(merged.values())


def make_fact_sheet_validator(corpus: list[dict], section: str) -> Callable[[Any], str | None]:
    by_id = {d["id"]: d for d in corpus}

    def v(o: Any) -> str | None:
        if not isinstance(o, dict) or not isinstance(o.get("claims"), list):
            return "FINAL must be {'section', 'claims': [{'text', 'sources'}]}"
        if o.get("section") != section:
            return f"'section' must be {section!r}"
        for i, c in enumerate(o["claims"]):
            if not isinstance(c, dict) or not c.get("text") or not isinstance(c.get("sources"), list) or not c["sources"]:
                return f"claim {i} needs 'text' and non-empty 'sources'"
            if not any(s in by_id and _grounded(c["text"], by_id[s]["text"]) for s in c["sources"]):
                return f"claim {i} is not grounded in its sources"
        return None
    return v


def fact_sheet_task(domain: str, section: str, profile: dict, corpus: list[dict]) -> DecisionTask:
    return DecisionTask(
        "fact_sheet", f"{domain}/factsheet/{section}",
        f"Gather the grounded facts for the proposal section '{section}' (keywords: {SECTION_QUERIES[section]}). "
        f"FINAL({{'section': '{section}', 'claims': [{{'text': quoted sentence, 'sources': [doc ids]}}]}}). "
        "Only quote sentences from `corpus`.",
        {"corpus": corpus, "profile": profile, "section": section, "keywords": SECTION_QUERIES[section]}, "set",
        make_fact_sheet_validator(corpus, section), meta={"domain": domain, "section": section})


# ----------------------------------------------------------------------------- ground-truth scoring (synth + eval)
def score_field(out: Any, truth: dict) -> float:
    if not isinstance(out, dict):
        return 0.0
    f = 1.0 if norm_str(out.get("field")) == norm_str(truth["field"]) else 0.0
    c = 1.0 if norm_str(out.get("country")) == norm_str(truth["country"]) else 0.0
    return 0.7 * f + 0.3 * c


def score_evidence(out: Any, relevant: set[str]) -> float:
    """F1 of returned facts vs. relevant sentences (normalized)."""
    if not isinstance(out, list) or not out:
        return 0.0
    got = {norm_str(f.get("fact", "")) for f in out if isinstance(f, dict)}
    rel = {norm_str(r) for r in relevant}
    tp = len(got & rel)
    if not tp:
        return 0.0
    p, r = tp / len(got), tp / len(rel)
    return 2 * p * r / (p + r)


def score_projects(out: Any, gap_facts: set[str]) -> float:
    """1.0 if the top project comes from a true gap of the company's field; partial for the rest of top-3."""
    if not isinstance(out, list) or not out:
        return 0.0
    gaps = [keyfacts(g) for g in gap_facts]

    def from_gap(p: dict) -> bool:
        k = keyfacts(p.get("title", "") + " " + p.get("rationale", ""))
        return any(len(k & g) / max(len(g), 1) >= 0.5 for g in gaps)
    top = 1.0 if from_gap(out[0]) else 0.0
    rest = sum(from_gap(p) for p in out[1:3]) / max(len(out[1:3]), 1) if len(out) > 1 else 0.0
    return 0.7 * top + 0.3 * rest


def score_fact_sheet(out: Any, section: str) -> float:
    if not isinstance(out, dict) or not out.get("claims"):
        return 0.0
    kws = set(SECTION_QUERIES[section].split())
    rel = [bool(keyfacts(c["text"]) & kws) for c in out["claims"]]
    return sum(rel) / len(rel)
