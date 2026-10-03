"""Proposal assembly: prose for each section is written from its verified fact sheet.

Factual sentences come only from fact-sheet claims (each cited as [S#] to a fetched source). Planning
content (objectives, methodology, timeline, budget…) is proposal language, explicitly derived from the
selected projects, never presented as fact about the company. Sections whose fact sheet could not be
verified say "Not verified".

Writers:
  * TemplateWriter — deterministic composition (fallback; also generates SFT targets for section writing).
  * ModelWriter    — our model writes the prose from the fact sheet; a grounding check replaces any
                     sentence with unsupported numbers or names by "Not verified".
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from agent.srlm.engine import NOT_VERIFIED
from agent.srlm.tasks import FACT_SHEET_SECTIONS, PROPOSAL_SECTIONS

NV = "Not verified"


class SourceBook:
    """Assigns stable [S#] labels to source ids/URLs."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def cite(self, source_id: str, url: str = "", title: str = "") -> str:
        if source_id not in self.items:
            self.items[source_id] = {"label": f"S{len(self.items) + 1}", "id": source_id, "url": url, "title": title}
        elif url and not self.items[source_id]["url"]:
            self.items[source_id]["url"] = url
        return self.items[source_id]["label"]

    def as_list(self) -> list[dict]:
        return list(self.items.values())


def _size_number(size: str) -> int | None:
    m = re.search(r"(\d[\d,]*)", size or "")
    return int(m.group(1).replace(",", "")) if m else None


def budget_range(size: str, n_projects: int) -> tuple[str, str]:
    """Indicative budget band (planning assumption) from the company's stated size."""
    n = _size_number(size)
    if n is None:
        return "EUR 60,000 – 180,000", "company size not stated; mid-range band assumed"
    if n < 50:
        lo, hi = 40, 90
    elif n < 250:
        lo, hi = 90, 220
    else:
        lo, hi = 220, 600
    k = max(1, min(n_projects, 3))
    scale = 1 + 0.5 * (k - 1)          # each additional project adds half a base budget
    return (f"EUR {int(lo * scale):,},000 – {int(hi * scale):,},000",
            f"based on stated size ({size}) and {k} project(s)")


def _claims(sheet: Any) -> list[dict]:
    if not isinstance(sheet, dict):
        return []
    return [c for c in sheet.get("claims", []) if isinstance(c, dict) and c.get("text")]


def _sentence(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if t.endswith((".", "!", "?")) else t + "."


class TemplateWriter:
    name = "template"

    def facts_paragraph(self, sheet: Any, book: SourceBook, corpus_urls: dict[str, str], limit: int = 5) -> list[str]:
        out = []
        for c in _claims(sheet)[:limit]:
            labels = ", ".join(book.cite(s, corpus_urls.get(s, "")) for s in c["sources"])
            out.append(f"{_sentence(c['text'])} [{labels}]")
        return out

    def write(self, ctx: dict) -> list[dict]:
        """Return a list of section dicts: {title, paragraphs, bullets, table, verified}."""
        profile, projects, sheets = ctx["profile"], ctx["projects"], ctx["fact_sheets"]
        book: SourceBook = ctx["book"]
        urls = ctx["corpus_urls"]
        name = profile.get("name") or ctx["domain"]
        field = profile.get("field", "unknown")
        proj_ok = isinstance(projects, list) and projects
        titles = [p["title"] for p in projects] if proj_ok else []
        main = titles[0] if titles else NV
        sections: list[dict] = []

        def facts(sec: str, limit: int = 5) -> list[str]:
            return self.facts_paragraph(sheets.get(sec), book, urls, limit)

        def sec(title: str, paragraphs=None, bullets=None, table=None, verified=True):
            sections.append({"title": title, "paragraphs": paragraphs or [], "bullets": bullets or [],
                             "table": table, "verified": verified})

        def nv(title: str) -> None:
            sec(title, [f"{NV}: no grounded information could be verified for this section."], verified=False)

        today = dt.date.today().isoformat()
        sec("Cover", [f"Project Proposal for {name}", f"Field: {field} · Country: {profile.get('country', NV)}",
                      f"Prepared {today} by ProposalAgent (automated research; facts cite public sources)"])
        for title in PROPOSAL_SECTIONS[1:-1]:
            sheet = sheets.get(title)
            if title in FACT_SHEET_SECTIONS and (sheet is None or sheet == NOT_VERIFIED):
                nv(title)
                continue
            f = facts(title)
            if title == "Executive Summary":
                sec(title, [f"{name} works in {field}. This proposal recommends: {main}."
                            + (f" Further options: {'; '.join(titles[1:])}." if len(titles) > 1 else "")] + f[:3])
            elif title == "Company Overview":
                sec(title, f or [f"{NV}: the company's own pages did not state these details."],
                    table={"headers": ["Item", "Value"], "rows": [
                        ["Name", name], ["Country", profile.get("country", NV)], ["Field", field],
                        ["Size", profile.get("size") or "not stated"],
                        ["Services", "; ".join(profile.get("services", [])[:5]) or NV]]})
            elif title == "Field and Market Analysis":
                if f:
                    sec(title, [f"Public sources on {field} describe the following trends and existing work:"], f)
                else:
                    nv(title)
            elif title == "Problem/Opportunity Analysis":
                if f:
                    sec(title, ["The evidence points to these unsolved problems and opportunities:"], f)
                else:
                    nv(title)
            elif title == "Proposed Project(s)":
                rows = [[str(i + 1), p["title"], _sentence(p.get("rationale", "")) + " [" + ", ".join(
                    book.cite(e, urls.get(e, "")) for e in p.get("evidence_ids", [])) + "]"]
                        for i, p in enumerate(projects or [])] if proj_ok else []
                sec(title, [f"Recommended project: {main}." if proj_ok else f"{NV}: no project could be selected."] + f[:2],
                    table={"headers": ["#", "Project", "Rationale (evidence)"], "rows": rows} if rows else None,
                    verified=bool(proj_ok))
            elif title == "Objectives":
                sec(title, ["The project objectives (proposed, to be agreed with the client):"],
                    [f"Deliver a working solution for: {t}." for t in titles[:3]] +
                    ["Measure baseline and post-project KPIs agreed in the discovery phase.",
                     "Transfer knowledge so the team can operate the solution independently."] + f[:2])
            elif title == "Scope":
                svc = profile.get("services", [])[:4]
                sec(title, ["In scope: " + (", ".join(svc) if svc else "the operations described on the company's website")
                            + f", as they relate to {main}.",
                            "Out of scope: changes to unrelated business units, hardware procurement and long-term operations "
                            "beyond the hand-over period."] + f[:2])
            elif title == "Methodology":
                sec(title, ["A phased, evidence-driven approach:"],
                    ["Discovery: interviews, data inventory and KPI baseline.",
                     f"Design: solution architecture for {main}, reviewed with stakeholders.",
                     "Build: iterative delivery in two-week sprints with demos.",
                     "Pilot: limited roll-out, measurement against the baseline.",
                     "Hand-over: documentation, training and support plan."] + f[:2])
            elif title == "Timeline and Milestones":
                sec(title, ["Indicative plan (proposal assumption):"],
                    table={"headers": ["Phase", "Weeks", "Milestone"], "rows": [
                        ["Discovery", "1–3", "Baseline report"], ["Design", "4–6", "Approved architecture"],
                        ["Build", "7–14", "Feature-complete release"], ["Pilot", "15–18", "Pilot results vs. KPIs"],
                        ["Hand-over", "19–20", "Operations hand-over"]]})
            elif title == "Deliverables":
                sec(title, [], [f"{t}: production-ready implementation and documentation." for t in titles[:3]] +
                    ["Baseline and pilot KPI reports.", "Training material and operations runbook."])
            elif title == "Team and Resources":
                sec(title, [f"Client side: a sponsor and domain experts from {name}" +
                            (f" (the company reports {profile['size']})." if profile.get("size") not in (None, "", "not stated") else ".")],
                    ["Project manager (0.5 FTE)", "Solution architect (0.5 FTE)", "Two engineers (2.0 FTE)",
                     "Data/analytics specialist (0.5 FTE)"] + f[:2])
            elif title == "Risks and Mitigation":
                rows = [["Data availability or quality is lower than expected", "Data audit in discovery; scope adjusted early"],
                        ["Adoption by staff", "Co-design with users; training; phased roll-out"],
                        ["Integration with existing systems", "Interface review in design; pilot on one site first"]]
                for c in _claims(sheet)[:2]:
                    rows.append([_sentence(c["text"]), "Addressed explicitly in design and pilot scope"])
                sec(title, f[:2], table={"headers": ["Risk", "Mitigation"], "rows": rows})
            elif title == "Budget Range":
                band, basis = budget_range(profile.get("size", ""), len(titles))
                sec(title, [f"Indicative budget: {band} (planning estimate, {basis}; not a quotation).",
                            "Final pricing follows the discovery phase."])
            elif title == "Expected Impact":
                sec(title, ["Expected benefits, supported by the cited evidence:"], f or
                    [f"{NV}: no quantified impact evidence was found; impact will be measured against the baseline."])
            elif title == "Conclusion":
                sec(title, [f"{name} is well placed to benefit from {main}. We propose to start with a three-week "
                            "discovery phase to confirm scope, data and KPIs."] + f[:1])
        refs = [f"[{s['label']}] {s['title'] or s['id']} — {s['url'] or 'company website'}" for s in book.as_list()]
        sec("Sources", ["Every factual statement above cites one of these public sources:"], refs or [NV])
        return sections


# Grounding check used for model-written prose.
_NUM = re.compile(r"\d[\d,.]*")
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")


def ungrounded_sentences(text: str, allowed_text: str) -> list[str]:
    """Sentences with numbers or proper names absent from the fact sheet / profile text."""
    bad = []
    allowed_low = allowed_text.lower()
    for s in re.split(r"(?<=[.!?])\s+", text):
        nums = [n.strip(".,") for n in _NUM.findall(s)]
        names = [n for i, n in enumerate(_PROPER.findall(s)) if not (i == 0 and s.startswith(n))]
        if any(n and n not in allowed_text for n in nums) or any(n.lower() not in allowed_low for n in names):
            bad.append(s)
    return bad


class ModelWriter(TemplateWriter):
    """Our model rewrites the factual paragraph of each section from its fact sheet."""

    name = "model"

    def __init__(self, model_policy, max_tokens: int = 200):
        self.policy = model_policy
        self.max_tokens = max_tokens

    def facts_paragraph(self, sheet: Any, book: SourceBook, corpus_urls: dict[str, str], limit: int = 5) -> list[str]:
        base = super().facts_paragraph(sheet, book, corpus_urls, limit)
        if not base:
            return base
        import torch
        tok = self.policy.tok
        facts = "\n".join(base)
        prompt = f"<|user|> WRITE SECTION from these cited facts only:\n{facts}\n<|step|><|thought|>"
        ids = torch.tensor([tok.encode(prompt)[-(self.policy.model.cfg.max_seq_len - self.max_tokens - 1):]])
        out = self.policy.model.generate(ids, self.max_tokens, temperature=0.3, top_p=0.9,
                                         stop_ids={tok.special["<|end|>"], tok.eos_id})
        text = tok.decode(out, skip_special=True).strip()
        bad = ungrounded_sentences(text, facts)
        if not text or bad:
            return base  # fall back to the verified sentences rather than publishing unsupported claims
        return [text]


def build_record(domain: str, url: str, profile: dict, projects: Any, evidence: list[dict], sections: list[dict],
                 decisions: dict, book: SourceBook) -> dict:
    """JSON record saved next to the .docx."""
    return {"domain": domain, "url": url, "profile": profile, "projects": projects,
            "evidence": evidence, "sources": book.as_list(), "sections": sections, "decisions": decisions,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
