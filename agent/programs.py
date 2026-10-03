"""Library of programmatic REPL programs for each decision unit.

These are used in two places:
  * agent/synth: to generate synthetic SFT traces (good, flawed, and recovering programs) whose
    confidence targets are then calibrated against ground truth;
  * agent/policy.TemplatePolicy: the deterministic fallback policy when no trained checkpoint exists.

A strategy is a list of step functions `f(history) -> (thought, code, confidence)`. `history` is the list of
previous {"text", "observation"} items of this program. Confidences here are runtime heuristics derived
from what the program has observed; synthetic data replaces them with calibrated targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Step = Callable[[list], tuple[str, str, float]]


def _last(history: list) -> str:
    return history[-1]["observation"] if history else ""


def _c(text: str, good: float, bad: float, markers: tuple[str, ...]) -> float:
    return bad if any(m in text for m in markers) else good


FIELD_FINAL = ('FINAL({"field": info["field"], "country": info["country"], "services": info["services"], '
               '"name": info["name"], "size": info["size"]})')


@dataclass
class Strategy:
    name: str
    steps: list[Step]
    weight: float = 1.0
    flawed: bool = False


# ============================================================================ field extraction
FIELD = [
    Strategy("extract_all", [
        lambda h: ("Run the field extractor over the full site text held in `context`.",
                   "info = extract_fields(context)\nprint(info)", 70.0),
        lambda h: ("The extractor returned a label, country and services; return them.", FIELD_FINAL,
                   _c(_last(h), 88.0, 20.0, ("'field': 'unknown'", "[error]"))),
    ], 2.0),
    Strategy("about_services", [
        lambda h: ("Locate the home/about pages and the services/products pages.",
                   'about = [u for u in pages if "about" in u or u.rstrip("/").count("/") == 2]\n'
                   'svc = [u for u in pages if "service" in u or "product" in u]\nprint(about, svc)', 75.0),
        lambda h: ("Classify from the about pages and read services from the services pages.",
                   "info = extract_fields({u: pages[u] for u in about})\n"
                   'services = extract_fields({u: pages[u] for u in svc})["services"]\n'
                   'print(info["field"], info["country"], services)', 72.0),
        lambda h: ("Combine both readings into the final answer.",
                   'FINAL({"field": info["field"], "country": info["country"], "services": services or '
                   'info["services"], "name": info["name"], "size": info["size"]})',
                   _c(_last(h), 85.0, 25.0, ("unknown", "[error]"))),
    ]),
    Strategy("bm25", [
        lambda h: ("Rank page chunks by relevance to industry, services and customers.",
                   'top = bm25("company industry services products customers", k=4, var="pages")\n'
                   'for h in top:\n    print(h["source"], h["score"], h["text"][:160])', 70.0),
        lambda h: ("Classify the top chunks and, as a check, the whole site.",
                   'info = extract_fields(" ".join(h["text"] for h in top))\nfull = extract_fields(context)\n'
                   'print(info["field"], full["field"], info["country"], full["country"])', 74.0),
        lambda h: ("Prefer the chunk-level label; fall back to the full-site reading when it is unknown.",
                   'field = info["field"] if info["field"] != "unknown" else full["field"]\n'
                   'country = info["country"] if info["country"] != "unknown" else full["country"]\n'
                   'FINAL({"field": field, "country": country, "services": full["services"], "name": full["name"], '
                   '"size": full["size"]})',
                   _c(_last(h), 86.0, 30.0, ("unknown unknown", "[error]"))),
    ]),
    Strategy("regex", [
        lambda h: ("Search the pages for service and company-description phrases.",
                   'hits = search_context(r"(?i)(our services|we offer|products|founded)", var="pages", window=200)\n'
                   'print(len(hits))\nfor h in hits[:4]:\n    print(h["source"], h["snippet"][:200])', 68.0),
        lambda h: ("Classify the matched snippets together with the full text.",
                   'info = extract_fields(" ".join(h["snippet"] for h in hits) + "\\n" + context)\nprint(info)', 72.0),
        lambda h: ("Return the classification.", FIELD_FINAL,
                   _c(_last(h), 84.0, 20.0, ("'field': 'unknown'", "[error]"))),
    ]),
    Strategy("wrong_slice", [
        lambda h: ("Read the beginning of the context; it should describe the company.",
                   "head = slice(context, 0, 160)\nprint(head)", 60.0),
        lambda h: ("Classify the slice.", "info = extract_fields(head)\nprint(info)", 55.0),
        lambda h: ("Return the classification of the slice.", FIELD_FINAL,
                   _c(_last(h), 60.0, 15.0, ("'field': 'unknown'",))),
    ], 0.6, True),
    Strategy("missed_page", [
        lambda h: ("Use the contact page; it names the company and location.",
                   'last = [u for u in pages if "contact" in u] or list(pages)[-1:]\nprint(last)', 55.0),
        lambda h: ("Classify the contact page.", "info = extract_fields(pages[last[0]])\n" + FIELD_FINAL, 50.0),
    ], 0.5, True),
    Strategy("buggy_call", [
        lambda h: ("Extract the fields from the context.", "info = extract_field(context)\nprint(info)", 65.0),
    ], 0.4, True),
]

# ============================================================================ evidence extraction
EVIDENCE = [
    Strategy("field_docs", [
        lambda h: ("Keep the evidence documents whose own field matches the company's field.",
                   'rel = [e for e in evidence if extract_fields(e.get("title", "") + " " + e["text"])["field"] == profile["field"]]\n'
                   'print([(e["id"], e.get("title", "")) for e in rel])', 72.0),
        lambda h: ("Quote every sentence of the relevant documents as a fact with its source.",
                   'facts = []\nfor e in rel:\n    for s in sentences(e["text"]):\n'
                   '        facts.append({"fact": s["text"], "source_id": e["id"], "url": e["url"]})\n'
                   'print(len(facts))\nprint(facts[:2])', 75.0),
        lambda h: ("Return the quoted facts.", "FINAL(facts[:12])",
                   _c(_last(h), 84.0, 15.0, ("\n0\n", "[error]", "[]"))),
    ], 2.0),
    Strategy("overlap", [
        lambda h: ("Split all evidence into sentences and build a query from field and services.",
                   'q = profile["field"] + " " + " ".join(profile.get("services", []))\nsents = sentences("evidence")\n'
                   'print(len(sents), "sentences;", q)', 70.0),
        lambda h: ("Rank sentences by word overlap with the query and keep the relevant ones.",
                   'urls = {e["id"]: e["url"] for e in evidence}\n'
                   'keep = [s for s in sorted(sents, key=lambda s: -overlap(s["text"], q)) if overlap(s["text"], q) > 0][:8]\n'
                   'for s in keep:\n    print(s["source"], round(overlap(s["text"], q), 3), s["text"][:120])', 72.0),
        lambda h: ("Return the ranked facts with their sources.",
                   'FINAL([{"fact": s["text"], "source_id": s["source"], "url": urls[s["source"]]} for s in keep])',
                   _c(_last(h), 78.0, 20.0, ("[error]",))),
    ]),
    Strategy("bm25", [
        lambda h: ("Retrieve the evidence chunks most relevant to the company profile.",
                   'q = profile["field"] + " " + " ".join(profile.get("services", []))\n'
                   'same = [e for e in evidence if extract_fields(e["text"])["field"] == profile["field"]] or evidence\n'
                   'hits = bm25(q, k=6, var=same)\nfor h in hits:\n    print(h["source"], h["score"], h["text"][:100])',
                   70.0),
        lambda h: ("Quote the sentences of the retrieved chunks.",
                   'urls = {e["id"]: e["url"] for e in evidence}\n'
                   'facts = [{"fact": s["text"], "source_id": h["source"], "url": urls[h["source"]]} '
                   'for h in hits for s in sentences(h["text"])]\nFINAL(facts[:12])',
                   _c(_last(h), 76.0, 20.0, ("[]", "[error]"))),
    ]),
    Strategy("paraphrase", [
        lambda h: ("Summarise the market in one sentence.",
                   'FINAL([{"fact": "The " + profile["field"] + " market is growing quickly.", '
                   '"source_id": evidence[0]["id"], "url": evidence[0]["url"]}])', 70.0),
    ], 0.5, True),
    Strategy("first_doc", [
        lambda h: ("The first evidence document should be the most relevant; quote it.",
                   'e = evidence[0]\nFINAL([{"fact": s["text"], "source_id": e["id"], "url": e["url"]} '
                   'for s in sentences(e["text"])])', 60.0),
    ], 0.5, True),
]

# ============================================================================ project selection
GAP_PATTERN = r'(?i)\b(few|still|limited|bottleneck|barrier|remains|lack|manual|spreadsheets|whiteboards)\b'
PROJECTS = [
    Strategy("gaps_first", [
        lambda h: ("Separate gap statements (unsolved problems) from general trends.",
                   "import re\n"
                   f'gaps = [f for f in facts if re.search(r"{GAP_PATTERN}", f["fact"])]\n'
                   'trends = [f for f in facts if f not in gaps]\nprint(len(gaps), "gaps;", len(trends), "trends")\n'
                   'for g in gaps:\n    print("-", g["fact"][:140])', 74.0),
        lambda h: ("Turn gaps into project ideas first, then trends ranked by fit with the company's services.",
                   'svc = " ".join(profile.get("services", [])) + " " + profile["field"]\n'
                   'ranked = gaps + sorted(trends, key=lambda f: -overlap(f["fact"], svc))\n'
                   'cands = [{"title": project_from_gap(f["fact"]), "rationale": f["fact"], '
                   '"evidence_ids": [f["source_id"]]} for f in ranked]\nprint([c["title"] for c in cands[:5]])', 78.0),
        lambda h: ("Return the top three projects.", "FINAL(cands[:3])",
                   _c(_last(h), 82.0, 20.0, ("[]", "[error]"))),
    ], 2.0),
    Strategy("fit_ranked", [
        lambda h: ("Score every fact by fit with the company's services, with a bonus for gap statements.",
                   "import re\nsvc = \" \".join(profile.get(\"services\", [])) + \" \" + profile[\"field\"]\n"
                   f'score = lambda f: overlap(f["fact"], svc) + (0.5 if re.search(r"{GAP_PATTERN}", f["fact"]) else 0)\n'
                   'ranked = sorted(facts, key=lambda f: -score(f))\nfor f in ranked[:5]:\n'
                   '    print(round(score(f), 3), f["fact"][:120])', 72.0),
        lambda h: ("Return the three best-scoring ideas as projects.",
                   'FINAL([{"title": project_from_gap(f["fact"]), "rationale": f["fact"], '
                   '"evidence_ids": [f["source_id"]]} for f in ranked[:3]])',
                   _c(_last(h), 80.0, 20.0, ("[error]",))),
    ]),
    Strategy("trends_only", [
        lambda h: ("Use the most recent trends as project ideas.",
                   'FINAL([{"title": project_from_gap(f["fact"]), "rationale": f["fact"], '
                   '"evidence_ids": [f["source_id"]]} for f in facts[:3]])', 65.0),
    ], 0.6, True),
]

# ============================================================================ fact sheets
FACT_SHEET = [
    Strategy("bm25", [
        lambda h: ("Retrieve corpus passages that match the section keywords.",
                   'hits = bm25(keywords, k=6, var="corpus")\nfor h in hits:\n'
                   '    print(h["source"], h["score"], h["text"][:120])', 72.0),
        lambda h: ("Quote the sentences that contain section keywords, with their document ids.",
                   'claims = []\nfor h in hits:\n    for s in sentences(h["text"]):\n'
                   '        if overlap(s["text"], keywords) > 0 and len(claims) < 6:\n'
                   '            claims.append({"text": s["text"], "sources": [h["source"]]})\nprint(len(claims))', 75.0),
        lambda h: ("Return the fact sheet.", 'FINAL({"section": section, "claims": claims})',
                   _c(_last(h), 82.0, 25.0, ("\n0", "[error]"))),
    ], 2.0),
    Strategy("regex", [
        lambda h: ("Build a keyword regex and count matching passages.",
                   'import re\npat = "(?i)\\\\b(" + "|".join(keywords.split()) + ")\\\\b"\n'
                   'hits = search_context(pat, var="corpus", window=0, max_hits=40)\nprint(len(hits))', 70.0),
        lambda h: ("Collect distinct matching sentences with their sources and return them.",
                   'claims, seen = [], set()\nfor s in sentences("corpus"):\n'
                   '    if re.search(pat, s["text"]) and s["text"] not in seen and len(claims) < 6:\n'
                   '        seen.add(s["text"])\n        claims.append({"text": s["text"], "sources": [s["source"]]})\n'
                   'FINAL({"section": section, "claims": claims})',
                   _c(_last(h), 78.0, 25.0, ("\n0", "[error]"))),
    ]),
    Strategy("unsourced", [
        lambda h: ("Write the key claims for this section.",
                   'FINAL({"section": section, "claims": [{"text": profile["name"] + " needs this project.", '
                   '"sources": ["unknown"]}]})', 70.0),
    ], 0.5, True),
]

LIBRARY: dict[str, list[Strategy]] = {
    "field_extraction": FIELD, "evidence_extraction": EVIDENCE,
    "project_selection": PROJECTS, "fact_sheet": FACT_SHEET,
}

RECOVERY_THOUGHT = "The previous step failed ({err}). Switching to a more reliable approach."
