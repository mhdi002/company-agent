"""Generate synthetic SRLM agent traces for supervised agent-format training — in code, no external LLMs.

Episodes are built around company websites with known ground truth: programmatically generated
companies whose pages embed real cleaned text (background paragraphs from `data/cleaned`), plus field
evidence documents with distractors. For every decision unit we sample K candidate programs at
temperature from the program library (good, flawed, and recovering programs), EXECUTE them in the
sandboxed REPL, score the final output against ground truth, and set each step's confidence target from
actual correctness (calibration). Each step becomes one SFT example: prompt (as seen at runtime) → step.

Usage:
    python -m agent.synth.generate --companies 400 --k 4 --out data/sft
The bundled fixture companies (data/sample/companies.py) are always held out for evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from agent.policy import TemplatePolicy
from agent.programs import LIBRARY
from agent.srlm.engine import CandidateTrace, SRLMEngine
from agent.srlm.helpers import HELPER_DOC, project_title
from agent.srlm.protocol import build_prompt, render_step
from agent.srlm.tasks import (FACT_SHEET_SECTIONS, GAP_RE, build_corpus, evidence_task, fact_sheet_task, field_task,
                              project_task, score_evidence, score_fact_sheet, score_field, score_projects)
from core.config import load_config, resolve
from core.logging import get_logger
from data.sample.companies import COMPANIES, FIELD_KB, company_pages
from tools.extract import html_to_text

log = get_logger("training")

SERVICE_BANK = {
    "renewable energy": ["solar park development", "rooftop PV installation", "battery storage integration",
                         "wind farm operation", "grid connection studies", "O&M services", "energy yield assessment"],
    "logistics": ["container drayage", "bonded warehousing", "customs brokerage", "last-mile delivery",
                  "rail freight forwarding", "contract logistics", "cold chain transport"],
    "software": ["custom web application development", "cloud migration", "mobile app development",
                 "QA automation", "API integration", "SaaS product development"],
    "agriculture": ["soil moisture sensing", "precision irrigation design", "drone crop scouting",
                    "farm management software", "greenhouse climate control"],
    "manufacturing": ["precision CNC machining", "sheet metal fabrication", "contract assembly",
                      "industrial robot integration", "machine vision inspection", "surface treatment"],
    "healthcare": ["remote patient monitoring", "clinical data integration", "telehealth platform operation",
                   "medical device distribution", "diagnostic imaging services"],
    "finance": ["credit risk modelling", "regulatory reporting automation", "payment processing",
                "financial data engineering", "insurance claims analytics"],
    "construction": ["commercial general contracting", "design-build", "prefabricated timber framing",
                     "civil engineering works", "building renovation"],
    "education": ["online vocational courses", "corporate training programs", "learning management system hosting",
                  "exam preparation courses", "curriculum design"],
    "retail": ["grocery retail", "online grocery delivery", "private label sourcing", "e-commerce fulfilment",
               "store merchandising"],
}
CITIES = {"Germany": ["Munich", "Leipzig", "Bremen"], "United Kingdom": ["Leeds", "Bristol", "Glasgow"],
          "Canada": ["Vancouver", "Ottawa", "Halifax"], "Australia": ["Perth", "Adelaide", "Brisbane"],
          "Netherlands": ["Utrecht", "Groningen", "Delft"], "Singapore": ["Singapore"],
          "United States": ["Denver", "Austin", "Columbus"], "France": ["Lyon", "Nantes", "Lille"],
          "Sweden": ["Malmö", "Uppsala"], "India": ["Pune", "Chennai"], "Ireland": ["Cork", "Galway"]}
SUFFIX = {"Germany": "GmbH", "United Kingdom": "Ltd", "Canada": "Inc.", "Australia": "Pty Ltd",
          "Netherlands": "B.V.", "Singapore": "Pte Ltd", "United States": "LLC", "France": "SAS",
          "Sweden": "AB", "India": "Pvt Ltd", "Ireland": "Ltd"}
SYLL = ["nor", "vel", "ta", "ri", "kon", "sa", "mer", "lin", "da", "vo", "tri", "bel", "ar", "quin", "zen", "ola"]
FIELD_NOUN = {"renewable energy": "Energy", "logistics": "Logistics", "software": "Software", "agriculture": "Agri",
              "manufacturing": "Works", "healthcare": "Health", "finance": "Analytics", "construction": "Build",
              "education": "Learning", "retail": "Market"}


def synth_company(rng: random.Random, background: list[str]) -> dict:
    field = rng.choice(list(SERVICE_BANK))
    country = rng.choice(list(CITIES))
    base = "".join(rng.choice(SYLL) for _ in range(rng.randint(2, 3))).capitalize()
    name = f"{base} {FIELD_NOUN[field]} {SUFFIX[country]}"
    services = rng.sample(SERVICE_BANK[field], rng.randint(2, 4))
    emp = rng.choice([12, 25, 40, 85, 150, 320, 900])
    c = dict(name=name, domain=f"{base.lower()}-{FIELD_NOUN[field].lower()}.example", country=country,
             city=rng.choice(CITIES[country]), field=field, founded=rng.randint(1975, 2021),
             employees=f"{emp} employees", services=services,
             products=[f"{base} {rng.choice(['Cloud', 'Pro', 'One', 'Link', 'Grid'])}"],
             clients=rng.choice(["regional businesses", "public sector bodies", "mid-size enterprises",
                                 "international customers"]),
             about=f"{name} provides {services[0]} and {services[-1]} for its customers.")
    pages = {u: html_to_text(h) for u, h in company_pages(c).items() if not u.endswith("robots.txt")}
    # Variation: embed real cleaned text as a news/blog page, sometimes drop the about page.
    if background:
        pages[f"https://{c['domain']}/news"] = " ".join(rng.sample(background, min(2, len(background))))[:2500]
    if rng.random() < 0.2:
        pages.pop(f"https://{c['domain']}/about", None)
    return {"truth": c, "pages": pages}


def fixture_episode(c: dict) -> dict:
    pages = {u: html_to_text(h) for u, h in company_pages(c).items() if not u.endswith("robots.txt")}
    return {"truth": c, "pages": pages}


def evidence_docs(field: str, rng: random.Random) -> tuple[list[dict], set[str]]:
    docs = [dict(a, field=field) for a in FIELD_KB.get(field, [])]
    others = [f for f in FIELD_KB if f != field]
    for f in rng.sample(others, 2):
        docs.append(dict(rng.choice(FIELD_KB[f]), field=f))
    rng.shuffle(docs)
    ev = [{"id": f"E{i + 1}", "url": d["url"], "title": d["title"], "text": d["text"]} for i, d in enumerate(docs)]
    relevant = {s.strip() + "." for d in docs if d["field"] == field for s in d["text"].split(".") if s.strip()}
    return ev, relevant


def gold_facts(ev: list[dict], relevant: set[str]) -> list[dict]:
    out = []
    for e in ev:
        for s in re.split(r"(?<=[.!?])\s+", e["text"]):
            if s.strip() in relevant:
                out.append({"fact": s.strip(), "source_id": e["id"], "url": e["url"]})
    return out


def step_targets(trace: CandidateTrace, score: float) -> list[float]:
    """Calibrated confidence target per step from actual correctness of the program.

    Steps that errored get a low target (their own output was wrong); other steps get the program's
    final correctness (graded for partial answers). No valid output → all steps low.
    """
    targets = []
    for st in trace.steps:
        if not trace.valid:
            t = 5.0 if st.error else 12.0
        elif st.error:
            t = min(10.0, 100.0 * score)
        else:
            t = 100.0 * score
        targets.append(round(min(max(t, 2.0), 98.0), 3))
    return targets


def examples_from_trace(task, trace: CandidateTrace, score: float, episode: str) -> list[dict]:
    exs, history = [], []
    for st, target in zip(trace.steps, step_targets(trace, score)):
        prompt = build_prompt(task.name, task.query, task.variables, history, HELPER_DOC)
        exs.append({"prompt": prompt, "target": render_step(st.thought, st.code, target),
                    "meta": {"episode": episode, "task": task.name, "unit": task.unit_id, "t": st.t,
                             "conf": target, "correct": round(score, 4), "valid": trace.valid,
                             "step_error": bool(st.error), "candidate": trace.index}})
        history.append({"text": st.raw.replace("<|end|>", "").strip(), "observation":
                        "\n".join(x for x in [st.stdout.rstrip(), f"[error] {st.error}" if st.error else ""] if x)
                        or "[no output]"})
    return exs


def section_writing_examples(corpus: list[dict], episode: str, rng: random.Random, n: int = 3) -> list[dict]:
    exs = []
    for d in rng.sample(corpus, min(n, len(corpus))):
        sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", d["text"]) if len(s) > 20][:3]
        if not sents:
            continue
        facts = "\n".join(f"{s.rstrip('.')}. [S{i + 1}]" for i, s in enumerate(sents))
        exs.append({"prompt": f"<|user|> WRITE SECTION from these cited facts only:\n{facts}\n<|step|><|thought|>",
                    "target": " ".join(f"{s.rstrip('.')} [S{i + 1}]." for i, s in enumerate(sents)) + " <|end|>",
                    "meta": {"episode": episode, "task": "section_writing"}})
    return exs


def run_episode(ep: dict, engine: SRLMEngine, K: int, rng: random.Random, name: str) -> list[dict]:
    c, pages = ep["truth"], ep["pages"]
    out: list[dict] = []

    def sample(task, scorer):
        for attempt in range(1):
            for tr in engine._sample(task, K, attempt + rng.randint(0, 10**6), 0.8, 0.95):
                score = scorer(tr.output) if tr.valid else 0.0
                out.extend(examples_from_trace(task, tr, score, name))

    sample(field_task(c["domain"], pages), lambda o: score_field(o, c))
    profile = {"name": c["name"], "field": c["field"], "country": c["country"], "services": c["services"],
               "size": c["employees"]}
    ev, relevant = evidence_docs(c["field"], rng)
    sample(evidence_task(c["domain"], profile, ev), lambda o: score_evidence(o, relevant))
    facts = gold_facts(ev, relevant)
    gaps = {f["fact"] for f in facts if GAP_RE.search(f["fact"])} or {f["fact"] for f in facts}
    sample(project_task(c["domain"], profile, facts), lambda o: score_projects(o, gaps))
    projects = [{"title": project_title(f["fact"]), "rationale": f["fact"], "evidence_ids": [f["source_id"]]}
                for f in facts if f["fact"] in gaps][:3]
    corpus = build_corpus(pages, facts, projects)
    for section in rng.sample(FACT_SHEET_SECTIONS, 2):
        sample(fact_sheet_task(c["domain"], section, profile, corpus), lambda o, s=section: score_fact_sheet(o, s))
    out.extend(section_writing_examples(corpus, name, rng))
    return out


def load_background(path: Path, n: int = 400, seed: int = 0) -> list[str]:
    if not path.exists():
        return []
    rng = random.Random(seed)
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            if 200 < len(t) < 3000:
                texts.append(t)
    rng.shuffle(texts)
    return texts[:n]


def sample_texts(n: int = 2000) -> list[str]:
    """Rendered prompts/steps (without execution) so the tokenizer learns the agent format."""
    rng = random.Random(0)
    texts = []
    fake_hist = [{"observation": "{'field': 'logistics', 'country': 'Germany'}"}]
    for _ in range(max(1, n // 50)):
        c = rng.choice(COMPANIES)
        pages = fixture_episode(c)["pages"]
        task = field_task(c["domain"], pages)
        texts.append(build_prompt(task.name, task.query, task.variables, [], HELPER_DOC))
        for strategies in LIBRARY.values():
            for s in strategies:
                for f in s.steps:
                    th, code, conf = f(fake_hist)
                    texts.append(render_step(th, code, round(rng.uniform(2, 98), 3)))
    return texts[:n]


def main(argv: list[str] | None = None) -> dict:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--companies", type=int, default=400)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--background", default=str(resolve(cfg["paths"]["data_cleaned"]) / "cleaned.jsonl"))
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out_dir = resolve(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    srlm_cfg = dict(cfg["srlm"], max_steps=6, step_timeout_s=20)
    engine = SRLMEngine(TemplatePolicy(flawed_scale=1.0, seed=a.seed), lambda s: len(s.split()), srlm_cfg)
    background = load_background(Path(a.background))
    stats = {"train": 0, "val": 0}
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as ftr, open(out_dir / "val.jsonl", "w", encoding="utf-8") as fva:
        for i in range(a.companies):
            ep = synth_company(rng, background)
            for ex in run_episode(ep, engine, a.k, rng, f"synth-{i}"):
                ftr.write(json.dumps(ex, ensure_ascii=False) + "\n")
                stats["train"] += 1
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{a.companies} companies, {stats['train']} examples", flush=True)
        for c in COMPANIES:   # held-out fixture companies → validation / calibration set
            for ex in run_episode(fixture_episode(c), engine, a.k, rng, f"fixture-{c['domain']}"):
                fva.write(json.dumps(ex, ensure_ascii=False) + "\n")
                stats["val"] += 1
    (out_dir / "stats.json").write_text(json.dumps(stats))
    log.event("synth.generate", "done", details=stats)
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
