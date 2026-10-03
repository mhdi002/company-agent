"""Paper-style ablation: direct baseline vs. single signals vs. full SRLM on held-out companies.

Variants (config switches):
  direct          one program, no selection (srlm.direct_baseline)
  +verb_conf      K programs, select by VC only (no self-consistency, no length)
  +trace_len      K programs, select by Len only
  +self_consist   K programs, plurality vote only (index tie-break)
  srlm            full: self-consistency + VC·Len
Metrics per variant: field accuracy, evidence precision / F1, project hit rate, hallucination rate
(ungrounded factual sentences in the written proposal), proposal completeness (verified sections /
17), wall-clock per company.

Usage:
    python -m training.ablation --policy template --flawed 1.0 --synthetic 18 --k 8
    python -m training.ablation --policy model --checkpoint training/checkpoints/sft-tiny/latest.pt --units field --k 4
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time

from agent.policy import TemplatePolicy
from agent.proposal import SourceBook, TemplateWriter
from agent.srlm.canonical import norm_str
from agent.srlm.engine import SRLMEngine
from agent.srlm.helpers import project_title
from agent.srlm.tasks import (FACT_SHEET_SECTIONS, GAP_RE, build_corpus, evidence_task, fact_sheet_task, field_task,
                              project_task, score_field, score_projects)
from agent.synth.generate import evidence_docs, fixture_episode, gold_facts, load_background, synth_company
from core.config import load_config, resolve
from core.logging import configure, get_logger
from data.sample.companies import COMPANIES

log = get_logger("training")

VARIANTS = {
    "direct": dict(direct_baseline=True),
    "+verb_conf": dict(use_self_consistency=False, use_verbalized_confidence=True, use_trace_length=False),
    "+trace_len": dict(use_self_consistency=False, use_verbalized_confidence=False, use_trace_length=True),
    "+self_consist": dict(use_self_consistency=True, use_verbalized_confidence=False, use_trace_length=False),
    "srlm": dict(use_self_consistency=True, use_verbalized_confidence=True, use_trace_length=True),
}


def hallucination_rate(sections: list[dict], allowed_text: str) -> tuple[int, int]:
    """Count factual sentences (those carrying a [S#] citation) whose content is not in the sources."""
    allowed = norm_str(allowed_text)
    total = bad = 0
    for s in sections:
        if s["title"] == "Sources":
            continue
        for line in s.get("paragraphs", []) + s.get("bullets", []):
            if not re.search(r"\[S\d+", line):
                continue
            total += 1
            body = norm_str(re.sub(r"\[S[\d, S]+\]", "", line))
            if body and body not in allowed:
                bad += 1
    return bad, total


def run_company(engine: SRLMEngine, ep: dict, rng: random.Random, units: set[str]) -> dict:
    c, pages = ep["truth"], ep["pages"]
    out: dict = {"domain": c["domain"]}
    t0 = time.time()
    d = engine.run(field_task(c["domain"], pages))
    out["field_ok"] = float(d.verified and score_field(d.output, c) >= 0.7)
    out["field_status"] = d.status
    if units == {"field"}:
        out["seconds"] = time.time() - t0
        return out
    profile = {"name": c["name"], "field": c["field"], "country": c["country"], "services": c["services"],
               "size": c["employees"]}
    ev, relevant = evidence_docs(c["field"], rng)
    de = engine.run(evidence_task(c["domain"], profile, ev))
    facts = de.output if de.verified else []
    rel = {norm_str(r) for r in relevant}
    got = [norm_str(f["fact"]) for f in facts]
    out["evidence_precision"] = sum(g in rel for g in got) / len(got) if got else 0.0
    out["evidence_recall"] = len(set(got) & rel) / len(rel) if rel else 0.0
    gold = gold_facts(ev, relevant)
    gaps = {f["fact"] for f in gold if GAP_RE.search(f["fact"])} or {f["fact"] for f in gold}
    dp = engine.run(project_task(c["domain"], profile, facts or gold))
    projects = dp.output if dp.verified else []
    out["project_score"] = score_projects(projects, gaps) if projects else 0.0
    corpus = build_corpus(pages, facts, projects)
    sheets = {}
    for section in FACT_SHEET_SECTIONS:
        ds = engine.run(fact_sheet_task(c["domain"], section, profile, corpus))
        sheets[section] = ds.output
    book = SourceBook()
    sections = TemplateWriter().write({"profile": profile, "projects": projects, "fact_sheets": sheets, "book": book,
                                       "corpus_urls": {x["id"]: x["url"] for x in corpus}, "domain": c["domain"]})
    allowed = " ".join(pages.values()) + " " + " ".join(e["text"] for e in ev) + " " + " ".join(
        f"{p['title']}. {p.get('rationale', '')}" for p in projects)
    bad, total = hallucination_rate(sections, allowed)
    out["hallucinated"], out["factual_sentences"] = bad, total
    out["completeness"] = sum(1 for s in sections if s.get("verified")) / len(sections)
    out["seconds"] = time.time() - t0
    return out


def summarize(rows: list[dict]) -> dict:
    def mean(k):
        vals = [r[k] for r in rows if k in r]
        return round(statistics.mean(vals), 4) if vals else None
    hall = sum(r.get("hallucinated", 0) for r in rows)
    fact = sum(r.get("factual_sentences", 0) for r in rows)
    return {"n": len(rows), "field_accuracy": mean("field_ok"), "evidence_precision": mean("evidence_precision"),
            "evidence_recall": mean("evidence_recall"), "project_score": mean("project_score"),
            "hallucination_rate": round(hall / fact, 4) if fact else None, "completeness": mean("completeness"),
            "seconds_per_company": mean("seconds"),
            "low_agreement": sum(r.get("field_status") == "low_agreement" for r in rows),
            "not_verified": sum(r.get("field_status") == "not_verified" for r in rows)}


def main(argv=None) -> dict:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["template", "model"], default="template")
    ap.add_argument("--checkpoint", default=str(resolve(cfg["agent"]["checkpoint"])))
    ap.add_argument("--flawed", type=float, default=1.0, help="template policy: weight multiplier of flawed programs")
    ap.add_argument("--synthetic", type=int, default=18, help="held-out synthetic companies (in addition to 12 fixtures)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--units", default="all", choices=["all", "field"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="docs/eval")
    a = ap.parse_args(argv)
    configure(cfg["paths"]["logs"])
    rng = random.Random(a.seed)
    background = load_background(resolve(cfg["paths"]["data_cleaned"]) / "cleaned.jsonl", seed=a.seed)
    episodes = [fixture_episode(c) for c in COMPANIES] + [synth_company(rng, background) for _ in range(a.synthetic)]
    if a.policy == "model":
        from agent.policy import ModelPolicy
        policy = ModelPolicy.from_checkpoint(a.checkpoint, resolve(cfg["paths"]["tokenizer"]), max_new_tokens=256)
        count = policy.tok.count
    else:
        policy, count = TemplatePolicy(flawed_scale=a.flawed, seed=a.seed), (lambda s: len(s.split()))
    units = {"field"} if a.units == "field" else {"all"}
    results = {}
    for name in a.variants:
        srlm_cfg = dict(cfg["srlm"], K=a.k, max_steps=a.max_steps, max_retries=0, log_prompts=False, **VARIANTS[name])
        engine = SRLMEngine(policy, count, srlm_cfg)
        rows = [run_company(engine, ep, random.Random(a.seed + i), units) for i, ep in enumerate(episodes)]
        results[name] = summarize(rows)
        print(name, json.dumps(results[name]), flush=True)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.policy}" + (f"_flawed{a.flawed}" if a.policy == "template" else "") + f"_{a.units}_k{a.k}"
    payload = {"policy": a.policy, "k": a.k, "flawed": a.flawed, "companies": len(episodes), "results": results}
    (out / f"ablation_{tag}.json").write_text(json.dumps(payload, indent=2))
    lines = [f"### Ablation — policy={a.policy}" + (f" (flawed×{a.flawed})" if a.policy == "template" else "") +
             f", K={a.k}, {len(episodes)} held-out companies", "",
             "| Variant | Field acc. | Evidence P | Evidence R | Project | Halluc. rate | Completeness | s/company |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['field_accuracy']} | {r['evidence_precision']} | {r['evidence_recall']} | "
                     f"{r['project_score']} | {r['hallucination_rate']} | {r['completeness']} | {r['seconds_per_company']} |")
    (out / f"ablation_{tag}.md").write_text("\n".join(lines) + "\n")
    log.event("ablation.done", "done", details=payload)
    return payload


if __name__ == "__main__":
    main()
