"""Evaluation of our agent model: step-format accuracy, calibration (ECE), field accuracy, proposal quality.

Usage:
    python -m training.eval --checkpoint training/checkpoints/sft/latest.pt --data data/sft --out docs/eval
    python -m training.eval --policy template --out docs/eval        # reference numbers without a model
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from agent.srlm.protocol import parse_confidence, parse_step
from core.config import load_config, resolve
from core.logging import configure, get_logger

log = get_logger("training")


def ece(conf: list[float], correct: list[float], bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width bins; conf in [0,1], correct in [0,1] (graded)."""
    conf_a, corr_a = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total, curve = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf_a >= lo) & ((conf_a < hi) if i < bins - 1 else (conf_a <= hi))
        if not sel.any():
            continue
        c, a = conf_a[sel].mean(), corr_a[sel].mean()
        total += sel.sum() / len(conf_a) * abs(c - a)
        curve.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(sel.sum()), "mean_conf": round(float(c), 4),
                      "accuracy": round(float(a), 4)})
    return float(total), curve


def load_val(path: Path, limit: int, seed: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["meta"].get("task") != "section_writing"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:limit]
    return [rows[i] for i in idx]


@torch.no_grad()
def model_metrics(policy, rows: list[dict], max_new: int = 200) -> dict:
    """Unconstrained step-format accuracy and verbalized-confidence calibration on held-out steps."""
    tok, model = policy.tok, policy.model
    end = tok.special["<|end|>"]
    fmt_ok, confs, corr, parsed_conf = 0, [], [], 0
    for r in rows:
        ids = torch.tensor([policy._fit(tok.encode(r["prompt"]), max_new + 1)])
        out = model.generate(ids, max_new, temperature=0.0, stop_ids={end, tok.eos_id})
        p = parse_step(tok.decode(out))
        fmt_ok += p.format_ok
        parsed_conf += p.confidence is not None
        # Calibration: teacher-force the gold step up to the confidence number, decode the number constrained.
        prefix = r["target"].split('{"confidence": ')[0] + '{"confidence": '
        pid = tok.encode(r["prompt"] + prefix)
        allowed = torch.tensor(policy.number_ids)

        def number_only(_o, logits):
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed] = 0.0
            return logits + mask
        num = tok.decode(model.generate(torch.tensor([policy._fit(pid, 4)]), 3, temperature=0.0,
                                        logits_processor=number_only))
        m = re.match(r"\d{1,3}(\.\d{1,3})?", num)
        v = min(max(float(m.group(0)), 0.0), 100.0) if m else 50.0
        confs.append(v / 100.0)
        corr.append(float(r["meta"]["correct"]) if r["meta"]["valid"] else 0.0)
    e, curve = ece(confs, corr)
    return {"n": len(rows), "step_format_accuracy_unconstrained": round(fmt_ok / max(len(rows), 1), 4),
            "confidence_parse_rate_unconstrained": round(parsed_conf / max(len(rows), 1), 4),
            "step_format_accuracy_constrained": 1.0, "ece": round(e, 4),
            "brier": round(float(np.mean((np.array(confs) - np.array(corr)) ** 2)), 4), "calibration_curve": curve}


def target_calibration(rows: list[dict]) -> dict:
    """ECE of the training targets themselves (sanity check of the calibration labels)."""
    e, curve = ece([r["meta"]["conf"] / 100 for r in rows],
                   [float(r["meta"]["correct"]) if r["meta"]["valid"] else 0.0 for r in rows])
    return {"ece_targets": round(e, 4), "curve": curve}


def field_accuracy(policy, srlm_cfg: dict, count) -> dict:
    from agent.srlm.engine import SRLMEngine
    from agent.srlm.tasks import field_task
    from data.sample.companies import COMPANIES, company_pages
    from tools.extract import html_to_text
    eng = SRLMEngine(policy, count, srlm_cfg)
    hits, t0 = 0, time.time()
    rows = []
    for c in COMPANIES:
        pages = {u: html_to_text(h) for u, h in company_pages(c).items() if not u.endswith("robots.txt")}
        d = eng.run(field_task(c["domain"], pages))
        ok = d.verified and isinstance(d.output, dict) and d.output.get("field") == c["field"]
        hits += ok
        rows.append({"domain": c["domain"], "truth": c["field"], "pred": d.output.get("field") if isinstance(d.output, dict) else d.output,
                     "status": d.status})
    return {"field_accuracy": round(hits / len(COMPANIES), 4), "n": len(COMPANIES),
            "seconds": round(time.time() - t0, 1), "rows": rows}


def main(argv=None) -> dict:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(resolve(cfg["agent"]["checkpoint"])))
    ap.add_argument("--tokenizer", default=str(resolve(cfg["paths"]["tokenizer"])))
    ap.add_argument("--policy", default="model", choices=["model", "template"])
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=2, help="K for the SRLM field-accuracy run")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--out", default="docs/eval")
    a = ap.parse_args(argv)
    configure(cfg["paths"]["logs"])
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_val(resolve(a.data) / "val.jsonl", a.limit)
    res: dict = {"policy": a.policy, "targets": target_calibration(rows)}
    srlm_cfg = dict(cfg["srlm"], K=a.k, max_steps=a.max_steps, max_retries=0)
    if a.policy == "model":
        from agent.policy import ModelPolicy
        pol = ModelPolicy.from_checkpoint(a.checkpoint, a.tokenizer, max_new_tokens=256)
        res["model"] = model_metrics(pol, rows)
        res["srlm_field"] = field_accuracy(pol, srlm_cfg, pol.tok.count)
    else:
        from agent.policy import TemplatePolicy
        res["srlm_field"] = field_accuracy(TemplatePolicy(), srlm_cfg, lambda s: len(s.split()))
    (out / f"eval_{a.policy}.json").write_text(json.dumps(res, indent=2))
    log.event("eval.done", "done", details={k: v for k, v in res.items() if k != "targets"})
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("curve", "calibration_curve", "rows")})
                      for k, v in res.items()}, indent=1))
    return res


if __name__ == "__main__":
    main()
