"""SRLM engine: Self-Reflective Program Search over context held in a sandboxed REPL.

For each decision unit: sample K independent candidate programs in parallel
(each a sequence of model-written REPL steps), execute them, then select the
output with self-consistency + verbalized confidence + trace length
(`agent/srlm/select.py`). No verifier, reward model, or other LLM is used.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agent.srlm.canonical import cluster
from agent.srlm.helpers import HELPER_DOC
from agent.srlm.protocol import build_prompt, parse_step
from agent.srlm.repl import ExecResult, Sandbox
from agent.srlm.select import CandidateSummary, Selection, needs_retry, select
from agent.srlm.tasks import DecisionTask
from core.logging import get_logger

NOT_VERIFIED = "Not verified"


@dataclass
class StepRecord:
    t: int
    raw: str
    thought: str
    code: str
    confidence: float | None
    tokens: int
    stdout: str
    error: str | None
    violation: str | None
    format_issues: list[str]
    seconds: float


@dataclass
class CandidateTrace:
    index: int
    attempt: int
    steps: list[StepRecord] = field(default_factory=list)
    output: Any = None
    valid: bool = False
    seconds: float = 0.0
    stop_reason: str = ""


@dataclass
class Decision:
    unit_id: str
    task: str
    output: Any
    verified: bool
    status: str                    # selected | low_agreement | not_verified | direct
    selected_index: int | None
    selection: dict
    candidates: list[CandidateTrace]
    retries: int
    seconds: float
    policy: str

    def summary(self) -> dict:
        return {"unit_id": self.unit_id, "status": self.status, "verified": self.verified,
                "selected": self.selected_index, "retries": self.retries, "seconds": round(self.seconds, 2),
                "p_hat": self.selection.get("p_hat")}


class SRLMEngine:
    def __init__(self, policy, count_tokens: Callable[[str], int], cfg: dict, logger=None):
        self.policy = policy
        self.count = count_tokens
        self.cfg = cfg
        self.log = logger or get_logger("srlm")

    # ------------------------------------------------------------------ one candidate program
    def run_candidate(self, task: DecisionTask, k: int, attempt: int, temperature: float, top_p: float) -> CandidateTrace:
        cfg = self.cfg
        trace = CandidateTrace(index=k, attempt=attempt)
        t0 = time.time()
        max_steps = int(cfg.get("max_steps", 30))
        budget = float(cfg.get("step_timeout_s", 60)) * max_steps
        subcall = self._subcall if cfg.get("use_subcalls") else None
        history: list[dict] = []
        with Sandbox(task.variables, cfg, subcall=subcall) as sb:
            for t in range(max_steps):
                if time.time() - t0 > budget:
                    trace.stop_reason = "time budget exceeded"
                    break
                prompt = build_prompt(task.name, task.query, task.variables, history, HELPER_DOC)
                ts = time.time()
                raw = self.policy.generate(prompt, task=task, history=history, k=k, attempt=attempt,
                                           temperature=temperature, top_p=top_p)
                if not raw.strip():
                    trace.stop_reason = "policy stopped"
                    break
                ps = parse_step(raw)
                res = sb.execute(ps.code) if ps.code.strip() else ExecResult(error="no program code in step")
                if res.has_final:
                    err = task.validate(res.final)
                    if err:
                        res.error = f"FINAL rejected: {err}"
                        res.has_final = False
                trace.steps.append(StepRecord(t, raw, ps.thought, ps.code, ps.confidence, self.count(raw),
                                              res.stdout, res.error, res.violation, ps.issues, time.time() - ts))
                if res.violation:
                    self.log.event("srlm.sandbox_violation", "warn", details=res.violation, unit=task.unit_id,
                                   candidate=k)
                elif res.error:
                    self.log.event("srlm.repl_error", "warn", details=res.error[:300], unit=task.unit_id,
                                   candidate=k)
                history.append({"text": raw.replace("<|end|>", "").strip(), "observation": res.observation()})
                if res.has_final:
                    trace.output, trace.valid, trace.stop_reason = res.final, True, "final"
                    break
            else:
                trace.stop_reason = "max steps"
        trace.seconds = time.time() - t0
        return trace

    def _subcall(self, query: str, text: str) -> str:
        """Optional recursive sub-query (ablation only; OFF by default)."""
        sub_cfg = dict(self.cfg, use_subcalls=False, K=1, max_steps=3, direct_baseline=True)
        sub = SRLMEngine(self.policy, self.count, sub_cfg, self.log)
        t = DecisionTask("sub_call", "sub", query, {"context": text}, "free_text", lambda o: None)
        return str(sub.run(t).output)

    # ------------------------------------------------------------------ K candidates + selection
    def _sample(self, task: DecisionTask, K: int, attempt: int, temperature: float, top_p: float) -> list[CandidateTrace]:
        if K == 1 or not self.cfg.get("parallel", True):
            return [self.run_candidate(task, k, attempt, temperature, top_p) for k in range(K)]
        with ThreadPoolExecutor(max_workers=min(K, int(self.cfg.get("max_workers", 8)))) as ex:
            futs = [ex.submit(self.run_candidate, task, k, attempt, temperature, top_p) for k in range(K)]
            return [f.result() for f in futs]

    def _summaries(self, task: DecisionTask, cands: list[CandidateTrace]) -> list[CandidateSummary]:
        outs = [c.output if c.valid else None for c in cands]
        key = "set_agreement_threshold" if task.kind == "set" else "agreement_threshold"
        ids = cluster(outs, task.kind, float(self.cfg.get(key, self.cfg.get("agreement_threshold", 0.6))),
                      self.cfg.get("free_text_agreement", "keyfacts"), task.label_fn)
        return [CandidateSummary(index=i, output=c.output, valid=c.valid,
                                 confidences=[s.confidence for s in c.steps], step_tokens=[s.tokens for s in c.steps],
                                 cluster=ids[i]) for i, c in enumerate(cands)]

    def run(self, task: DecisionTask) -> Decision:
        cfg = self.cfg
        t0 = time.time()
        direct = bool(cfg.get("direct_baseline", False))
        K = 1 if direct else int(cfg.get("K", 8))
        # The direct baseline is one program sampled with the same decoding settings, without selection.
        temperature = float(cfg.get("temperature", 0.8))
        top_p = float(cfg.get("top_p", 0.95))
        max_retries = 0 if direct else int(cfg.get("max_retries", 1))
        all_cands: list[CandidateTrace] = []
        sel = Selection(None, None)
        summaries: list[CandidateSummary] = []
        attempt = 0
        for attempt in range(max_retries + 1):
            cands = self._sample(task, K, attempt, temperature, top_p)
            all_cands.extend(cands)
            summaries = self._summaries(task, cands)
            sel = select(summaries, K, bool(cfg.get("use_self_consistency", True)),
                         bool(cfg.get("use_verbalized_confidence", True)), bool(cfg.get("use_trace_length", True)))
            if direct or not needs_retry(sel, float(cfg.get("min_score", -400)), float(cfg.get("min_prob", 0.25))):
                break
            self.log.event("srlm.retry", "retry", details=f"{task.unit_id}: best_s={sel.best_score:.2f} "
                                                          f"p_hat={sel.p_hat:.2f}", unit=task.unit_id, attempt=attempt)
        last = all_cands[-K:]
        failed = sel.selected is None or (not direct and needs_retry(sel, float(cfg.get("min_score", -400)),
                                                                      float(cfg.get("min_prob", 0.25))))
        if failed:
            output, status, verified = NOT_VERIFIED, "not_verified", False
        else:
            output = last[sel.selected].output
            status = "direct" if direct else ("low_agreement" if sel.low_agreement else "selected")
            verified = True
        decision = Decision(task.unit_id, task.name, output, verified, status, sel.selected,
                            {"prob": sel.prob, "consistent_set": sel.consistent_set, "low_agreement": sel.low_agreement,
                             "best_score": sel.best_score, "p_hat": sel.p_hat, "reason": sel.reason},
                            all_cands, attempt, time.time() - t0, getattr(self.policy, "name", "unknown"))
        self._log(task, decision, summaries, K)
        return decision

    # ------------------------------------------------------------------ srlm.jsonl
    def _log(self, task: DecisionTask, d: Decision, summaries: list[CandidateSummary], K: int) -> None:
        from agent.srlm.canonical import canon_json
        cands = []
        last = d.candidates[-K:]
        for c, s in zip(last, summaries):
            cands.append({"index": c.index, "attempt": c.attempt, "valid": c.valid, "stop_reason": c.stop_reason,
                          "code": [st.code for st in c.steps], "thoughts": [st.thought for st in c.steps],
                          "step_outputs": [st.stdout[:500] + (f" [error] {st.error}" if st.error else "") for st in c.steps],
                          "confidences": [st.confidence for st in c.steps], "step_tokens": [st.tokens for st in c.steps],
                          "VC": round(s.vc, 4), "Len": s.length, "s": round(s.score, 4), "cluster": s.cluster,
                          "canonical_output": canon_json(c.output)[:500] if c.valid else None,
                          "violations": [st.violation for st in c.steps if st.violation], "seconds": round(c.seconds, 3)})
        self.log.event("srlm.decision", "done" if d.verified else "not_verified", duration=d.seconds,
                       company=task.meta.get("domain"), details=f"{task.unit_id} → {d.status}",
                       unit=task.unit_id, task=task.name, query=task.query, K=K, policy=d.policy,
                       candidates=cands, prob={str(k): v for k, v in d.selection["prob"].items()},
                       consistent_set=d.selection["consistent_set"], selected=d.selected_index,
                       low_agreement=d.selection["low_agreement"], retries=d.retries,
                       retry_occurred=d.retries > 0, reason=d.selection["reason"],
                       output=canon_json(d.output)[:1000], wall_clock_s=round(d.seconds, 3))


def decision_to_dict(d: Decision) -> dict:
    out = asdict(d)
    return out
