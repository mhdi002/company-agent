import math
import time

import pytest

from agent.srlm import select as S
from agent.srlm.canonical import cluster, item_set, set_similarity
from agent.srlm.engine import NOT_VERIFIED, SRLMEngine
from agent.srlm.protocol import CONFIDENCE_INSTRUCTION, build_prompt, parse_confidence, parse_step, render_step
from agent.srlm.repl import Sandbox
from agent.srlm.tasks import DecisionTask

SB_CFG = {"allowed_imports": ["re", "json", "math"], "max_output_chars": 500, "memory_limit_mb": 512,
          "step_timeout_s": 2, "max_steps": 5}


def cand(i, out, confs, toks, cl=None, valid=True):
    return S.CandidateSummary(index=i, output=out, valid=valid, confidences=confs, step_tokens=toks,
                              cluster=i if cl is None else cl)


# ------------------------------------------------------------------ signals and sign convention
def test_vc_is_nonpositive_and_monotone():
    assert S.verbalized_confidence([100.0, 100.0]) == 0.0
    assert S.verbalized_confidence([90.0]) < 0
    assert S.verbalized_confidence([90.0]) > S.verbalized_confidence([50.0])
    assert math.isclose(S.verbalized_confidence([50.0, 25.0]), math.log(0.5) + math.log(0.25))


def test_joint_score_sign_convention():
    # s = VC * Len <= 0; closer to zero is better: confident + concise beats unsure + long
    good = S.joint_score(S.verbalized_confidence([95, 95]), 100)
    bad_conf = S.joint_score(S.verbalized_confidence([40, 40]), 100)
    bad_len = S.joint_score(S.verbalized_confidence([95, 95]), 1000)
    assert good <= 0 and bad_conf <= 0 and bad_len <= 0
    assert good > bad_conf and good > bad_len
    sel = S.select([cand(0, "a", [40, 40], [100], 0), cand(1, "a", [95, 95], [100], 0),
                    cand(2, "a", [95, 95], [1000], 0)], K=3)
    assert sel.selected == 1


def test_missing_confidence_fill():
    assert S.fill_missing([80.0, None, 60.0]) == [80.0, 70.0, 60.0]
    assert S.fill_missing([None, None]) == [S.DEFAULT_CONFIDENCE] * 2
    assert math.isclose(S.verbalized_confidence([80.0, None, 60.0]),
                        math.log(0.8) + math.log(0.7) + math.log(0.6))


def test_consistent_set_and_plurality():
    cands = [cand(0, "x", [99], [10], 0), cand(1, "y", [60], [50], 1), cand(2, "y", [70], [50], 1),
             cand(3, "y", [50], [50], 1), cand(4, None, [90], [5], None, valid=False)]
    sel = S.select(cands, K=5)
    assert sel.consistent_set == [1, 2, 3]
    assert sel.prob == {0: 0.2, 1: 0.6}
    assert sel.selected == 2           # best s(p) inside S, even though cand 0 has a better score overall
    assert not sel.low_agreement
    assert sel.p_hat == 0.6


def test_single_consistent_candidate_taken():
    sel = S.select([cand(0, "x", [50], [10], 0), cand(1, "x", [50], [10], 0), cand(2, "z", [99], [5], 2)], K=3)
    assert sel.selected in (0, 1) and sel.consistent_set == [0, 1]
    sel = S.select([cand(0, "x", [50], [10], 0)], K=1)
    assert sel.selected == 0 and sel.reason == "single consistent candidate"


def test_low_agreement_takes_best_overall():
    cands = [cand(0, "a", [50], [100], 0), cand(1, "b", [90], [100], 1), cand(2, "c", [90], [20], 2)]
    sel = S.select(cands, K=3)
    assert sel.low_agreement and sel.selected == 2


def test_tie_breaks():
    # identical s(p): higher mean confidence wins
    a = cand(0, "a", [100, 50], [10, 10], 0)   # VC = log .5, Len 20
    b = cand(1, "a", [50, 100], [10, 10], 0)
    c = cand(2, "a", [100, 100, 50], [5, 5, 10], 0)  # same VC & Len, higher mean confidence
    assert S.select([a, b, c], K=3).selected == 2
    # identical s(p) and mean confidence: shorter wins; then lower index
    d = cand(0, "a", [100], [10], 0)
    e = cand(1, "a", [100], [5], 0)
    assert S.select([d, e], K=2).selected == 1          # s = 0 for both; shorter wins
    f = cand(0, "a", [100], [5], 0)
    g = cand(1, "a", [100], [5], 0)
    assert S.select([f, g], K=2).selected == 0


def test_tied_plurality_uses_best_member():
    cands = [cand(0, "a", [50], [10], 0), cand(1, "a", [50], [10], 0),
             cand(2, "b", [99], [10], 2), cand(3, "b", [99], [10], 2)]
    sel = S.select(cands, K=4)
    assert sel.selected == 2 and sel.consistent_set == [2, 3]


def test_ablation_switches():
    cands = [cand(0, "a", [99], [500], 0), cand(1, "a", [60], [10], 0), cand(2, "b", [99], [5], 2)]
    assert S.select(cands, 3, use_vc=True, use_len=False).selected == 0    # VC only
    assert S.select(cands, 3, use_vc=False, use_len=True).selected == 1    # Len only (shorter)
    assert S.select(cands, 3, use_self_consistency=False).selected == 2    # best over all K
    assert S.select(cands, 3, use_vc=False, use_len=False).selected == 0   # constant → lowest index


def test_retry_trigger():
    sel = S.Selection(0, 0, best_score=-900.0, p_hat=0.125)
    assert S.needs_retry(sel, -400, 0.25)
    sel.p_hat = 0.5
    assert not S.needs_retry(sel, -400, 0.25)
    assert S.needs_retry(S.Selection(None, None), -400, 0.25)


# ------------------------------------------------------------------ protocol
def test_parse_confidence_robust():
    assert parse_confidence('text\n```json\n{"confidence": 87.125}\n```') == 87.125
    assert parse_confidence('```json {"confidence": 42} ```') == 42.0
    assert parse_confidence('"confidence": 12.5') == 12.5
    assert parse_confidence('```json\n{"confidence": 0}\n```') is None
    assert parse_confidence('```json\n{"confidence": 150}\n```') is None
    assert parse_confidence("no block") is None


def test_parse_and_render_roundtrip():
    raw = render_step("look at pages", "print(len(context))", 77.5)
    p = parse_step(raw)
    assert p.format_ok and p.thought == "look at pages" and p.code == "print(len(context))" and p.confidence == 77.5
    bad = parse_step("<|thought|> x <|program|>\n```python\nprint(1)\n```")
    assert bad.confidence is None and "missing/invalid confidence" in bad.issues


def test_prompt_contains_instruction_and_not_context():
    secret = "UNIQUE-CONTEXT-STRING " * 200
    p = build_prompt("field_extraction", "q?", {"context": secret}, [], "helpers")
    assert CONFIDENCE_INSTRUCTION in p
    assert "UNIQUE-CONTEXT-STRING" not in p and "context: str (4400 chars)" in p


# ------------------------------------------------------------------ canonical agreement
def test_label_cluster_normalizes():
    ids = cluster([{"field": "Logistics ", "country": "UK"}, {"field": "logistics", "country": "uk"}, None],
                  "label", label_fn=lambda o: o["field"].strip().lower() + o["country"].lower())
    assert ids == [0, 0, None]


def test_set_and_free_text_clusters():
    a = [{"fact": "Grid queues are a major bottleneck."}, {"fact": "Storage raises revenue."}]
    b = [{"fact": "grid queues are a major bottleneck"}, {"fact": "Storage raises revenue"}]
    c = [{"fact": "Completely unrelated sentence about farming."}]
    assert set_similarity(item_set(a), item_set(b)) == 1.0
    assert cluster([a, b, c], "set") == [0, 0, 2]
    assert cluster(["the solar market grows fast", "solar market grows fast", "rivers flow"], "free_text") == [0, 0, 2]
    emb = cluster(["solar market grows quickly in germany", "solar market grows quickly in germany today", "rivers"],
                  "free_text", 0.6, "embedding")
    assert emb[0] == emb[1] != emb[2]


# ------------------------------------------------------------------ sandbox
def test_sandbox_state_helpers_and_final():
    with Sandbox({"context": "Polderwerk B.V. does CNC machining in Eindhoven, Netherlands.", "pages": {"u": "x"}}, SB_CFG) as sb:
        assert sb.execute("x = 41").error is None
        assert sb.execute("print(x + 1)").stdout.strip() == "42"
        r = sb.execute("info = extract_fields(context)\nprint(info['field'], info['country'])")
        assert r.stdout.strip() == "manufacturing Netherlands"
        r = sb.execute("FINAL({'a': 1})")
        assert r.has_final and r.final == {"a": 1}
        assert not sb.execute("print(1)").has_final   # FINAL applies to its step only


@pytest.mark.parametrize("code,needle", [
    ("import os", "not allowed"), ("import socket", "not allowed"), ("open('x', 'w')", "not allowed"),
    ("().__class__.__bases__", "not allowed"), ("eval('1')", "not allowed"),
    ("g = (i for i in []); g.gi_frame", "not allowed"), ("x = '__import__'", "not allowed"),
])
def test_sandbox_violations(code, needle):
    with Sandbox({"context": ""}, SB_CFG) as sb:
        r = sb.execute(code)
        assert r.violation and needle in r.error


def test_sandbox_timeout_and_memory():
    with Sandbox({"context": ""}, SB_CFG) as sb:
        t = time.time()
        r = sb.execute("while True:\n    pass")
        assert r.timed_out and time.time() - t < 6
        assert sb.execute("print('alive')").stdout.strip() == "alive"   # restarted
        r = sb.execute("x = 'a' * (2 * 1024 ** 3)")
        assert r.violation == "memory limit exceeded"


def test_sandbox_output_truncated():
    with Sandbox({"context": ""}, SB_CFG) as sb:
        r = sb.execute("print('y' * 5000)")
        assert "truncated" in r.stdout and len(r.stdout) < 700


# ------------------------------------------------------------------ engine
class ScriptPolicy:
    """Candidate k returns answers[k] with confidence confs[k], after `sleep` seconds."""
    name = "script"

    def __init__(self, answers, confs, sleep=0.0, steps=1):
        self.answers, self.confs, self.sleep, self.steps = answers, confs, sleep, steps

    def generate(self, prompt, task, history, k, attempt, temperature, top_p):
        time.sleep(self.sleep)
        if len(history) < self.steps - 1:
            return render_step("explore", "print(len(context))", self.confs[k])
        return render_step("answer", f"FINAL({self.answers[k]!r})", self.confs[k])


def make_task(validate=lambda o: None):
    return DecisionTask("field_extraction", "t/field", "q", {"context": "abc"}, "label", validate,
                        lambda o: str(o).lower(), {"domain": "t"})


def engine_cfg(**kw):
    cfg = dict(SB_CFG, K=4, temperature=0.8, top_p=0.95, max_steps=4, parallel=True, max_workers=8,
               use_self_consistency=True, use_verbalized_confidence=True, use_trace_length=True,
               min_score=-1e9, min_prob=0.0, max_retries=1, agreement_threshold=0.6)
    cfg.update(kw)
    return cfg


def test_engine_selects_plurality(tmp_logs):
    pol = ScriptPolicy(["A", "b", "a", "c"], [50, 99, 80, 99])
    d = SRLMEngine(pol, lambda s: len(s.split()), engine_cfg()).run(make_task())
    assert d.verified and d.output == "a" and d.selected_index == 2 and d.status == "selected"
    log = (tmp_logs / "srlm.jsonl").read_text()
    for key in ('"VC"', '"Len"', '"s"', '"consistent_set"', '"prob"', '"low_agreement"', '"wall_clock_s"', '"code"'):
        assert key in log


def test_engine_parallel_candidates(tmp_logs):
    pol = ScriptPolicy(["a"] * 4, [90] * 4, sleep=0.4, steps=2)
    t = time.time()
    d = SRLMEngine(pol, len, engine_cfg(K=4)).run(make_task())
    wall = time.time() - t
    assert d.verified and wall < 4 * 2 * 0.4 * 0.7   # well below sequential time (3.2 s)


def test_engine_not_verified_after_retries(tmp_logs):
    pol = ScriptPolicy(["a", "b", "c", "d"], [5, 5, 5, 5])
    d = SRLMEngine(pol, lambda s: 1000, engine_cfg(min_score=-1.0, min_prob=0.5, max_retries=1)).run(make_task())
    assert d.output == NOT_VERIFIED and not d.verified and d.retries == 1 and len(d.candidates) == 8


def test_engine_rejects_invalid_final_and_direct_baseline(tmp_logs):
    pol = ScriptPolicy(["bad"] * 4, [90] * 4)
    d = SRLMEngine(pol, len, engine_cfg(max_retries=0)).run(make_task(lambda o: "must not be bad" if o == "bad" else None))
    assert not d.verified
    d = SRLMEngine(ScriptPolicy(["a"], [70]), len, engine_cfg(direct_baseline=True)).run(make_task())
    assert d.status == "direct" and len(d.candidates) == 1


def test_engine_with_template_policy_on_fixture(tmp_logs):
    from agent.policy import TemplatePolicy
    from agent.srlm.tasks import field_task
    from data.sample.companies import COMPANIES, company_pages
    from tools.extract import html_to_text
    c = COMPANIES[1]
    pages = {u: html_to_text(h) for u, h in company_pages(c).items() if not u.endswith("robots.txt")}
    d = SRLMEngine(TemplatePolicy(), len, engine_cfg(K=6)).run(field_task(c["domain"], pages))
    assert d.verified and d.output["field"] == "logistics" and d.output["country"] == "United Kingdom"
