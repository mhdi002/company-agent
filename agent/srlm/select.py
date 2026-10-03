"""SRLM uncertainty signals and joint selection (spec §5.4–5.5).

Signals per candidate program p (all from the model itself):
  * self-consistency: prob(a) = count(out(p_k) == a) / K, plurality a_hat, consistent set S
  * verbalized confidence: VC(p) = Σ_t log(ν_t / 100) ≤ 0 (missing ν_t filled with the mean of the
    other steps of the same trajectory)
  * trace length: Len(p) = Σ_t l_t (reasoning + output tokens generated at each step)

Joint score s(p) = VC(p) · Len(p) ≤ 0. p* = argmax_{p ∈ S} s(p): the candidate closest to zero, i.e.
the most confident and most concise. Tie-breaks: higher mean confidence, shorter Len, lower index.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIDENCE = 50.0   # used only when no step of a trajectory has a parseable confidence


def fill_missing(confidences: list[float | None], default: float = DEFAULT_CONFIDENCE) -> list[float]:
    """Fill missing per-step confidences with the mean of the other steps (paper's rule)."""
    known = [c for c in confidences if c is not None]
    fill = sum(known) / len(known) if known else default
    return [fill if c is None else c for c in confidences]


def verbalized_confidence(confidences: list[float | None]) -> float:
    """VC(p) = Σ_t log(ν_t/100). Always ≤ 0; closer to 0 = more confident."""
    filled = fill_missing(confidences)
    return sum(math.log(min(max(c, 1e-3), 100.0) / 100.0) for c in filled)


def trace_length(step_tokens: list[int]) -> int:
    return int(sum(step_tokens))


def joint_score(vc: float, length: int, use_vc: bool = True, use_len: bool = True) -> float:
    """s(p) = VC(p)·Len(p) ≤ 0, with ablation switches.

    VC only → s = VC. Len only → s = −Len (shorter is better). Neither → constant −1 (index tie-break).
    """
    a = vc if use_vc else -1.0
    b = float(length) if use_len else 1.0
    return a * b


@dataclass
class CandidateSummary:
    index: int
    output: Any
    valid: bool
    confidences: list[float | None]
    step_tokens: list[int]
    cluster: int | None = None
    vc: float = 0.0
    length: int = 0
    score: float = 0.0
    mean_conf: float = 0.0


@dataclass
class Selection:
    selected: int | None
    answer_cluster: int | None
    prob: dict[int, float] = field(default_factory=dict)
    consistent_set: list[int] = field(default_factory=list)
    low_agreement: bool = False
    best_score: float = float("-inf")
    p_hat: float = 0.0
    reason: str = ""


def score_candidates(cands: list[CandidateSummary], use_vc: bool = True, use_len: bool = True) -> None:
    for c in cands:
        c.vc = verbalized_confidence(c.confidences) if c.confidences else math.log(DEFAULT_CONFIDENCE / 100)
        c.length = trace_length(c.step_tokens)
        c.score = joint_score(c.vc, c.length, use_vc, use_len)
        filled = fill_missing(c.confidences) if c.confidences else [DEFAULT_CONFIDENCE]
        c.mean_conf = sum(filled) / len(filled)


def _rank_key(c: CandidateSummary) -> tuple:
    # argmax s(p); then higher mean confidence; then shorter length; then lower index
    return (-c.score, -c.mean_conf, c.length, c.index)


def best_of(cands: list[CandidateSummary]) -> CandidateSummary:
    return sorted(cands, key=_rank_key)[0]


def select(cands: list[CandidateSummary], K: int, use_self_consistency: bool = True,
           use_vc: bool = True, use_len: bool = True) -> Selection:
    """Joint selection over K candidates. Candidates must carry `cluster` ids (None = invalid output)."""
    score_candidates(cands, use_vc, use_len)
    valid = [c for c in cands if c.valid and c.cluster is not None]
    if not valid:
        return Selection(None, None, reason="no valid candidate")
    counts = Counter(c.cluster for c in valid)
    prob = {a: n / K for a, n in counts.items()}
    top = max(counts.values())
    leaders = [a for a, n in counts.items() if n == top]
    if len(leaders) > 1:
        # Tied plurality: pick the cluster whose best member has the best joint score.
        leaders.sort(key=lambda a: _rank_key(best_of([c for c in valid if c.cluster == a])))
    a_hat = leaders[0]
    low_agreement = top == 1 and len(valid) > 1
    if not use_self_consistency:
        pool, reason = valid, "self-consistency disabled: best s(p) over all candidates"
    elif low_agreement:
        pool, reason = valid, "no plurality (all disagree): best s(p) over all K"
    else:
        pool = [c for c in valid if c.cluster == a_hat]
        reason = "single consistent candidate" if len(pool) == 1 else "argmax s(p) within consistent set"
    best = best_of(pool)
    S = [c.index for c in valid if c.cluster == a_hat]
    return Selection(best.index, best.cluster, prob, S, low_agreement, best.score, prob.get(best.cluster, 0.0), reason)


def needs_retry(sel: Selection, min_score: float, min_prob: float) -> bool:
    """Retry with fresh samples when both the best s(p) and prob(a_hat) fall below thresholds."""
    if sel.selected is None:
        return True
    return sel.best_score < min_score and sel.p_hat < min_prob
