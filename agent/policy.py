"""Policies that write SRLM program steps.

* ModelPolicy — our own trained transformer with constrained decoding (default runtime brain).
* TemplatePolicy — deterministic programmatic policy built from `agent/programs.py`. It is the same
  generator that produces the synthetic SFT traces; used in tests and when no checkpoint exists.

No external LLM is ever called.
"""
from __future__ import annotations

import hashlib
import random
import re
import threading
from pathlib import Path

import torch

from agent.programs import LIBRARY, RECOVERY_THOUGHT, Strategy
from agent.srlm.protocol import render_step
from core.logging import get_logger

log = get_logger("agent")


def stable_seed(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def choose_strategy(strategies: list[Strategy], rng: random.Random, allow_flawed: bool = True,
                    flawed_scale: float = 1.0) -> Strategy:
    pool = [s for s in strategies if allow_flawed or not s.flawed]
    weights = [s.weight * (flawed_scale if s.flawed else 1.0) for s in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


class TemplatePolicy:
    """Programmatic policy: picks a strategy per candidate, follows it, recovers from errors."""

    name = "template"

    def __init__(self, flawed_scale: float = 0.3, seed: int = 0):
        self.flawed_scale = flawed_scale
        self.seed = seed
        self._plans: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def plan_for(self, task, k: int, attempt: int, temperature: float) -> dict:
        key = (task.unit_id, k, attempt)
        with self._lock:
            if key not in self._plans:
                strategies = LIBRARY[task.name]
                if temperature <= 0:
                    strat = max((s for s in strategies if not s.flawed), key=lambda s: s.weight)
                else:
                    rng = random.Random(stable_seed(self.seed, task.unit_id, k, attempt))
                    strat = choose_strategy(strategies, rng, True, self.flawed_scale)
                self._plans[key] = {"strategy": strat, "offset": 0, "recovered": False}
            return self._plans[key]

    def next_step(self, task, history: list, k: int, attempt: int, temperature: float):
        """Return (thought, code, confidence, strategy_name) or None to stop."""
        plan = self.plan_for(task, k, attempt, temperature)
        t = len(history)
        last = history[-1]["observation"] if history else ""
        strat: Strategy = plan["strategy"]
        idx = t - plan["offset"]
        failed = "[error]" in last or (idx >= len(strat.steps) and "[FINAL set]" not in last)
        if failed and t > 0:
            if plan["recovered"]:
                return None
            canonical = max((s for s in LIBRARY[task.name] if not s.flawed), key=lambda s: s.weight)
            plan.update(strategy=canonical, offset=t, recovered=True)
            err = re.sub(r"\s+", " ", last.split("[error]")[-1]).strip()[:120] or "no valid output"
            thought, code, conf = canonical.steps[0](history)
            return RECOVERY_THOUGHT.format(err=err) + " " + thought, code, conf, canonical.name + "+recovery"
        if idx >= len(strat.steps):
            return None
        thought, code, conf = strat.steps[idx](history)
        return thought, code, conf, strat.name

    def generate(self, prompt: str, task, history: list, k: int, attempt: int, temperature: float = 0.8,
                 top_p: float = 0.95) -> str:
        step = self.next_step(task, history, k, attempt, temperature)
        if step is None:
            return ""
        thought, code, conf, _ = step
        return render_step(thought, code, conf)


# ============================================================================ model policy
class ModelPolicy:
    """Our transformer writes each step; decoding is constrained to the step grammar:
    thought → <|program|> ```python code ``` → ```json {"confidence": <number>} ``` → <|end|>.
    """

    name = "model"

    def __init__(self, model, tok, max_new_tokens: int = 512, device: str | torch.device = "cpu",
                 max_thought_tokens: int = 96):
        self.model = model.eval().to(device)
        self.tok = tok
        self.device = torch.device(device)
        self.max_new = max_new_tokens
        self.max_thought = max_thought_tokens
        self.special_ids = set(tok.special.values())
        vocab = tok.tk.get_vocab()
        self.number_ids = [i for t, i in vocab.items() if i not in self.special_ids
                           and re.fullmatch(r"[0-9.]+", tok.decode([i]) or "")]
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, path: str | Path, tokenizer_dir: str | Path, device: str = "cpu", **kw) -> "ModelPolicy":
        from tokenizer.tok import Tok
        from training.common import model_from_checkpoint
        return cls(model_from_checkpoint(Path(path), device), Tok.load(tokenizer_dir), device=device, **kw)

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def _fit(self, ids: list[int], reserve: int) -> list[int]:
        limit = self.model.cfg.max_seq_len - reserve
        if len(ids) <= limit:
            return ids
        head = ids[:min(384, limit // 2)]
        return head + ids[-(limit - len(head)):]

    def _gen(self, ids: list[int], n: int, temperature: float, top_p: float, stop: set[int],
             proc, gen: torch.Generator) -> list[int]:
        x = torch.tensor([self._fit(ids, n + 1)], device=self.device)
        with self._lock, torch.no_grad():
            return self.model.generate(x, n, temperature=temperature, top_p=top_p, stop_ids=stop,
                                       logits_processor=proc, generator=gen)

    def generate(self, prompt: str, task=None, history=None, k: int = 0, attempt: int = 0,
                 temperature: float = 0.8, top_p: float = 0.95) -> str:
        tok = self.tok
        gen = torch.Generator(device="cpu").manual_seed(stable_seed(getattr(task, "unit_id", ""), k, attempt,
                                                                    len(history or [])))
        ban = torch.tensor(sorted(self.special_ids), device=self.device)
        prog_id = tok.special["<|program|>"]

        def no_special_except(allowed: set[int]):
            def proc(_out, logits):
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask[ban] = True
                for a in allowed:
                    mask[a] = False
                return logits.masked_fill(mask, float("-inf"))
            return proc

        ids = tok.encode(prompt) + [tok.special["<|thought|>"]]
        # Phase A: thought, until <|program|>
        th = self._gen(ids, self.max_thought, temperature, top_p, {prog_id}, no_special_except({prog_id}), gen)
        if th and th[-1] == prog_id:
            th = th[:-1]
        thought = tok.decode(th)
        forced = "<|program|>\n```python\n"
        ids = ids + th + self._ids(forced)
        # Phase B: code, until the closing fence
        code_ids: list[int] = []
        budget = max(self.max_new - len(th) - 24, 32)
        while budget > 0:
            chunk = self._gen(ids + code_ids, min(64, budget), temperature, top_p, set(), no_special_except(set()), gen)
            code_ids += chunk
            budget -= len(chunk)
            if "```" in tok.decode(code_ids):
                break
        code = tok.decode(code_ids).split("```")[0]
        # Phase C: constrained confidence number
        prefix = f"<|thought|>{thought}{forced}{code.rstrip()}\n```\n```json\n{{\"confidence\": "
        ids = tok.encode(prompt) + self._ids(prefix)
        allowed = torch.tensor(self.number_ids, device=self.device)

        def number_only(out, logits):
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed] = 0.0
            return logits + mask
        num_ids = self._gen(ids, 3, temperature, top_p, set(), number_only, gen)
        num = re.match(r"\d{1,3}(\.\d{1,3})?", tok.decode(num_ids))
        val = float(num.group(0)) if num else 50.0
        val = min(max(val, 0.001), 100.0)
        return f"{prefix}{val:.3f}}}\n```\n<|end|>"
