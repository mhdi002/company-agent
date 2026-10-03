"""Parent side of the sandboxed REPL: spawns and drives a worker subprocess."""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.config import ROOT


@dataclass
class ExecResult:
    stdout: str = ""
    error: str | None = None
    final: object = None
    has_final: bool = False
    violation: str | None = None
    seconds: float = 0.0
    timed_out: bool = False

    def observation(self) -> str:
        """What the model sees after the step."""
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.error:
            parts.append(f"[error] {self.error}")
        if self.has_final:
            parts.append("[FINAL set]")
        return "\n".join(parts) if parts else "[no output]"


class Sandbox:
    """One persistent REPL per candidate program; context lives here, not in the prompt."""

    def __init__(self, variables: dict, cfg: dict, subcall: Callable[[str, str], str] | None = None):
        self.cfg = cfg
        self.variables = variables
        self.subcall = subcall
        self.tmp = tempfile.TemporaryDirectory(prefix="srlm_")
        self.timeout = float(cfg.get("step_timeout_s", 60))
        self.proc: subprocess.Popen | None = None
        self.violations: list[str] = []
        self._start()

    def _start(self) -> None:
        d = Path(self.tmp.name)
        (d / "vars.json").write_text(json.dumps(self.variables, ensure_ascii=False, default=str), encoding="utf-8")
        wcfg = {k: self.cfg.get(k) for k in ("allowed_imports", "max_output_chars", "memory_limit_mb", "use_subcalls")}
        wcfg["cpu_limit_s"] = int(self.timeout * int(self.cfg.get("max_steps", 30)) + 30)
        (d / "cfg.json").write_text(json.dumps(wcfg))
        env = {"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1"}
        self.proc = subprocess.Popen([sys.executable, "-m", "agent.srlm.sandbox_worker", "vars.json", "cfg.json"],
                                     cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env=env, text=True, bufsize=1)

    def _readline(self, deadline: float) -> str | None:
        assert self.proc and self.proc.stdout
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        r, _, _ = select.select([self.proc.stdout], [], [], remaining)
        if not r:
            return None
        return self.proc.stdout.readline()

    def execute(self, code: str) -> ExecResult:
        t0 = time.time()
        if self.proc is None or self.proc.poll() is not None:
            self.violations.append("worker exited")
            self._restart()
            return ExecResult(error="REPL worker had exited; state was reset — recompute any variables you need",
                              seconds=time.time() - t0, violation="worker exited")
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(json.dumps({"type": "exec", "code": code}) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self._restart()
            return ExecResult(error="REPL process died; state was reset", seconds=time.time() - t0,
                              violation="worker died")
        deadline = t0 + self.timeout
        while True:
            line = self._readline(deadline)
            if line is None:
                self.violations.append("timeout")
                self._restart()
                return ExecResult(error=f"TimeoutError: step exceeded {self.timeout:.0f}s; REPL state was reset",
                                  seconds=time.time() - t0, violation="timeout", timed_out=True)
            if not line:   # worker exited (e.g. CPU limit / segfault)
                self.violations.append("worker exited")
                self._restart()
                return ExecResult(error="REPL worker exited (resource limit); state was reset",
                                  seconds=time.time() - t0, violation="worker exited")
            msg = json.loads(line)
            if msg.get("type") == "subcall":
                out = self.subcall(msg["query"], msg["text"]) if self.subcall else "sub_call disabled"
                self.proc.stdin.write(json.dumps({"type": "subcall_result", "output": out}) + "\n")
                self.proc.stdin.flush()
                continue
            res = ExecResult(stdout=msg["stdout"], error=msg["error"], final=msg["final"],
                             has_final=msg["has_final"], violation=msg["violation"], seconds=time.time() - t0)
            if res.violation:
                self.violations.append(res.violation)
            return res

    def _restart(self) -> None:
        self._kill()
        self._start()

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.proc:
            for s in (self.proc.stdin, self.proc.stdout):
                try:
                    s and s.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._kill()
        self.tmp.cleanup()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
