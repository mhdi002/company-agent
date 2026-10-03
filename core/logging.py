"""Structured logging: JSONL + human-readable lines, separate channel files.

Channels: agent, tools, training, errors, srlm. Every event is written to
`logs/<channel>.jsonl` and a human line to `logs/<channel>.log`
(`srlm` is JSONL only, as `srlm.jsonl`). Events with status "error" are
mirrored to `errors.log`/`errors.jsonl`. A bounded in-memory ring buffer
feeds the UI live log view.
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import threading
import traceback
from pathlib import Path
from typing import Any

from core.config import resolve

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"run_id": "-", "dir": None}
RING: collections.deque = collections.deque(maxlen=5000)
TRUNC = 2000


def configure(log_dir: str | Path = "logs", run_id: str | None = None, ring_size: int | None = None,
              truncate_chars: int | None = None) -> None:
    """Set the log directory, current run id and (optionally) buffer/truncation sizes from config."""
    global RING, TRUNC
    with _LOCK:
        if ring_size and ring_size != RING.maxlen:
            RING = collections.deque(RING, maxlen=int(ring_size))
        if truncate_chars:
            TRUNC = int(truncate_chars)
        d = resolve(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = d
        if run_id:
            _STATE["run_id"] = run_id


def set_run_id(run_id: str) -> None:
    _STATE["run_id"] = run_id


def run_id() -> str:
    return _STATE["run_id"]


def log_dir() -> Path:
    if _STATE["dir"] is None:
        configure()
    return _STATE["dir"]


def truncate(v: Any, n: int | None = None) -> Any:
    """Truncate long strings (recursively) so logs stay readable."""
    n = n or TRUNC
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"...[+{len(v) - n} chars]"
    if isinstance(v, dict):
        return {k: truncate(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [truncate(x, n) for x in v[:200]]
    return v


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _write(name: str, line: str) -> None:
    with open(log_dir() / name, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def human_line(rec: dict) -> str:
    """Format: timestamp | run_id | company | step | tool | status | duration | details."""
    dur = rec.get("duration")
    dur_s = f"{dur:.2f}s" if isinstance(dur, (int, float)) else "-"
    details = rec.get("details", "")
    if not isinstance(details, str):
        details = json.dumps(details, ensure_ascii=False, default=str)
    details = details.replace("\n", " ⏎ ")[:500]
    return " | ".join(str(x) for x in [rec["timestamp"], rec["run_id"], rec.get("company") or "-",
                                       rec.get("step") or "-", rec.get("tool") or "-",
                                       rec.get("status") or "-", dur_s, details])


class EventLogger:
    """Logger bound to one channel."""

    def __init__(self, channel: str):
        self.channel = channel

    def event(self, step: str, status: str = "info", *, company: str | None = None,
              tool: str | None = None, duration: float | None = None, details: Any = "",
              **extra: Any) -> dict:
        rec = {"timestamp": _now(), "run_id": run_id(), "channel": self.channel,
               "company": company, "step": step, "tool": tool, "status": status,
               "duration": duration, "details": truncate(details)}
        rec.update({k: truncate(v) for k, v in extra.items()})
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _LOCK:
            _write(f"{self.channel}.jsonl", line)
            if self.channel != "srlm":
                _write(f"{self.channel}.log", human_line(rec))
            if status == "error" and self.channel != "errors":
                _write("errors.jsonl", line)
                _write("errors.log", human_line(rec))
            RING.append(rec)
        return rec

    def error(self, step: str, exc: BaseException | None = None, **kw: Any) -> dict:
        tb = "".join(traceback.format_exception(exc)) if exc else traceback.format_exc()
        kw.setdefault("details", f"{type(exc).__name__}: {exc}" if exc else "error")
        return self.event(step, "error", traceback=tb, **kw)


_LOGGERS: dict[str, EventLogger] = {}


def get_logger(channel: str) -> EventLogger:
    with _LOCK:
        if channel not in _LOGGERS:
            _LOGGERS[channel] = EventLogger(channel)
        return _LOGGERS[channel]


def recent(channel: str | None = None, status: str | None = None, company: str | None = None,
           text: str | None = None, limit: int = 300) -> list[dict]:
    """Filtered view over the in-memory ring buffer (for the UI)."""
    out = []
    for rec in reversed(RING):
        if channel and rec.get("channel") != channel:
            continue
        if status and rec.get("status") != status:
            continue
        if company and company.lower() not in str(rec.get("company") or "").lower():
            continue
        if text and text.lower() not in json.dumps(rec, default=str).lower():
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    return list(reversed(out))


def load_recent_from_files(max_lines: int = 3000) -> int:
    """Seed the ring buffer from the tail of the JSONL logs (e.g. after a UI server restart)."""
    if RING:
        return 0
    recs = []
    for name in ("agent.jsonl", "tools.jsonl", "errors.jsonl", "training.jsonl", "srlm.jsonl"):
        p = log_dir() / name
        if not p.exists():
            continue
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - 2_000_000, 0))
            lines = f.read().decode("utf-8", errors="replace").splitlines()[-max_lines:]
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if name == "errors.jsonl" and rec.get("channel") != "errors":
                continue   # mirrored error lines are already in their own channel file
            if rec.get("step") == "srlm.step":
                continue   # per-step prompt records are too verbose for the live view
            recs.append(rec)
    recs.sort(key=lambda r: r.get("timestamp", ""))
    for r in recs[-RING.maxlen:]:
        RING.append(r)
    return len(recs)
