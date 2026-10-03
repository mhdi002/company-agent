"""Persistent agent state (crash recovery): one JSON file written atomically after every change."""
from __future__ import annotations

import copy
import datetime as dt
import json
import threading
from pathlib import Path
from typing import Any


def today() -> str:
    return dt.date.today().isoformat()


class StateStore:
    """Thread-safe JSON state. Layout:

    {run_id, status, started_at, finished_at, counters{found,processed,failed,sent},
     todo[...], companies{domain: {...}}, processed_domains[...], daily{date: n}, current_company}
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                bad = self.path.with_suffix(".corrupt.json")
                self.path.replace(bad)
        return self.fresh()

    @staticmethod
    def fresh() -> dict:
        return {"run_id": None, "status": "idle", "started_at": None, "finished_at": None,
                "counters": {"found": 0, "processed": 0, "failed": 0, "sent": 0}, "todo": [], "companies": {},
                "processed_domains": [], "daily": {}, "current_company": None}

    def save(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(self.path)

    def update(self, **kw: Any) -> None:
        with self.lock:
            self.data.update(kw)
            self.save()

    def company(self, domain: str) -> dict:
        with self.lock:
            return self.data["companies"].setdefault(domain, {"domain": domain, "status": "pending", "steps": {}})

    def set_company(self, domain: str, **kw: Any) -> None:
        with self.lock:
            self.company(domain).update(kw)
            self.save()

    def save_step(self, domain: str, step: str, value: Any) -> None:
        with self.lock:
            self.company(domain)["steps"][step] = value
            self.save()

    def bump(self, counter: str, n: int = 1) -> None:
        with self.lock:
            self.data["counters"][counter] = self.data["counters"].get(counter, 0) + n
            self.save()

    def mark_processed(self, domain: str) -> None:
        with self.lock:
            if domain not in self.data["processed_domains"]:
                self.data["processed_domains"].append(domain)
            d = self.data["daily"]
            d[today()] = d.get(today(), 0) + 1
            self.save()

    def processed_today(self) -> int:
        return self.data["daily"].get(today(), 0)

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.data)
