"""The visible to-do list: run-level and per-company items with pending/in_progress/done/failed status."""
from __future__ import annotations

import datetime as dt

from core.logging import get_logger

log = get_logger("agent")

COMPANY_STEPS = [
    ("fields", "Fetch the companies' working fields"),
    ("research", "Research the field: existing work and possible needed projects"),
    ("text", "Find the text needed for the proposal"),
    ("docx_skill", "Fetch and read the Word docx skill"),
    ("finalize", "Finalize the proposal"),
    ("save_send", "Save everything (and send via Telegram)"),
]
STATUSES = ("pending", "in_progress", "done", "failed", "skipped")


class TodoList:
    """Backed by the StateStore's `todo` list so it survives restarts."""

    def __init__(self, store):
        self.store = store

    @property
    def items(self) -> list[dict]:
        return self.store.data["todo"]

    def ensure_run_item(self) -> None:
        with self.store.lock:
            if not any(i["id"] == "gather" for i in self.items):
                self.items.insert(0, {"id": "gather", "n": 1, "title": "Gather companies' websites", "company": None,
                                      "status": "pending", "updated": None, "error": None})
                self.store.save()

    def add_company(self, domain: str) -> None:
        with self.store.lock:
            if any(i["company"] == domain for i in self.items):
                return
            for n, (sid, title) in enumerate(COMPANY_STEPS, start=2):
                self.items.append({"id": f"{domain}:{sid}", "n": n, "title": title, "company": domain,
                                   "status": "pending", "updated": None, "error": None})
            self.store.save()

    def get(self, item_id: str) -> dict | None:
        return next((i for i in self.items if i["id"] == item_id), None)

    def set(self, item_id: str, status: str, error: str | None = None) -> None:
        assert status in STATUSES, status
        with self.store.lock:
            it = self.get(item_id)
            if it is None:
                return
            old = it["status"]
            it.update(status=status, error=error, updated=dt.datetime.now(dt.timezone.utc).isoformat())
            self.store.save()
        log.event(item_id, status, company=it["company"], tool="todo",
                  details=f"[{it['n']}] {it['title']}: {old} → {status}" + (f" ({error})" if error else ""))

    def company_items(self, domain: str) -> list[dict]:
        return [i for i in self.items if i["company"] == domain]
