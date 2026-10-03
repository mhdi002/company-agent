"""Telegram Bot API delivery: messages (chunked) and documents, with retry and delivery confirmation.

Token and chat id come from the environment (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, via .env).
Every call returns a receipt; delivery is confirmed only when the API answers ok=true with a message_id.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import requests

from core.config import secret
from core.logging import get_logger
from tools.http import with_retries

log = get_logger("tools")
API = "https://api.telegram.org/bot{token}/{method}"


class TelegramError(Exception):
    pass


def chunk_text(text: str, size: int = 3900) -> list[str]:
    """Split on paragraph/line boundaries where possible; never exceed `size` chars."""
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(buf) + len(line) > size:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)
    return chunks or [""]


class TelegramClient:
    def __init__(self, cfg: dict, token: str | None = None, chat_id: str | None = None,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep):
        tcfg = cfg.get("telegram", cfg)
        self.enabled = bool(tcfg.get("enabled", True))
        self.token = token or secret("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or secret("TELEGRAM_CHAT_ID")
        self.chunk = int(tcfg.get("chunk_chars", 3900))
        self.retries = int(tcfg.get("retries", 4))
        self.timeout = float(tcfg.get("timeout_s", 30))
        self.session = session or requests.Session()
        self.sleep = sleep

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.token and self.chat_id)

    def _call(self, method: str, data: dict, files: dict | None = None) -> dict:
        url = API.format(token=self.token, method=method)

        def once() -> dict:
            fh = {k: (Path(p).name, open(p, "rb")) for k, p in (files or {}).items()}
            try:
                r = self.session.post(url, data=data, files=fh or None, timeout=self.timeout)
            finally:
                for _, (_, f) in fh.items():
                    f.close()
            try:
                body = r.json()
            except ValueError:
                body = {"ok": False, "description": r.text[:200]}
            if r.status_code == 429 or r.status_code >= 500:
                err = TelegramError(f"{r.status_code}: {body.get('description')}")
                err.retry_after = (body.get("parameters") or {}).get("retry_after")  # type: ignore[attr-defined]
                raise err
            if not body.get("ok"):
                raise TelegramError(f"{r.status_code}: {body.get('description')}")
            return body
        return with_retries(once, self.retries, 1.0, (TelegramError, requests.RequestException), self.sleep, "telegram")

    def _receipt(self, kind: str, body: dict | None, error: str | None = None) -> dict:
        mid = (body or {}).get("result", {}).get("message_id") if body else None
        return {"kind": kind, "delivered": bool(body and body.get("ok") and mid), "message_id": mid, "error": error}

    def send_message(self, text: str, company: str | None = None) -> list[dict]:
        if not self.configured:
            log.event("telegram.send_message", "skipped", tool="telegram", company=company,
                      details="telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
            return [self._receipt("message", None, "not configured")]
        receipts = []
        for part in chunk_text(text, self.chunk):
            t0 = time.time()
            try:
                body = self._call("sendMessage", {"chat_id": self.chat_id, "text": part, "disable_web_page_preview": True})
                receipts.append(self._receipt("message", body))
                log.event("telegram.send_message", "ok", tool="telegram", company=company, duration=time.time() - t0,
                          details=f"{len(part)} chars, message_id={receipts[-1]['message_id']}")
            except Exception as e:
                receipts.append(self._receipt("message", None, str(e)))
                log.error("telegram.send_message", e, tool="telegram", company=company)
        return receipts

    def send_document(self, path: str | Path, caption: str = "", company: str | None = None) -> dict:
        if not self.configured:
            log.event("telegram.send_document", "skipped", tool="telegram", company=company,
                      details=f"not configured; {Path(path).name} kept locally")
            return self._receipt("document", None, "not configured")
        t0 = time.time()
        try:
            body = self._call("sendDocument", {"chat_id": self.chat_id, "caption": caption[:1000]}, {"document": str(path)})
            rec = self._receipt("document", body)
            log.event("telegram.send_document", "ok", tool="telegram", company=company, duration=time.time() - t0,
                      details=f"{Path(path).name} message_id={rec['message_id']}")
            return rec
        except Exception as e:
            log.error("telegram.send_document", e, tool="telegram", company=company, details=f"{Path(path).name}: {e}")
            return self._receipt("document", None, str(e))
