"""Telegram bot wired to the app: connect/pair from the UI, and control the agent from the chat.

Pairing: the user pastes the bot token in the UI → `connect()` verifies it with getMe and stores it in
.env → the user opens the bot in Telegram and presses Start (sends /start) → the listener adopts that
chat as the agent's chat (stored in .env) and confirms. From then on only that chat is obeyed.

Commands (from the paired chat): /run  /stop  /status  /logs  /help
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from core.config import save_env, secret
from core.logging import get_logger
from tools.telegram import TelegramClient

log = get_logger("tools")

HELP = ("ProposalAgent bot\n/run — start (or resume) the agent\n/stop — stop after the current step\n"
        "/status — counters and current company\n/logs — send the log files\n/help — this message")


class TelegramBot:
    def __init__(self, cfg: dict, actions: dict[str, Callable[[], str]], client: TelegramClient | None = None,
                 env_writer: Callable[[dict], None] = save_env):
        self.cfg = cfg
        self.tcfg = cfg.get("telegram", {})
        self.client = client or TelegramClient(cfg)
        self.actions = actions            # run, stop, status, logs → reply text
        self.env_writer = env_writer
        self.username: str | None = None
        self.chat_title: str | None = None
        self.pairing = False
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset: int | None = None

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        return {"token_set": bool(self.client.token), "bot_username": self.username,
                "chat_set": bool(self.client.chat_id), "chat_title": self.chat_title, "pairing": self.pairing,
                "listening": bool(self._thread and self._thread.is_alive()), "last_error": self.last_error,
                "link": f"https://t.me/{self.username}" if self.username else None}

    # ------------------------------------------------------------------ connect / pair
    def connect(self, token: str) -> dict:
        """Verify a bot token, store it, and wait for the user to press Start in the bot's chat."""
        token = token.strip()
        old = (self.client.token, self.client.chat_id)
        self.client.reconfigure(token, None)
        try:
            me = self.client.get_me()
        except Exception as e:
            self.client.reconfigure(*old)
            raise ValueError(f"Telegram rejected the token: {e}") from e
        self.username = me.get("username")
        self.env_writer({"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": ""})
        self.pairing = True
        self.chat_title = None
        self.last_error = None
        log.event("telegram.connect", "ok", tool="telegram", details=f"bot @{self.username}; waiting for /start")
        self.start_listening()
        return self.status()

    def disconnect(self) -> None:
        self.stop_listening()
        self.client.reconfigure(None, None)
        self.username = self.chat_title = None
        self.pairing = False
        self.env_writer({"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""})
        log.event("telegram.disconnect", "ok", tool="telegram")

    def restore(self) -> None:
        """On app start: reconnect to a previously configured bot."""
        token, chat = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
        if not token:
            return
        self.client.reconfigure(token, chat)
        try:
            self.username = self.client.get_me().get("username")
        except Exception as e:
            self.last_error = str(e)
            log.error("telegram.restore", e, tool="telegram")
            return
        self.pairing = not chat
        self.start_listening()

    # ------------------------------------------------------------------ listener
    def start_listening(self) -> None:
        if not self.tcfg.get("commands", True) and not self.pairing:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        try:
            self.client.delete_webhook()
        except Exception as e:
            self.last_error = f"deleteWebhook: {e}"
        self._thread = threading.Thread(target=self._loop, name="telegram-bot", daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        timeout = int(self.tcfg.get("poll_timeout_s", 25))
        while not self._stop.is_set() and self.client.token:
            try:
                updates = self.client.get_updates(self._offset, timeout)
                self.last_error = None
            except Exception as e:
                self.last_error = str(e)
                log.event("telegram.poll", "retry", tool="telegram", details=str(e)[:200])
                self._stop.wait(10)
                continue
            for u in updates:
                self._offset = u["update_id"] + 1
                try:
                    self.handle(u)
                except Exception as e:  # a bad message never kills the listener
                    log.error("telegram.handle", e, tool="telegram")

    def handle(self, update: dict) -> None:
        msg = update.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        text = (msg.get("text") or "").strip()
        cmd = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
        if not chat_id:
            return
        if self.pairing and cmd in ("/start", "/connect"):
            self.client.reconfigure(self.client.token, chat_id)
            self.env_writer({"TELEGRAM_CHAT_ID": chat_id})
            self.pairing = False
            self.chat_title = chat.get("title") or chat.get("username") or chat.get("first_name") or chat_id
            log.event("telegram.paired", "ok", tool="telegram", details=f"chat {self.chat_title}")
            self.client.send_message("✅ Connected to ProposalAgent.\n" + HELP)
            return
        if chat_id != str(self.client.chat_id):
            return                       # only the paired chat may control the agent
        if self.chat_title is None:
            self.chat_title = chat.get("title") or chat.get("username") or chat.get("first_name") or chat_id
        reply = {"/run": "run", "/stop": "stop", "/status": "status", "/logs": "logs"}.get(cmd)
        if reply:
            log.event("telegram.command", "ok", tool="telegram", details=cmd)
            self.client.send_message(self.actions[reply]())
        elif cmd in ("/start", "/help"):
            self.client.send_message(HELP)

    def wait_paired(self, timeout_s: float) -> bool:
        """Block until pairing completes (used by tests and CLI)."""
        end = time.time() + timeout_s
        while time.time() < end:
            if not self.pairing and self.client.chat_id:
                return True
            time.sleep(0.1)
        return False
