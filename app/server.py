"""FastAPI UI server: Start/Stop, live to-do list, counters, log viewer, settings, proposal downloads.

Run:  python -m app            (http://127.0.0.1:8000)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from core import logging as plog
from core.config import load_config, resolve, save_env, save_override, secret

STATIC = Path(__file__).parent / "static"

# Settings the UI may edit: section -> {key: type}
EDITABLE: dict[str, dict[str, type]] = {
    "agent": {"countries": list, "industries": list, "companies_per_run": int, "daily_limit": int,
              "concurrency": int, "policy": str, "send_logs_every": int, "continuous": bool,
              "run_interval_minutes": float},
    "srlm": {"K": int, "temperature": float, "top_p": float, "max_steps": int, "step_timeout_s": int,
             "use_self_consistency": bool, "use_verbalized_confidence": bool, "use_trace_length": bool,
             "use_subcalls": bool, "direct_baseline": bool, "max_retries": int},
    "search": {"provider": str, "seed_file": str},
    "research": {"provider": str, "max_sources": int},
    "fetch": {"min_delay_s": float, "max_pages_per_site": int, "respect_robots": bool},
    "telegram": {"enabled": bool, "heartbeat_minutes": float, "commands": bool},
    "docx": {"backend": str},
}


class AppState:
    def __init__(self, agent_factory=None, bot_factory=None):
        self.lock = threading.Lock()
        self.factory = agent_factory or self._default_factory
        self.bot_factory = bot_factory
        self.agent = None
        self.bot = None

    def get_bot(self):
        """The Telegram bot shares the agent's Telegram client, so pairing updates delivery immediately."""
        if self.bot is None:
            from agent.telegram_bot import TelegramBot
            a = self.get()
            actions = {
                "run": lambda: f"Started run {self.get().start(background=True)}",
                "stop": lambda: (self.get().stop(), "Stopping after the current step.")[1],
                "status": lambda: self.get().status_text(),
                "logs": lambda: (self.get()._send_logs(final=False), "Log files sent.")[1],
            }
            self.bot = (self.bot_factory or TelegramBot)(a.cfg, actions, client=a.telegram)
        return self.bot

    @staticmethod
    def _default_factory(cfg: dict):
        from agent.loop import Agent
        from tools.http import HttpClient
        http = None
        if cfg["search"]["provider"] == "offline":   # offline mode: bundled fixture websites
            from data.sample.companies import all_pages
            http = HttpClient(cfg["fetch"], cache_dir=resolve(cfg["paths"]["cache"]) / "http", offline=all_pages())
        return Agent(cfg, http=http)

    def get(self):
        with self.lock:
            if self.agent is None:
                cfg = load_config()
                plog.configure(cfg["paths"]["logs"])
                plog.load_recent_from_files()
                self.agent = self.factory(cfg)
            return self.agent

    def rebuild(self) -> None:
        with self.lock:
            if self.agent is not None and self.agent.running:
                raise HTTPException(409, "stop the run before changing settings")
            if self.bot is not None:
                self.bot.stop_listening()
            self.agent = None
            self.bot = None


def _coerce(val: Any, typ: type) -> Any:
    if typ is list:
        return [x.strip() for x in val.split(",") if x.strip()] if isinstance(val, str) else list(val)
    if typ is bool:
        return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
    return typ(val)


def create_app(agent_factory=None, bot_factory=None, restore_telegram: bool = True) -> FastAPI:
    app = FastAPI(title="ProposalAgent")
    state = AppState(agent_factory, bot_factory)
    app.state.pa = state

    @app.on_event("startup")
    def restore_bot() -> None:
        if restore_telegram:
            try:
                state.get_bot().restore()
            except Exception as e:  # the UI shows the error; the app still starts
                plog.get_logger("tools").error("telegram.restore", e, tool="telegram")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        a = state.get()
        s = a.snapshot()
        return {"run_id": s["run_id"], "status": s["status"], "running": s["running"], "policy": s["policy"],
                "counters": s["counters"], "current_company": s["current_company"], "todo": s["todo"],
                "companies": {d: {"name": c.get("name"), "status": c["status"], "error": c.get("error"),
                                  "files": c.get("files")} for d, c in s["companies"].items()},
                "started_at": s["started_at"], "finished_at": s["finished_at"],
                "processed_today": a.store.processed_today(), "heartbeat": s.get("heartbeat"),
                "next_run_at": s.get("next_run_at"), "continuous": bool(a.cfg["agent"].get("continuous"))}

    @app.get("/api/telegram")
    def telegram_status() -> dict:
        return state.get_bot().status()

    @app.post("/api/telegram/connect")
    def telegram_connect(body: dict = Body(...)) -> dict:
        token = str(body.get("token") or "").strip()
        if not token:
            raise HTTPException(400, "token is required")
        try:
            return state.get_bot().connect(token)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/telegram/test")
    def telegram_test() -> dict:
        bot = state.get_bot()
        if not bot.client.configured:
            raise HTTPException(409, "connect the bot and press Start in its chat first")
        rec = bot.client.send_message("👋 Test message from ProposalAgent. " + state.get().status_text())
        return {"delivered": all(r.get("delivered") for r in rec), "receipts": rec}

    @app.post("/api/telegram/disconnect")
    def telegram_disconnect() -> dict:
        state.get_bot().disconnect()
        return {"disconnected": True}

    @app.post("/api/start")
    def start() -> dict:
        a = state.get()
        return {"run_id": a.start(background=True), "running": a.running}

    @app.post("/api/stop")
    def stop() -> dict:
        a = state.get()
        a.stop()
        return {"stopping": True}

    @app.post("/api/reset")
    def reset() -> dict:
        a = state.get()
        try:
            a.reset()
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"reset": True}

    @app.get("/api/logs")
    def logs(channel: str | None = None, status: str | None = None, company: str | None = None,
             q: str | None = None, limit: int = 300) -> list[dict]:
        state.get()
        recs = plog.recent(channel or None, status or None, company or None, q or None, min(limit, 2000))
        return [{k: r.get(k) for k in ("timestamp", "run_id", "channel", "company", "step", "tool", "status",
                                       "duration", "details")} for r in recs if r.get("step") != "srlm.step"]

    @app.get("/api/settings")
    def get_settings() -> dict:
        cfg = load_config()
        out = {sec: {k: cfg.get(sec, {}).get(k) for k in keys} for sec, keys in EDITABLE.items()}
        out["secrets"] = {"telegram_configured": bool(secret("TELEGRAM_BOT_TOKEN") and secret("TELEGRAM_CHAT_ID"))}
        return out

    @app.post("/api/settings")
    def post_settings(body: dict = Body(...)) -> dict:
        values: dict = {}
        unknown = set(body) - set(EDITABLE) - {"secrets"}
        if unknown:
            raise HTTPException(400, f"not editable: {sorted(unknown)}")
        for sec, keys in EDITABLE.items():
            for k, v in (body.get(sec) or {}).items():
                if k not in keys:
                    raise HTTPException(400, f"{sec}.{k} is not editable")
                try:
                    values.setdefault(sec, {})[k] = _coerce(v, keys[k])
                except (TypeError, ValueError) as e:
                    raise HTTPException(400, f"{sec}.{k}: {e}") from e
        sec_in = body.get("secrets") or {}
        env = {k: str(sec_in[f]).strip() for f, k in (("telegram_token", "TELEGRAM_BOT_TOKEN"),
                                                      ("telegram_chat_id", "TELEGRAM_CHAT_ID")) if sec_in.get(f)}
        state.rebuild()
        if values:
            save_override(values)
        if env:
            save_env(env)
        plog.get_logger("agent").event("settings", "saved", details={"changed": values, "secrets_updated": sorted(env)})
        return {"saved": True, "changed": values, "secrets_updated": sorted(env)}

    @app.get("/api/files")
    def files() -> dict:
        cfg = load_config()
        out = {}
        for kind in ("proposals", "records"):
            d = resolve(cfg["paths"][kind])
            out[kind] = sorted((p.name for p in d.glob("*") if p.suffix in (".docx", ".json")), reverse=True) if d.exists() else []
        return out

    @app.get("/api/files/{kind}/{name}")
    def download(kind: str, name: str):
        if kind not in ("proposals", "records"):
            raise HTTPException(404)
        d = resolve(load_config()["paths"][kind]).resolve()
        p = (d / name).resolve()
        if p.parent != d or not p.exists():
            raise HTTPException(404)
        return FileResponse(p, filename=p.name)

    return app


app = create_app()
