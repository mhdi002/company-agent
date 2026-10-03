import time

import pytest
from fastapi.testclient import TestClient

import app.server as server
from agent.telegram_bot import TelegramBot
from tests.test_agent import make_agent, make_cfg
from tools.telegram import TelegramFatal as TelegramError


class FakeClient:
    """Stands in for TelegramClient: records sends, serves queued updates."""

    def __init__(self, valid_token="123:abc"):
        self.valid = valid_token
        self.token = self.chat_id = None
        self.sent, self.updates = [], []

    def reconfigure(self, token, chat_id):
        self.token, self.chat_id = token or None, (str(chat_id) if chat_id else None)

    def get_me(self):
        if self.token != self.valid:
            raise TelegramError("401: Unauthorized")
        return {"id": 1, "username": "pa_test_bot"}

    def delete_webhook(self):
        pass

    def get_updates(self, offset, timeout_s):
        if self.updates:
            u, self.updates = self.updates, []
            return u
        time.sleep(0.05)
        return []

    @property
    def configured(self):
        return bool(self.token and self.chat_id)

    def send_message(self, text, company=None):
        self.sent.append((self.chat_id, text))
        return [{"delivered": True, "message_id": len(self.sent)}]


def upd(i, chat_id, text):
    return {"update_id": i, "message": {"chat": {"id": chat_id, "first_name": "Mahdi"}, "text": text}}


def make_bot(env):
    calls = []
    actions = {"run": lambda: calls.append("run") or "Started",
               "stop": lambda: calls.append("stop") or "Stopping",
               "status": lambda: "Status: idle", "logs": lambda: "Log files sent."}
    bot = TelegramBot({"telegram": {"commands": True, "poll_timeout_s": 1}}, actions, client=FakeClient(),
                      env_writer=env.update)
    return bot, calls


def test_connect_pair_and_commands():
    env = {}
    bot, calls = make_bot(env)
    with pytest.raises(ValueError):
        bot.connect("bad-token")
    st = bot.connect("123:abc")
    assert st["bot_username"] == "pa_test_bot" and st["pairing"] and env["TELEGRAM_BOT_TOKEN"] == "123:abc"
    assert st["link"] == "https://t.me/pa_test_bot"
    bot.client.updates = [upd(1, 4242, "/start")]
    assert bot.wait_paired(5)
    assert env["TELEGRAM_CHAT_ID"] == "4242" and "Connected" in bot.client.sent[-1][1]
    bot.handle(upd(2, 999, "/run"))                 # a stranger: ignored
    assert calls == []
    bot.handle(upd(3, 4242, "/run@pa_test_bot"))
    bot.handle(upd(4, 4242, "/status"))
    bot.handle(upd(5, 4242, "/stop"))
    assert calls == ["run", "stop"]
    assert bot.client.sent[-2][1] == "Status: idle" and bot.client.sent[-1][1] == "Stopping"
    bot.disconnect()
    assert env["TELEGRAM_BOT_TOKEN"] == "" and bot.client.token is None


def test_app_telegram_endpoints(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(server, "load_config", lambda: cfg)
    env = {}
    fake = FakeClient()

    def bot_factory(cfg_, actions, client=None):
        return TelegramBot(cfg_, actions, client=fake, env_writer=env.update)
    app = server.create_app(agent_factory=lambda c: make_agent(cfg), bot_factory=bot_factory, restore_telegram=False)
    c = TestClient(app)
    assert c.get("/api/telegram").json()["token_set"] is False
    assert c.post("/api/telegram/connect", json={"token": "nope"}).status_code == 400
    assert c.post("/api/telegram/test").status_code == 409
    r = c.post("/api/telegram/connect", json={"token": "123:abc"}).json()
    assert r["pairing"] and r["bot_username"] == "pa_test_bot"
    fake.updates = [upd(1, 77, "/start")]
    deadline = time.time() + 5
    while time.time() < deadline and c.get("/api/telegram").json()["pairing"]:
        time.sleep(0.1)
    st = c.get("/api/telegram").json()
    assert st["chat_set"] and not st["pairing"] and env["TELEGRAM_CHAT_ID"] == "77"
    assert c.post("/api/telegram/test").json()["delivered"]
    assert "Test message" in fake.sent[-1][1]
    s = c.get("/api/status").json()
    assert "heartbeat" in s and "continuous" in s
    c.post("/api/telegram/disconnect")
    assert c.get("/api/telegram").json()["token_set"] is False
