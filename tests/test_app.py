import time

from fastapi.testclient import TestClient

import app.server as server
from tests.test_agent import make_agent, make_cfg


def client(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(server, "load_config", lambda: cfg)
    saved = {}
    monkeypatch.setattr(server, "save_override", lambda v: saved.setdefault("override", v))
    monkeypatch.setattr(server, "save_env", lambda v: saved.setdefault("env", v))
    app = server.create_app(agent_factory=lambda c: make_agent(cfg))
    return TestClient(app), saved


def wait_idle(c, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline and c.get("/api/status").json()["running"]:
        time.sleep(0.2)


def test_start_status_logs_files(tmp_path, monkeypatch):
    c, _ = client(tmp_path, monkeypatch)
    assert c.get("/").status_code == 200 and "ProposalAgent" in c.get("/").text
    s = c.get("/api/status").json()
    assert s["status"] == "idle" and s["counters"]["found"] == 0
    assert c.post("/api/start").json()["running"] is True
    wait_idle(c)
    s = c.get("/api/status").json()
    assert s["status"] == "done" and s["counters"]["processed"] == 3
    assert all(i["status"] == "done" for i in s["todo"])
    logs = c.get("/api/logs", params={"company": "nordwind", "status": "done"}).json()
    assert logs and all("nordwind" in (r["company"] or "") and r["status"] == "done" for r in logs)
    files = c.get("/api/files").json()
    assert len(files["proposals"]) == 3 and len(files["records"]) == 3
    r = c.get(f"/api/files/proposals/{files['proposals'][0]}")
    assert r.status_code == 200 and r.content[:2] == b"PK"
    assert c.get("/api/files/proposals/..%2Fstate.json").status_code == 404
    assert c.get("/api/files/other/x").status_code == 404


def test_settings_roundtrip(tmp_path, monkeypatch):
    c, saved = client(tmp_path, monkeypatch)
    s = c.get("/api/settings").json()
    assert s["srlm"]["K"] == 3 and "telegram_configured" in s["secrets"]
    r = c.post("/api/settings", json={"srlm": {"K": "5", "use_trace_length": "false"},
                                      "agent": {"countries": "Germany, France"},
                                      "secrets": {"telegram_token": "123:abc"}})
    assert r.status_code == 200
    assert saved["override"] == {"srlm": {"K": 5, "use_trace_length": False}, "agent": {"countries": ["Germany", "France"]}}
    assert saved["env"] == {"TELEGRAM_BOT_TOKEN": "123:abc"}
    assert "123:abc" not in r.text                     # secrets are never echoed back
    assert c.post("/api/settings", json={"paths": {"logs": "/etc"}}).status_code == 400


def test_stop_and_reset(tmp_path, monkeypatch):
    c, _ = client(tmp_path, monkeypatch)
    c.post("/api/start")
    c.post("/api/stop")
    wait_idle(c)
    assert c.get("/api/status").json()["status"] in ("stopped", "done")
    assert c.post("/api/reset").json()["reset"] is True
    assert c.get("/api/status").json()["status"] == "idle"
