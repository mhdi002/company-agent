import json
import time
from pathlib import Path

import pytest
import requests

from agent.loop import Agent
from agent.policy import TemplatePolicy
from agent.proposal import TemplateWriter
from core.config import load_config
from data.sample.companies import all_pages
from tools.http import HttpClient
from tools.telegram import TelegramClient


class OkResp:
    status_code = 200
    text = ""

    def __init__(self, n):
        self.n = n

    def json(self):
        return {"ok": True, "result": {"message_id": self.n}}


class TgSession:
    def __init__(self, fail=False):
        self.n, self.fail, self.headers, self.sent = 0, fail, {}, []

    def post(self, url, data=None, files=None, timeout=None):
        if self.fail:
            raise requests.ConnectionError("telegram down")
        self.n += 1
        self.sent.append((url.rsplit("/", 1)[-1], data, list((files or {}).keys())))
        return OkResp(self.n)


def make_cfg(tmp_path: Path, **agent_kw) -> dict:
    cfg = load_config(overrides={
        "paths": {"state": str(tmp_path / "state.json"), "proposals": str(tmp_path / "proposals"),
                  "records": str(tmp_path / "records"), "cache": str(tmp_path / "cache"), "logs": str(tmp_path / "logs")},
        "agent": {"countries": ["Germany", "United Kingdom"], "industries": ["logistics", "renewable energy"],
                  "companies_per_run": 10, "daily_limit": 50, "concurrency": 2, "policy": "template",
                  "send_logs_every": 2, "continuous": False, **agent_kw},
        "search": {"provider": "offline"}, "research": {"provider": "offline"},
        "srlm": {"K": 3, "max_steps": 6, "max_retries": 0},
        "fetch": {"min_delay_s": 0.0, "cache": False},
        "docx": {"backend": "python-docx"},
    })
    return cfg


def make_agent(cfg, pages=None, tg_session=None, policy=None) -> Agent:
    http = HttpClient(cfg["fetch"], offline=all_pages() if pages is None else pages)
    tg = TelegramClient(cfg, token="T", chat_id="C", session=tg_session or TgSession(), sleep=lambda s: None)
    return Agent(cfg, http=http, telegram=tg, policy=policy or TemplatePolicy(), count_tokens=lambda s: len(s.split()),
                 writer=TemplateWriter())


def test_full_offline_run(tmp_path):
    cfg = make_cfg(tmp_path)
    tg = TgSession()
    agent = make_agent(cfg, tg_session=tg)
    agent.start(background=False)
    s = agent.snapshot()
    assert s["status"] == "done"
    assert s["counters"] == {"found": 3, "processed": 3, "failed": 0, "skipped": 0, "sent": 3}
    assert {d for d, c in s["companies"].items() if c["status"] == "done"} == {
        "nordwind-solar.example", "rheinland-freight.example", "harbourline-logistics.example"}
    assert all(i["status"] == "done" for i in s["todo"])
    assert len([i for i in s["todo"] if i["company"]]) == 3 * 6
    docs = list((tmp_path / "proposals").glob("*.docx"))
    recs = list((tmp_path / "records").glob("*.json"))
    assert len(docs) == 3 and len(recs) == 3
    rec = json.loads(recs[0].read_text())
    titles = [x["title"] for x in rec["sections"]]
    assert titles[0] == "Cover" and titles[-1] == "Sources" and "Budget Range" in titles and len(titles) == 17
    kinds = [k for k, _, _ in tg.sent]
    assert kinds.count("sendDocument") >= 6 and "sendMessage" in kinds
    assert any(f == ["document"] and "agent.log" in str(d) for k, d, f in tg.sent)   # logs sent
    logs = tmp_path / "logs"
    for name in ("agent.log", "tools.log", "srlm.jsonl", "agent.jsonl"):
        assert (logs / name).exists(), name
    line = (logs / "agent.log").read_text().splitlines()[0]
    assert line.count(" | ") == 7    # timestamp | run_id | company | step | tool | status | duration | details


def test_dedup_across_runs_and_daily_limit(tmp_path):
    cfg = make_cfg(tmp_path, companies_per_run=1)
    a = make_agent(cfg)
    a.start(background=False)
    first = set(a.snapshot()["companies"])
    a2 = make_agent(cfg)
    a2.start(background=False)
    second = set(a2.snapshot()["companies"])
    assert len(first) == 1 and len(second) == 1 and first != second       # dedup by domain
    cfg3 = make_cfg(tmp_path, daily_limit=2)
    a3 = make_agent(cfg3)
    a3.start(background=False)
    s = a3.snapshot()
    assert s["counters"]["found"] == 0 and s["todo"][0]["status"] == "skipped"   # daily limit reached


def test_per_company_isolation(tmp_path):
    cfg = make_cfg(tmp_path)
    pages = {u: h for u, h in all_pages().items() if "rheinland-freight" not in u}   # site unreachable
    a = make_agent(cfg, pages=pages)
    a.start(background=False)
    s = a.snapshot()
    assert s["companies"]["rheinland-freight.example"]["status"] == "skipped"
    assert s["counters"]["processed"] == 2 and s["counters"]["skipped"] == 1 and s["status"] == "done"
    assert "rheinland-freight" in (tmp_path / "logs" / "agent.log").read_text()


def test_unreachable_site_replaced_by_reserve(tmp_path):
    cfg = make_cfg(tmp_path, companies_per_run=2, reserve_companies=2)
    pages = {u: h for u, h in all_pages().items() if "nordwind-solar" not in u}
    a = make_agent(cfg, pages=pages)
    a.start(background=False)
    s = a.snapshot()
    assert s["counters"]["processed"] == 2 and s["counters"]["skipped"] <= 1
    assert sum(c["status"] == "done" for c in s["companies"].values()) == 2


def test_failure_after_fetch_is_isolated(tmp_path):
    cfg = make_cfg(tmp_path)

    class BrokenWriter(TemplateWriter):
        def write(self, ctx):
            if ctx["domain"] == "harbourline-logistics.example":
                raise RuntimeError("writer exploded")
            return super().write(ctx)
    a = make_agent(cfg)
    a.writer = BrokenWriter()
    a.start(background=False)
    s = a.snapshot()
    assert s["companies"]["harbourline-logistics.example"]["status"] == "failed"
    assert s["counters"]["processed"] == 2 and s["counters"]["failed"] == 1
    assert "writer exploded" in (tmp_path / "logs" / "errors.log").read_text()


def test_telegram_down_does_not_fail_companies(tmp_path):
    cfg = make_cfg(tmp_path)
    a = make_agent(cfg, tg_session=TgSession(fail=True))
    a.start(background=False)
    s = a.snapshot()
    assert s["counters"]["processed"] == 3 and s["counters"]["sent"] == 0 and s["counters"]["failed"] == 0
    assert "telegram down" in (tmp_path / "logs" / "errors.log").read_text()


def test_search_network_down(tmp_path):
    cfg = make_cfg(tmp_path)
    a = make_agent(cfg)

    class Down:
        name = "down"

        def search(self, *a):
            raise requests.ConnectionError("network down")
    a.search = Down()
    a.start(background=False)
    s = a.snapshot()
    assert s["todo"][0]["status"] == "failed" and s["status"] == "failed"


def test_stop_and_resume(tmp_path):
    cfg = make_cfg(tmp_path, concurrency=1)

    class SlowPolicy(TemplatePolicy):
        def generate(self, *a, **kw):
            time.sleep(0.02)
            return super().generate(*a, **kw)
    a = make_agent(cfg, policy=SlowPolicy())
    a.start(background=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        if any(i["status"] == "done" and i["id"].endswith(":fields") for i in a.store.data["todo"]):
            break
        time.sleep(0.05)
    a.stop()
    a.wait(60)
    s = a.snapshot()
    assert s["status"] == "stopped"
    assert any(c["status"] == "pending" for c in s["companies"].values())
    run_id = s["run_id"]
    b = make_agent(cfg)                       # new process-equivalent, same state file
    b.start(background=False)
    s2 = b.snapshot()
    assert s2["run_id"] == run_id and s2["status"] == "done" and s2["counters"]["processed"] == 3


class Crash(BaseException):
    pass


def test_crash_recovery_reuses_completed_steps(tmp_path):
    cfg = make_cfg(tmp_path, concurrency=1)

    class CrashOnSend(TgSession):
        def post(self, url, data=None, files=None, timeout=None):
            if "sendDocument" in url and "Proposal" in str((data or {}).get("caption", "")):
                raise Crash()
            return super().post(url, data, files, timeout)
    a = make_agent(cfg, tg_session=CrashOnSend())
    with pytest.raises(Crash):
        a.start(background=False)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "running"                       # process "died" mid-run
    crashed = [d for d, c in state["companies"].items() if c["status"] == "in_progress"][0]
    done_items = {i["id"].split(":")[1] for i in state["todo"] if i["company"] == crashed and i["status"] == "done"}
    assert {"fields", "research", "text", "docx_skill", "finalize"} <= done_items

    fetched = []

    class CountingHttp(HttpClient):
        def get(self, url, *a, **kw):
            fetched.append(url)
            return super().get(url, *a, **kw)
    b = make_agent(cfg)
    b.http = CountingHttp(cfg["fetch"], offline=all_pages())
    b.start(background=False)
    s = b.snapshot()
    assert s["status"] == "done" and s["companies"][crashed]["status"] == "done"
    assert not any(crashed in u for u in fetched)              # completed steps were not redone


def test_continuous_mode_and_heartbeat(tmp_path):
    cfg = make_cfg(tmp_path, continuous=True, run_interval_minutes=0.01, companies_per_run=1)
    a = make_agent(cfg)
    first = a.start(background=True)
    deadline = time.time() + 90
    while time.time() < deadline and (a.store.data["run_id"] == first or a.store.data["status"] != "waiting"):
        time.sleep(0.1)
    assert a.store.data["run_id"] != first                  # a second run started on its own
    assert a.store.data.get("heartbeat", {}).get("last")
    assert "Status:" in a.status_text()
    a.stop()
    a.wait(60)
    assert not a.running and a.store.data["status"] in ("done", "stopped")
    assert len(a.store.data["processed_domains"]) >= 2      # dedup across cycles
