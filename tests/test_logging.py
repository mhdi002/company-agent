import json

from core import logging as plog


def test_channels_human_and_jsonl(tmp_logs):
    log = plog.get_logger("tools")
    log.event("fetch", "ok", company="a.example", tool="http", duration=1.234, details="x" * 5000)
    human = (tmp_logs / "tools.log").read_text().strip()
    parts = human.split(" | ")
    assert len(parts) == 8 and parts[1] == "test" and parts[2] == "a.example" and parts[6] == "1.23s"
    rec = json.loads((tmp_logs / "tools.jsonl").read_text())
    assert rec["details"].endswith("chars]") and len(rec["details"]) < 2100   # truncated


def test_errors_mirrored_with_traceback(tmp_logs):
    try:
        1 / 0
    except ZeroDivisionError as e:
        plog.get_logger("agent").error("step", e, company="b.example")
    err = json.loads((tmp_logs / "errors.jsonl").read_text())
    assert "ZeroDivisionError" in err["traceback"] and err["channel"] == "agent"
    assert "b.example" in (tmp_logs / "errors.log").read_text()


def test_srlm_channel_is_jsonl_only(tmp_logs):
    plog.get_logger("srlm").event("srlm.decision", "done", unit="u", K=8)
    assert (tmp_logs / "srlm.jsonl").exists() and not (tmp_logs / "srlm.log").exists()


def test_ring_filters(tmp_logs):
    plog.RING.clear()
    a = plog.get_logger("agent")
    a.event("s1", "ok", company="alpha.example", details="hello world")
    a.event("s2", "error", company="beta.example", details="boom")
    plog.get_logger("tools").event("s3", "ok", company="alpha.example")
    assert [r["step"] for r in plog.recent(company="alpha")] == ["s1", "s3"]
    assert [r["step"] for r in plog.recent(status="error")] == ["s2"]
    assert [r["step"] for r in plog.recent(channel="tools")] == ["s3"]
    assert [r["step"] for r in plog.recent(text="hello")] == ["s1"]
    plog.RING.clear()
    assert plog.load_recent_from_files() >= 3
