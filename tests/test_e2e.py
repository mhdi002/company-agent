"""End-to-end runs on 10 companies.

* test_e2e_offline_10: always runs — 10 fixture companies through the whole agent (fixture websites, offline research).
* test_e2e_real_10: real internet (DuckDuckGo search, real company websites, Wikipedia research). Skipped
  unless PA_NETWORK_TESTS=1 because it needs outbound access.
"""
import json
import os
import zipfile

import pytest

from agent.loop import Agent
from tests.test_agent import make_agent, make_cfg

ALL_FIELDS = ["renewable energy", "logistics", "software", "agriculture", "manufacturing", "healthcare",
              "finance", "construction", "education", "retail"]
ALL_COUNTRIES = ["Germany", "United Kingdom", "Canada", "Australia", "Netherlands", "Singapore"]


def check_outputs(tmp_path, n):
    docs = sorted((tmp_path / "proposals").glob("*.docx"))
    recs = sorted((tmp_path / "records").glob("*.json"))
    assert len(docs) == n and len(recs) == n
    for d, r in zip(docs, recs):
        with zipfile.ZipFile(d) as z:
            assert "word/document.xml" in z.namelist()
        rec = json.loads(r.read_text())
        titles = [s["title"] for s in rec["sections"]]
        assert titles[0] == "Cover" and titles[-1] == "Sources" and len(titles) == 17
        # every cited label resolves to a source with a URL or the company website
        labels = {s["label"] for s in rec["sources"]}
        for s in rec["sections"]:
            for line in s["paragraphs"] + s["bullets"]:
                for lab in __import__("re").findall(r"\[(S\d+)", line):
                    assert lab in labels


def test_e2e_offline_10(tmp_path):
    cfg = make_cfg(tmp_path, countries=ALL_COUNTRIES, industries=ALL_FIELDS, companies_per_run=10, concurrency=3)
    a = make_agent(cfg)
    a.start(background=False)
    s = a.snapshot()
    assert s["status"] == "done" and s["counters"]["found"] == 10 and s["counters"]["processed"] == 10
    check_outputs(tmp_path, 10)
    srlm = [json.loads(line) for line in (tmp_path / "logs" / "srlm.jsonl").read_text().splitlines()]
    decisions = [r for r in srlm if r["step"] == "srlm.decision"]
    assert len(decisions) == 10 * (1 + 2 + 15)        # field + evidence + projects + 15 fact sheets
    assert all(len(r["candidates"]) == 3 for r in decisions)


@pytest.mark.network
def test_e2e_real_10(tmp_path):
    from core.config import load_config
    cfg = make_cfg(tmp_path, countries=ALL_COUNTRIES, industries=["software", "logistics"], companies_per_run=10,
                   policy=os.environ.get("PA_POLICY", "template"))
    cfg["search"]["provider"] = "duckduckgo"
    cfg["research"]["provider"] = "wikipedia"
    cfg["fetch"].update(load_config()["fetch"])
    a = Agent(cfg)
    a.start(background=False)
    s = a.snapshot()
    assert s["counters"]["found"] == 10
    assert s["counters"]["processed"] >= 8          # some real sites may block or be down
    check_outputs(tmp_path, s["counters"]["processed"])
