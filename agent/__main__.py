"""CLI: run the agent headless.

    python -m agent run                      # real providers from config.yaml
    python -m agent run --offline            # bundled fixtures (no network), template policy if no checkpoint
    python -m agent status
"""
from __future__ import annotations

import argparse
import json

from core.config import load_config


def offline_overrides(cfg: dict) -> dict:
    cfg["search"]["provider"] = "offline"
    cfg["research"]["provider"] = "offline"
    return cfg


def make_agent(offline: bool, overrides: dict | None = None):
    from agent.loop import Agent
    from tools.http import HttpClient
    cfg = load_config(overrides=overrides)
    http = None
    if offline:
        from core.config import resolve
        from data.sample.companies import all_pages
        cfg = offline_overrides(cfg)
        http = HttpClient(cfg["fetch"], cache_dir=resolve(cfg["paths"]["cache"]) / "http", offline=all_pages())
    return Agent(cfg, http=http)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--companies", type=int)
    ap.add_argument("--policy", choices=["model", "template"])
    ap.add_argument("--continuous", action="store_true", help="keep running new cycles until Ctrl+C")
    a = ap.parse_args(argv)
    ov: dict = {"agent": {"continuous": bool(a.continuous)}}
    if a.companies:
        ov["agent"]["companies_per_run"] = a.companies
    if a.policy:
        ov["agent"]["policy"] = a.policy
    agent = make_agent(a.offline, ov)
    if a.cmd == "status":
        print(json.dumps(agent.snapshot()["counters"]))
        return
    agent.start(background=False)
    s = agent.snapshot()
    print(json.dumps({"status": s["status"], "counters": s["counters"],
                      "companies": {d: c["status"] for d, c in s["companies"].items()}}, indent=1))


if __name__ == "__main__":
    main()
