"""Run planning: how many companies to gather and which (country, industry) queries to issue."""
from __future__ import annotations

import random


def plan_run(cfg: dict, processed_today: int, seed: int | None = None) -> dict:
    a = cfg["agent"]
    remaining_today = max(int(a["daily_limit"]) - processed_today, 0)
    limit = min(int(a["companies_per_run"]), remaining_today)
    countries, industries = list(a["countries"]), list(a["industries"])
    if seed is not None:   # vary query order between runs so different companies surface
        rng = random.Random(seed)
        rng.shuffle(countries)
        rng.shuffle(industries)
    return {"limit": limit, "countries": countries, "industries": industries,
            "daily_remaining": remaining_today, "concurrency": max(1, int(a["concurrency"]))}
