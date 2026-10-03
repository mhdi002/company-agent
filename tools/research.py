"""Research a field: existing work and trends from public sources, stored as evidence with URLs.

Providers:
  wikipedia — MediaWiki API search + plain-text extracts (public, CC BY-SA); several queries per field
  offline   — the bundled field knowledge base (fictional .example URLs) for tests / air-gapped runs
"""
from __future__ import annotations

import hashlib
import json
import re

from core.logging import get_logger
from tools.http import HttpClient

log = get_logger("tools")


def _clean(text: str, max_chars: int = 4000) -> str:
    text = re.sub(r"\n=+ [^=]+ =+\n", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


class WikipediaProvider:
    name = "wikipedia"

    def __init__(self, http: HttpClient, cfg: dict):
        self.http = http
        self.api = cfg["wikipedia_api"]
        self.templates = cfg.get("queries") or ["{field} industry"]
        self.per_query = int(cfg.get("results_per_query", 3))
        self.max_chars = int(cfg.get("max_chars_per_source", 4000))

    def queries(self, field: str, services: list[str]) -> list[str]:
        out = []
        for t in self.templates:
            if "{service}" in t:
                out += [t.format(field=field, service=s) for s in services[:2]]
            else:
                out.append(t.format(field=field))
        return out

    def research(self, field: str, services: list[str], max_sources: int) -> list[dict]:
        docs, seen = [], set()
        for q in self.queries(field, services):
            r = self.http.get(self.api, params={"action": "query", "list": "search", "srsearch": q, "srlimit": self.per_query,
                                                "format": "json"}, check_robots=False)
            for hit in json.loads(r.text).get("query", {}).get("search", []):
                title = hit["title"]
                if title in seen:
                    continue
                seen.add(title)
                ex = self.http.get(self.api, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                                     "titles": title, "format": "json", "exsectionformat": "plain"},
                                   check_robots=False)
                pages = json.loads(ex.text).get("query", {}).get("pages", {})
                text = next(iter(pages.values()), {}).get("extract", "")
                if len(text) > 200:
                    docs.append({"url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", "title": title,
                                 "text": _clean(text, self.max_chars)})
                if len(docs) >= max_sources:
                    return docs
        return docs


class OfflineResearchProvider:
    name = "offline"

    def research(self, field: str, services: list[str], max_sources: int) -> list[dict]:
        from data.sample.companies import FIELD_KB
        docs = [dict(d) for d in FIELD_KB.get(field, [])]
        # include one off-field document so evidence extraction has to discriminate
        others = [d for f, v in FIELD_KB.items() if f != field for d in v]
        if others:
            docs.append(dict(others[int(hashlib.md5(field.encode()).hexdigest(), 16) % len(others)]))
        return docs[:max_sources]


def make_research_provider(cfg: dict, http: HttpClient):
    name = cfg["research"]["provider"]
    if name == "wikipedia":
        return WikipediaProvider(http, cfg["research"])
    if name == "offline":
        return OfflineResearchProvider()
    raise ValueError(f"unknown research provider {name!r}")


def research_field(provider, field: str, services: list[str], max_sources: int = 6, company: str | None = None) -> list[dict]:
    """Return evidence documents [{id, url, title, text}] with stable ids E1..En."""
    docs = provider.research(field, services, max_sources)
    evidence = [{"id": f"E{i + 1}", "url": d["url"], "title": d.get("title", ""), "text": d["text"]}
                for i, d in enumerate(docs)]
    log.event("research", "ok", tool=f"research:{provider.name}", company=company,
              details={"field": field, "sources": [e["url"] for e in evidence]})
    return evidence
