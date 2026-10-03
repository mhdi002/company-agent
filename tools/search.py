"""Find company websites by country and industry. Providers are pluggable.

Providers:
  duckduckgo  — DuckDuckGo HTML endpoint (free, no key); results filtered to company homepages
  seedfile    — a JSON file of {name, url, country, industry} (open directories exported by the user)
  offline     — the bundled fictional fixture companies (tests / air-gapped runs)
"""
from __future__ import annotations

import json
import re
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from core.config import resolve
from core.logging import get_logger
from tools.http import HttpClient

log = get_logger("tools")

# Aggregators, social networks and directories are not company websites.
EXCLUDE = re.compile(r"(linkedin|facebook|twitter|x\.com|instagram|youtube|wikipedia|yelp|glassdoor|indeed|crunchbase|"
                     r"bloomberg|zoominfo|dnb\.com|yellowpages|kompass|clutch\.co|amazon|reddit|medium\.com|"
                     r"duckduckgo|google\.|bing\.com|tripadvisor|trustpilot|github\.com|gov\.|\.gov|europa\.eu)", re.I)


def domain_of(url: str) -> str:
    host = urlparse(url if "://" in url else "https://" + url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def homepage(url: str) -> str:
    p = urlparse(url if "://" in url else "https://" + url)
    return f"{p.scheme or 'https'}://{p.netloc}/"


class SearchProvider(Protocol):
    name: str

    def search(self, country: str, industry: str, limit: int) -> list[dict]: ...


class DuckDuckGoProvider:
    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, country: str, industry: str, limit: int) -> list[dict]:
        q = f"{industry} company {country} official website"
        resp = self.http.get(self.ENDPOINT, params={"q": q}, check_robots=False)
        soup = BeautifulSoup(resp.text, "lxml")
        out, seen = [], set()
        for a in soup.select("a.result__a"):
            href = a.get("href", "")
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            if not href.startswith("http") or EXCLUDE.search(href):
                continue
            d = domain_of(href)
            if d in seen:
                continue
            seen.add(d)
            out.append({"name": a.get_text(" ", strip=True)[:120], "url": homepage(href), "domain": d,
                        "country_hint": country, "industry_hint": industry, "source": self.name})
            if len(out) >= limit:
                break
        return out


class SeedFileProvider:
    name = "seedfile"

    def __init__(self, path: str):
        self.items = json.loads(resolve(path).read_text(encoding="utf-8"))

    def search(self, country: str, industry: str, limit: int) -> list[dict]:
        out = []
        for it in self.items:
            if country.lower() not in it.get("country", "").lower():
                continue
            if industry and industry.lower() not in it.get("industry", "").lower():
                continue
            out.append({"name": it["name"], "url": homepage(it["url"]), "domain": domain_of(it["url"]),
                        "country_hint": it.get("country", country), "industry_hint": it.get("industry", industry),
                        "source": self.name})
        return out[:limit]


class OfflineProvider:
    name = "offline"

    def search(self, country: str, industry: str, limit: int) -> list[dict]:
        from data.sample.companies import COMPANIES
        out = [{"name": c["name"], "url": f"https://{c['domain']}/", "domain": c["domain"], "country_hint": c["country"],
                "industry_hint": c["field"], "source": self.name}
               for c in COMPANIES if c["country"] == country and (not industry or c["field"] == industry)]
        return out[:limit]


def make_provider(cfg: dict, http: HttpClient) -> SearchProvider:
    name = cfg["search"]["provider"]
    if name == "duckduckgo":
        return DuckDuckGoProvider(http)
    if name == "seedfile":
        return SeedFileProvider(cfg["search"]["seed_file"])
    if name == "offline":
        return OfflineProvider()
    raise ValueError(f"unknown search provider {name!r}")


def gather_companies(provider: SearchProvider, countries: list[str], industries: list[str], limit: int,
                     seen_domains: set[str] | None = None, per_query: int = 10) -> list[dict]:
    """Search every (country, industry) pair; deduplicate by domain; stop at `limit`."""
    seen = set(seen_domains or ())
    out: list[dict] = []
    for industry in industries:
        for country in countries:
            try:
                results = provider.search(country, industry, per_query)
            except Exception as e:  # one failing query never stops the search
                log.error("search", e, tool=provider.name, details=f"{industry} / {country}: {e}")
                continue
            log.event("search", "ok", tool=provider.name, details=f"{industry} / {country}: {len(results)} results")
            for r in results:
                if r["domain"] in seen:
                    continue
                seen.add(r["domain"])
                out.append(r)
                if len(out) >= limit:
                    return out
    return out
