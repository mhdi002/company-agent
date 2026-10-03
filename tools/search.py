"""Find company websites by country and industry. Providers are pluggable.

Providers:
  wikidata    — Wikidata SPARQL (open data, CC0): companies with country (P17), industry (P452) and an
                official website (P856). Default: free, keyless, and not blocked for server IPs.
  duckduckgo  — DuckDuckGo HTML endpoint (free, no key); results filtered to company homepages.
                Datacenter IPs often get a bot challenge, which is reported as SearchBlocked.
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


class SearchBlocked(Exception):
    """The search engine answered with a bot challenge instead of results."""


class WikidataProvider:
    name = "wikidata"

    def __init__(self, http: HttpClient, cfg: dict):
        self.http = http
        self.endpoint = cfg["wikidata_sparql"]
        self.api = cfg["wikidata_api"]
        self.qids = dict(cfg.get("country_qids") or {})
        self.keywords = {k.lower(): v for k, v in (cfg.get("industry_keywords") or {}).items()}

    def _country_qid(self, country: str) -> str | None:
        if country in self.qids:
            return self.qids[country]
        r = self.http.get(self.api, params={
            "action": "wbsearchentities", "search": country, "language": "en", "type": "item", "format": "json"},
            check_robots=False)
        hits = json.loads(r.text).get("search", [])
        if hits:
            self.qids[country] = hits[0]["id"]
        return hits[0]["id"] if hits else None

    def query(self, qid: str, industry: str, limit: int) -> str:
        kws = self.keywords.get(industry.lower(), [industry.lower()])
        cond = " || ".join(f'CONTAINS(LCASE(?industryLabel), "{k.replace(chr(34), "")}")' for k in kws)
        return f"""SELECT DISTINCT ?item ?itemLabel ?website ?industryLabel WHERE {{
  ?item wdt:P17 wd:{qid}; wdt:P856 ?website; wdt:P452 ?industry.
  ?industry rdfs:label ?industryLabel. FILTER(LANG(?industryLabel) = "en")
  FILTER({cond})
  ?item rdfs:label ?itemLabel. FILTER(LANG(?itemLabel) = "en")
}} LIMIT {int(limit) * 3}"""

    def search(self, country: str, industry: str, limit: int) -> list[dict]:
        qid = self._country_qid(country)
        if not qid:
            return []
        r = self.http.get(self.endpoint, params={"query": self.query(qid, industry, limit), "format": "json"},
                          check_robots=False)
        out, seen = [], set()
        for b in json.loads(r.text).get("results", {}).get("bindings", []):
            url = b["website"]["value"]
            if not url.startswith("http") or EXCLUDE.search(url):
                continue
            d = domain_of(url)
            if d in seen:
                continue
            seen.add(d)
            out.append({"name": b["itemLabel"]["value"], "url": homepage(url), "domain": d, "country_hint": country,
                        "industry_hint": industry, "source": self.name, "wikidata": b["item"]["value"]})
            if len(out) >= limit:
                break
        return out


class DuckDuckGoProvider:
    name = "duckduckgo"

    def __init__(self, http: HttpClient, cfg: dict):
        self.http = http
        self.endpoint = cfg["duckduckgo_endpoint"]
        self.template = cfg.get("duckduckgo_query", "{industry} company {country}")

    def search(self, country: str, industry: str, limit: int) -> list[dict]:
        q = self.template.format(industry=industry, country=country)
        resp = self.http.get(self.endpoint, params={"q": q}, check_robots=False)
        if "anomaly" in resp.text and "result__a" not in resp.text:
            raise SearchBlocked("DuckDuckGo returned a bot challenge; use search.provider: wikidata or seedfile")
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
    if name == "wikidata":
        return WikidataProvider(http, cfg["search"])
    if name == "duckduckgo":
        return DuckDuckGoProvider(http, cfg["search"])
    if name == "seedfile":
        return SeedFileProvider(cfg["search"]["seed_file"])
    if name == "offline":
        return OfflineProvider()
    raise ValueError(f"unknown search provider {name!r}")


def gather_companies(provider: SearchProvider, countries: list[str], industries: list[str], limit: int,
                     seen_domains: set[str] | None = None, per_query: int = 10) -> list[dict]:
    """Search every (country, industry) pair, interleave results round-robin so the run covers many
    countries and industries, deduplicate by domain, and stop at `limit`."""
    seen = set(seen_domains or ())
    buckets: list[list[dict]] = []
    for industry in industries:
        for country in countries:
            try:
                results = provider.search(country, industry, per_query)
            except Exception as e:  # one failing query never stops the search
                log.error("search", e, tool=provider.name, details=f"{industry} / {country}: {e}")
                continue
            log.event("search", "ok", tool=provider.name,
                      details={"query": f"{industry} / {country}", "results": [r["domain"] for r in results]})
            buckets.append(results)
    out: list[dict] = []
    for rank in range(max((len(b) for b in buckets), default=0)):
        for b in buckets:
            if rank < len(b) and b[rank]["domain"] not in seen:
                seen.add(b[rank]["domain"])
                out.append(b[rank])
                if len(out) >= limit:
                    return out
    return out
