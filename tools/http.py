"""Polite HTTP client shared by all network tools.

* Identifies itself with a configurable User-Agent.
* Obeys robots.txt (cached per host); disallowed URLs are never fetched.
* Per-host rate limiting (`min_delay_s` between requests to the same host).
* Timeouts, retries with exponential backoff (429/5xx/connection errors), size cap.
* Optional on-disk cache keyed by URL.
* An offline transport (dict url → body) for tests and air-gapped runs.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from core.logging import get_logger

log = get_logger("tools")


class FetchError(Exception):
    pass


class RobotsDisallowed(FetchError):
    pass


@dataclass
class Response:
    url: str
    status: int
    text: str
    content_type: str = "text/html"
    from_cache: bool = False


def with_retries(fn: Callable, retries: int = 3, base_delay: float = 1.0, retry_on=(Exception,),
                 sleep: Callable[[float], None] = time.sleep, what: str = "call"):
    """Call fn() with exponential backoff; re-raise the last error."""
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_on as e:  # noqa: PERF203
            last = e
            if attempt == retries:
                break
            delay = getattr(e, "retry_after", None) or base_delay * (2 ** attempt)
            log.event("retry", "retry", tool=what, details=f"attempt {attempt + 1}: {type(e).__name__}: {e}; sleep {delay:.1f}s")
            sleep(delay)
    raise last  # type: ignore[misc]


class HttpClient:
    def __init__(self, cfg: dict, cache_dir: Path | None = None,
                 offline: dict[str, str] | Callable[[str, dict | None], str] | None = None,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep):
        self.ua = cfg.get("user_agent", "ProposalAgentBot/0.1")
        self.timeout = float(cfg.get("timeout_s", 20))
        self.retries = int(cfg.get("retries", 3))
        self.delay = float(cfg.get("min_delay_s", 2.0))
        self.max_bytes = int(cfg.get("max_bytes", 2_000_000))
        self.respect_robots = bool(cfg.get("respect_robots", True))
        self.cache_dir = cache_dir if cfg.get("cache", True) else None
        self.offline = offline
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.ua, "Accept-Language": "en"})
        self.sleep = sleep
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- transport
    def _raw_get(self, url: str, params: dict | None = None) -> Response:
        if callable(self.offline):
            try:
                return Response(url, 200, self.offline(url, params))
            except KeyError as e:
                raise FetchError(f"404 (offline) {url}") from e
        if self.offline is not None:
            if url in self.offline:
                return Response(url, 200, self.offline[url])
            alt = url.rstrip("/") if url.endswith("/") else url + "/"
            if alt in self.offline:
                return Response(url, 200, self.offline[alt])
            raise FetchError(f"404 (offline) {url}")
        r = self.session.get(url, params=params, timeout=self.timeout, stream=True, allow_redirects=True)
        if r.status_code == 429 or r.status_code >= 500:
            err = FetchError(f"HTTP {r.status_code} for {url}")
            ra = r.headers.get("Retry-After")
            err.retry_after = float(ra) if ra and ra.isdigit() else None  # type: ignore[attr-defined]
            raise err
        if r.status_code >= 400:
            raise FetchError(f"HTTP {r.status_code} for {url}")
        body = b""
        for chunk in r.iter_content(65536):
            body += chunk
            if len(body) > self.max_bytes:
                break
        enc = r.encoding or "utf-8"
        return Response(r.url, r.status_code, body.decode(enc, errors="replace"), r.headers.get("Content-Type", ""))

    # ------------------------------------------------------------- politeness
    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.time()
            wait = self._last.get(host, 0) + self.delay - now
            self._last[host] = max(now, self._last.get(host, 0) + self.delay)
        if wait > 0 and self.offline is None:
            self.sleep(wait)

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self._raw_get(root + "/robots.txt")
                rp.parse(resp.text.splitlines())
            except Exception:
                rp.parse([])   # no robots.txt → everything allowed
            self._robots[root] = rp
        rp = self._robots[root]
        return rp.can_fetch(self.ua, url) if rp else True

    # ------------------------------------------------------------- cache
    def _cache_path(self, url: str, params: dict | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, url: str, params: dict | None = None, check_robots: bool = True) -> Response:
        cp = self._cache_path(url, params)
        if cp and cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            return Response(d["url"], d["status"], d["text"], d.get("content_type", ""), True)
        if check_robots and not self.allowed(url):
            log.event("fetch", "skipped", tool="http", details=f"robots.txt disallows {url}")
            raise RobotsDisallowed(url)
        host = urlparse(url).netloc
        t0 = time.time()

        def once() -> Response:
            self._wait_turn(host)
            return self._raw_get(url, params)
        resp = with_retries(once, self.retries, 1.0, (FetchError, requests.RequestException), self.sleep, "http")
        log.event("fetch", "ok", tool="http", duration=time.time() - t0, details=f"{url} ({len(resp.text)} chars)")
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps({"url": resp.url, "status": resp.status, "text": resp.text,
                                      "content_type": resp.content_type}), encoding="utf-8")
        return resp
