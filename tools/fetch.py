"""Polite site crawler: homepage + key pages (about, services, products), clean text per page."""
from __future__ import annotations

import time

from core.logging import get_logger
from tools.extract import find_links, html_to_text, key_pages, page_meta
from tools.http import FetchError, HttpClient, RobotsDisallowed

log = get_logger("tools")


def fetch_site(http: HttpClient, url: str, max_pages: int = 6, company: str | None = None) -> dict:
    """Return {"pages": {url: text}, "meta": {...}, "errors": [...]} for a company website.

    Raises FetchError only when the homepage itself cannot be fetched.
    """
    t0 = time.time()
    errors: list[str] = []
    home = http.get(url)
    pages = {home.url: html_to_text(home.text)}
    meta = page_meta(home.text)
    links = find_links(home.text, home.url)
    for u in key_pages(links, limit=max(max_pages - 1, 0)):
        if u in pages:
            continue
        try:
            r = http.get(u)
            text = html_to_text(r.text)
            if len(text) > 40:
                pages[u] = text
        except RobotsDisallowed:
            errors.append(f"robots disallow {u}")
        except (FetchError, Exception) as e:  # one bad page never fails the site
            errors.append(f"{u}: {e}")
    log.event("fetch_site", "ok", tool="fetch", company=company, duration=time.time() - t0,
              details={"url": url, "pages": list(pages), "errors": errors[:5]})
    return {"pages": pages, "meta": meta, "errors": errors}


__all__ = ["fetch_site", "FetchError"]
