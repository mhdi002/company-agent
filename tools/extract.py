"""Clean text extraction and key-page discovery (home, about, services, products)."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from data.clean.normalize import normalize_ws, remove_boilerplate

KEY_PAGES = {
    "about": re.compile(r"about|company|who-we-are|ueber-uns|uber-uns|qui-sommes|over-ons|our-story", re.I),
    "services": re.compile(r"services?|solutions|what-we-do|leistungen|capabilities|expertise", re.I),
    "products": re.compile(r"products?|produkte|portfolio|platform", re.I),
}
SKIP = re.compile(r"\.(pdf|jpg|jpeg|png|gif|svg|zip|mp4|docx?|xlsx?)$|login|signin|account|cart|checkout|privacy|"
                  r"cookie|terms|impressum|imprint|mailto:|tel:|javascript:", re.I)
DROP = ["script", "style", "noscript", "nav", "footer", "header", "form", "iframe", "svg", "aside"]


def html_to_text(html: str) -> str:
    """Main visible text with navigation, scripts, cookie banners and legal boilerplate removed."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(DROP):
        t.decompose()
    for t in soup.select("[class*=cookie], [id*=cookie], [class*=banner], [class*=newsletter], [role=navigation]"):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n")
    return normalize_ws(remove_boilerplate(text))


def page_meta(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    desc = soup.find("meta", attrs={"name": "description"})
    lang = soup.html.get("lang") if soup.html else None
    return {"title": title, "description": desc.get("content", "") if desc else "", "lang": lang}


def find_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    host = urlparse(base_url).netloc
    out = []
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"]).split("#")[0]
        if urlparse(u).netloc == host and not SKIP.search(u) and u not in out:
            out.append(u)
    return out


def key_pages(links: list[str], limit: int = 5) -> list[str]:
    """Pick at most one link per key-page type, in a stable order."""
    chosen = []
    for _kind, pat in KEY_PAGES.items():
        for u in links:
            if pat.search(urlparse(u).path) and u not in chosen:
                chosen.append(u)
                break
    return chosen[:limit]
