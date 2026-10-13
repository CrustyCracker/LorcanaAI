"""
Inkdecks Metagame Scraper for Emerald/Steel (Green Steel).
Scrapes live tournament statistics from https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats
Extracts presence percentage, average copies, meta multipliers, and categories for all cards.

Designed to be zero-dependency at runtime:
- Automatically loads pre-bundled JSON cache from data_cache/inkdecks_emerald_steel_stats.json.
- BeautifulSoup4 is optional: if absent on HPC clusters like Athena, loads cache or uses regex fallback.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import re
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
INKDECKS_URL = "https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats"


def _parse_with_bs4(html: str) -> list[dict[str, Any]]:
    """Parse Inkdecks HTML using BeautifulSoup4 if installed."""
    from bs4 import BeautifulSoup  # Lazy import only when called

    soup = BeautifulSoup(html, "html.parser")
    items = soup.find_all(class_="card-stat-item")

    cards: list[dict[str, Any]] = []
    for it in items:
        parts = it.get_text(" | ", strip=True).split(" | ")
        category = parts[0] if parts else "Unknown"
        copies_str = ""
        pct_str = ""
        mult_str = ""
        img = it.find("img")
        name = img.get("alt", "").strip() if img else ""

        for p in parts[1:]:
            if p.startswith("×") or p.startswith("\xd7"):
                copies_str = p
            elif p.endswith("%"):
                pct_str = p
            elif p.endswith("x") and (p[:-1].replace(".", "").isdigit()):
                mult_str = p
            elif not name and p != category:
                name = p.strip()

        if not name:
            continue

        try:
            pct_val = float(pct_str.replace("%", "").strip())
        except Exception:
            pct_val = 0.0

        try:
            copies_val = float(copies_str.replace("×", "").replace("\xd7", "").strip())
        except Exception:
            copies_val = 0.0

        cards.append({
            "name": name,
            "category": category,
            "presence": pct_str,
            "presence_pct": pct_val,
            "copies": copies_str,
            "avg_copies": copies_val,
            "meta_mult": mult_str,
        })
    return cards


def _parse_with_regex(html: str) -> list[dict[str, Any]]:
    """Standard library regex fallback if beautifulsoup4 is not installed."""
    cards: list[dict[str, Any]] = []
    blocks = re.findall(
        r'<div[^>]*class="[^"]*card-stat-item[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*card-stat-item|\Z)',
        html,
        re.DOTALL,
    )
    for block in blocks:
        img_match = re.search(r'<img[^>]+alt="([^"]+)"', block)
        name = img_match.group(1).strip() if img_match else ""
        if not name:
            continue

        clean_text = re.sub(r'<[^>]+>', ' ', block)
        tokens = [t.strip() for t in clean_text.split() if t.strip()]

        category = tokens[0] if tokens else "Unknown"
        copies_str = ""
        pct_str = ""
        mult_str = ""

        for tok in tokens:
            if "%" in tok:
                pct_str = tok
            elif tok.startswith("×") or tok.startswith("\xd7"):
                copies_str = tok
            elif tok.endswith("x") and (tok[:-1].replace(".", "").isdigit()):
                mult_str = tok

        try:
            pct_val = float(pct_str.replace("%", "").strip())
        except Exception:
            pct_val = 0.0

        try:
            copies_val = float(copies_str.replace("×", "").replace("\xd7", "").strip())
        except Exception:
            copies_val = 0.0

        cards.append({
            "name": name,
            "category": category,
            "presence": pct_str,
            "presence_pct": pct_val,
            "copies": copies_str,
            "avg_copies": copies_val,
            "meta_mult": mult_str,
        })
    return cards


def parse_inkdecks_html(html: str) -> list[dict[str, Any]]:
    """Parse Inkdecks HTML using BeautifulSoup if available, or regex fallback."""
    try:
        return _parse_with_bs4(html)
    except ImportError:
        logger.info("bs4 (BeautifulSoup4) not installed; using standard library regex parser.")
        return _parse_with_regex(html)


def scrape_inkdecks_emerald_steel(
    cache_file: Path | None = None,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Scrapes or loads cached card stats from Inkdecks for Emerald/Steel.
    Returns list of dicts with keys: name, category, copies, presence_pct, meta_mult, avg_copies.

    If cache_file exists and force_refresh is False, loads directly from cache.
    Does not require bs4 or network connection when using cached data.
    """
    if cache_file is None:
        cache_file = CACHE_DIR / "inkdecks_emerald_steel_stats.json"

    # Fast path: Load bundled cache immediately without network or external deps
    if cache_file.exists() and not force_refresh:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data and isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to load cached Inkdecks data from %s: %s", cache_file, exc)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    html: str | None = None
    try:
        req = urllib.request.Request(INKDECKS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
    except Exception as exc:
        if cache_file.exists():
            logger.info("Network fetch failed (%s); using cached Inkdecks file %s", exc, cache_file)
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Failed to fetch Inkdecks data ({exc}) and no cache file found at {cache_file}."
        ) from exc

    cards = parse_inkdecks_html(html)

    if cards:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    elif cache_file.exists():
        # Fallback to cache if parsed empty
        return json.loads(cache_file.read_text(encoding="utf-8"))

    return cards


if __name__ == "__main__":
    data = scrape_inkdecks_emerald_steel()
    print(f"Loaded/Scraped {len(data)} cards from Inkdecks.")
    for c in data[:20]:
        print(f"[{c['category']:7s}] {c['name']:40s} | {c['presence']:6s} | {c['copies']}")
