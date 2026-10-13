"""
Inkdecks Metagame Scraper for Emerald/Steel (Green Steel).
Scrapes live tournament statistics from https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats
Extracts presence percentage, average copies, meta multipliers, and categories for all cards.
"""

from __future__ import annotations
import json
from pathlib import Path
import re
import urllib.request
from typing import Any
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
INKDECKS_URL = "https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats"


def scrape_inkdecks_emerald_steel(cache_file: Path | None = None) -> list[dict[str, Any]]:
    """
    Scrapes or loads cached card stats from Inkdecks for Emerald/Steel.
    Returns list of dicts with keys: name, category, copies, presence_pct, meta_mult, avg_copies_float.
    """
    if cache_file is None:
        cache_file = CACHE_DIR / "inkdecks_emerald_steel_stats.json"

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
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data
        raise RuntimeError(f"Failed to fetch Inkdecks data and no cache available: {exc}") from exc

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
            if p.startswith("×"):
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
            copies_val = float(copies_str.replace("×", "").strip())
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

    if cards:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")

    return cards


if __name__ == "__main__":
    data = scrape_inkdecks_emerald_steel()
    print(f"Scraped {len(data)} cards from Inkdecks.")
    for c in data[:20]:
        print(f"[{c['category']:7s}] {c['name']:40s} | {c['presence']:6s} | {c['copies']}")
