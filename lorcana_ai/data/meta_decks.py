"""
Real Season 14 Core Constructed (Sets 9-14) Metagame Scraper & Loader from Duels.ink:
- Source URL: https://duels.ink/stats/core-bo1 (API: https://duels.ink/api/stats/meta?queue=core-bo1&period=all_time&era=set14)
- Offline Cache: data_cache/duels_ink_core_bo1.json (51,360+ ranked games, 73 archetype profiles)
- Maps Duels.ink centroidCounts ('set-number') through LorcanaJSON non-promo printings to Sets 9-14 Core Constructed cards
  (automatically resolving Set 9 Fabled reprints such as Ursula - Vanessa, Strike a Good Match, Stitch - Carefree Surfer, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re

from lorcana_ai.config import DATA_DIR, PROJECT_ROOT
from lorcana_ai.data.card_db import CardDatabase, GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck

DUELS_INK_META_API = "https://duels.ink/api/stats/meta"
DUELS_INK_CACHE_FILE = DATA_DIR / "duels_ink_core_bo1.json"
EXPORTED_DECKS_DIR = PROJECT_ROOT / "decks" / "duels_ink_core_bo1"


@dataclass
class DuelsInkArchetypeProfile:
    """Represents a scraped meta archetype profile from https://duels.ink/stats/core-bo1."""
    profile_id: str
    name: str
    ink_pair: str
    games_played: int
    win_rate: float
    meta_share: float
    signature_cards: list[str]
    deck: Deck


def _build_set_num_to_fullname_map() -> dict[str, str]:
    """
    Maps Duels.ink 'setCode-collectorNumber' IDs (e.g. '14-89', '9-26', '4-25') to canonical fullName
    using non-promo cards from data_cache/allCards.json.
    """
    all_cards_path = DATA_DIR / "allCards.json"
    mapping: dict[str, str] = {}
    if not all_cards_path.exists():
        return mapping
    raw = json.loads(all_cards_path.read_text(encoding="utf-8"))
    for c in raw.get("cards", []):
        if c.get("promoGrouping"):
            continue
        key = f"{c.get('setCode')}-{c.get('number')}"
        full_name = c.get("fullName") or c.get("name")
        if key not in mapping and full_name:
            mapping[key] = str(full_name)
    return mapping


def sync_duels_ink_meta(
    queue: str = "core-bo1",
    period: str = "all_time",
    era: str = "set14",
    export_txt_files: bool = True,
    timeout: int = 20,
) -> list[DuelsInkArchetypeProfile]:
    """
    Fetches live Core Constructed Season 14 metagame statistics & centroid decklists from
    https://duels.ink/stats/core-bo1 (via /api/stats/meta), caches the JSON in data_cache/,
    and optionally exports Dreamborn-format .txt decklists into decks/duels_ink_core_bo1/.
    """
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({"queue": queue, "period": period, "era": era})
    req = urllib.request.Request(
        f"{DUELS_INK_META_API}?{query}",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LorcanaAlpha/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_text = resp.read().decode("utf-8")

    cache_path = DATA_DIR / f"duels_ink_{queue.replace('-', '_')}.json"
    cache_path.write_text(raw_text, encoding="utf-8")

    profiles = load_duels_ink_profiles(
        cache_path=cache_path,
        top_n=25,
        deduplicate_by_name=False,
    )
    if export_txt_files:
        export_profiles_to_txt(profiles, EXPORTED_DECKS_DIR)
    return profiles


def export_profiles_to_txt(profiles: list[DuelsInkArchetypeProfile], target_dir: Path | None = None) -> list[Path]:
    """Writes scraped Duels.ink meta decks as Dreamborn-compatible .txt files."""
    out_dir = target_dir or EXPORTED_DECKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, prof in enumerate(profiles, 1):
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", prof.name).strip("_").lower()
        fname = out_dir / f"{idx:02d}_{safe_slug}_{prof.games_played}g.txt"
        header = (
            f"# {prof.name} ({prof.ink_pair})\n"
            f"# Source: https://duels.ink/stats/core-bo1 (ID: {prof.profile_id})\n"
            f"# Games Played: {prof.games_played} | Win Rate: {prof.win_rate:.2f}% | Meta Share: {prof.meta_share*100:.2f}%\n"
        )
        fname.write_text(header + prof.deck.to_dreamborn_text() + "\n", encoding="utf-8")
        written.append(fname)
    return written


def load_duels_ink_profiles(
    db: CardDatabase | None = None,
    cache_path: Path | None = None,
    top_n: int = 12,
    deduplicate_by_name: bool = True,
) -> list[DuelsInkArchetypeProfile]:
    """
    Loads scraped Duels.ink Core Set 14 meta profiles from `data_cache/duels_ink_core_bo1.json`.
    """
    db = db or GLOBAL_CARD_DB
    path = cache_path or DUELS_INK_CACHE_FILE
    if not path.exists():
        return []

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    id_to_fullname = _build_set_num_to_fullname_map()

    raw_profiles = [
        p for p in raw_data.get("profiles", [])
        if not p.get("isMixed") and sum(p.get("centroidCounts", {}).values()) == 60
    ]
    raw_profiles.sort(key=lambda x: int(x.get("gamesPlayed", 0)), reverse=True)

    selected_raw: list[tuple[str, str, dict]] = []
    seen_titles: set[str] = set()

    for p in raw_profiles:
        inks = "/".join(str(c).capitalize() for c in p.get("colors", []))
        raw_arch = str(p.get("archetypeName") or p.get("name") or "Meta").strip()
        base_title = f"{inks} {raw_arch}" if inks.lower() not in raw_arch.lower() else raw_arch

        if deduplicate_by_name:
            if base_title in seen_titles:
                continue
            seen_titles.add(base_title)
            title = base_title
        else:
            title = base_title
            v_idx = 2
            while title in seen_titles:
                title = f"{base_title} (v{v_idx})"
                v_idx += 1
            seen_titles.add(title)

        selected_raw.append((title, inks, p))
        if len(selected_raw) >= top_n:
            break

    total_selected_games = max(1, sum(int(item[2].get("gamesPlayed", 0)) for item in selected_raw))
    result: list[DuelsInkArchetypeProfile] = []

    for title, inks, p in selected_raw:
        counts: dict[int, int] = {}
        for cid_str, qty in p.get("centroidCounts", {}).items():
            full_name = id_to_fullname.get(cid_str, "")
            card_tmpl = db.get_by_name(full_name) if full_name else None
            if card_tmpl is not None:
                counts[card_tmpl.card_id] = counts.get(card_tmpl.card_id, 0) + int(qty)

        deck = Deck(
            name=title,
            card_counts=counts,
            archetype_tag=inks,
            db=db,
        )
        sig_names = [
            id_to_fullname.get(sc, sc) for sc in (p.get("flagshipCards") or p.get("signatureCards") or [])
        ]
        games = int(p.get("gamesPlayed", 0))
        wr = float(p.get("winRate", 50.0))
        result.append(
            DuelsInkArchetypeProfile(
                profile_id=str(p.get("id", "")),
                name=title,
                ink_pair=inks,
                games_played=games,
                win_rate=wr,
                meta_share=round(games / total_selected_games, 4),
                signature_cards=sig_names,
                deck=deck,
            )
        )

    return result


def get_reference_meta_decks(db: CardDatabase | None = None, top_n: int = 10) -> dict[str, Deck]:
    """
    Returns the Season 14 Core Constructed reference meta decks scraped from https://duels.ink/stats/core-bo1.
    Falls back to synthetic seed decks only if the Duels.ink cache is missing.
    """
    db = db or GLOBAL_CARD_DB
    scraped = load_duels_ink_profiles(db=db, top_n=top_n, deduplicate_by_name=True)
    if scraped:
        return {prof.name: prof.deck for prof in scraped}

    # Fallback seed decks if cache file was deleted
    fallback = Deck.from_dreamborn_text(
        """
        4 Strike a Good Match
        4 Ursula - Vanessa
        4 Prince Phillip - Vanquisher of Foes
        4 Circle of Life
        4 Max Goof - Chart Topper
        4 Malicious, Mean, and Scary
        4 Angel - Siren Singer
        4 White Rabbit - Late Again
        4 Chernabog - Unnatural Force
        4 Lenny - Toy Binoculars
        4 Milo Thatch - Getting His Hands Dirty
        4 With a Few Good Friends
        4 Tod - Clever Fox
        4 Don Karnage - Debonair Pirate
        4 Max Goof - Karaoke Star
        """,
        name="Amber/Emerald Circle of Life",
        archetype_tag="Amber/Emerald",
        db=db,
    )
    return {fallback.name: fallback}


def get_metagame_shares(top_n: int = 10) -> dict[str, float]:
    """Returns empirical metagame play-rate weights computed from Duels.ink games played."""
    scraped = load_duels_ink_profiles(top_n=top_n, deduplicate_by_name=True)
    if scraped:
        return {prof.name: prof.meta_share for prof in scraped}
    return {"Amber/Emerald Circle of Life": 1.0}


DEFAULT_METAGAME_SHARES: dict[str, float] = get_metagame_shares(top_n=10)

# User-defined 44-card locked core for Amber/Emerald Circle of Life (leaves 16 flex/tech slots)
CIRCLE_OF_LIFE_CORE_TEXT: str = """
4 Strike a Good Match
4 Malicious, Mean, and Scary
4 Akood et Emuti
4 Circle of Life
4 Angel - Siren Singer
4 Ursula - Vanessa
4 Max Goof - Karaoke Star
4 Tod - Clever Fox
4 Meilin Lee - Losing Control
4 Prince Phillip - Vanquisher of Foes
4 Milo Thatch - Getting His Hands Dirty
"""

# User-defined 44-card locked core for Amber/Emerald Princess Aggro (leaves 16 flex/tech slots)
PRINCESS_AGGRO_CORE_TEXT: str = """
4 Mike Wazowski - Heroic Climber
4 Elinor - Renowned Diplomat
4 Grandmother Willow - Ancient Advisor
4 Bobby Zimuruski - Spray Cheese Kid
4 Dale - Ready for His Shot
4 Mickey Mouse - Best in Town
4 Aurora - Holding Court
4 Pocahontas - Guiding the Tribe
4 Pocahontas - Peacekeeper
4 Thomas - Wide-Eyed Recruit
4 Ursula - Deceiver
"""

# User's starting 60-card baseline version for Amber/Emerald Princess Aggro (44 Core + 16 Starting Flex)
PRINCESS_AGGRO_BASELINE_TEXT: str = """
4 Mike Wazowski - Heroic Climber (13-226)
4 Elinor - Renowned Diplomat (12-86)
4 Grandmother Willow - Ancient Advisor (11-206)
4 Bobby Zimuruski - Spray Cheese Kid (9-78)
4 Dale - Ready for His Shot (12-22)
4 Mickey Mouse - Expedition Leader (12-26)
4 Mickey Mouse - Best in Town (14-23)
4 Aurora - Holding Court (9-206)
4 Pocahontas - Guiding the Tribe (13-12)
4 Pocahontas - Peacekeeper (11-241)
4 Lionheart - Incumbent Mayor (14-18)
4 Thomas - Wide-Eyed Recruit (11-1)
4 Finnick - Tiny Terror (10-74)
2 Pocahontas - Finding the Way (11-8)
4 Ursula - Deceiver (3-90)
2 The Prince - Searching for His Love (12-23)
"""


# User-defined 40-card locked core for Emerald/Steel Green Steel (leaves 20 flex/tech slots)
GREEN_STEEL_CORE_TEXT: str = """
4 Ursula - Deceiver
1 Pegasus - Gift for Hercules
3 Max Goof - Rebellious Teen
4 Vixey - Forest Friend
4 Tod - All Alone
1 White Rabbit - Late Again
4 Elinor - Renowned Diplomat
2 The Queen - Devious Disguise
2 Mushu - Stealthy Dragon
4 Strength of a Raging Fire
3 Angela - Night Warrior
4 Flash - Efficient Clerk
2 Woolter & Jesse - Bellwether's Henchmen
2 Angel - Experiment 624
"""

# User's starting 60-card baseline version for Emerald/Steel Green Steel (most consistent)
GREEN_STEEL_BASELINE_TEXT: str = """
4 Ursula - Deceiver
4 Pegasus - Gift for Hercules
4 Max Goof - Rebellious Teen
4 Flynn Rider - Spectral Scoundrel
4 Vixey - Forest Friend
4 Tod - All Alone
4 White Rabbit - Late Again
4 Elinor - Renowned Diplomat
4 The Queen - Devious Disguise
4 Mushu - Stealthy Dragon
4 Strength of a Raging Fire
4 Angela - Night Warrior
4 Angel - Experiment 624
4 Flash - Efficient Clerk
4 Woolter & Jesse - Bellwether's Henchmen
"""


def get_circle_of_life_core_counts(db: CardDatabase | None = None) -> dict[int, int]:
    """Parses the 44-card locked core for Amber/Emerald Circle of Life into {card_id: count}."""
    db = db or GLOBAL_CARD_DB
    core_deck = Deck.from_dreamborn_text(
        CIRCLE_OF_LIFE_CORE_TEXT,
        name="Amber/Emerald Circle of Life (44-Card Core)",
        archetype_tag="Amber/Emerald",
        db=db,
    )
    return dict(core_deck.card_counts)


def get_princess_aggro_core_counts(db: CardDatabase | None = None) -> dict[int, int]:
    """Parses the 44-card locked core for Amber/Emerald Princess Aggro into {card_id: count}."""
    db = db or GLOBAL_CARD_DB
    core_deck = Deck.from_dreamborn_text(
        PRINCESS_AGGRO_CORE_TEXT,
        name="Amber/Emerald Princess Aggro (44-Card Core)",
        archetype_tag="Amber/Emerald",
        db=db,
    )
    return dict(core_deck.card_counts)


def get_princess_aggro_baseline_deck(db: CardDatabase | None = None) -> Deck:
    """Returns the user's 60-card starting baseline deck for Amber/Emerald Princess Aggro."""
    db = db or GLOBAL_CARD_DB
    return Deck.from_dreamborn_text(
        PRINCESS_AGGRO_BASELINE_TEXT,
        name="Amber/Emerald Princess Aggro (User Starting Baseline)",
        archetype_tag="Amber/Emerald",
        db=db,
    )


def get_green_steel_core_counts(db: CardDatabase | None = None) -> dict[int, int]:
    """Parses the 40-card locked core for Emerald/Steel Green Steel into {card_id: count}."""
    db = db or GLOBAL_CARD_DB
    core_deck = Deck.from_dreamborn_text(
        GREEN_STEEL_CORE_TEXT,
        name="Emerald/Steel Green Steel (40-Card Core)",
        archetype_tag="Emerald/Steel",
        db=db,
    )
    return dict(core_deck.card_counts)


def get_green_steel_baseline_deck(db: CardDatabase | None = None) -> Deck:
    """Returns the user's 60-card starting baseline deck for Emerald/Steel Green Steel."""
    db = db or GLOBAL_CARD_DB
    return Deck.from_dreamborn_text(
        GREEN_STEEL_BASELINE_TEXT,
        name="Emerald/Steel Green Steel (Most Consistent Baseline)",
        archetype_tag="Emerald/Steel",
        db=db,
    )


def get_archetype_core_and_baseline(
    archetype_name: str,
    db: CardDatabase | None = None,
) -> tuple[dict[int, int], Deck | None]:
    """Returns (core_card_counts, optional_baseline_deck) for a named archetype."""
    db = db or GLOBAL_CARD_DB
    name_low = archetype_name.lower()
    if "princess" in name_low:
        return get_princess_aggro_core_counts(db), get_princess_aggro_baseline_deck(db)
    if "steel" in name_low or "green" in name_low:
        return get_green_steel_core_counts(db), get_green_steel_baseline_deck(db)
    return get_circle_of_life_core_counts(db), None



