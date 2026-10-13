"""
Card Database & LorcanaJSON Parser for LorcanaAlpha (Season 14 Core Constructed: Sets 9-14).
Loads all 1,228 unique cards from Sets 9, 10, 11, 12, 13, and 14:
- Set 9: Fabled
- Set 10: Whispers in the Well
- Set 11: Winterspell
- Set 12: Wilds Unknown
- Set 13: Attack of the Vine!
- Set 14: Hyperia City
Supports Single-Ink and Dual-Ink cards and all Comprehensive Rules v2.3.0 (Oct 16, 2026) mechanics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

from lorcana_ai.config import DATA_DIR, ClusterConfig, EngineConfig
from lorcana_ai.data.effect_dsl import (
    CompiledAbilities,
    EffectDSLCompiler,
)

INK_COLORS: tuple[str, ...] = (
    "Amber",
    "Amethyst",
    "Emerald",
    "Ruby",
    "Sapphire",
    "Steel",
)

CORE_CONSTRUCTED_SETS: tuple[str, ...] = EngineConfig().core_constructed_sets


@dataclass
class CardTemplate:
    """Immutable definition of a Disney Lorcana card (Season 14 Core Constructed)."""
    card_id: int
    full_name: str
    base_name: str
    ink: str                        # e.g. "Amber" or "Amber-Ruby"
    inks: tuple[str, ...]           # e.g. ("Amber",) or ("Amber", "Ruby")
    cost: int
    inkable: bool
    card_type: str                  # "Character", "Action", "Item", "Location"
    subtypes: tuple[str, ...] = ()
    is_song: bool = False
    strength: int = 0
    willpower: int = 0
    lore: int = 0
    move_cost: int = 0
    set_code: str = "9"
    rarity: str = "Common"
    full_text: str = ""
    abilities: CompiledAbilities = field(default_factory=CompiledAbilities)

    @property
    def is_dual_ink(self) -> bool:
        return len(self.inks) > 1

    def matches_base_name_for_shift(self, target_base_name: str) -> bool:
        """Checks if this card's name matches `target_base_name` for Standard/Combo Shift."""
        if self.base_name == target_base_name:
            return True
        if (self.abilities.is_combo_shift or self.abilities.is_duo_shift) and "&" in self.base_name:
            parts = [p.strip() for p in self.base_name.split("&")]
            return target_base_name in parts
        return False

    def can_shift_onto_single(self, target_card: CardTemplate) -> bool:
        """
        Comprehensive Rules 8.10.2 & 8.10.8: Checks if this card can Shift onto a single card in play.
        - Duo Shift (8.10.8.3) requires TWO characters in play, so returns False for single target.
        - Potato Shift (8.10.8.6) shifts onto an Item named 'Potato'.
        - Universal Shift (8.10.8.2) shifts onto any character.
        - [Classification] Shift (8.10.8.1) shifts onto a character with that subtype (e.g. Madrigal, Floodborn, Red Panda).
        - Combo Shift (8.10.8.4) shifts onto a character matching either name.
        - Standard/Temporary Shift (8.10.2 / 8.10.8.5) shifts onto a character with the same base name.
        """
        if self.abilities.shift_cost is None:
            return False
        if self.abilities.is_duo_shift:
            return False
        if self.abilities.is_potato_shift:
            return target_card.card_type == "Item" and target_card.base_name.lower() == "potato"
        if target_card.card_type != "Character":
            return False
        if target_card.abilities.is_universal_shift or target_card.base_name == "Morph":
            return True
        if self.abilities.is_universal_shift:
            return True
        if self.abilities.shift_classification:
            req_sub = self.abilities.shift_classification.lower()
            return any(s.lower() == req_sub for s in target_card.subtypes)
        return self.matches_base_name_for_shift(target_card.base_name)

    def can_shift_onto_pair(self, char1: CardTemplate, char2: CardTemplate) -> bool:
        """
        Comprehensive Rules 8.10.8.3 (Duo Shift) & 8.10.8.4 (Combo Shift):
        Checks if this card can Shift onto two characters in play that each match one of the names on the card.
        """
        if self.abilities.shift_cost is None:
            return False
        if not (self.abilities.is_duo_shift or self.abilities.is_combo_shift):
            return False
        if char1.card_type != "Character" or char2.card_type != "Character":
            return False
        if "&" not in self.base_name:
            return False
        parts = [p.strip() for p in self.base_name.split("&")]
        if len(parts) != 2:
            return False
        return (
            (char1.base_name == parts[0] and char2.base_name == parts[1])
            or (char1.base_name == parts[1] and char2.base_name == parts[0])
        )

    def to_rust_tuple(self) -> tuple[int, int, bool, bool, bool, int, int, int, int, int]:
        flags, eff_val = self.abilities.to_rust_bitflags()
        return (
            self.card_id,
            self.cost,
            self.inkable,
            self.card_type == "Character",
            self.is_song,
            self.strength,
            self.willpower,
            self.lore,
            flags,
            eff_val,
        )


def parse_inks(raw_color: str) -> tuple[str, ...]:
    if not raw_color:
        return ()
    parts = [p.strip().capitalize() for p in raw_color.split("-") if p.strip()]
    valid = tuple(p for p in parts if p in INK_COLORS)
    return valid


class CardDatabase:
    """
    Central registry of Disney Lorcana cards for Season 14 Core Constructed (Sets 9, 10, 11, 12, 13, 14).
    Loads all 1,228 unique Core Constructed cards from `data_cache/allCards.json` (offline-ready for PLGrid Athena).
    """

    def __init__(
        self,
        allowed_sets: Iterable[str] | None = None,
        auto_load_cache: bool = True,
    ) -> None:
        self.allowed_sets: set[str] = set(allowed_sets or CORE_CONSTRUCTED_SETS)
        self.by_id: dict[int, CardTemplate] = {}
        self.by_name: dict[str, CardTemplate] = {}
        self.sets_metadata: dict[str, str] = {}

        if auto_load_cache:
            cache_file = DATA_DIR / "allCards.json"
            if cache_file.exists():
                self.load_from_lorcana_json_file(cache_file)

    def register(self, card: CardTemplate) -> None:
        self.by_id[card.card_id] = card
        self.by_name[card.full_name.lower()] = card
        norm_name = card.full_name.lower().replace("’", "'")
        self.by_name[norm_name] = card

    def get_by_name(self, name: str) -> CardTemplate | None:
        key = name.strip().lower().replace("’", "'")
        if key in self.by_name:
            return self.by_name[key]
        for k, card in self.by_name.items():
            if key in k or k in key:
                return card
        return None

    def get_by_inks(self, inks: Iterable[str]) -> list[CardTemplate]:
        """
        Returns all Core Constructed cards legal in a deck with the given ink colors.
        Includes Dual-Ink cards only if BOTH of the card's inks are in `allowed`.
        """
        allowed = {i.capitalize() for i in inks}
        return [
            c for c in self.by_id.values()
            if c.inks and set(c.inks).issubset(allowed)
        ]

    def sync_from_lorcana_json(self, url: str | None = None, timeout: int = 30) -> int:
        """Downloads latest allCards.json from LorcanaJSON and loads Sets 9-14 Core Constructed cards."""
        import urllib.request

        target_url = url or ClusterConfig().lorcana_json_url
        cache_file = DATA_DIR / "allCards.json"
        req = urllib.request.Request(target_url, headers={"User-Agent": "LorcanaAlpha/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cache_file.write_bytes(resp.read())
        return self.load_from_lorcana_json_file(cache_file)

    def load_from_lorcana_json_file(self, path: Path) -> int:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for scode, sinfo in raw.get("sets", {}).items():
            if str(scode) in self.allowed_sets:
                self.sets_metadata[str(scode)] = sinfo.get("name", f"Set {scode}")

        cards_list = raw.get("cards", [])
        self.by_id.clear()
        self.by_name.clear()

        added = 0
        next_id = 1
        for item in cards_list:
            set_code = str(item.get("setCode", ""))
            if set_code not in self.allowed_sets:
                continue

            full_name = item.get("fullName") or item.get("name")
            raw_color = str(item.get("color", ""))
            inks = parse_inks(raw_color)
            if not full_name or not inks:
                continue

            # Deduplicate reprints within Sets 9-14 by fullName
            norm_key = full_name.strip().lower().replace("’", "'")
            if norm_key in self.by_name:
                continue

            base_name = str(item.get("name") or full_name.split(" - ")[0]).strip()
            cost = int(item.get("cost", 1) or 1)
            inkable = bool(item.get("inkwell", True))
            ctype = str(item.get("type", "Character"))
            subtypes = tuple(item.get("subtypes", []))
            is_song = "Song" in subtypes or "Song" in str(item.get("subtypesText", ""))
            strength = int(item.get("strength", 0) or 0)
            willpower = int(item.get("willpower", 0) or 0)
            lore = int(item.get("lore", 0) or 0)
            move_cost = int(item.get("moveCost", 1) or 1) if ctype == "Location" else 0
            rarity = str(item.get("rarity", "Common"))
            full_text = " ".join(str(item.get("fullText", "") or "").split())
            kw_list = list(item.get("keywordAbilities", []) or [])

            abilities = EffectDSLCompiler.compile_from_text(
                full_text=full_text,
                card_type=ctype,
                cost=cost,
                keyword_list=kw_list,
            )

            card = CardTemplate(
                card_id=next_id,
                full_name=full_name,
                base_name=base_name,
                ink=raw_color,
                inks=inks,
                cost=cost,
                inkable=inkable,
                card_type=ctype,
                subtypes=subtypes,
                is_song=is_song,
                strength=strength,
                willpower=willpower,
                lore=lore,
                move_cost=move_cost,
                set_code=set_code,
                rarity=rarity,
                full_text=full_text,
                abilities=abilities,
            )
            self.register(card)
            next_id += 1
            added += 1

        return added


GLOBAL_CARD_DB = CardDatabase()
