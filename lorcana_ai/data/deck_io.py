"""
Deck representation, Season 14 Core Constructed (Sets 9-14) tournament legality validation,
and Dreamborn/Inkdecks parser/exporter.
Compliant with Official Disney Lorcana Comprehensive Rules v2.3.0 (Rule 1.10.1.1 - 1.10.1.3):
- Minimum 60 cards (1.10.1.1)
- Max 2 ink types, with Dual-Ink cards requiring both ink types (1.10.1.1.1)
- Max 4 copies per full name (1.10.1.2)
- Deck Construction Modifiers (1.10.1.3):
  - 'Christopher Robin - Hunny Sage' allows Hunny characters of any ink type in an Amethyst/Sapphire deck.
  - Custom copy limits parsed from rules text (e.g. 'up to 99 copies', 'any number of cards named').
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import re

from lorcana_ai.config import EngineConfig
from lorcana_ai.data.card_db import CardDatabase, CardTemplate, GLOBAL_CARD_DB

_RE_UP_TO_COPIES = re.compile(r"up\s+to\s+(\d+)\s+copies", re.IGNORECASE)


@dataclass
class Deck:
    """Represents a 60-card Disney Lorcana deck (Core Constructed Sets 9-14)."""
    name: str
    card_counts: dict[int, int]  # card_id -> count (1..4+)
    archetype_tag: str = "Custom"
    db: CardDatabase = field(default_factory=lambda: GLOBAL_CARD_DB, repr=False)

    @property
    def total_cards(self) -> int:
        return sum(self.card_counts.values())

    def has_hunny_sage_modifier(self) -> bool:
        """
        CR Rule 1.10.1.3: Checks if the deck includes 'Christopher Robin - Hunny Sage'
        ('you may also include Hunny characters of any ink type in your deck').
        """
        for cid, cnt in self.card_counts.items():
            if cnt > 0 and cid in self.db.by_id:
                c = self.db.by_id[cid]
                if "hunny characters of any ink" in c.full_text.lower() or "hunny sage" in c.full_name.lower():
                    return True
        return False

    @property
    def inks(self) -> tuple[str, ...]:
        has_hunny_mod = self.has_hunny_sage_modifier()
        colors: set[str] = set()
        for cid, cnt in self.card_counts.items():
            if cnt <= 0 or cid not in self.db.by_id:
                continue
            c = self.db.by_id[cid]
            if has_hunny_mod and c.card_type == "Character" and "Hunny" in c.subtypes:
                # Off-color Hunny characters are permitted by Christopher Robin - Hunny Sage in Amethyst/Sapphire
                for ink in c.inks:
                    if ink in ("Amethyst", "Sapphire"):
                        colors.add(ink)
                continue
            colors.update(c.inks)
        return tuple(sorted(colors))

    @property
    def uninkable_count(self) -> int:
        return sum(
            cnt for cid, cnt in self.card_counts.items()
            if cid in self.db.by_id and not self.db.by_id[cid].inkable
        )

    @property
    def average_cost(self) -> float:
        tot = self.total_cards
        if tot == 0:
            return 0.0
        weighted = sum(self.db.by_id[cid].cost * cnt for cid, cnt in self.card_counts.items())
        return weighted / tot

    @property
    def aggro_control_index(self) -> float:
        """
        Behavioral descriptor in [0.0 (Pure Aggro), 1.0 (Pure Control/Ramp)] for MAP-Elites grid.
        """
        tot = max(1, self.total_cards)
        high_cost = sum(
            cnt for cid, cnt in self.card_counts.items() if self.db.by_id[cid].cost >= 5
        )
        low_lore_quester = sum(
            cnt for cid, cnt in self.card_counts.items()
            if self.db.by_id[cid].cost <= 2 and self.db.by_id[cid].lore >= 2
        )
        raw = 0.5 + (high_cost - low_lore_quester) / (tot * 0.6)
        return max(0.0, min(1.0, raw))

    def expand_cards(self) -> list[CardTemplate]:
        cards: list[CardTemplate] = []
        for cid, cnt in self.card_counts.items():
            template = self.db.by_id[cid]
            cards.extend([template] * cnt)
        return cards

    def to_rust_tuples(self) -> list[tuple[int, int, bool, bool, bool, int, int, int, int, int]]:
        return [c.to_rust_tuple() for c in self.expand_cards()]

    @staticmethod
    def max_copies_for_card(card: CardTemplate, default_max: int = 4) -> int:
        """CR Rule 1.10.1.3: Checks if a card's rules text modifies the 4-copy limit."""
        lower = card.full_text.lower()
        if "any number of cards named" in lower:
            return 99
        m = _RE_UP_TO_COPIES.search(card.full_text)
        if m:
            return int(m.group(1))
        return default_max

    def validate(self, cfg: EngineConfig | None = None) -> tuple[bool, list[str]]:
        cfg = cfg or EngineConfig()
        errors: list[str] = []
        if self.total_cards < cfg.min_deck_size:
            errors.append(f"Deck has {self.total_cards} cards (minimum is {cfg.min_deck_size}).")
        if len(self.inks) > cfg.max_ink_colors:
            errors.append(f"Deck uses {len(self.inks)} inks {self.inks} (max is {cfg.max_ink_colors}).")
        allowed_sets = set(cfg.core_constructed_sets)
        for cid, cnt in self.card_counts.items():
            if cid not in self.db.by_id:
                errors.append(f"Unknown card ID {cid}.")
                continue
            card = self.db.by_id[cid]
            card_max = self.max_copies_for_card(card, cfg.max_copies_per_card)
            if cnt > card_max:
                errors.append(f"Card '{card.full_name}' has {cnt} copies (max is {card_max}).")
            if card.set_code not in allowed_sets:
                errors.append(
                    f"Card '{card.full_name}' (Set {card.set_code}) is not legal in Core Constructed Sets {cfg.core_constructed_sets}."
                )
        return (len(errors) == 0, errors)

    def clone(self, new_name: str | None = None) -> Deck:
        return Deck(
            name=new_name or self.name,
            card_counts=dict(self.card_counts),
            archetype_tag=self.archetype_tag,
            db=self.db,
        )

    def to_dreamborn_text(self) -> str:
        lines: list[str] = []
        sorted_items = sorted(
            self.card_counts.items(),
            key=lambda kv: (self.db.by_id[kv[0]].cost, self.db.by_id[kv[0]].full_name),
        )
        for cid, cnt in sorted_items:
            if cnt > 0:
                lines.append(f"{cnt} {self.db.by_id[cid].full_name}")
        return "\n".join(lines)

    @classmethod
    def from_dreamborn_text(
        cls,
        text: str,
        name: str = "Imported Deck",
        archetype_tag: str = "Imported",
        db: CardDatabase | None = None,
    ) -> Deck:
        db = db or GLOBAL_CARD_DB
        counts: Counter[int] = Counter()
        line_re = re.compile(r"^\s*(\d+)\s*x?\s+(.+?)\s*$")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            m = line_re.match(line)
            if not m:
                continue
            qty = int(m.group(1))
            card_name = re.sub(r"\s*[\(\[][0-9A-Za-z-]+[\)\]]\s*$", "", m.group(2)).strip()
            card = db.get_by_name(card_name)
            if card is not None:
                counts[card.card_id] += qty
        return cls(name=name, card_counts=dict(counts), archetype_tag=archetype_tag, db=db)
