"""
Module 2: MAP-Elites Quality-Diversity + Genetic Coevolution Deckbuilder.
Discovers brand-new competitive 60-card decks across all 15 Two-Ink combinations
and 3 tactical profiles (Aggro, Midrange, Control/Ramp).
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import random

from lorcana_ai.config import DeckbuilderConfig, RESULTS_DIR
from lorcana_ai.data.card_db import CardDatabase, GLOBAL_CARD_DB, INK_COLORS
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import DEFAULT_METAGAME_SHARES, get_reference_meta_decks
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator
from lorcana_ai.engine.vectorizer import INK_PAIRS


@dataclass
class MapEliteCell:
    ink_pair: str
    style_profile: str  # "Aggro", "Midrange", "Control"
    deck: Deck
    expected_meta_winrate: float
    play_winrate: float
    draw_winrate: float
    generation_found: int
    matchup_breakdown: dict[str, float]


class MAPElitesDeckbuilder:
    """
    Maintains a 2D Quality-Diversity archive:
      Axis 1: 15 Two-Ink Pairs (e.g. Amber/Amethyst, Ruby/Sapphire, ...)
      Axis 2: 3 Curve/Style Profiles ("Aggro" [avg_cost < 3.1], "Midrange" [3.1..3.65], "Control" [> 3.65])
    Evolves new decks via synergistic crossover and curve-preserving mutation.
    """

    def __init__(
        self,
        db: CardDatabase | None = None,
        cfg: DeckbuilderConfig | None = None,
        evaluator: MatchupEvaluator | None = None,
    ) -> None:
        self.db = db or GLOBAL_CARD_DB
        self.cfg = cfg or DeckbuilderConfig()
        self.evaluator = evaluator or MatchupEvaluator()
        self.archive: dict[tuple[str, str], MapEliteCell] = {}
        self.meta_decks = get_reference_meta_decks(self.db)
        self.meta_shares = dict(DEFAULT_METAGAME_SHARES)

    @staticmethod
    def classify_style(deck: Deck) -> str:
        idx = deck.aggro_control_index
        if idx < 0.42 or deck.average_cost < 3.1:
            return "Aggro"
        if idx > 0.58 or deck.average_cost > 3.65:
            return "Control"
        return "Midrange"

    def generate_random_synergistic_deck(
        self,
        ink1: str,
        ink2: str,
        target_style: str = "Midrange",
        rng: random.Random | None = None,
    ) -> Deck:
        rng = rng or random.Random()
        pool = self.db.get_by_inks((ink1, ink2))
        if not pool:
            raise ValueError(f"No cards found for inks {ink1}/{ink2}")

        # Weight cards by target style curve preference
        weights: list[float] = []
        for c in pool:
            w = 1.0
            if target_style == "Aggro":
                if c.cost <= 2 and c.lore >= 2:
                    w += 4.0
                elif c.cost <= 3:
                    w += 2.0
                elif c.cost >= 6:
                    w *= 0.25
            elif target_style == "Control":
                if c.cost >= 5 or "banish" in c.full_text.lower() or "inkwell" in c.full_text.lower():
                    w += 3.5
                elif c.cost <= 2 and "draw" in c.full_text.lower():
                    w += 2.5
            else:
                if 2 <= c.cost <= 5:
                    w += 2.5
            weights.append(w)

        counts: dict[int, int] = {}
        total = 0
        uninkables = 0
        attempts = 0

        while total < 60 and attempts < 400:
            attempts += 1
            chosen = rng.choices(pool, weights=weights, k=1)[0]
            curr = counts.get(chosen.card_id, 0)
            if curr >= 4:
                continue
            add_qty = min(4 - curr, 60 - total, rng.choice([2, 3, 4]))
            if not chosen.inkable and (uninkables + add_qty > self.cfg.max_uninkable_count):
                continue
            counts[chosen.card_id] = curr + add_qty
            total += add_qty
            if not chosen.inkable:
                uninkables += add_qty

        # Fill remaining up to 60 with inkable cards
        if total < 60:
            inkables = [c for c in pool if c.inkable]
            for c in inkables:
                curr = counts.get(c.card_id, 0)
                if curr < 4:
                    take = min(4 - curr, 60 - total)
                    counts[c.card_id] = curr + take
                    total += take
                    if total >= 60:
                        break

        pair_str = f"{ink1}/{ink2}"
        return Deck(
            name=f"AI-{pair_str}-{target_style}",
            card_counts=counts,
            archetype_tag=pair_str,
            db=self.db,
        )

    def mutate_deck(self, parent: Deck, rng: random.Random) -> Deck:
        child = parent.clone(new_name=f"{parent.name}-Evolved")
        inks = child.inks
        if len(inks) < 2:
            other_inks = [i for i in INK_COLORS if i not in inks]
            inks = (inks[0], rng.choice(other_inks)) if inks else ("Amethyst", "Ruby")

        pool = self.db.get_by_inks(inks)
        present_ids = list(child.card_counts.keys())
        if not present_ids:
            return child

        # Swap 2 to 4 copies between 1-2 card slots
        num_swaps = rng.choice([1, 2])
        for _ in range(num_swaps):
            present_ids = [cid for cid, q in child.card_counts.items() if q > 0]
            if not present_ids:
                break
            cut_id = rng.choice(present_ids)
            cut_qty = min(child.card_counts[cut_id], rng.choice([2, 3, 4]))

            candidates = [
                c for c in pool
                if c.card_id != cut_id and child.card_counts.get(c.card_id, 0) + cut_qty <= 4
            ]
            if not candidates:
                continue
            add_card = rng.choice(candidates)

            # Check uninkable limit before applying
            unink_delta = (0 if add_card.inkable else cut_qty) - (
                0 if self.db.by_id[cut_id].inkable else cut_qty
            )
            if child.uninkable_count + unink_delta > self.cfg.max_uninkable_count:
                continue

            if child.card_counts[cut_id] == cut_qty:
                del child.card_counts[cut_id]
            else:
                child.card_counts[cut_id] -= cut_qty
            child.card_counts[add_card.card_id] = child.card_counts.get(add_card.card_id, 0) + cut_qty

        return child

    def crossover_decks(self, parent_a: Deck, parent_b: Deck, rng: random.Random) -> Deck:
        """Combines two decks that share the same 2-ink combination."""
        if parent_a.inks != parent_b.inks:
            return self.mutate_deck(parent_a, rng)

        combined_pool = list(set(parent_a.card_counts.keys()) | set(parent_b.card_counts.keys()))
        rng.shuffle(combined_pool)
        counts: dict[int, int] = {}
        total = 0
        uninkables = 0

        for cid in combined_pool:
            if total >= 60:
                break
            card = self.db.by_id[cid]
            pref_qty = max(parent_a.card_counts.get(cid, 0), parent_b.card_counts.get(cid, 0))
            take = min(pref_qty, 60 - total, 4)
            if not card.inkable and uninkables + take > self.cfg.max_uninkable_count:
                continue
            if take > 0:
                counts[cid] = take
                total += take
                if not card.inkable:
                    uninkables += take

        # Top up from parent_a ink pool if < 60
        if total < 60:
            for c in self.db.get_by_inks(parent_a.inks):
                if c.inkable and counts.get(c.card_id, 0) < 4:
                    take = min(4 - counts.get(c.card_id, 0), 60 - total)
                    counts[c.card_id] = counts.get(c.card_id, 0) + take
                    total += take
                    if total >= 60:
                        break

        pair_str = "/".join(parent_a.inks)
        return Deck(
            name=f"AI-{pair_str}-Hybrid",
            card_counts=counts,
            archetype_tag=pair_str,
            db=self.db,
        )

    def evaluate_against_metagame(
        self,
        candidate: Deck,
        games_per_matchup: int = 8,
        seed: int = 1001,
    ) -> tuple[float, float, float, dict[str, float]]:
        weighted_bo3 = 0.0
        weighted_play = 0.0
        weighted_draw = 0.0
        total_w = 0.0
        breakdown: dict[str, float] = {}

        for idx, (m_name, m_deck) in enumerate(self.meta_decks.items()):
            share = self.meta_shares.get(m_name, 1.0 / len(self.meta_decks))
            stats = self.evaluator.evaluate_matchup(
                candidate,
                m_deck,
                num_games=games_per_matchup,
                seed=seed + idx * 97,
            )
            weighted_bo3 += share * stats.bo3_winrate
            weighted_play += share * stats.play_winrate
            weighted_draw += share * stats.draw_winrate
            total_w += share
            breakdown[m_name] = stats.bo3_winrate

        norm = max(1e-6, total_w)
        return (
            round(weighted_bo3 / norm, 4),
            round(weighted_play / norm, 4),
            round(weighted_draw / norm, 4),
            breakdown,
        )

    def run_evolution(
        self,
        generations: int = 6,
        offspring_per_gen: int = 6,
        games_per_matchup: int = 6,
        target_ink_pairs: list[tuple[str, str]] | None = None,
        seed: int = 42,
    ) -> list[MapEliteCell]:
        rng = random.Random(seed)
        pairs_to_explore = target_ink_pairs or INK_PAIRS

        # 1. Seed archive with reference metagame decks & initial random archetypes
        for m_deck in self.meta_decks.values():
            if len(m_deck.inks) == 2 and (
                (m_deck.inks[0], m_deck.inks[1]) in pairs_to_explore
                or (m_deck.inks[1], m_deck.inks[0]) in pairs_to_explore
            ):
                self._try_insert(m_deck, generation=0, games_per_matchup=games_per_matchup, seed=seed)

        for ink1, ink2 in pairs_to_explore:
            style = rng.choice(["Aggro", "Midrange", "Control"])
            init_deck = self.generate_random_synergistic_deck(ink1, ink2, target_style=style, rng=rng)
            self._try_insert(init_deck, generation=0, games_per_matchup=games_per_matchup, seed=seed + 17)

        # 2. MAP-Elites Evolutionary Loop
        for gen in range(1, generations + 1):
            elites = list(self.archive.values())
            for off_idx in range(offspring_per_gen):
                if elites and rng.random() < 0.75:
                    parent_a = rng.choice(elites).deck
                    same_pair = [e.deck for e in elites if e.deck.inks == parent_a.inks]
                    if len(same_pair) >= 2 and rng.random() < 0.4:
                        parent_b = rng.choice(same_pair)
                        child = self.crossover_decks(parent_a, parent_b, rng)
                    else:
                        child = self.mutate_deck(parent_a, rng)
                else:
                    ink1, ink2 = rng.choice(pairs_to_explore)
                    style = rng.choice(["Aggro", "Midrange", "Control"])
                    child = self.generate_random_synergistic_deck(ink1, ink2, target_style=style, rng=rng)

                self._try_insert(
                    child,
                    generation=gen,
                    games_per_matchup=games_per_matchup,
                    seed=seed + gen * 1009 + off_idx * 37,
                )

        sorted_cells = sorted(
            self.archive.values(),
            key=lambda c: c.expected_meta_winrate,
            reverse=True,
        )
        self.save_archive(RESULTS_DIR / "map_elites_archive.json")
        return sorted_cells

    def _try_insert(
        self,
        deck: Deck,
        generation: int,
        games_per_matchup: int,
        seed: int,
    ) -> bool:
        valid, _ = deck.validate()
        if not valid:
            return False

        pair_str = "/".join(deck.inks)
        style = self.classify_style(deck)
        key = (pair_str, style)

        ev_wr, play_wr, draw_wr, breakdown = self.evaluate_against_metagame(
            deck, games_per_matchup=games_per_matchup, seed=seed
        )
        existing = self.archive.get(key)
        if existing is None or ev_wr > existing.expected_meta_winrate:
            deck.name = f"AI {pair_str} ({style} v{generation})"
            self.archive[key] = MapEliteCell(
                ink_pair=pair_str,
                style_profile=style,
                deck=deck,
                expected_meta_winrate=ev_wr,
                play_winrate=play_wr,
                draw_winrate=draw_wr,
                generation_found=generation,
                matchup_breakdown=breakdown,
            )
            return True
        return False

    def save_archive(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for cell in sorted(self.archive.values(), key=lambda c: c.expected_meta_winrate, reverse=True):
            payload.append(
                {
                    "ink_pair": cell.ink_pair,
                    "style_profile": cell.style_profile,
                    "deck_name": cell.deck.name,
                    "expected_meta_winrate": cell.expected_meta_winrate,
                    "play_winrate": cell.play_winrate,
                    "draw_winrate": cell.draw_winrate,
                    "uninkable_count": cell.deck.uninkable_count,
                    "average_cost": round(cell.deck.average_cost, 2),
                    "generation_found": cell.generation_found,
                    "matchup_breakdown": cell.matchup_breakdown,
                    "decklist": cell.deck.to_dreamborn_text(),
                }
            )
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
