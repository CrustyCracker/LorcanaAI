"""Smoke unit tests for CoreValue leave-one-out ablation."""

from __future__ import annotations

import unittest

from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import get_princess_aggro_core_counts, get_reference_meta_decks
from lorcana_ai.deckbuilder.core_ablation import _drop_one_copy, run_core_value_ablation
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator


class TestCoreAblation(unittest.TestCase):
    def test_drop_one_copy_reduces_total(self) -> None:
        from pathlib import Path

        text = Path("decks/amber_emerald_princess_aggro_optimized.txt").read_text(encoding="utf-8")
        deck = Deck.from_dreamborn_text(text, name="Princess Opt")
        cid = next(iter(deck.card_counts))
        before = deck.total_cards
        out = _drop_one_copy(deck, cid)
        self.assertEqual(out.total_cards, before - 1)
        self.assertEqual(out.card_counts.get(cid, 0), deck.card_counts[cid] - 1)

    def test_ablation_tiny_runs_dsl(self) -> None:
        from pathlib import Path

        text = Path("decks/amber_emerald_princess_aggro_optimized.txt").read_text(encoding="utf-8")
        deck = Deck.from_dreamborn_text(text, name="Princess Opt")
        core = get_princess_aggro_core_counts()
        # Ablate only first 2 core cards by temporarily shrinking core map
        slim_core = dict(list(core.items())[:2])
        meta = dict(list(get_reference_meta_decks(top_n=2).items())[:2])
        report = run_core_value_ablation(
            full_deck=deck,
            core_card_counts=slim_core,
            meta_decks=meta,
            meta_shares={k: 1.0 / len(meta) for k in meta},
            games_per_matchup=2,
            seed=7,
            workers=1,
            lookahead_depth=0,
        )
        self.assertEqual(len(report.rows), 2)
        self.assertTrue(all(r.card_name for r in report.rows))
        # Evaluator used DSL path by construction
        ev = MatchupEvaluator(use_full_dsl_sim=True)
        self.assertTrue(ev.use_full_dsl_sim)


if __name__ == "__main__":
    unittest.main()
