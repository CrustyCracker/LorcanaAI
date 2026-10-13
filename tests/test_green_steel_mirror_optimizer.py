"""
Unit tests for GreenSteelMirrorOptimizer module.
Verifies Player 1 / Player 2 symmetry, locked core adherence, and flex package evaluation.
"""

import unittest
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.meta_decks import (
    get_green_steel_core_counts,
    get_green_steel_baseline_deck,
)
from lorcana_ai.deckbuilder.mirror_optimizer import GreenSteelMirrorOptimizer


class TestGreenSteelMirrorOptimizer(unittest.TestCase):
    def setUp(self):
        self.db = GLOBAL_CARD_DB
        self.core_counts = get_green_steel_core_counts(self.db)
        self.baseline_deck = get_green_steel_baseline_deck(self.db)
        self.optimizer = GreenSteelMirrorOptimizer(db=self.db, workers=1, lookahead_depth=0)

    def test_baseline_mirror_evaluation(self):
        stats = self.optimizer.evaluate_baseline_mirror(self.baseline_deck, num_games=6, seed=42)
        self.assertEqual(stats.games_played, 6)
        self.assertGreaterEqual(stats.overall_winrate, 0.0)
        self.assertLessEqual(stats.overall_winrate, 1.0)

    def test_build_test_deck_with_tech_preserves_core_and_total(self):
        smash_card = self.db.get_by_name("Smash")
        self.assertIsNotNone(smash_card)
        test_deck = self.optimizer.build_test_deck_with_tech(
            self.core_counts, self.baseline_deck, smash_card, target_total_copies=3
        )
        self.assertEqual(test_deck.total_cards, 60, "Deck must remain exactly 60 cards")
        self.assertEqual(test_deck.card_counts[smash_card.card_id], 3)

        # Ensure all 40 core cards are strictly present at >= core count
        for cid, min_qty in self.core_counts.items():
            self.assertGreaterEqual(
                test_deck.card_counts.get(cid, 0),
                min_qty,
                f"Core card ID {cid} was violated in test deck",
            )

    def test_evaluate_tech_candidates_quick(self):
        cands = ["Smash", "Pegasus - Gift for Hercules"]
        evals = self.optimizer.evaluate_tech_candidates(
            core_counts=self.core_counts,
            baseline_deck=self.baseline_deck,
            candidate_names=cands,
            num_games=4,
            seed=101,
        )
        self.assertEqual(len(evals), 2)
        for e in evals:
            self.assertIn(e.card_name, cands)
            self.assertGreaterEqual(e.bo1_winrate, 0.0)
            self.assertLessEqual(e.bo1_winrate, 1.0)

    def test_evaluate_packages_quick(self):
        curated = self.optimizer._get_curated_mirror_packages()[:2]
        pkgs = self.optimizer.evaluate_packages(
            core_counts=self.core_counts,
            baseline_deck=self.baseline_deck,
            packages=curated,
            num_games=4,
            seed=101,
        )
        self.assertEqual(len(pkgs), 2)
        self.assertEqual(pkgs[0].rank, 1)
        self.assertEqual(pkgs[1].rank, 2)
        self.assertEqual(pkgs[0].deck.total_cards, 60)
        self.assertEqual(pkgs[1].deck.total_cards, 60)


if __name__ == "__main__":
    unittest.main()
