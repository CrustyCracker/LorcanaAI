"""
Test suite verifying Green Steel (Emerald/Steel) core and baseline integrity,
as well as the multi-package flex slot optimization system in TechCardScanner.
"""

import unittest
from pathlib import Path

from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import (
    get_green_steel_core_counts,
    get_green_steel_baseline_deck,
    get_archetype_core_and_baseline,
    GREEN_STEEL_CORE_TEXT,
    GREEN_STEEL_BASELINE_TEXT,
)
from lorcana_ai.deckbuilder.tech_scanner import (
    TechCardScanner,
    FlexPackageRecommendation,
    CoreOptimizationReport,
)


class TestGreenSteelAndTechPackages(unittest.TestCase):
    def setUp(self):
        self.db = GLOBAL_CARD_DB

    def test_green_steel_core_cards_and_counts(self):
        core_counts = get_green_steel_core_counts(self.db)
        total_cards = sum(core_counts.values())
        self.assertEqual(total_cards, 40, "Green Steel core must have exactly 40 cards")

        # Verify key cards in core
        card_names = {self.db.by_id[cid].full_name: cnt for cid, cnt in core_counts.items()}
        expected_cards = {
            "Ursula - Deceiver": 4,
            "Pegasus - Gift for Hercules": 1,
            "Max Goof - Rebellious Teen": 3,
            "Vixey - Forest Friend": 4,
            "Tod - All Alone": 4,
            "White Rabbit - Late Again": 1,
            "Elinor - Renowned Diplomat": 4,
            "The Queen - Devious Disguise": 2,
            "Mushu - Stealthy Dragon": 2,
            "Strength of a Raging Fire": 4,
            "Angela - Night Warrior": 3,
            "Flash - Efficient Clerk": 4,
            "Woolter & Jesse - Bellwether's Henchmen": 2,
            "Angel - Experiment 624": 2,
        }
        for name, expected_qty in expected_cards.items():
            self.assertIn(name, card_names, f"Core must contain {name}")
            self.assertEqual(card_names[name], expected_qty, f"{name} count in core must be {expected_qty}")

    def test_green_steel_baseline_deck(self):
        baseline = get_green_steel_baseline_deck(self.db)
        self.assertEqual(baseline.total_cards, 60, "Baseline deck must have exactly 60 cards")
        self.assertEqual(baseline.uninkable_count, 12, "Baseline deck must have exactly 12 uninkables")
        self.assertEqual(baseline.inks, ("Emerald", "Steel"), "Deck inks must be Emerald/Steel")

        card_names = {self.db.by_id[cid].full_name: cnt for cid, cnt in baseline.card_counts.items()}
        self.assertEqual(len(card_names), 15, "Baseline must contain exactly 15 unique cards at 4 copies each")
        for name, cnt in card_names.items():
            self.assertEqual(cnt, 4, f"Each card in baseline must have 4 copies, got {cnt} for {name}")

        # Check Flynn Rider is present at 4 copies
        self.assertEqual(card_names.get("Flynn Rider - Spectral Scoundrel"), 4)

    def test_green_steel_flex_difference(self):
        core_counts = get_green_steel_core_counts(self.db)
        baseline = get_green_steel_baseline_deck(self.db)

        flex_cards: dict[str, int] = {}
        for cid, cnt in baseline.card_counts.items():
            diff = cnt - core_counts.get(cid, 0)
            if diff > 0:
                flex_cards[self.db.by_id[cid].full_name] = diff

        total_flex = sum(flex_cards.values())
        self.assertEqual(total_flex, 20, "Difference between baseline and core must be exactly 20 flex slots")

        expected_flex = {
            "Flynn Rider - Spectral Scoundrel": 4,
            "Pegasus - Gift for Hercules": 3,
            "White Rabbit - Late Again": 3,
            "The Queen - Devious Disguise": 2,
            "Mushu - Stealthy Dragon": 2,
            "Angel - Experiment 624": 2,
            "Woolter & Jesse - Bellwether's Henchmen": 2,
            "Max Goof - Rebellious Teen": 1,
            "Angela - Night Warrior": 1,
        }
        self.assertEqual(flex_cards, expected_flex)

    def test_get_archetype_core_and_baseline_routing(self):
        core_counts, baseline = get_archetype_core_and_baseline("Green Steel", self.db)
        self.assertEqual(sum(core_counts.values()), 40)
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.total_cards, 60)

        core_gs, base_gs = get_archetype_core_and_baseline("Emerald / Steel", self.db)
        self.assertEqual(sum(core_gs.values()), 40)
        self.assertIsNotNone(base_gs)

    def test_multi_package_recommendations(self):
        core_counts = get_green_steel_core_counts(self.db)
        baseline = get_green_steel_baseline_deck(self.db)

        # Create 2 small test opponents
        opp1 = baseline.clone(new_name="Meta Opponent 1")
        opp2 = baseline.clone(new_name="Meta Opponent 2")
        meta_decks = {"Meta Opponent 1": opp1, "Meta Opponent 2": opp2}
        meta_shares = {"Meta Opponent 1": 0.5, "Meta Opponent 2": 0.5}

        scanner = TechCardScanner(db=self.db, lookahead_depth=1)
        report = scanner.optimize_core_for_duels_ink_meta(
            core_card_counts=core_counts,
            meta_decks=meta_decks,
            meta_shares=meta_shares,
            archetype_name="Green Steel",
            games_per_matchup=2,
            package_candidates=3,
            seed=42,
            baseline_deck=baseline,
        )

        self.assertIsInstance(report, CoreOptimizationReport)
        self.assertGreaterEqual(len(report.top_flex_packages), 1)
        self.assertLessEqual(len(report.top_flex_packages), 5)

        for pkg in report.top_flex_packages:
            self.assertIsInstance(pkg, FlexPackageRecommendation)
            self.assertGreater(pkg.rank, 0)
            self.assertIsInstance(pkg.package_name, str)
            self.assertIsInstance(pkg.flex_cards, dict)
            self.assertEqual(sum(pkg.flex_cards.values()), 20)
            self.assertGreaterEqual(pkg.expected_bo1_winrate, 0.0)
            self.assertLessEqual(pkg.expected_bo1_winrate, 1.0)
            self.assertGreater(len(pkg.tactical_summary), 5)

        # Check that top package matches report.selected_flex_package
        top_pkg = report.top_flex_packages[0]
        self.assertEqual(top_pkg.flex_cards, report.selected_flex_package)


if __name__ == "__main__":
    unittest.main()
