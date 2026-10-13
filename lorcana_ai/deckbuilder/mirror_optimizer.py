"""
Module: Green/Steel (Emerald/Steel) Mirror Match Optimizer.
Evaluates individual candidate tech cards and 20-slot flex packages exclusively in the Green/Steel mirror match.
Player 2 (Opponent) is strictly locked to the 60-card Baseline list.
Player 1 (Evaluated Deck) retains the 40-card locked core while varying the 20 flex/tech slots.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Sequence

from lorcana_ai.data.card_db import CardDatabase, CardTemplate, GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import (
    get_green_steel_core_counts,
    get_green_steel_baseline_deck,
)
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator, MatchupStats


@dataclass
class MirrorTechEvaluation:
    """Evaluation of a single candidate tech card in the Green/Steel mirror match."""
    card_name: str
    card_id: int
    cost: int
    inkable: bool
    card_type: str
    copies_tested: int
    is_core_tech: bool
    core_copies: int
    bo1_winrate: float
    delta_bo1_winrate: float
    bo3_winrate: float
    delta_bo3_winrate: float
    play_winrate: float
    draw_winrate: float
    games_simulated: int
    mirror_tactical_role: str


@dataclass
class MirrorFlexPackage:
    """A ranked 20-card flex package evaluated against the baseline Green/Steel mirror."""
    rank: int
    package_name: str
    flex_cards: dict[str, int]
    bo1_winrate: float
    delta_bo1_winrate: float
    bo3_winrate: float
    delta_bo3_winrate: float
    play_winrate: float
    draw_winrate: float
    uninkable_count: int
    average_cost: float
    deck: Deck
    tactical_summary: str


@dataclass
class MirrorOptimizationReport:
    """Complete report of Green/Steel mirror optimization."""
    archetype_name: str
    locked_core_count: int
    flex_slots_count: int
    baseline_bo1_winrate: float
    baseline_bo3_winrate: float
    baseline_play_winrate: float
    baseline_draw_winrate: float
    optimized_bo1_winrate: float
    optimized_bo3_winrate: float
    optimized_play_winrate: float
    optimized_draw_winrate: float
    delta_bo1_winrate: float
    delta_bo3_winrate: float
    top_tech_cards: list[MirrorTechEvaluation]
    top_flex_packages: list[MirrorFlexPackage]
    selected_flex_package: dict[str, int]
    optimized_deck: Deck
    baseline_deck: Deck


class GreenSteelMirrorOptimizer:
    """
    Simulates and ranks tech cards and flex packages strictly within the Green/Steel mirror match.
    Player 2 always plays the fixed 60-card Baseline Green Steel deck.
    Player 1 keeps the 40-card locked core and adjusts the remaining 20 flex/tech slots.
    """

    def __init__(
        self,
        db: CardDatabase | None = None,
        evaluator: MatchupEvaluator | None = None,
        workers: int | None = None,
        lookahead_depth: int = 1,
    ) -> None:
        self.db = db or GLOBAL_CARD_DB
        self.evaluator = evaluator or MatchupEvaluator(workers=workers, lookahead_depth=lookahead_depth)

    def evaluate_baseline_mirror(
        self,
        baseline_deck: Deck,
        num_games: int = 120,
        seed: int = 2137,
    ) -> MatchupStats:
        """Evaluates baseline mirror (Baseline vs Baseline). Expected symmetrical winrate ~50.0%."""
        return self.evaluator.evaluate_matchup(
            baseline_deck,
            baseline_deck,
            num_games=num_games,
            seed=seed,
        )

    def build_test_deck_with_tech(
        self,
        core_counts: dict[int, int],
        baseline_deck: Deck,
        tech_card: CardTemplate,
        target_total_copies: int,
    ) -> Deck:
        """
        Builds a 60-card test deck for Player 1 with `tech_card` set to `target_total_copies`.
        Cuts are made exclusively from the 20 flex slots of `baseline_deck`, never touching the locked core.
        """
        counts = dict(baseline_deck.card_counts)
        current_copies = counts.get(tech_card.card_id, 0)
        core_copies = core_counts.get(tech_card.card_id, 0)

        if target_total_copies > current_copies:
            to_add = target_total_copies - current_copies
            flex_cids = [
                cid for cid, q in counts.items()
                if cid != tech_card.card_id and q > core_counts.get(cid, 0)
            ]
            # Cut flex cards prioritizing uninkable balance and higher costs
            flex_cids.sort(
                key=lambda cid: (
                    (not self.db.by_id[cid].inkable) == (not tech_card.inkable),
                    self.db.by_id[cid].cost,
                    counts[cid] - core_counts.get(cid, 0),
                ),
                reverse=True,
            )
            removed = 0
            for f_cid in flex_cids:
                if removed >= to_add:
                    break
                removable = counts[f_cid] - core_counts.get(f_cid, 0)
                cut = min(removable, to_add - removed)
                if cut > 0:
                    counts[f_cid] -= cut
                    if counts[f_cid] <= 0:
                        del counts[f_cid]
                    removed += cut
            counts[tech_card.card_id] = current_copies + removed
        elif target_total_copies < current_copies:
            to_cut = current_copies - target_total_copies
            counts[tech_card.card_id] = target_total_copies
            if counts[tech_card.card_id] <= 0:
                del counts[tech_card.card_id]
            # Fill with solid inkable flex cards (e.g. White Rabbit or Pegasus)
            filler_cids = [
                cid for cid, q in counts.items()
                if cid != tech_card.card_id and q < 4 and self.db.by_id[cid].inkable
            ]
            added = 0
            for f_cid in filler_cids:
                if added >= to_cut:
                    break
                can_add = 4 - counts[f_cid]
                add_qty = min(can_add, to_cut - added)
                counts[f_cid] += add_qty
                added += add_qty

        return Deck(
            name=f"Green Steel [Tech: {tech_card.full_name}]",
            card_counts=counts,
            archetype_tag="Emerald/Steel",
            db=self.db,
        )

    def evaluate_tech_candidates(
        self,
        core_counts: dict[int, int],
        baseline_deck: Deck,
        candidate_names: Sequence[str],
        num_games: int = 80,
        seed: int = 2137,
    ) -> list[MirrorTechEvaluation]:
        """
        Evaluates each candidate tech card individually against Player 2's Baseline Deck in the GS mirror.
        Uses Common Random Numbers (CRN) for precise paired-comparison delta winrates.
        """
        base_stats = self.evaluate_baseline_mirror(baseline_deck, num_games=num_games, seed=seed)
        base_bo1 = base_stats.overall_winrate
        base_bo3 = base_stats.bo3_winrate

        evals: list[MirrorTechEvaluation] = []

        for cname in candidate_names:
            card = self.db.get_by_name(cname)
            if card is None:
                continue

            is_core = card.card_id in core_counts and 0 < core_counts[card.card_id] < 4
            core_copies = core_counts.get(card.card_id, 0)

            if is_core:
                target_copies = 4
                tested_deck = self.build_test_deck_with_tech(
                    core_counts, baseline_deck, card, target_copies
                )
            else:
                target_copies = 3 if card.cost <= 3 else 2
                tested_deck = self.build_test_deck_with_tech(
                    core_counts, baseline_deck, card, target_copies
                )

            stats = self.evaluator.evaluate_matchup(
                tested_deck,
                baseline_deck,
                num_games=num_games,
                seed=seed,
            )

            d_bo1 = round(stats.overall_winrate - base_bo1, 4)
            d_bo3 = round(stats.bo3_winrate - base_bo3, 4)

            evals.append(
                MirrorTechEvaluation(
                    card_name=card.full_name,
                    card_id=card.card_id,
                    cost=card.cost,
                    inkable=card.inkable,
                    card_type=card.card_type,
                    copies_tested=target_copies,
                    is_core_tech=is_core,
                    core_copies=core_copies,
                    bo1_winrate=stats.overall_winrate,
                    delta_bo1_winrate=d_bo1,
                    bo3_winrate=stats.bo3_winrate,
                    delta_bo3_winrate=d_bo3,
                    play_winrate=stats.play_winrate,
                    draw_winrate=stats.draw_winrate,
                    games_simulated=num_games,
                    mirror_tactical_role=self.explain_mirror_role(card),
                )
            )

        evals.sort(key=lambda te: (te.bo1_winrate + te.bo3_winrate, te.delta_bo1_winrate), reverse=True)
        return evals

    def evaluate_packages(
        self,
        core_counts: dict[int, int],
        baseline_deck: Deck,
        packages: list[tuple[str, dict[str, int], str]],
        num_games: int = 120,
        seed: int = 2137,
    ) -> list[MirrorFlexPackage]:
        """
        Evaluates full 20-card flex packages combined with the 40-card core vs Player 2's Baseline Deck.
        """
        base_stats = self.evaluate_baseline_mirror(baseline_deck, num_games=num_games, seed=seed)
        base_bo1 = base_stats.overall_winrate
        base_bo3 = base_stats.bo3_winrate

        results: list[MirrorFlexPackage] = []

        for pkg_name, flex_dict, summary in packages:
            # Build 60-card deck
            deck_counts = dict(core_counts)
            for cname, qty in flex_dict.items():
                c = self.db.get_by_name(cname)
                if c is not None:
                    deck_counts[c.card_id] = deck_counts.get(c.card_id, 0) + qty

            deck = Deck(
                name=f"Green Steel Mirror [{pkg_name}]",
                card_counts=deck_counts,
                archetype_tag="Emerald/Steel",
                db=self.db,
            )

            stats = self.evaluator.evaluate_matchup(
                deck,
                baseline_deck,
                num_games=num_games,
                seed=seed,
            )

            d_bo1 = round(stats.overall_winrate - base_bo1, 4)
            d_bo3 = round(stats.bo3_winrate - base_bo3, 4)

            results.append(
                MirrorFlexPackage(
                    rank=0,
                    package_name=pkg_name,
                    flex_cards=flex_dict,
                    bo1_winrate=stats.overall_winrate,
                    delta_bo1_winrate=d_bo1,
                    bo3_winrate=stats.bo3_winrate,
                    delta_bo3_winrate=d_bo3,
                    play_winrate=stats.play_winrate,
                    draw_winrate=stats.draw_winrate,
                    uninkable_count=deck.uninkable_count,
                    average_cost=round(deck.average_cost, 2),
                    deck=deck,
                    tactical_summary=summary,
                )
            )

        results.sort(key=lambda p: (p.bo1_winrate + p.bo3_winrate, p.delta_bo1_winrate), reverse=True)
        for idx, p in enumerate(results, 1):
            p.rank = idx

        return results

    def run_full_mirror_optimization(
        self,
        candidate_tech_names: Sequence[str] | None = None,
        num_games_tech: int = 80,
        num_games_packages: int = 140,
        seed: int = 2137,
    ) -> MirrorOptimizationReport:
        """Executes full Phase 1 (Single Techs) and Phase 2 (Flex Packages) mirror optimization."""
        core_counts = get_green_steel_core_counts(self.db)
        baseline_deck = get_green_steel_baseline_deck(self.db)

        base_stats = self.evaluate_baseline_mirror(baseline_deck, num_games=num_games_packages, seed=seed)

        if candidate_tech_names is None:
            candidate_tech_names = [
                # Internal core cards scaling to 4
                "Pegasus - Gift for Hercules",
                "Max Goof - Rebellious Teen",
                "White Rabbit - Late Again",
                "The Queen - Devious Disguise",
                "Mushu - Stealthy Dragon",
                "Angela - Night Warrior",
                "Woolter & Jesse - Bellwether's Henchmen",
                "Angel - Experiment 624",
                # External candidates
                "Flynn Rider - Spectral Scoundrel",
                "Smash",
                "Fire the Cannons!",
                "Look What You've Done",
                "Malicious, Mean, and Scary",
                "Sudden Chill",
                "Strike a Good Match",
                "Robin Hood - Champion of Sherwood",
                "Tinker Bell - Giant Fairy",
                "Cursed Merfolk - Ursula's Handiwork",
                "Tod & Copper - Best of Friends",
                "Inspector Tezuka - Resolute Officer",
                "Robin Hood - Daydreamer",
                "Three Arrows",
                "Bobby Zimuruski - Spray Cheese Kid",
                "Diablo - Protecting His Mistress",
                "Mother Knows Best",
                "To Wither a Flower",
                "Benja - Guardian of the Dragon Gem",
                "Rapunzel - Tower Defender",
                "Broadway - Sturdy and Strong",
                "Tinker Bell - Tiny Tactician",
            ]

        # 1. Phase 1: Tech Cards
        top_techs = self.evaluate_tech_candidates(
            core_counts=core_counts,
            baseline_deck=baseline_deck,
            candidate_names=candidate_tech_names,
            num_games=num_games_tech,
            seed=seed,
        )

        # 2. Phase 2: Curated 20-slot Flex Packages for the Mirror Match
        packages_def = self._get_curated_mirror_packages()
        top_packages = self.evaluate_packages(
            core_counts=core_counts,
            baseline_deck=baseline_deck,
            packages=packages_def,
            num_games=num_games_packages,
            seed=seed,
        )

        best_package = top_packages[0]

        return MirrorOptimizationReport(
            archetype_name="Emerald/Steel Green Steel Mirror",
            locked_core_count=40,
            flex_slots_count=20,
            baseline_bo1_winrate=base_stats.overall_winrate,
            baseline_bo3_winrate=base_stats.bo3_winrate,
            baseline_play_winrate=base_stats.play_winrate,
            baseline_draw_winrate=base_stats.draw_winrate,
            optimized_bo1_winrate=best_package.bo1_winrate,
            optimized_bo3_winrate=best_package.bo3_winrate,
            optimized_play_winrate=best_package.play_winrate,
            optimized_draw_winrate=best_package.draw_winrate,
            delta_bo1_winrate=best_package.delta_bo1_winrate,
            delta_bo3_winrate=best_package.delta_bo3_winrate,
            top_tech_cards=top_techs,
            top_flex_packages=top_packages,
            selected_flex_package=best_package.flex_cards,
            optimized_deck=best_package.deck,
            baseline_deck=baseline_deck,
        )

    @staticmethod
    def _get_curated_mirror_packages() -> list[tuple[str, dict[str, int], str]]:
        """Returns 12 curated 20-card flex packages designed to break the Green/Steel mirror."""
        return [
            (
                "Baseline Mirror (Wersja Wyjściowa / Maksymalna Konsystencja)",
                {
                    "Flynn Rider - Spectral Scoundrel": 4,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "The Queen - Devious Disguise": 2,
                    "Mushu - Stealthy Dragon": 2,
                    "Angel - Experiment 624": 2,
                    "Woolter & Jesse - Bellwether's Henchmen": 2,
                    "Max Goof - Rebellious Teen": 1,
                    "Angela - Night Warrior": 1,
                },
                "Wersja wyjściowa (15 kart po 4 kopie). Zrównoważona, agresywna krzywa z pełnym Evasive i Ward.",
            ),
            (
                "Removal & Tempo Mirror-Breaker (Smash + Fire the Cannons! + Robin Hood)",
                {
                    "Smash": 3,
                    "Fire the Cannons!": 3,
                    "Robin Hood - Champion of Sherwood": 2,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "The Queen - Devious Disguise": 2,
                    "Mushu - Stealthy Dragon": 2,
                    "Angel - Experiment 624": 2,
                },
                "Likwiduje wczesne 1- i 2-tury rywala (Flynn, Tod, White Rabbit). Robin Hood daje swing +2 lore i kartę.",
            ),
            (
                "Sweeper & AoE Ping (Tinker Bell Giant Fairy + Malicious Mean & Scary + Smash)",
                {
                    "Tinker Bell - Giant Fairy": 2,
                    "Malicious, Mean, and Scary": 3,
                    "Smash": 3,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Mushu - Stealthy Dragon": 2,
                    "Angel - Experiment 624": 2,
                    "Woolter & Jesse - Bellwether's Henchmen": 2,
                },
                "Zamiata szerokie stoły rywala (Tod, Flynn, Ursula, Flash). Osłabia siłę 'Strength of a Raging Fire' przeciwnika.",
            ),
            (
                "Hand Starvation & Disruption (Sudden Chill + Cursed Merfolk + Strike a Good Match)",
                {
                    "Sudden Chill": 3,
                    "Cursed Merfolk - Ursula's Handiwork": 3,
                    "Strike a Good Match": 3,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Flynn Rider - Spectral Scoundrel": 3,
                    "The Queen - Devious Disguise": 2,
                },
                "W mirrorze rywal szybko wypstrykuje się z kart. Odrzucanie kart odcina go od Strength i top-endów.",
            ),
            (
                "Evasive Aerial Dominance (Robin Hood Daydreamer + Diablo + Pegasus + White Rabbit)",
                {
                    "Robin Hood - Daydreamer": 2,
                    "Diablo - Protecting His Mistress": 2,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Mushu - Stealthy Dragon": 2,
                    "Flynn Rider - Spectral Scoundrel": 4,
                    "Angel - Experiment 624": 2,
                    "The Queen - Devious Disguise": 2,
                },
                "Maksymalizacja lataczy Evasive — uziemione postacie rywala (Tod, Flash, Elinor) nie mogą ich wyzwać.",
            ),
            (
                "Control Removal & Card Advantage (Smash + Three Arrows + Strike a Good Match + Robin Hood)",
                {
                    "Smash": 3,
                    "Three Arrows": 3,
                    "Strike a Good Match": 3,
                    "Robin Hood - Champion of Sherwood": 2,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Mushu - Stealthy Dragon": 2,
                    "The Queen - Devious Disguise": 1,
                },
                "Czyste usuwanie zagrożeń rywala, rotacja kart przez Strike i przewaga zasobów dzięki Robin Hoodowi.",
            ),
            (
                "Early Suppression & Anti-Aggro Pings (Fire the Cannons! + Look What You've Done + Smash)",
                {
                    "Fire the Cannons!": 3,
                    "Look What You've Done": 3,
                    "Smash": 3,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Angel - Experiment 624": 2,
                    "The Queen - Devious Disguise": 2,
                    "Woolter & Jesse - Bellwether's Henchmen": 1,
                },
                "Absolutna dominacja tur 1-3. Niszczy tempo rywala na draw poprzez Fire the Cannons i tanie pingi.",
            ),
            (
                "Midrange Value & Heavy Shifting (The Queen 4x + Mushu 4x + Robin Hood 2x + Smash 3x)",
                {
                    "The Queen - Devious Disguise": 2,  # up to 4x total
                    "Mushu - Stealthy Dragon": 2,      # up to 4x total
                    "Robin Hood - Champion of Sherwood": 2,
                    "Smash": 3,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Angel - Experiment 624": 2,
                    "Woolter & Jesse - Bellwether's Henchmen": 1,
                    "Angela - Night Warrior": 1,
                    "Max Goof - Rebellious Teen": 1,
                },
                "Zwiększa obecność twardych ciał 4/5 i debuffów Queen (-4 siły) w turach 4-6, deklasując stół rywala.",
            ),
            (
                "Bounce & Tempo Disruption (Mother Knows Best + Smash + Flynn Rider)",
                {
                    "Mother Knows Best": 3,
                    "Smash": 3,
                    "Flynn Rider - Spectral Scoundrel": 4,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "The Queen - Devious Disguise": 2,
                    "Mushu - Stealthy Dragon": 2,
                },
                "Cofa wysokie Shifty i drogie postacie rywala (Elinor, The Queen) niszcząc ich tempo gry.",
            ),
            (
                "Card Cycling & Filtering (Tinker Bell Tiny Tactician + Strike a Good Match + Pegasus)",
                {
                    "Tinker Bell - Tiny Tactician": 3,
                    "Strike a Good Match": 3,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Flynn Rider - Spectral Scoundrel": 4,
                    "The Queen - Devious Disguise": 2,
                    "Angel - Experiment 624": 2,
                },
                "Filtruje rękę co turę, odrzucając zbędne atramenty i gwarantując znalezienie kluczowych odpowiedzi.",
            ),
            (
                "Heavy Ward & Anti-Evasive (Woolter & Jesse 4x + Angela 4x + Smash 3x)",
                {
                    "Woolter & Jesse - Bellwether's Henchmen": 2,  # up to 4x total
                    "Angela - Night Warrior": 1,                 # up to 4x total
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Smash": 3,
                    "The Queen - Devious Disguise": 2,
                    "Mushu - Stealthy Dragon": 2,
                    "Angel - Experiment 624": 2,
                    "Max Goof - Rebellious Teen": 1,
                    "Flynn Rider - Spectral Scoundrel": 1,
                },
                "Maksymalizuje Ward: Woolter i Angela są nietykalni dla Strength of a Raging Fire oraz Smash rywala.",
            ),
            (
                "Evasive Aggro & Direct Burn (Flynn Rider 4x + Fire the Cannons! 3x + Smash 3x)",
                {
                    "Flynn Rider - Spectral Scoundrel": 4,
                    "Fire the Cannons!": 3,
                    "Smash": 3,
                    "Pegasus - Gift for Hercules": 3,
                    "White Rabbit - Late Again": 3,
                    "Mushu - Stealthy Dragon": 2,
                    "Angel - Experiment 624": 2,
                },
                "Szybka agresja wczesnych tur podparta 6 tanimi usunięciami, które torują drogę dla questów.",
            ),
        ]

    @staticmethod
    def explain_mirror_role(card: CardTemplate) -> str:
        """Explains the tactical role of a card specifically in the Green/Steel mirror match."""
        name = card.full_name
        if "Smash" in name:
            return (
                "Zadaje bezwarunkowo 3 obrażenia za 3 ink. Likwiduje kluczowe zagrożenia rywala "
                "(Flash, Ursula, White Rabbit, Angel, Mushu) omijając walkę i nie tracąc własnego ciała."
            )
        if "Fire the Cannons!" in name:
            return (
                "Kluczowa odpowiedź za 1 ink zadająca 2 obrażenia. Na Draw karze rywala w 1. turze za Flynn Ridera (1/2) "
                "oraz Tod - All Alone (2/2), odwracając przewagę tempa."
            )
        if "Robin Hood - Champion of Sherwood" in name:
            return (
                "Shift 3 na ciele 4/4/2. W mirrorze banishuje stół rywala zyskując +2 Lore za wygnanie, "
                "a przy banishmentu sam dobiera kartę — generuje potężną przewagę surowcową."
            )
        if "Tinker Bell - Giant Fairy" in name:
            return (
                "Shift 4 / 6-drop (4/5). Na wejściu zadaje 1 obrażenie wszystkim postaciom rywala (zamiata Flynn/Cursed Merfolk), "
                "a w wyzwaniu dobija kolejną postać za 2 obrażenia — niszczy mirrorowe stoły."
            )
        if "Malicious, Mean, and Scary" in name:
            return (
                "Pieśń/Akcja za 3 ink zadająca 1 obrażenie wszystkim wrogim postaciom. Czyści stół z 1-HP rywali "
                "i osłabia całą armię rywala, neutralizując jego Strength of a Raging Fire."
            )
        if "Sudden Chill" in name:
            return (
                "Pieśń za 2 ink zmuszająca rywala do odrzucenia karty. W mirrorze gracze szybko wchodzą w topdeck — "
                "odrzucenie kluczowego Strength lub finishera wygrywa późną grę."
            )
        if "Strike a Good Match" in name:
            return (
                "Pieśń za 2 ink (dobierz 2, odrzuć 1). Przewija talię, odrzuca nadmiar uninkables i błyskawicznie "
                "dostarcza Strength of a Raging Fire do ręki."
            )
        if "Cursed Merfolk - Ursula's Handiwork" in name:
            return (
                "1-drop za 1 ink (0/1) questujący za 2 Lore. Jeśli rywal wyzwie go Todem lub Flynnem, "
                "musi odrzucić kartę — wywiera presję na zasoby przeciwnika."
            )
        if "Look What You've Done" in name:
            return "2-drop removal/damage. Pozwala tanio dobić wczesne postacie rywala zanim rozwiną szeroki stół."
        if "Mother Knows Best" in name:
            return "Cofa na rękę drogą postać rywala (Elinor, The Queen, Woolter), niszcząc zainwestowane tempo."
        if "Three Arrows" in name:
            return "Zadaje 3 obrażenia, a jeśli cel był ranny, dobiera kartę. Bezpieczne usuwanie średnich ciał rywala."
        if "Tod & Copper - Best of Friends" in name:
            return "Postać z Evasive omijająca naziemnych wojowników przeciwnika i bezpiecznie gromadząca punkty Lore."
        if "Inspector Tezuka - Resolute Officer" in name:
            return "Wchodzi wytapowany jako Bodyguard, absorbując ataki wrogich Todów i osłaniając Twoje White Rabbity."
        if "Robin Hood - Daydreamer" in name:
            return "Evasive 2/3 quester, który wygrywa wyścigi w powietrzu przeciwko wrogim Evasive."
        if "Bobby Zimuruski - Spray Cheese Kid" in name:
            return "1-drop z lootem (dobierz 1, odrzuć 1). Wygładza rękę początkową i odrzuca zbędne uninkables."
        if "Diablo - Protecting His Mistress" in name:
            return "Evasive 2/3. Jeśli rywal dobiera dodatkowe karty z Flasha, Diablo również dobiera kartę."
        if "White Rabbit - Late Again" in name:
            return (
                "Kluczowy 2-drop z Evasive i Underdog (kosztuje 1 ink na Draw). W mirrorze nieuchwytny dla naziemnych "
                "postaci rywala, bezpiecznie zbierający Lore."
            )
        if "Pegasus - Gift for Hercules" in name:
            return "1-drop z Evasive (1/2). Blokuje lub wyzywa wczesne Evasive rywala (White Rabbit) i bezpiecznie questuje."
        if "The Queen - Devious Disguise" in name:
            return "Potężne ciało 4/5 za 5 ink. Daje -4 siły wybranej postaci rywala, neutralizując jego największego questera."
        if "Mushu - Stealthy Dragon" in name:
            return "Evasive 3/3 za 4 ink z dobieraniem kart. Zapewnia dominację w powietrzu i paliwo na późne tury."
        if "Angel - Experiment 624" in name:
            return "Evasive 3/3 za 4 ink z zakłócaniem ręki. Kolejny latacz wygrywający wyścig punktowy."
        if "Woolter & Jesse - Bellwether's Henchmen" in name:
            return (
                "Postać z WARD (2/4 za 3 ink). W mirrorze jest całkowicie odporna na wrogie 'Strength of a Raging Fire' "
                "oraz 'Smash', gwarantując bezpieczne punkty Lore."
            )
        if "Max Goof - Rebellious Teen" in name:
            return "Singer 3 i postać 2/3. Pozwala śpiewać 'Strength of a Raging Fire' już w 3. turze za 0 ink."
        if "Angela - Night Warrior" in name:
            return "Postać z WARD (1/3 za 3 ink) questująca za 2 Lore. Wzmocnienie siły przy wejściu i odporność na removal."
        if "Flynn Rider - Spectral Scoundrel" in name:
            return "1-drop (1/2) z Boost 2 (staje się 3/2 z 2 Lore). Wywiera natychmiastową presję w mirrorze."

        return f"Wspiera krzywą kosztu ({card.cost} ink) w mirrorze Green/Steel."
