"""
Module 1: Tech-Card Scanner & Locked-Core Flex-Slot Optimizer (UCB1 Bandit + Duels.ink Meta Gauntlet).
Supports:
1. `scan_for_matchup`: Finds the highest-impact 1-4 card swaps for a specific matchup (respecting locked core cards).
2. `optimize_core_for_duels_ink_meta`: Takes a locked core (e.g., 44 cards = 11x4 of Amber/Emerald Circle of Life)
   and evaluates candidate tech cards and 16-card flex packages against the live Duels.ink Core Set 14 metagame.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import random

from lorcana_ai.config import DeckbuilderConfig
from lorcana_ai.data.card_db import CardDatabase, CardTemplate, GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.effect_dsl import EffectOpType, KeywordFlag
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator


@dataclass
class TechCardRecommendation:
    card_in_name: str
    card_out_name: str
    copies_swapped: int
    baseline_bo3_winrate: float
    new_bo3_winrate: float
    delta_bo3_winrate: float
    play_winrate: float
    draw_winrate: float
    games_simulated: int
    uninkable_count: int
    tactical_rationale: str


@dataclass
class MetaTechCardEvaluation:
    """Performance and per-matchup percentage winrate impact of a candidate tech card against the Duels.ink metagame."""
    card_name: str
    card_id: int
    cost: int
    inkable: bool
    card_type: str
    recommended_copies: int
    expected_bo1_winrate: float
    expected_bo3_winrate: float
    delta_bo1_winrate: float
    delta_bo3_winrate: float
    play_winrate: float
    draw_winrate: float
    matchup_winrates: dict[str, float] = field(default_factory=dict)
    matchup_deltas: dict[str, float] = field(default_factory=dict)
    best_against: str = ""
    tactical_role: str = ""


@dataclass
class CoreOptimizationReport:
    """Result of optimizing the flex slots (e.g. 16 cards) around a locked core (e.g. 44 cards)."""
    archetype_name: str
    locked_core_count: int
    flex_slots_count: int
    baseline_meta_bo1_wr: float
    baseline_meta_bo3_wr: float
    optimized_meta_bo1_wr: float
    optimized_meta_bo3_wr: float
    optimized_play_wr: float
    optimized_draw_wr: float
    top_tech_cards: list[MetaTechCardEvaluation]
    selected_flex_package: dict[str, int]
    optimized_deck: Deck
    matchup_breakdown: dict[str, dict[str, float]]


class TechCardScanner:
    """
    Uses Upper Confidence Bound (UCB1) Multi-Armed Bandit search and Meta Gauntlet evaluation
    to find the strongest tech cards and flex-slot packages.
    """

    def __init__(
        self,
        db: CardDatabase | None = None,
        cfg: DeckbuilderConfig | None = None,
        evaluator: MatchupEvaluator | None = None,
        workers: int | None = None,
    ) -> None:
        self.db = db or GLOBAL_CARD_DB
        self.cfg = cfg or DeckbuilderConfig()
        self.evaluator = evaluator or MatchupEvaluator(workers=workers)

    def scan_for_matchup(
        self,
        base_deck: Deck,
        opponent_deck: Deck,
        copies_to_swap: int = 3,
        total_simulation_budget: int = 240,
        top_k: int = 5,
        locked_core_ids: set[int] | None = None,
    ) -> list[TechCardRecommendation]:
        locked = locked_core_ids or set()
        # 1. Measure baseline performance
        base_stats = self.evaluator.evaluate_matchup(base_deck, opponent_deck, num_games=24, seed=101)
        baseline_wr = base_stats.bo3_winrate

        # 2. Identify candidate flex slots to cut in base_deck (never cutting locked_core_ids!)
        cut_candidates = [
            cid for cid, cnt in base_deck.card_counts.items()
            if cnt >= copies_to_swap and cid not in locked
        ]
        if not cut_candidates:
            cut_candidates = [
                cid for cid in sorted(base_deck.card_counts.keys(), key=lambda c: base_deck.card_counts[c], reverse=True)
                if cid not in locked
            ][:3]
        if not cut_candidates:
            cut_candidates = sorted(base_deck.card_counts.keys(), key=lambda c: base_deck.card_counts[c], reverse=True)[:2]

        cut_candidates.sort(
            key=lambda cid: (
                not self.db.by_id[cid].inkable,
                abs(self.db.by_id[cid].cost - 4),
            ),
            reverse=True,
        )
        flex_cut_ids = cut_candidates[:2]

        # 3. Gather candidate tech cards in base_deck's 2 inks not already at 4 copies
        legal_pool = self.db.get_by_inks(base_deck.inks)
        arms: list[tuple[int, int, Deck]] = []

        for cut_id in flex_cut_ids:
            for cand in legal_pool:
                if cand.card_id == cut_id or cand.card_id in locked:
                    continue
                current_qty = base_deck.card_counts.get(cand.card_id, 0)
                if current_qty + copies_to_swap > 4:
                    continue
                mod_deck = base_deck.clone(new_name=f"{base_deck.name} [+{copies_to_swap} {cand.full_name}]")
                rem = mod_deck.card_counts[cut_id] - copies_to_swap
                if rem <= 0:
                    del mod_deck.card_counts[cut_id]
                else:
                    mod_deck.card_counts[cut_id] = rem
                mod_deck.card_counts[cand.card_id] = current_qty + copies_to_swap

                if mod_deck.uninkable_count <= self.cfg.max_uninkable_count:
                    arms.append((cut_id, cand.card_id, mod_deck))

        if not arms:
            return []

        opp_avg_cost = opponent_deck.average_cost
        arms.sort(
            key=lambda item: self._StaticTacticalPrior(self.db.by_id[item[1]], opp_avg_cost),
            reverse=True,
        )
        arms = arms[:14]

        # 4. UCB1 Bandit Allocation Loop
        n_arms = len(arms)
        arm_games = [0] * n_arms
        arm_wins_play = [0.0] * n_arms
        arm_games_play = [0] * n_arms
        arm_wins_draw = [0.0] * n_arms
        arm_games_draw = [0] * n_arms

        batch_games = 6
        games_used = 0
        step_idx = 0

        while games_used < total_simulation_budget:
            step_idx += 1
            chosen_idx = 0
            best_ucb = -1e9
            for idx in range(n_arms):
                if arm_games[idx] == 0:
                    chosen_idx = idx
                    break
                p_pl = arm_wins_play[idx] / max(1, arm_games_play[idx])
                p_dr = arm_wins_draw[idx] / max(1, arm_games_draw[idx])
                bo3_est = MatchupEvaluator.compute_bo3_match_winrate(p_pl, p_dr)
                explore = self.cfg.ucb_exploration_c * math.sqrt(math.log(max(1, games_used)) / arm_games[idx])
                ucb_val = bo3_est + explore
                if ucb_val > best_ucb:
                    best_ucb = ucb_val
                    chosen_idx = idx

            _, _, mod_deck = arms[chosen_idx]
            stats = self.evaluator.evaluate_matchup(
                mod_deck,
                opponent_deck,
                num_games=batch_games,
                seed=5000 + step_idx * 313 + chosen_idx * 17,
            )
            half = batch_games // 2
            arm_games[chosen_idx] += batch_games
            arm_games_play[chosen_idx] += half
            arm_games_draw[chosen_idx] += (batch_games - half)
            arm_wins_play[chosen_idx] += stats.play_winrate * half
            arm_wins_draw[chosen_idx] += stats.draw_winrate * (batch_games - half)
            games_used += batch_games

        # 5. Build ranked recommendations
        recs: list[TechCardRecommendation] = []
        for idx, (cut_id, in_id, mod_deck) in enumerate(arms):
            if arm_games[idx] == 0:
                continue
            p_pl = arm_wins_play[idx] / max(1, arm_games_play[idx])
            p_dr = arm_wins_draw[idx] / max(1, arm_games_draw[idx])
            bo3_wr = MatchupEvaluator.compute_bo3_match_winrate(p_pl, p_dr)
            c_in = self.db.by_id[in_id]
            c_out = self.db.by_id[cut_id]

            recs.append(
                TechCardRecommendation(
                    card_in_name=c_in.full_name,
                    card_out_name=c_out.full_name,
                    copies_swapped=copies_to_swap,
                    baseline_bo3_winrate=round(baseline_wr, 4),
                    new_bo3_winrate=round(bo3_wr, 4),
                    delta_bo3_winrate=round(bo3_wr - baseline_wr, 4),
                    play_winrate=round(p_pl, 4),
                    draw_winrate=round(p_dr, 4),
                    games_simulated=arm_games[idx],
                    uninkable_count=mod_deck.uninkable_count,
                    tactical_rationale=self._explain_tech_card(c_in, opponent_deck),
                )
            )

        recs.sort(key=lambda r: (r.new_bo3_winrate, r.games_simulated), reverse=True)
        return recs[:top_k]

    def optimize_core_for_duels_ink_meta(
        self,
        core_card_counts: dict[int, int],
        meta_decks: dict[str, Deck],
        meta_shares: dict[str, float],
        archetype_name: str = "Amber/Emerald Circle of Life",
        games_per_matchup: int = 12,
        package_candidates: int = 10,
        seed: int = 2026,
        baseline_deck: Deck | None = None,
    ) -> CoreOptimizationReport:
        """
        Given a locked core (e.g. 44 cards = 11x4 for Amber/Emerald Princess Aggro or Circle of Life),
        evaluates all high-synergy tech candidates (computing exact per-matchup percentage winrate impact
        delta_WR against the baseline deck) and searches for the optimal 16-card flex package.
        """
        rng = random.Random(seed)
        locked_total = sum(core_card_counts.values())
        flex_needed = max(0, 60 - locked_total)
        is_princess_aggro = "princess" in archetype_name.lower()

        inks_set: set[str] = set()
        core_base_names: set[str] = set()
        for cid, cnt in core_card_counts.items():
            if cnt > 0 and cid in self.db.by_id:
                c = self.db.by_id[cid]
                inks_set.update(c.inks)
                core_base_names.add(c.base_name)
        inks = tuple(sorted(inks_set)) or ("Amber", "Emerald")

        # 1. Identify high-value candidate tech cards for the flex slots
        legal_pool = [
            c for c in self.db.get_by_inks(inks)
            if core_card_counts.get(c.card_id, 0) < 4
        ]
        if is_princess_aggro:
            legal_pool.sort(
                key=lambda c: self._score_princess_aggro_synergy(c, core_base_names),
                reverse=True,
            )
            shortlist = legal_pool[:14]
        else:
            legal_pool.sort(
                key=lambda c: self._score_circle_of_life_synergy(c, core_base_names),
                reverse=True,
            )
            shortlist = legal_pool[:16]

        # 2. Build or use the provided 60-card baseline deck and evaluate per-matchup baseline winrates
        if baseline_deck is None:
            baseline_deck = self._build_deck_with_flex(
                core_card_counts,
                shortlist[6:11],
                flex_needed,
                name=f"{archetype_name} (Un-Tuned Baseline)",
                inks_tag="/".join(inks),
            )
        base_bo1, base_bo3, _, _, base_m_wrs, base_breakdown = self._evaluate_vs_meta_gauntlet(
            baseline_deck, meta_decks, meta_shares, games_per_matchup=games_per_matchup, seed=seed
        )

        # 3. Evaluate each shortlisted Tech Card individually across the Duels.ink meta gauntlet
        #    using Common Random Numbers (seed=seed) so matchup_deltas reflect pure card swap impact
        tech_evals: list[MetaTechCardEvaluation] = []
        if is_princess_aggro:
            balanced_shell_names = [
                "Mickey Mouse - Expedition Leader",
                "Lionheart - Incumbent Mayor",
                "Finnick - Tiny Terror",
                "Pocahontas - Finding the Way",
                "The Prince - Searching for His Love",
                "David - Protective Snowboarder",
                "Nani - Stage Manager",
                "Rapunzel - Tower Defender",
            ]
        else:
            balanced_shell_names = [
                "Max Goof - Chart Topper",
                "Ursula - Deceiver",
                "White Rabbit - Late Again",
                "Chernabog - Unnatural Force",
                "Lenny - Toy Binoculars",
                "To Wither a Flower",
                "Stitch - Carefree Surfer",
                "Aurora - Delightful Musician",
            ]
        balanced_fillers = [
            self.db.get_by_name(n) for n in balanced_shell_names if self.db.get_by_name(n) is not None
        ]

        best_single_deck = baseline_deck
        best_single_bo1 = base_bo1
        best_single_bo3 = base_bo3
        best_single_play = sum(b["play_winrate"] * b["meta_share"] for b in base_breakdown.values()) / max(1e-6, sum(b["meta_share"] for b in base_breakdown.values()))
        best_single_draw = sum(b["draw_winrate"] * b["meta_share"] for b in base_breakdown.values()) / max(1e-6, sum(b["meta_share"] for b in base_breakdown.values()))
        best_single_breakdown: dict[str, dict[str, float]] = dict(base_breakdown)

        eval_count = 14 if is_princess_aggro else 12
        for idx, cand in enumerate(shortlist[:eval_count]):
            rec_copies = self._recommended_copies_for_tech(cand, is_princess_aggro=is_princess_aggro)
            test_deck = self._build_single_tech_test_deck(
                core_card_counts=core_card_counts,
                tech_card=cand,
                tech_copies=rec_copies,
                balanced_fillers=balanced_fillers,
                flex_needed=flex_needed,
                name=f"{archetype_name} [Tech: {cand.full_name}]",
                inks_tag="/".join(inks),
                baseline_deck=baseline_deck,
            )
            # If the card was already at full copies in baseline_deck, compare against a deck where it is cut
            if test_deck.card_counts == baseline_deck.card_counts:
                without_deck = self._build_deck_without_tech(
                    core_card_counts=core_card_counts,
                    baseline_deck=baseline_deck,
                    tech_card=cand,
                    balanced_fillers=balanced_fillers,
                    inks_tag="/".join(inks),
                )
                wo_bo1, wo_bo3, _, _, wo_m_wrs, _ = self._evaluate_vs_meta_gauntlet(
                    without_deck,
                    meta_decks,
                    meta_shares,
                    games_per_matchup=games_per_matchup,
                    seed=seed,
                )
                bo1_wr, bo3_wr = base_bo1, base_bo3
                pl_wr = sum(b["play_winrate"] * b["meta_share"] for b in base_breakdown.values()) / max(1e-6, sum(b["meta_share"] for b in base_breakdown.values()))
                dr_wr = sum(b["draw_winrate"] * b["meta_share"] for b in base_breakdown.values()) / max(1e-6, sum(b["meta_share"] for b in base_breakdown.values()))
                m_wrs = dict(base_m_wrs)
                m_deltas = {k: round(m_wrs[k] - wo_m_wrs.get(k, m_wrs[k]), 4) for k in m_wrs}
                d_bo1 = round(bo1_wr - wo_bo1, 4)
                d_bo3 = round(bo3_wr - wo_bo3, 4)
            else:
                bo1_wr, bo3_wr, pl_wr, dr_wr, m_wrs, test_breakdown = self._evaluate_vs_meta_gauntlet(
                    test_deck,
                    meta_decks,
                    meta_shares,
                    games_per_matchup=games_per_matchup,
                    seed=seed,
                )
                m_deltas = {k: round(m_wrs[k] - base_m_wrs.get(k, m_wrs[k]), 4) for k in m_wrs}
                d_bo1 = round(bo1_wr - base_bo1, 4)
                d_bo3 = round(bo3_wr - base_bo3, 4)
                if (bo1_wr + bo3_wr) > (best_single_bo1 + best_single_bo3):
                    best_single_deck = test_deck
                    best_single_bo1 = bo1_wr
                    best_single_bo3 = bo3_wr
                    best_single_play = pl_wr
                    best_single_draw = dr_wr
                    best_single_breakdown = test_breakdown

            if m_deltas:
                best_opp_key, best_delta = max(m_deltas.items(), key=lambda kv: (kv[1], m_wrs.get(kv[0], 0.0)))
                sign_str = "+" if best_delta >= 0 else ""
                best_opp = f"{best_opp_key} ({sign_str}{best_delta*100:.1f}%)"
            else:
                best_opp = "Meta"

            role_str = (
                self._explain_princess_aggro_role(cand)
                if is_princess_aggro
                else self._explain_circle_of_life_role(cand)
            )
            s_d1 = "+" if d_bo1 >= 0 else ""
            s_d3 = "+" if d_bo3 >= 0 else ""
            print(
                f"  [Tech {idx+1:2d}/{eval_count}] {cand.full_name:<38} -> "
                f"BO1: {bo1_wr*100:5.1f}% ({s_d1}{d_bo1*100:5.1f}%) | BO3: {bo3_wr*100:5.1f}% ({s_d3}{d_bo3*100:5.1f}%) | Best: {best_opp}",
                flush=True,
            )
            tech_evals.append(
                MetaTechCardEvaluation(
                    card_name=cand.full_name,
                    card_id=cand.card_id,
                    cost=cand.cost,
                    inkable=cand.inkable,
                    card_type=cand.card_type,
                    recommended_copies=rec_copies,
                    expected_bo1_winrate=round(bo1_wr, 4),
                    expected_bo3_winrate=round(bo3_wr, 4),
                    delta_bo1_winrate=d_bo1,
                    delta_bo3_winrate=d_bo3,
                    play_winrate=round(pl_wr, 4),
                    draw_winrate=round(dr_wr, 4),
                    matchup_winrates={k: round(v, 4) for k, v in m_wrs.items()},
                    matchup_deltas=m_deltas,
                    best_against=best_opp,
                    tactical_role=role_str,
                )
            )

        tech_evals.sort(
            key=lambda x: (x.expected_bo1_winrate + x.expected_bo3_winrate, x.delta_bo1_winrate),
            reverse=True,
        )

        # 4. Search over candidate 16-card Flex Packages combining the top-performing tech cards
        best_deck = best_single_deck
        best_bo1 = best_single_bo1
        best_bo3 = best_single_bo3
        best_play = best_single_play
        best_draw = best_single_draw
        best_breakdown: dict[str, dict[str, float]] = dict(best_single_breakdown)

        top_pool = [self.db.by_id[te.card_id] for te in tech_evals[:10]]

        for pkg_idx in range(package_candidates):
            cand_deck = self._sample_synergistic_flex_package(
                core_card_counts=core_card_counts,
                top_pool=top_pool,
                flex_needed=flex_needed,
                pkg_idx=pkg_idx,
                rng=rng,
                archetype_name=archetype_name,
                inks_tag="/".join(inks),
                allowed_inks=inks_set,
                is_princess_aggro=is_princess_aggro,
            )
            bo1_wr, bo3_wr, pl_wr, dr_wr, _, pkg_breakdown = self._evaluate_vs_meta_gauntlet(
                cand_deck,
                meta_decks,
                meta_shares,
                games_per_matchup=games_per_matchup,
                seed=seed,
            )
            is_new_best = (bo1_wr + bo3_wr) > (best_bo1 + best_bo3)
            if is_new_best:
                best_deck = cand_deck
                best_bo1 = bo1_wr
                best_bo3 = bo3_wr
                best_play = pl_wr
                best_draw = dr_wr
                best_breakdown = pkg_breakdown
            pkg_d1 = bo1_wr - base_bo1
            pkg_d3 = bo3_wr - base_bo3
            ps1 = "+" if pkg_d1 >= 0 else ""
            ps3 = "+" if pkg_d3 >= 0 else ""
            star = " ★ NEW BEST!" if is_new_best else ""
            print(
                f"  [Pkg  {pkg_idx+1:2d}/{package_candidates:2d}] BO1: {bo1_wr*100:5.1f}% ({ps1}{pkg_d1*100:5.1f}%) | "
                f"BO3: {bo3_wr*100:5.1f}% ({ps3}{pkg_d3*100:5.1f}%){star}",
                flush=True,
            )

        # Enrich best_breakdown with baseline winrates and exact percentage deltas per matchup
        for opp_name, m_info in best_breakdown.items():
            b_info = base_breakdown.get(opp_name, {})
            b_bo1 = b_info.get("bo1_winrate", m_info["bo1_winrate"])
            b_bo3 = b_info.get("bo3_winrate", m_info["bo3_winrate"])
            m_info["baseline_bo1_winrate"] = round(b_bo1, 4)
            m_info["delta_bo1_winrate"] = round(m_info["bo1_winrate"] - b_bo1, 4)
            m_info["baseline_bo3_winrate"] = round(b_bo3, 4)
            m_info["delta_bo3_winrate"] = round(m_info["bo3_winrate"] - b_bo3, 4)

        selected_flex: dict[str, int] = {}
        for cid, cnt in best_deck.card_counts.items():
            base_cnt = core_card_counts.get(cid, 0)
            if cnt > base_cnt:
                selected_flex[self.db.by_id[cid].full_name] = cnt - base_cnt

        return CoreOptimizationReport(
            archetype_name=archetype_name,
            locked_core_count=locked_total,
            flex_slots_count=flex_needed,
            baseline_meta_bo1_wr=round(base_bo1, 4),
            baseline_meta_bo3_wr=round(base_bo3, 4),
            optimized_meta_bo1_wr=round(best_bo1, 4),
            optimized_meta_bo3_wr=round(best_bo3, 4),
            optimized_play_wr=round(best_play, 4),
            optimized_draw_wr=round(best_draw, 4),
            top_tech_cards=tech_evals,
            selected_flex_package=selected_flex,
            optimized_deck=best_deck,
            matchup_breakdown=best_breakdown,
        )

    def _evaluate_vs_meta_gauntlet(
        self,
        deck: Deck,
        meta_decks: dict[str, Deck],
        meta_shares: dict[str, float],
        games_per_matchup: int,
        seed: int,
    ) -> tuple[float, float, float, float, dict[str, float], dict[str, dict[str, float]]]:
        total_w = 0.0
        weighted_bo1 = 0.0
        weighted_bo3 = 0.0
        weighted_play = 0.0
        weighted_draw = 0.0
        per_opp: dict[str, float] = {}
        breakdown: dict[str, dict[str, float]] = {}

        batch_stats = self.evaluator.evaluate_gauntlet_batch(
            deck, meta_decks, num_games=games_per_matchup, seed=seed
        )
        for opp_name, st in batch_stats.items():
            w = meta_shares.get(opp_name, 1.0 / max(1, len(meta_decks)))
            bo1_wr = 0.5 * st.play_winrate + 0.5 * st.draw_winrate
            weighted_bo1 += w * bo1_wr
            weighted_bo3 += w * st.bo3_winrate
            weighted_play += w * st.play_winrate
            weighted_draw += w * st.draw_winrate
            total_w += w
            per_opp[opp_name] = bo1_wr
            breakdown[opp_name] = {
                "meta_share": round(w, 4),
                "bo1_winrate": round(bo1_wr, 4),
                "bo3_winrate": round(st.bo3_winrate, 4),
                "play_winrate": round(st.play_winrate, 4),
                "draw_winrate": round(st.draw_winrate, 4),
            }

        norm = max(1e-6, total_w)
        return (
            weighted_bo1 / norm,
            weighted_bo3 / norm,
            weighted_play / norm,
            weighted_draw / norm,
            per_opp,
            breakdown,
        )

    def _build_single_tech_test_deck(
        self,
        core_card_counts: dict[int, int],
        tech_card: CardTemplate,
        tech_copies: int,
        balanced_fillers: list[CardTemplate],
        flex_needed: int,
        name: str,
        inks_tag: str,
        baseline_deck: Deck | None = None,
    ) -> Deck:
        if baseline_deck is not None and baseline_deck.total_cards == 60:
            # Swap tech_card directly into baseline_deck's non-core flex slots so delta_WR is measured against baseline_deck
            counts = dict(baseline_deck.card_counts)
            cur = counts.get(tech_card.card_id, 0)
            want_total = min(4, max(cur, tech_copies))
            need_to_add = want_total - cur
            if need_to_add > 0:
                flex_cids = [
                    cid for cid, q in counts.items()
                    if cid != tech_card.card_id and q > core_card_counts.get(cid, 0)
                ]
                # Prefer cutting uninkable flex if tech_card is uninkable, or higher-cost flex slots
                flex_cids.sort(
                    key=lambda cid: (
                        (not self.db.by_id[cid].inkable) == (not tech_card.inkable),
                        self.db.by_id[cid].cost,
                        counts[cid] - core_card_counts.get(cid, 0),
                    ),
                    reverse=True,
                )
                removed = 0
                for f_cid in flex_cids:
                    if removed >= need_to_add:
                        break
                    removable = counts[f_cid] - core_card_counts.get(f_cid, 0)
                    cut = min(removable, need_to_add - removed)
                    if cut > 0:
                        counts[f_cid] -= cut
                        if counts[f_cid] <= 0:
                            del counts[f_cid]
                        removed += cut
                if removed > 0:
                    counts[tech_card.card_id] = cur + removed
            return Deck(name=name, card_counts=counts, archetype_tag=inks_tag, db=self.db)

        counts = dict(core_card_counts)
        rem = flex_needed
        cur = counts.get(tech_card.card_id, 0)
        add_tech = min(4 - cur, tech_copies, rem)
        if add_tech > 0:
            counts[tech_card.card_id] = cur + add_tech
            rem -= add_tech

        for f in balanced_fillers:
            if rem <= 0:
                break
            if f.card_id == tech_card.card_id:
                continue
            cur_f = counts.get(f.card_id, 0)
            cur_unink = sum(
                q for cid, q in counts.items() if cid in self.db.by_id and not self.db.by_id[cid].inkable
            )
            want = 2
            add = min(4 - cur_f, want, rem)
            if not f.inkable:
                add = min(add, max(0, 9 - cur_unink))
            if add > 0:
                counts[f.card_id] = cur_f + add
                rem -= add

        return Deck(name=name, card_counts=counts, archetype_tag=inks_tag, db=self.db)

    def _build_deck_without_tech(
        self,
        core_card_counts: dict[int, int],
        baseline_deck: Deck,
        tech_card: CardTemplate,
        balanced_fillers: list[CardTemplate],
        inks_tag: str,
    ) -> Deck:
        counts = dict(baseline_deck.card_counts)
        base_core_qty = core_card_counts.get(tech_card.card_id, 0)
        cur_qty = counts.get(tech_card.card_id, 0)
        rem = max(0, cur_qty - base_core_qty)
        if base_core_qty <= 0:
            counts.pop(tech_card.card_id, None)
        else:
            counts[tech_card.card_id] = base_core_qty

        for f in balanced_fillers:
            if rem <= 0:
                break
            if f.card_id == tech_card.card_id:
                continue
            cur_f = counts.get(f.card_id, 0)
            add = min(4 - cur_f, rem)
            if add > 0:
                counts[f.card_id] = cur_f + add
                rem -= add
        return Deck(
            name=f"{baseline_deck.name} [-{tech_card.full_name}]",
            card_counts=counts,
            archetype_tag=inks_tag,
            db=self.db,
        )

    def _build_deck_with_flex(
        self,
        core_card_counts: dict[int, int],
        flex_cards: list[CardTemplate],
        flex_needed: int,
        name: str,
        inks_tag: str,
    ) -> Deck:
        counts = dict(core_card_counts)
        rem = flex_needed
        for c in flex_cards:
            if rem <= 0:
                break
            cur = counts.get(c.card_id, 0)
            add = min(4 - cur, rem)
            if add > 0:
                counts[c.card_id] = cur + add
                rem -= add
        return Deck(name=name, card_counts=counts, archetype_tag=inks_tag, db=self.db)

    def _sample_synergistic_flex_package(
        self,
        core_card_counts: dict[int, int],
        top_pool: list[CardTemplate],
        flex_needed: int,
        pkg_idx: int,
        rng: random.Random,
        archetype_name: str,
        inks_tag: str,
        allowed_inks: set[str] | None = None,
        is_princess_aggro: bool = False,
    ) -> Deck:
        counts = dict(core_card_counts)
        rem = flex_needed
        valid_inks = allowed_inks or {"Amber", "Emerald"}

        if is_princess_aggro:
            if pkg_idx == 0:
                # 1. Duels.ink Pro + Top Discovered Techs (Pluto + Nani + David Bodyguard + Percy 3-Lore + Pocahontas Finding the Way)
                preferred = [
                    ("Pluto - Friendly Pooch", 3),
                    ("Nani - Stage Manager", 3),
                    ("David - Protective Snowboarder", 3),
                    ("Percy - Pupsicle", 3),
                    ("Pocahontas - Finding the Way", 2),
                    ("Lionheart - Incumbent Mayor", 2),
                ]
            elif pkg_idx == 1:
                # 2. Percy + Pluto + Finding the Way + Finnick + Expedition Leader + Lionheart (cuts 2x The Prince for 2x Pluto)
                preferred = [
                    ("Percy - Pupsicle", 3),
                    ("Pocahontas - Finding the Way", 2),
                    ("Finnick - Tiny Terror", 4),
                    ("Mickey Mouse - Expedition Leader", 4),
                    ("Pluto - Friendly Pooch", 2),
                    ("Lionheart - Incumbent Mayor", 1),
                ]
            elif pkg_idx == 2:
                # 3. Percy + Monterey Jack + Finding the Way + Finnick + Expedition Leader + Lionheart (cuts 2x The Prince for 2x Monterey Jack)
                preferred = [
                    ("Percy - Pupsicle", 3),
                    ("Monterey Jack - Hypnotized by Cheese", 2),
                    ("Pocahontas - Finding the Way", 2),
                    ("Finnick - Tiny Terror", 4),
                    ("Mickey Mouse - Expedition Leader", 4),
                    ("Lionheart - Incumbent Mayor", 1),
                ]
            elif pkg_idx == 3:
                # 4. Percy + Pluto + Nani + Finding the Way + Finnick + Expedition Leader + Lionheart
                preferred = [
                    ("Percy - Pupsicle", 3),
                    ("Pocahontas - Finding the Way", 3),
                    ("Finnick - Tiny Terror", 3),
                    ("Mickey Mouse - Expedition Leader", 2),
                    ("Pluto - Friendly Pooch", 2),
                    ("Nani - Stage Manager", 2),
                    ("Lionheart - Incumbent Mayor", 1),
                ]
            elif pkg_idx == 4:
                # 5. Upgraded User Baseline (Percy 3-Lore + Pocahontas Finding the Way + Finnick + Lionheart + Pluto + Nani)
                preferred = [
                    ("Percy - Pupsicle", 3),
                    ("Pocahontas - Finding the Way", 3),
                    ("Finnick - Tiny Terror", 3),
                    ("Lionheart - Incumbent Mayor", 3),
                    ("Pluto - Friendly Pooch", 2),
                    ("Nani - Stage Manager", 2),
                ]
            elif pkg_idx == 5:
                # 6. Refined Baseline with Percy + Pluto + Nani (cutting The Prince and excess 6-drop Lionheart)
                preferred = [
                    ("Percy - Pupsicle", 3),
                    ("Pocahontas - Finding the Way", 2),
                    ("Finnick - Tiny Terror", 4),
                    ("Mickey Mouse - Expedition Leader", 3),
                    ("Pluto - Friendly Pooch", 2),
                    ("Nani - Stage Manager", 1),
                    ("Lionheart - Incumbent Mayor", 1),
                ]
            else:
                preferred = []
                shuffled = list(top_pool)
                rng.shuffle(shuffled)
                for c in shuffled:
                    qty = rng.choice([2, 3, 4]) if c.inkable else rng.choice([2, 3])
                    preferred.append((c.full_name, qty))
        else:
            # Curated competitive flex distributions for Circle of Life (pkg_idx == 0..4), then stochastic sampling
            if pkg_idx == 0:
                preferred = [
                    ("Max Goof - Chart Topper", 3),
                    ("To Wither a Flower", 3),
                    ("Chernabog - Unnatural Force", 3),
                    ("Max Goof - Rebellious Teen", 2),
                    ("Lenny - Toy Binoculars", 2),
                    ("Don Karnage - Debonair Pirate", 2),
                    ("Don Karnage - Khan's Courier", 1),
                ]
            elif pkg_idx == 1:
                preferred = [
                    ("Max Goof - Chart Topper", 3),
                    ("To Wither a Flower", 3),
                    ("White Rabbit - Late Again", 4),
                    ("Don Karnage - Debonair Pirate", 2),
                    ("Don Karnage - Khan's Courier", 2),
                    ("Miguel Rivera - Street Musician", 2),
                ]
            elif pkg_idx == 2:
                preferred = [
                    ("Max Goof - Chart Topper", 3),
                    ("Lenny - Toy Binoculars", 3),
                    ("Ursula - Deceiver", 3),
                    ("Chernabog - Unnatural Force", 3),
                    ("Stitch - Carefree Surfer", 2),
                    ("White Rabbit - Late Again", 2),
                ]
            elif pkg_idx == 3:
                preferred = [
                    ("Max Goof - Chart Topper", 3),
                    ("Ursula - Deceiver", 3),
                    ("Lenny - Toy Binoculars", 2),
                    ("To Wither a Flower", 2),
                    ("Chernabog - Unnatural Force", 2),
                    ("White Rabbit - Late Again", 2),
                    ("Stitch - Carefree Surfer", 2),
                ]
            elif pkg_idx == 4:
                preferred = [
                    ("Max Goof - Chart Topper", 3),
                    ("Aurora - Delightful Musician", 3),
                    ("White Rabbit - Late Again", 3),
                    ("To Wither a Flower", 3),
                    ("Chernabog - Unnatural Force", 2),
                    ("Lenny - Toy Binoculars", 2),
                ]
            else:
                preferred = []
                shuffled = list(top_pool)
                rng.shuffle(shuffled)
                for c in shuffled:
                    qty = rng.choice([2, 3, 4]) if c.inkable else rng.choice([2, 3])
                    preferred.append((c.full_name, qty))

        max_unink_cap = 16 if is_princess_aggro else 11
        for cname, want_qty in preferred:
            if rem <= 0:
                break
            c = self.db.get_by_name(cname)
            if c is None or not set(c.inks).issubset(valid_inks):
                continue
            cur = counts.get(c.card_id, 0)
            cur_unink = sum(
                q for cid, q in counts.items() if cid in self.db.by_id and not self.db.by_id[cid].inkable
            )
            max_add = min(4 - cur, want_qty, rem)
            if not c.inkable:
                max_add = min(max_add, max(0, max_unink_cap - cur_unink))
            if max_add > 0:
                counts[c.card_id] = cur + max_add
                rem -= max_add

        for c in top_pool:
            if rem <= 0:
                break
            if not c.inkable or not set(c.inks).issubset(valid_inks):
                continue
            cur = counts.get(c.card_id, 0)
            add = min(4 - cur, rem)
            if add > 0:
                counts[c.card_id] = cur + add
                rem -= add

        return Deck(
            name=f"{archetype_name} [Optimized v{pkg_idx + 1}]",
            card_counts=counts,
            archetype_tag=inks_tag,
            db=self.db,
        )

    @staticmethod
    def _recommended_copies_for_tech(card: CardTemplate, is_princess_aggro: bool = False) -> int:
        if is_princess_aggro:
            if "Pluto - Friendly Pooch" in card.full_name:
                return 3
            if "Nani - Stage Manager" in card.full_name or "David - Protective Snowboarder" in card.full_name:
                return 3
            if not card.inkable:
                return 2
            if card.cost >= 4:
                return 2
            return 3
        if not card.inkable:
            return 2 if card.cost >= 5 else 3
        if (card.abilities.shift_cost is not None and card.cost >= 6) or card.cost >= 7:
            return 3
        return 4

    @staticmethod
    def _score_princess_aggro_synergy(card: CardTemplate, core_base_names: set[str]) -> float:
        name = card.full_name
        curated_scores = {
            "Pluto - Friendly Pooch": 35.0,
            "Nani - Stage Manager": 34.5,
            "David - Protective Snowboarder": 34.0,
            "Rapunzel - Tower Defender": 31.5,
            "Monterey Jack - Hypnotized by Cheese": 31.0,
            "Percy - Pupsicle": 30.5,
            "Mickey Mouse - Expedition Leader": 30.0,
            "Lionheart - Incumbent Mayor": 29.5,
            "Finnick - Tiny Terror": 29.0,
            "Pocahontas - Finding the Way": 28.5,
            "The Prince - Searching for His Love": 28.0,
            "Lenny - Toy Binoculars": 27.5,
            "Cursed Merfolk - Ursula's Handiwork": 27.0,
            "Mr. Manchas - Service with a Smile": 26.5,
        }
        if name in curated_scores:
            return curated_scores[name]

        score = 0.0
        if card.card_type == "Character" and card.cost <= 3:
            score += 8.0 + card.lore * 3.0
            if card.willpower >= 4:
                score += 4.0  # Synergizes with Dale - Ready for His Shot & Monterey Jack!
            if "Princess" in card.subtypes:
                score += 4.0  # Discounted by Aurora - Holding Court!
            if bool(card.abilities.keywords & KeywordFlag.BODYGUARD):
                score += 5.0  # Protects wide board & enters exerted for Elinor!
        return score

    @staticmethod
    def _explain_princess_aggro_role(card: CardTemplate) -> str:
        name = card.full_name
        if "Pluto - Friendly Pooch" in name:
            return (
                "Kluczowy 1-drop (0/2/1): może wejść za darmo z 'Pocahontas - Guiding the Tribe' w 2. turze, "
                "a jego umiejętność '⟳ — płacisz 1 mniej za następną postać' przyspiesza wyrzucanie ręki "
                "i bezpiecznie tapuje Pluto pod warunek 3+ wytapowanych postaci 'Elinor - Renowned Diplomat'."
            )
        if "Nani - Stage Manager" in name:
            return (
                "Najlepszy silnik dobierający do Princess Aggro (3 ink, 3/3/1, Inkable): na wejściu patrzy na 4 wierzchu talii "
                "i dobiera postać o koszcie <=2 (w tej talii to prawie każda karta: Pocahontas, Dale, Willow, Aurora, Pluto, David)."
            )
        if "David - Protective Snowboarder" in name:
            return (
                "Kluczowy 2-drop Bodyguard (1/4/1, Inkable): wchodzi wytapowany chroniąc Twoje 1-dropy (Pluto, Aurora, Pocahontas), "
                "od razu wlicza się do 'Elinor' (3+ exerted), a z 'Dale - Ready for His Shot' bije w wyzwaniach za 4 obrażenia (4 ⛉)!"
            )
        if "Rapunzel - Tower Defender" in name:
            return (
                "Księżniczka (4 ink — kosztuje 3 ink po queście 'Aurora - Holding Court'!, 3/3/1, Inkable): "
                "na wejściu pozwala odrzucić kartę, aby cofnąć dowolną postać rywala na rękę (usuwa Bodyguardów i duże Shifty)."
            )
        if "Monterey Jack - Hypnotized by Cheese" in name:
            return (
                "Ukryty gem (2 ink, 0/3, aż 3 LORE!): może questować tylko gdy masz postać z 4+ Willpower — "
                " a w Princess Aggro masz aż 16-20 postaci 0/4 i 1/4 od 1. tury (Mickey, Thomas, Mike Wazowski, David)!"
            )
        if "Percy - Pupsicle" in name:
            return (
                "Inkowalny 3-drop (1/4, aż 3 LORE!): nie może wyzywać, ale za ścianą Bodyguardów i Shiftem 'Pocahontas - Peacekeeper' "
                "zdobywa 3 Lore co turę, a przy obronie z 'Dale - Ready for His Shot' zadaje 4 obrażenia (4 ⛉)!"
            )
        if "Mickey Mouse - Expedition Leader" in name:
            return (
                "4-drop z Twojej wersji bazowej (2/3/2, Inkable): może wejść wytapowany (pod 'Elinor') i sprawia, że każdy quest "
                "Twojej innej postaci daje wybranej postaci rywala -2 ¤ do początku Twojej następnej tury."
            )
        if "Lionheart - Incumbent Mayor" in name:
            return (
                "Finisher z Twojej wersji bazowej (6 ink, 4/7/2 Bodyguard, Inkable): na wejściu daje +1 Lore dwóm Twoim postaciom "
                "(natychmiastowy burst +2 Lore do zamknięcia gry), ale przez koszt 6 bywa ciężki w rękach bez rampy."
            )
        if "Finnick - Tiny Terror" in name:
            return (
                "1-drop z Twojej wersji bazowej (1/2/1, Inkable): w 3+ turze pozwala dopłacić 2 ink na wejściu, "
                "aby cofnąć postać rywala o sile <=2 na rękę."
            )
        if "Pocahontas - Finding the Way" in name:
            return (
                "Księżniczka z Twojej wersji bazowej (3 ink, 2/5/1, Inkable): daje +1 Lore wybranej postaci na wejściu, "
                "ma aż 5 Willpower pod 'Dale - Ready for His Shot' i stanowi dodatkową bazę pod Shift dla 'Pocahontas - Peacekeeper'."
            )
        if "The Prince - Searching for His Love" in name:
            return (
                "Agresywny 1-drop z Twojej wersji bazowej (0/2/2, Uninkable): questuje za 2 Lore już od 2. tury "
                "i może wejść za darmo z 'Pocahontas - Guiding the Tribe'."
            )
        if "Lenny - Toy Binoculars" in name:
            return (
                "Tech na Control oraz Circle of Life (3 ink, 0/2/1, Inkable): na wejściu odkrywa rękę przeciwnika "
                "i odrzuca jego kluczową Akcję/Pieśń (np. 'Under the Sea', 'Malicious, Mean, and Scary', 'The Horseman Strikes!')."
            )
        if "Cursed Merfolk - Ursula's Handiwork" in name:
            return (
                "1-drop (0/1/2, Uninkable) wchodzący za darmo z 'Pocahontas - Guiding the Tribe': questuje za 2 Lore "
                "i wymusza odrzucenie karty przez rywala, jeśli ten zdecyduje się go wyzwać."
            )
        if "Mr. Manchas - Service with a Smile" in name:
            return (
                "Inkowalny 3-drop (1/4/2): questuje za 2 Lore, obniża koszt kolejnej postaci o 1 ink "
                "i ma 4 Willpower pod 'Dale - Ready for His Shot'."
            )
        return f"Wspiera agresywną krzywą ({card.cost} ink) i synergię Princess/Willpower w talii Amber/Emerald Princess Aggro."

    @staticmethod
    def _score_circle_of_life_synergy(card: CardTemplate, core_base_names: set[str]) -> float:
        score = 0.0
        ops = {cl.op for cl in card.abilities.clauses}
        txt = " ".join(card.full_text.lower().split())
        name = card.full_name

        # 1. Direct Shift onto a locked core character (Max Goof - Chart Topper onto Max Goof - Karaoke Star!)
        if card.abilities.shift_cost is not None and card.cost <= 6 and card.base_name in core_base_names:
            score += 26.0

        # 2. Replaying Songs from discard (Max Goof - Chart Topper) or returning Songs from discard (Max Goof - Rebellious Teen / Aurora)
        if EffectOpType.REPLAY_SONG_FROM_DISCARD in ops:
            score += 25.0
        if "return a song card" in txt:
            score += 22.0

        # 3. Synergizes with Malicious, Mean, and Scary (1 damage AoE) & Prince Phillip (To Wither a Flower / Don Karnage)
        if EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP in ops:
            score += 24.5
        if EffectOpType.DEAL_DAMAGE_ALL_OPP in ops and card.abilities.shift_cost is not None:
            score += 23.5

        # 4. Early Underdog / Singer / Targeted Hand Disruption (White Rabbit, Miguel Rivera, Lenny, Ursula - Deceiver, Bobby Zimuruski)
        if card.abilities.has_underdog:
            score += 24.0
        if "gains singer 3" in txt:
            score += 21.5
        if "discards a song card" in txt or "discards an action card" in txt:
            score += 22.5
        if EffectOpType.LOOT_SELF in ops and card.cost <= 2:
            score += 19.5

        # 5. High-impact reanimation / tempo targets (Chernabog - Unnatural Force, Stitch - Carefree Surfer, Under the Sea)
        if "shuffle chosen opposing character" in txt:
            score += 23.0
        if "2 or more other characters in play, you may draw 2 cards" in txt:
            score += 22.5
        if "Khan's Courier" in name:
            score += 20.0
        if "Under the Sea" in name:
            score += 19.0

        if card.inkable:
            score += 2.0
        return score

    @staticmethod
    def _explain_circle_of_life_role(card: CardTemplate) -> str:
        name = card.full_name
        txt = " ".join(card.full_text.lower().split())
        if "Chart Topper" in name:
            return (
                "Shift 4 na 'Max Goof - Karaoke Star'; przy każdym queście zagrywa za darmo pieśń o koszcie <=4 z cmentarza "
                "(np. 'To Wither a Flower', 'Malicious, Mean, and Scary', 'Akood et Emuti' lub 'Strike a Good Match')."
            )
        if "Stitch - Carefree Surfer" in name:
            return (
                "Inkowalny cel pod 'Circle of Life' (4/8, 2 Lore, Singer 7): przy wejściu (gdy masz min. 2 inne postacie — np. po Sing Together) "
                "dobiera 2 karty i daje potężne ciało stabilizujące stół."
            )
        if "To Wither a Flower" in name:
            return (
                "Pieśń (koszt 4, śpiewana przez 'Ursula - Vanessa'): zadaje 2 obrażenia wszystkim zranionym postaciom rywala "
                "— w combo z 'Malicious, Mean, and Scary' czyści stoły Aggro/Midrange (3 obrażenia AoE)."
            )
        if "White Rabbit - Late Again" in name:
            return (
                "Kluczowy 2-drop z 'Underdog' (kosztuje 1 ink na Draw w 1. turze!) oraz 'Evasive' — broni przed Evasive zagrożeniami "
                "i daje dodatkowe ciało do chóru 'Sing Together 8' dla 'Circle of Life'."
            )
        if "Chernabog - Unnatural Force" in name:
            return (
                "Potężny cel pod 'Circle of Life' oraz odpowiedź (5/4, 2 Lore) wtasowująca największe zagrożenie rywala do jego talii "
                "(omija Ward przy wybranych celach / niszczy tempo rywala)."
            )
        if "Don Karnage - Debonair Pirate" in name:
            return (
                "Shift 4 (3/6, 2 Lore): przy wejściu zadaje 1 obrażenie wszystkim postaciom rywala (ignorując Resist!) "
                "— uruchamia masowe wygnanie z 'Prince Phillip - Vanquisher of Foes' i obniża Lore zranionych wrogów o -1."
            )
        if "Lenny - Toy Binoculars" in name:
            return (
                "Tech na Control oraz Mirror: na wejściu odkrywa rękę przeciwnika i odrzuca jego kluczową Akcję/Pieśń "
                "(np. 'Circle of Life', 'The Horseman Strikes!'), a sam odtapowuje się po zagraniu akcji."
            )
        if "Ursula - Deceiver" in name:
            return (
                "Inkowalny 2-drop (1/3) wyrzucający pieśń z ręki przeciwnika — niszczy plan gry w mirrorze 'Circle of Life' "
                "oraz przeciwko 'Amber/Sapphire' i 'Amber/Steel'."
            )
        if "Max Goof - Rebellious Teen" in name:
            return (
                "Dodatkowa baza pod Shift dla 'Max Goof - Chart Topper', która na wejściu pozwala odzyskać z cmentarza "
                "kluczową pieśń o koszcie <=3 ('Strike a Good Match', 'Malicious, Mean, and Scary', 'Akood et Emuti')."
            )
        if "Aurora - Delightful Musician" in name:
            return (
                "Inkowalny 3-drop (2/3): na wejściu zwraca na rękę zaśpiewaną w tej turze pieśń o koszcie <=3 "
                "oraz daje +1 Lore na koniec tury, w której zagrałeś pieśń."
            )
        if "Miguel Rivera - Street Musician" in name:
            return (
                "Inkowalny 1-drop (1/2), który przy choćby 1 pieśni na cmentarzu zyskuje +1 Lore (questuje za 2!) oraz 'Singer 3' "
                "— błyskawicznie przyspiesza 'Sing Together 8' dla 'Circle of Life'."
            )
        if "Khan's Courier" in name:
            return (
                "Agresywny 1-drop 2/2/1 (inkowalny), który wymienia się w 1-2 turze z postaciami Princess Aggro "
                "i stanowi bazę pod Shift 4 dla 'Don Karnage - Debonair Pirate'."
            )
        if "Bobby Zimuruski" in name:
            return (
                "1-drop (1/2/1) z efektem Looting (dobierz 1, odrzuć 1) — już w 1. turze pozwala wrzucić 'Prince Phillip' "
                "lub 'Milo Thatch' na cmentarz pod szybkie 'Circle of Life'."
            )
        if "Under the Sea" in txt or "Under the Sea" in name:
            return "Sing Together 8: masowo chowa na spód talii wszystkie postacie rywala o sile <=2 (świetne z 'Malicious' / debuffami)."
        return f"Wspiera krzywą kosztu ({card.cost} ink) i synergię pieśni/reanimacji w talii Amber/Emerald Circle of Life."

    @staticmethod
    def _StaticTacticalPrior(card: CardTemplate, opp_avg_cost: float) -> float:
        flags, eff_val = card.abilities.to_rust_bitflags()
        score = float(eff_val)
        if opp_avg_cost <= 3.3:
            if (flags & 32) and card.cost <= 5:
                score += 4.0
            if card.cost <= 3 and card.strength >= 3:
                score += 2.5
        else:
            if flags & 64:
                score += 3.5
            if card.lore >= 2:
                score += 3.0
        return score

    @staticmethod
    def _explain_tech_card(card: CardTemplate, opponent_deck: Deck) -> str:
        reasons: list[str] = []
        txt = card.full_text.lower()
        if "banish" in txt or "damage" in txt:
            reasons.append("Zapewnia bezpośrednią odpowiedź (removal/board control) na kluczowe postacie rywala")
        if "draw" in txt:
            reasons.append("Zwiększa przewagę kart (card advantage) w dłuższych wymianach")
        if "discard" in txt:
            reasons.append("Wyciąga kluczowe pieśni/sweepery z ręki przeciwnika przed ich zagraniem")
        if "inkwell" in txt:
            reasons.append("Przyspiesza krzywą inka (ramp), pozwalając wyprzedzić tempo talii rywala")
        if card.lore >= 2:
            reasons.append(f"Wywiera silną presję punktową (+{card.lore} Lore na quest) wymuszając niekorzystne wymiany")
        if not reasons:
            reasons.append(f"Poprawia krzywą kosztu ({card.cost} ink) oraz stabilność inkowania w matchupie vs {opponent_deck.name}")
        return "; ".join(reasons) + "."
