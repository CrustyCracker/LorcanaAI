"""
Module 3: Nash Equilibrium Metagame Solver & Matchup Mulligan Advisor.
Computes the unexploitable Nash Equilibrium distribution over a pool of decks (meta + newly discovered AI decks)
using Replicator Dynamics, and generates Play/Draw Mulligan Guides for any matchup.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Any

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import ActionType, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


@dataclass
class NashEquilibriumReport:
    deck_names: list[str]
    payoff_matrix_bo3: Any         # (N, N) Bo3 winrate of row deck vs col deck
    play_matrix: Any               # (N, N) On-the-Play winrate
    draw_matrix: Any               # (N, N) On-the-Draw winrate
    nash_weights: dict[str, float]
    expected_winrate_vs_current_meta: dict[str, float]
    best_counter_deck: str


@dataclass
class MulliganGuide:
    my_deck_name: str
    opponent_deck_name: str
    best_strategy_on_play: str
    best_strategy_on_draw: str
    strategy_winrates_play: dict[str, float]
    strategy_winrates_draw: dict[str, float]
    priority_keep_cards: list[str]


class NashMetaSolver:
    """
    Solves the metagame game-theoretic matrix using Replicator Dynamics and provides
    matchup-specific Play/Draw Mulligan optimization.
    """

    def __init__(self, evaluator: MatchupEvaluator | None = None) -> None:
        self.evaluator = evaluator or MatchupEvaluator()

    def solve_metagame(
        self,
        decks: list[Deck],
        current_meta_shares: dict[str, float] | None = None,
        games_per_pair: int = 12,
        replicator_steps: int = 500,
    ) -> NashEquilibriumReport:
        n = len(decks)
        names = [d.name for d in decks]
        bo3_mat = [[0.5 for _ in range(n)] for _ in range(n)]
        play_mat = [[0.5 for _ in range(n)] for _ in range(n)]
        draw_mat = [[0.5 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                stats = self.evaluator.evaluate_matchup(
                    decks[i], decks[j], num_games=games_per_pair, seed=900 + i * 31 + j * 7
                )
                bo3_mat[i][j] = round(stats.bo3_winrate, 4)
                bo3_mat[j][i] = round(1.0 - stats.bo3_winrate, 4)
                play_mat[i][j] = round(stats.play_winrate, 4)
                draw_mat[i][j] = round(stats.draw_winrate, 4)
                play_mat[j][i] = round(1.0 - stats.draw_winrate, 4)
                draw_mat[j][i] = round(1.0 - stats.play_winrate, 4)

        # 1. Expected winrate against current human tournament metagame
        prior_vec = [
            current_meta_shares.get(nm, 1.0 / n) if current_meta_shares else (1.0 / n)
            for nm in names
        ]
        s_prior = sum(prior_vec)
        if s_prior <= 0:
            prior_vec = [1.0 / n for _ in range(n)]
        else:
            prior_vec = [v / s_prior for v in prior_vec]

        ev_vs_current: dict[str, float] = {}
        for i, nm in enumerate(names):
            ev_vs_current[nm] = round(sum(bo3_mat[i][j] * prior_vec[j] for j in range(n)), 4)

        best_counter = max(ev_vs_current.items(), key=lambda kv: kv[1])[0]

        # 2. Replicator Dynamics to find unexploitable Nash Equilibrium metagame mix
        dist = [1.0 / n for _ in range(n)]
        lr = 0.25
        for _ in range(replicator_steps):
            fitness = [sum(bo3_mat[i][j] * dist[j] for j in range(n)) for i in range(n)]
            avg_fitness = sum(dist[i] * fitness[i] for i in range(n))
            dist = [max(1e-5, dist[i] * math.exp(lr * (fitness[i] - avg_fitness))) for i in range(n)]
            s_dist = sum(dist)
            dist = [v / s_dist for v in dist]

        nash_weights = {names[i]: round(float(dist[i]), 4) for i in range(n)}
        return NashEquilibriumReport(
            deck_names=names,
            payoff_matrix_bo3=bo3_mat,
            play_matrix=play_mat,
            draw_matrix=draw_mat,
            nash_weights=nash_weights,
            expected_winrate_vs_current_meta=ev_vs_current,
            best_counter_deck=best_counter,
        )

    def generate_mulligan_guide(
        self,
        my_deck: Deck,
        opponent_deck: Deck,
        games_per_strategy: int = 12,
    ) -> MulliganGuide:
        """
        Tests each of the 4 structured mulligan strategies (On the Play & On the Draw)
        for `my_deck` vs `opponent_deck` in a known-ink matchup, and identifies priority keep cards.
        """
        agent = TacticalHeuristicAgent()
        cards_me = my_deck.expand_cards()
        cards_opp = opponent_deck.expand_cards()

        strategy_names = [
            "Balanced Curve (Keep <=3 cost)",
            "Fast Early Game (Keep 1-2 drops & removal)",
            "Engine & Ramp Hunt",
            "Keep All 7",
        ]

        wr_play: dict[str, float] = {}
        wr_draw: dict[str, float] = {}

        for s_idx, s_label in enumerate(strategy_names):
            for is_play in (True, False):
                wins = 0
                trials = max(2, games_per_strategy // 2)
                for g in range(trials):
                    seed = 7700 + s_idx * 101 + g * 19 + (0 if is_play else 500)
                    rng = random.Random(seed)
                    state = GameState.create_new_game(
                        cards_me,
                        cards_opp,
                        starting_player=0 if is_play else 1,
                        know_opponent_inks=True,
                        seed=seed,
                    )
                    steps = 0
                    while state.phase != "GAME_OVER" and steps < 160:
                        steps += 1
                        legal = MoveGenerator.generate_legal_actions(state)
                        if not legal:
                            break
                        if state.active_player == 0 and state.phase.startswith("MULLIGAN"):
                            # Force strategy s_idx for player 0's mulligan
                            chosen = legal[min(s_idx, len(legal) - 1)]
                        else:
                            chosen = agent.select_action(state, legal)
                        LorcanaSimulator.step(state, chosen, rng=rng)

                    winner = state.winner if state.winner is not None else (
                        0 if state.players[0].lore >= state.players[1].lore else 1
                    )
                    if winner == 0:
                        wins += 1

                if is_play:
                    wr_play[s_label] = round(wins / trials, 4)
                else:
                    wr_draw[s_label] = round(wins / trials, 4)

        best_play = max(wr_play.items(), key=lambda kv: kv[1])[0]
        best_draw = max(wr_draw.items(), key=lambda kv: kv[1])[0]

        # Rank top priority keep cards in my_deck for this matchup
        distinct_cards = [my_deck.db.by_id[cid] for cid in my_deck.card_counts]
        distinct_cards.sort(
            key=lambda c: (
                c.cost <= 3 and c.inkable,
                "ramp" in c.full_text.lower() or "inkwell" in c.full_text.lower() or "draw" in c.full_text.lower() or "banish" in c.full_text.lower(),
                c.lore,
                -c.cost,
            ),
            reverse=True,
        )
        top_keeps = [f"{c.full_name} ({c.cost} ink)" for c in distinct_cards[:5]]

        return MulliganGuide(
            my_deck_name=my_deck.name,
            opponent_deck_name=opponent_deck.name,
            best_strategy_on_play=best_play,
            best_strategy_on_draw=best_draw,
            strategy_winrates_play=wr_play,
            strategy_winrates_draw=wr_draw,
            priority_keep_cards=top_keeps,
        )
