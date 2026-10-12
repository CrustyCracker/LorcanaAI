"""
High-speed Matchup Evaluator for LorcanaAlpha.
Computes Play Winrate, Draw Winrate, and Best-of-3 (Bo3) Match Winrate with 95% confidence intervals.
Automatically uses native Rust Rayon engine (lorcana_rs) if compiled, or fast tactical Python engine.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import MoveGenerator
from lorcana_ai.engine.simulator import HAS_RUST_ENGINE, LorcanaSimulator

if HAS_RUST_ENGINE:
    import lorcana_rs  # type: ignore


@dataclass
class MatchupStats:
    deck_a_name: str
    deck_b_name: str
    games_played: int
    play_winrate: float
    draw_winrate: float
    overall_winrate: float
    bo3_winrate: float
    ci_95_halfwidth: float
    avg_turns: float


class MatchupEvaluator:
    """Evaluates head-to-head matchups between two 60-card Lorcana decks."""

    def __init__(self, use_full_dsl_sim: bool = True) -> None:
        self.use_full_dsl_sim = use_full_dsl_sim
        self.agent = TacticalHeuristicAgent()

    def evaluate_matchup(
        self,
        deck_a: Deck,
        deck_b: Deck,
        num_games: int = 24,
        seed: int = 42,
    ) -> MatchupStats:
        num_games = max(2, num_games)

        if HAS_RUST_ENGINE and not self.use_full_dsl_sim:
            t1 = deck_a.to_rust_tuples()
            t2 = deck_b.to_rust_tuples()
            w_play, g_play, w_draw, g_draw, avg_turns = lorcana_rs.simulate_batch_matchup_rs(
                t1, t2, num_games, seed
            )
        else:
            w_play = 0
            g_play = 0
            w_draw = 0
            g_draw = 0
            total_turns = 0
            cards_a = deck_a.expand_cards()
            cards_b = deck_b.expand_cards()

            for i in range(num_games):
                start_p = i % 2
                game_seed = seed + i * 7919
                rng = random.Random(game_seed)
                state = GameState.create_new_game(
                    cards_a,
                    cards_b,
                    starting_player=start_p,
                    know_opponent_inks=(i % 3 != 0),  # 1/3 G1 blind, 2/3 G2/G3 known inks
                    seed=game_seed,
                )
                steps = 0
                while state.phase != "GAME_OVER" and steps < 180:
                    steps += 1
                    legal = MoveGenerator.generate_legal_actions(state)
                    if not legal:
                        break
                    act = self.agent.select_action(state, legal)
                    LorcanaSimulator.step(state, act, rng=rng)

                winner = state.winner if state.winner is not None else (
                    0 if state.players[0].lore >= state.players[1].lore else 1
                )
                total_turns += state.turn_number
                if start_p == 0:
                    g_play += 1
                    if winner == 0:
                        w_play += 1
                else:
                    g_draw += 1
                    if winner == 0:
                        w_draw += 1

            avg_turns = total_turns / max(1, num_games)

        p_play = w_play / max(1, g_play)
        p_draw = w_draw / max(1, g_draw)
        overall = (w_play + w_draw) / max(1, g_play + g_draw)

        # Exact Best-of-3 (Bo3) Match Winrate knowing loser of previous game chooses Play in next game!
        bo3_wr = self.compute_bo3_match_winrate(p_play, p_draw)
        ci = 1.96 * math.sqrt(max(1e-4, overall * (1.0 - overall)) / max(1, num_games))

        return MatchupStats(
            deck_a_name=deck_a.name,
            deck_b_name=deck_b.name,
            games_played=num_games,
            play_winrate=round(p_play, 4),
            draw_winrate=round(p_draw, 4),
            overall_winrate=round(overall, 4),
            bo3_winrate=round(bo3_wr, 4),
            ci_95_halfwidth=round(ci, 4),
            avg_turns=round(avg_turns, 2),
        )

    @staticmethod
    def compute_bo3_match_winrate(p_play: float, p_draw: float) -> float:
        """
        Calculates exact Bo3 match win probability in Lorcana tournament rules:
        - 50% chance Deck A wins die roll (starts G1 on the Play), 50% starts G1 on the Draw.
        - Loser of Game 1 chooses to be on the Play in Game 2.
        - Loser of Game 2 chooses to be on the Play in Game 3.
        """
        # Case 1: A wins die roll (G1 on Play)
        # A wins G1 (prob p_play) -> G2 A is on Draw: A wins G2 (p_draw) OR loses G2 (1-p_draw) and wins G3 on Play (p_play)
        # A loses G1 (prob 1-p_play) -> G2 A is on Play: A must win G2 (p_play) AND then win G3 on Draw (p_draw)
        win_given_die_win = p_play * (p_draw + (1.0 - p_draw) * p_play) + (1.0 - p_play) * (p_play * p_draw)

        # Case 2: A loses die roll (G1 on Draw)
        # A wins G1 (prob p_draw) -> G2 A is on Draw: A wins G2 (p_draw) OR loses G2 (1-p_draw) and wins G3 on Play (p_play)
        # A loses G1 (prob 1-p_draw) -> G2 A is on Play: A must win G2 (p_play) AND then win G3 on Draw (p_draw)
        win_given_die_loss = p_draw * (p_draw + (1.0 - p_draw) * p_play) + (1.0 - p_draw) * (p_play * p_draw)

        return 0.5 * win_given_die_win + 0.5 * win_given_die_loss
