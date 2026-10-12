"""
High-speed Parallel Matchup Evaluator for LorcanaAlpha.
Computes Play Winrate, Draw Winrate, and Best-of-3 (Bo3) Match Winrate with 95% confidence intervals.
Supports deterministic Common Random Numbers (CRN) across multi-core ProcessPoolExecutor workers
(auto-scaling to SLURM_CPUS_PER_TASK on PLGrid Athena or local CPU cores).
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import os
import random

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import MoveGenerator
from lorcana_ai.engine.simulator import HAS_RUST_ENGINE, LorcanaSimulator

if HAS_RUST_ENGINE:
    import lorcana_rs  # type: ignore

_WORKER_AGENT: TacticalHeuristicAgent | None = None


def _simulate_game_range_worker(
    args: tuple[str, dict[int, int], dict[int, int], int, int, int],
) -> tuple[str, int, int, int, int, int]:
    """
    Worker function for multi-core simulation.
    Preserves exact per-game deterministic seed (`seed + i * 7919`) for Common Random Numbers (CRN).
    """
    global _WORKER_AGENT
    if _WORKER_AGENT is None:
        _WORKER_AGENT = TacticalHeuristicAgent()

    opp_key, counts_a, counts_b, start_i, end_i, base_seed = args
    cards_a = [
        GLOBAL_CARD_DB.by_id[cid]
        for cid, qty in counts_a.items()
        for _ in range(qty)
        if cid in GLOBAL_CARD_DB.by_id
    ]
    cards_b = [
        GLOBAL_CARD_DB.by_id[cid]
        for cid, qty in counts_b.items()
        for _ in range(qty)
        if cid in GLOBAL_CARD_DB.by_id
    ]

    w_play = 0
    g_play = 0
    w_draw = 0
    g_draw = 0
    total_turns = 0

    for i in range(start_i, end_i):
        start_p = i % 2
        game_seed = base_seed + i * 7919
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
            act = _WORKER_AGENT.select_action(state, legal)
            LorcanaSimulator.step(state, act, rng=rng)

        winner = (
            state.winner
            if state.winner is not None
            else (0 if state.players[0].lore >= state.players[1].lore else 1)
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

    return opp_key, w_play, g_play, w_draw, g_draw, total_turns


def get_default_worker_count() -> int:
    """Detects SLURM_CPUS_PER_TASK on PLGrid Athena or local CPU count."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return max(1, min(16, (os.cpu_count() or 4)))


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

    _SHARED_POOL: ProcessPoolExecutor | None = None
    _SHARED_POOL_WORKERS: int = 0

    def __init__(self, use_full_dsl_sim: bool = True, workers: int | None = None) -> None:
        self.use_full_dsl_sim = use_full_dsl_sim
        self.agent = TacticalHeuristicAgent()
        self.workers = workers if workers is not None else get_default_worker_count()

    @classmethod
    def _get_pool(cls, workers: int) -> ProcessPoolExecutor:
        if cls._SHARED_POOL is None or cls._SHARED_POOL_WORKERS != workers:
            if cls._SHARED_POOL is not None:
                cls._SHARED_POOL.shutdown(wait=False)
            cls._SHARED_POOL = ProcessPoolExecutor(max_workers=workers)
            cls._SHARED_POOL_WORKERS = workers
        return cls._SHARED_POOL

    def _build_stats(
        self,
        deck_a_name: str,
        deck_b_name: str,
        num_games: int,
        w_play: int,
        g_play: int,
        w_draw: int,
        g_draw: int,
        avg_turns: float,
    ) -> MatchupStats:
        p_play = w_play / max(1, g_play)
        p_draw = w_draw / max(1, g_draw)
        overall = (w_play + w_draw) / max(1, g_play + g_draw)
        bo3_wr = self.compute_bo3_match_winrate(p_play, p_draw)
        ci = 1.96 * math.sqrt(max(1e-4, overall * (1.0 - overall)) / max(1, num_games))
        return MatchupStats(
            deck_a_name=deck_a_name,
            deck_b_name=deck_b_name,
            games_played=num_games,
            play_winrate=round(p_play, 4),
            draw_winrate=round(p_draw, 4),
            overall_winrate=round(overall, 4),
            bo3_winrate=round(bo3_wr, 4),
            ci_95_halfwidth=round(ci, 4),
            avg_turns=round(avg_turns, 2),
        )

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
        elif self.workers > 1 and num_games >= 16:
            eff_workers = min(self.workers, max(1, num_games // 4))
            chunk_size = math.ceil(num_games / eff_workers)
            tasks: list[tuple[str, dict[int, int], dict[int, int], int, int, int]] = []
            for w_idx in range(eff_workers):
                s_i = w_idx * chunk_size
                e_i = min(num_games, (w_idx + 1) * chunk_size)
                if s_i < e_i:
                    tasks.append((deck_b.name, deck_a.card_counts, deck_b.card_counts, s_i, e_i, seed))

            pool = self._get_pool(self.workers)
            w_play = g_play = w_draw = g_draw = total_turns = 0
            for _, wp, gp, wd, gd, tt in pool.map(_simulate_game_range_worker, tasks):
                w_play += wp
                g_play += gp
                w_draw += wd
                g_draw += gd
                total_turns += tt
            avg_turns = total_turns / max(1, num_games)
        else:
            _, w_play, g_play, w_draw, g_draw, total_turns = _simulate_game_range_worker(
                (deck_b.name, deck_a.card_counts, deck_b.card_counts, 0, num_games, seed)
            )
            avg_turns = total_turns / max(1, num_games)

        return self._build_stats(
            deck_a.name, deck_b.name, num_games, w_play, g_play, w_draw, g_draw, avg_turns
        )

    def evaluate_gauntlet_batch(
        self,
        deck: Deck,
        meta_decks: dict[str, Deck],
        num_games: int = 24,
        seed: int = 42,
    ) -> dict[str, MatchupStats]:
        """
        Evaluates `deck` against all `meta_decks` in a single parallel batch across all CPU cores,
        preserving exact per-opponent and per-game CRN seeds (`seed + idx * 37 + i * 7919`).
        """
        num_games = max(2, num_games)
        total_gauntlet_games = len(meta_decks) * num_games

        if self.workers <= 1 or total_gauntlet_games < 24:
            return {
                opp_name: self.evaluate_matchup(deck, opp_deck, num_games=num_games, seed=seed + idx * 37)
                for idx, (opp_name, opp_deck) in enumerate(meta_decks.items())
            }

        # Split each opponent's games into chunks so all workers stay saturated
        chunks_per_opp = max(1, math.ceil((self.workers * 2) / max(1, len(meta_decks))))
        chunk_size = max(4, math.ceil(num_games / chunks_per_opp))

        tasks: list[tuple[str, dict[int, int], dict[int, int], int, int, int]] = []
        for idx, (opp_name, opp_deck) in enumerate(meta_decks.items()):
            opp_seed = seed + idx * 37
            for s_i in range(0, num_games, chunk_size):
                e_i = min(num_games, s_i + chunk_size)
                tasks.append((opp_name, deck.card_counts, opp_deck.card_counts, s_i, e_i, opp_seed))

        pool = self._get_pool(self.workers)
        agg: dict[str, list[int]] = {opp_name: [0, 0, 0, 0, 0] for opp_name in meta_decks}
        for opp_key, wp, gp, wd, gd, tt in pool.map(_simulate_game_range_worker, tasks):
            row = agg[opp_key]
            row[0] += wp
            row[1] += gp
            row[2] += wd
            row[3] += gd
            row[4] += tt

        results: dict[str, MatchupStats] = {}
        for opp_name, opp_deck in meta_decks.items():
            wp, gp, wd, gd, tt = agg[opp_name]
            results[opp_name] = self._build_stats(
                deck.name, opp_deck.name, num_games, wp, gp, wd, gd, tt / max(1, num_games)
            )
        return results

    @staticmethod
    def compute_bo3_match_winrate(p_play: float, p_draw: float) -> float:
        """
        Calculates exact Bo3 match win probability in Lorcana tournament rules:
        - 50% chance Deck A wins die roll (starts G1 on the Play), 50% starts G1 on the Draw.
        - Loser of Game 1 chooses to be on the Play in Game 2.
        - Loser of Game 2 chooses to be on the Play in Game 3.
        """
        win_given_die_win = p_play * (p_draw + (1.0 - p_draw) * p_play) + (1.0 - p_play) * (p_play * p_draw)
        win_given_die_loss = p_draw * (p_draw + (1.0 - p_draw) * p_play) + (1.0 - p_draw) * (p_play * p_draw)
        return 0.5 * win_given_die_win + 0.5 * win_given_die_loss
