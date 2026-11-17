"""
Thin IS-MCTS spotcheck helper for Top-K tech / deck validation.

Full MatchupEvaluator gauntlets always use TacticalHeuristicAgent (L1).
This module runs a tiny number of IS-MCTS games as a play-quality spotcheck —
not a substitute for meta rankings. Prefer Heuristic L1 for scans; use this
only to sanity-check a shortlist of decks or tech swaps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

from lorcana_ai.agents.is_mcts import ISMCTSAgent
from lorcana_ai.config import MCTSConfig
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


@dataclass
class ISMCTSSpotResult:
    deck_a_name: str
    deck_b_name: str
    games: int
    wins_a: int
    winrate_a: float
    avg_turns: float
    simulations: int
    determinizations: int


def spotcheck_matchup(
    deck_a: Deck,
    deck_b: Deck,
    games: int = 2,
    seed: int = 2137,
    simulations: int = 8,
    determinizations: int = 2,
    max_steps: int = 160,
) -> ISMCTSSpotResult:
    """
    Play `games` short matches with IS-MCTS on both sides (very slow vs heuristic).
    Intended for Top-K validation only (2–4 games).
    """
    cfg = MCTSConfig(
        num_determinizations=max(1, determinizations),
        simulations_per_determinization=max(1, simulations),
    )
    agent = ISMCTSAgent(cfg=cfg)
    wins = 0
    turns = 0
    n = max(1, games)
    for i in range(n):
        g_seed = seed + i * 7919
        rng = random.Random(g_seed)
        start_p = i % 2
        state = GameState.create_new_game(
            deck_a.expand_cards(),
            deck_b.expand_cards(),
            starting_player=start_p,
            know_opponent_inks=True,
            seed=g_seed,
        )
        steps = 0
        while state.phase != "GAME_OVER" and steps < max_steps:
            steps += 1
            legal = MoveGenerator.generate_legal_actions(state)
            if not legal:
                break
            act, _ = agent.select_action(state, use_mcts=True, rng=rng)
            LorcanaSimulator.step(state, act, rng=rng)
        winner = (
            state.winner
            if state.winner is not None
            else (0 if state.players[0].lore >= state.players[1].lore else 1)
        )
        if winner == 0:
            wins += 1
        turns += state.turn_number
    return ISMCTSSpotResult(
        deck_a_name=deck_a.name,
        deck_b_name=deck_b.name,
        games=n,
        wins_a=wins,
        winrate_a=round(wins / n, 4),
        avg_turns=round(turns / n, 2),
        simulations=cfg.simulations_per_determinization,
        determinizations=cfg.num_determinizations,
    )


def spotcheck_top_k_from_json(
    tech_scan_json: Path,
    my_deck: Deck,
    opp_deck: Deck,
    top_k: int = 3,
    games: int = 2,
    seed: int = 2137,
) -> list[dict]:
    """
    Load a scan-tech JSON (list of TechCardRecommendation dicts), rebuild each
    Top-K swap into a modified deck, and IS-MCTS-spotcheck vs opp_deck.
    """
    raw = json.loads(tech_scan_json.read_text(encoding="utf-8"))
    results: list[dict] = []
    for entry in raw[: max(1, top_k)]:
        in_name = entry.get("card_in_name") or entry.get("card_name")
        out_name = entry.get("card_out_name")
        copies = int(entry.get("copies_swapped") or entry.get("recommended_copies") or 1)
        if not in_name:
            continue
        mod = my_deck.clone(new_name=f"{my_deck.name} +{copies} {in_name}")
        counts = dict(mod.card_counts)
        in_card = my_deck.db.get_by_name(str(in_name))
        if in_card is None:
            continue
        if out_name:
            out_card = my_deck.db.get_by_name(str(out_name))
            if out_card is not None and counts.get(out_card.card_id, 0) >= copies:
                counts[out_card.card_id] -= copies
                if counts[out_card.card_id] <= 0:
                    del counts[out_card.card_id]
        counts[in_card.card_id] = counts.get(in_card.card_id, 0) + copies
        mod.card_counts = counts
        spot = spotcheck_matchup(mod, opp_deck, games=games, seed=seed)
        results.append({"tech": entry, "ismcts": asdict(spot)})
    return results
