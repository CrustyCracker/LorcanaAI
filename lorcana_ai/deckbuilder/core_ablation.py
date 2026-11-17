"""
CoreValue leave-one-out ablation (−1 copy) against a Duels.ink meta gauntlet.

For each unique card in the locked core (or full deck when no core is given),
drops one copy, evaluates the resulting list with MatchupEvaluator(use_full_dsl_sim=True),
and ranks cards by how much meta winrate falls when they are removed (weakest = smallest drop).
Optional replace probe swaps the removed copy for a named filler instead of playing at 59.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator, MatchupStats


@dataclass
class CoreAblationRow:
    card_id: int
    card_name: str
    copies_in_core: int
    baseline_bo1: float
    ablated_bo1: float
    delta_bo1: float
    baseline_bo3: float
    ablated_bo3: float
    delta_bo3: float
    matchup_deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class CoreAblationReport:
    deck_name: str
    core_card_count: int
    flex_slots: int
    games_per_matchup: int
    meta_opponents: list[str]
    baseline_bo1: float
    baseline_bo3: float
    replace_card: str | None
    rows: list[CoreAblationRow]

    def to_dict(self) -> dict:
        return {
            "deck_name": self.deck_name,
            "core_card_count": self.core_card_count,
            "flex_slots": self.flex_slots,
            "games_per_matchup": self.games_per_matchup,
            "meta_opponents": self.meta_opponents,
            "baseline_bo1": self.baseline_bo1,
            "baseline_bo3": self.baseline_bo3,
            "replace_card": self.replace_card,
            "rows": [asdict(r) for r in self.rows],
        }


def _weighted_meta_wr(
    stats_by_opp: dict[str, MatchupStats],
    shares: dict[str, float],
) -> tuple[float, float]:
    """Meta-share-weighted BO1 / BO3 winrate; uniform if shares missing."""
    total_share = sum(max(0.0, shares.get(name, 0.0)) for name in stats_by_opp)
    use_shares = total_share > 1e-9
    bo1 = 0.0
    bo3 = 0.0
    n = max(1, len(stats_by_opp))
    for name, st in stats_by_opp.items():
        w = (shares.get(name, 0.0) / total_share) if use_shares else (1.0 / n)
        bo1 += w * st.overall_winrate
        bo3 += w * st.bo3_winrate
    return bo1, bo3


def _drop_one_copy(deck: Deck, card_id: int, replace_id: int | None = None) -> Deck:
    """Return a clone with one fewer copy of card_id (optionally replaced by replace_id)."""
    out = deck.clone(new_name=f"{deck.name} (−1 {deck.db.by_id[card_id].full_name})")
    counts = dict(out.card_counts)
    if counts.get(card_id, 0) <= 0:
        raise ValueError(f"Card id {card_id} not present in deck")
    counts[card_id] -= 1
    if counts[card_id] <= 0:
        del counts[card_id]
    if replace_id is not None:
        counts[replace_id] = counts.get(replace_id, 0) + 1
    out.card_counts = counts
    return out


def run_core_value_ablation(
    full_deck: Deck,
    core_card_counts: dict[int, int] | None = None,
    meta_decks: dict[str, Deck] | None = None,
    meta_shares: dict[str, float] | None = None,
    games_per_matchup: int = 8,
    seed: int = 2137,
    workers: int | None = None,
    lookahead_depth: int = 0,
    replace_card_name: str | None = None,
) -> CoreAblationReport:
    """
    Leave-one-out CoreValue: for each unique core card, remove 1 copy and measure ΔWR vs meta.

    Always uses MatchupEvaluator(use_full_dsl_sim=True) (Python DSL path).
    Weakest core cards are those with the least-negative (or positive) delta_bo3 when removed.
    """
    from lorcana_ai.data.meta_decks import get_metagame_shares, get_reference_meta_decks

    meta_decks = meta_decks or get_reference_meta_decks(top_n=8)
    meta_shares = meta_shares if meta_shares is not None else get_metagame_shares(top_n=len(meta_decks))
    core = dict(core_card_counts) if core_card_counts else dict(full_deck.card_counts)
    core_total = sum(core.values())
    flex_slots = max(0, 60 - core_total)

    replace_id: int | None = None
    if replace_card_name:
        tmpl = GLOBAL_CARD_DB.get_by_name(replace_card_name)
        if tmpl is None:
            raise ValueError(f"Replace card not found: {replace_card_name}")
        replace_id = tmpl.card_id

    evaluator = MatchupEvaluator(
        use_full_dsl_sim=True,
        workers=workers,
        lookahead_depth=lookahead_depth,
    )

    baseline_stats = evaluator.evaluate_gauntlet_batch(
        full_deck, meta_decks, num_games=games_per_matchup, seed=seed
    )
    base_bo1, base_bo3 = _weighted_meta_wr(baseline_stats, meta_shares)

    rows: list[CoreAblationRow] = []
    unique_core_ids = sorted(
        (cid for cid, cnt in core.items() if cnt > 0 and cid in full_deck.card_counts and full_deck.card_counts.get(cid, 0) > 0),
        key=lambda cid: GLOBAL_CARD_DB.by_id[cid].full_name if cid in GLOBAL_CARD_DB.by_id else str(cid),
    )

    for idx, cid in enumerate(unique_core_ids):
        if cid not in GLOBAL_CARD_DB.by_id:
            continue
        if full_deck.card_counts.get(cid, 0) < 1:
            continue
        ablated = _drop_one_copy(full_deck, cid, replace_id=replace_id)
        abl_stats = evaluator.evaluate_gauntlet_batch(
            ablated,
            meta_decks,
            num_games=games_per_matchup,
            seed=seed + 1000 + idx * 17,
        )
        abl_bo1, abl_bo3 = _weighted_meta_wr(abl_stats, meta_shares)
        mu_deltas = {
            opp: abl_stats[opp].overall_winrate - baseline_stats[opp].overall_winrate
            for opp in abl_stats
        }
        rows.append(
            CoreAblationRow(
                card_id=cid,
                card_name=GLOBAL_CARD_DB.by_id[cid].full_name,
                copies_in_core=core.get(cid, full_deck.card_counts.get(cid, 0)),
                baseline_bo1=round(base_bo1, 4),
                ablated_bo1=round(abl_bo1, 4),
                delta_bo1=round(abl_bo1 - base_bo1, 4),
                baseline_bo3=round(base_bo3, 4),
                ablated_bo3=round(abl_bo3, 4),
                delta_bo3=round(abl_bo3 - base_bo3, 4),
                matchup_deltas={k: round(v, 4) for k, v in mu_deltas.items()},
            )
        )

    # Weakest first: removing them hurts least (or helps) → highest delta_bo3
    rows.sort(key=lambda r: (r.delta_bo3, r.delta_bo1), reverse=True)

    return CoreAblationReport(
        deck_name=full_deck.name,
        core_card_count=core_total,
        flex_slots=flex_slots,
        games_per_matchup=games_per_matchup,
        meta_opponents=list(meta_decks.keys()),
        baseline_bo1=round(base_bo1, 4),
        baseline_bo3=round(base_bo3, 4),
        replace_card=replace_card_name,
        rows=rows,
    )


def write_ablation_json(report: CoreAblationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
