"""Second-pass tech hunt around Finnick / Percy / Best-in-Town winners."""
from __future__ import annotations

import json
from pathlib import Path

from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import get_metagame_shares, get_reference_meta_decks
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator

ROOT = Path(__file__).resolve().parents[1]
USER_DECK = ROOT / "decks" / "user_amber_emerald_princess.txt"
OUT = ROOT / "scratch" / "princess_refine_results.json"


def load_deck(path: Path, name: str) -> Deck:
    return Deck.from_dreamborn_text(path.read_text(encoding="utf-8"), name=name)


def swap_copies(base: Deck, cuts: list[tuple[str, int]], adds: list[tuple[str, int]], name: str) -> Deck:
    counts = dict(base.card_counts)
    db = GLOBAL_CARD_DB
    for nm, n in cuts:
        card = db.get_by_name(nm)
        assert card, nm
        counts[card.card_id] = counts.get(card.card_id, 0) - n
        if counts[card.card_id] <= 0:
            del counts[card.card_id]
    for nm, n in adds:
        card = db.get_by_name(nm)
        assert card, nm
        counts[card.card_id] = counts.get(card.card_id, 0) + n
        assert counts[card.card_id] <= 4, (nm, counts[card.card_id])
    assert sum(counts.values()) == 60, sum(counts.values())
    return Deck(name=name, card_counts=counts, archetype_tag="Amber/Emerald", db=db)


def one_drop_count(deck: Deck) -> int:
    total = 0
    for cid, n in deck.card_counts.items():
        c = GLOBAL_CARD_DB.by_id[cid]
        if c.card_type == "Character" and c.cost == 1:
            total += n
    return total


def evaluate_deck(deck: Deck, meta: dict[str, Deck], shares: dict[str, float], games: int, seed: int) -> dict:
    ev = MatchupEvaluator(lookahead_depth=1)
    results = {}
    weighted = 0.0
    share_sum = sum(shares.get(k, 0.0) for k in meta) or 1.0
    for opp_name, opp in meta.items():
        stats = ev.evaluate_matchup(deck, opp, num_games=games, seed=seed)
        wr = stats.overall_winrate
        share = shares.get(opp_name, 0.0) / share_sum
        results[opp_name] = {
            "wr": wr,
            "play_wr": stats.play_winrate,
            "draw_wr": stats.draw_winrate,
            "share": share,
            "games": stats.games_played,
        }
        weighted += wr * share
    return {"matchups": results, "meta_wr": weighted, "one_drops": one_drop_count(deck)}


def main() -> None:
    user = load_deck(USER_DECK, "User Princess")
    meta = get_reference_meta_decks(top_n=8)
    shares = get_metagame_shares(top_n=8)

    variants = {
        "user_baseline": user,
        "finick2_cut_champ2": swap_copies(
            user,
            cuts=[("Mickey Mouse - Amber Champion", 2)],
            adds=[("Finnick - Tiny Terror", 2)],
            name="Finnick2",
        ),
        "finick2_bit1_cut_champ2_rap1": swap_copies(
            user,
            cuts=[("Mickey Mouse - Amber Champion", 2), ("Rapunzel - Tower Defender", 1)],
            adds=[("Finnick - Tiny Terror", 2), ("Mickey Mouse - Best in Town", 1)],
            name="Finnick2+BiT",
        ),
        "finick3_cut_champ2_nani1": swap_copies(
            user,
            cuts=[("Mickey Mouse - Amber Champion", 2), ("Nani - Stage Manager", 1)],
            adds=[("Finnick - Tiny Terror", 3)],
            name="Finnick3",
        ),
        "percy2_finick2_cut_champ2_rap1_nani1": swap_copies(
            user,
            cuts=[
                ("Mickey Mouse - Amber Champion", 2),
                ("Rapunzel - Tower Defender", 1),
                ("Nani - Stage Manager", 1),
            ],
            adds=[("Percy - Pupsicle", 2), ("Finnick - Tiny Terror", 2)],
            name="Percy2+Finnick2",
        ),
        "week_like_keep_champ2": swap_copies(
            user,
            cuts=[
                ("Mickey Mouse - Amber Champion", 1),
                ("Nani - Stage Manager", 1),
                ("Rapunzel - Tower Defender", 1),
            ],
            adds=[
                ("Mickey Mouse - Best in Town", 1),
                ("Mickey Mouse - Expedition Leader", 1),
                ("Lionheart - Incumbent Mayor", 1),
            ],
            name="WeekLike Champ2",
        ),
        "bobby4_finick2_cut_champ2_rap1": swap_copies(
            user,
            cuts=[("Mickey Mouse - Amber Champion", 2), ("Rapunzel - Tower Defender", 1)],
            adds=[("Bobby Zimuruski - Spray Cheese Kid", 1), ("Finnick - Tiny Terror", 2)],
            name="Bobby4+Finnick2",
        ),
        "prince2_cut_champ2": swap_copies(
            user,
            cuts=[("Mickey Mouse - Amber Champion", 2)],
            adds=[("The Prince - Searching for His Love", 2)],
            name="Prince2",
        ),
    }

    report = {"variants": {}}
    for name, deck in variants.items():
        print(f"Evaluating {name} (1-drops={one_drop_count(deck)})...")
        report["variants"][name] = evaluate_deck(deck, meta, shares, games=32, seed=4242)
        print(f"  meta_wr={report['variants'][name]['meta_wr']*100:.1f}%")

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
