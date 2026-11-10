"""Higher-sample head-to-head of top candidate lists vs week meta."""
from __future__ import annotations

import json
from pathlib import Path

from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import get_metagame_shares, get_reference_meta_decks
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator

ROOT = Path(__file__).resolve().parents[1]
USER = Deck.from_dreamborn_text(
    (ROOT / "decks" / "user_amber_emerald_princess.txt").read_text(encoding="utf-8"),
    name="User",
)


def swap(base: Deck, cuts, adds, name: str) -> Deck:
    counts = dict(base.card_counts)
    for nm, n in cuts:
        c = GLOBAL_CARD_DB.get_by_name(nm)
        counts[c.card_id] = counts.get(c.card_id, 0) - n
        if counts[c.card_id] <= 0:
            del counts[c.card_id]
    for nm, n in adds:
        c = GLOBAL_CARD_DB.get_by_name(nm)
        counts[c.card_id] = counts.get(c.card_id, 0) + n
    assert sum(counts.values()) == 60
    return Deck(name=name, card_counts=counts, archetype_tag="Amber/Emerald", db=GLOBAL_CARD_DB)


def eval_all(deck: Deck, meta, shares, games: int, seed: int):
    ev = MatchupEvaluator(lookahead_depth=1)
    share_sum = sum(shares.get(k, 0.0) for k in meta) or 1.0
    weighted = 0.0
    rows = {}
    for opp_name, opp in meta.items():
        st = ev.evaluate_matchup(deck, opp, num_games=games, seed=seed)
        share = shares.get(opp_name, 0.0) / share_sum
        rows[opp_name] = {
            "wr": st.overall_winrate,
            "play": st.play_winrate,
            "draw": st.draw_winrate,
            "share": share,
        }
        weighted += st.overall_winrate * share
    return {"meta_wr": weighted, "matchups": rows}


def main() -> None:
    meta = get_reference_meta_decks(top_n=8)
    shares = get_metagame_shares(top_n=8)
    candidates = {
        "user": USER,
        "finick2": swap(
            USER,
            [("Mickey Mouse - Amber Champion", 2)],
            [("Finnick - Tiny Terror", 2)],
            "Finnick2",
        ),
        "week_like_champ2": swap(
            USER,
            [
                ("Mickey Mouse - Amber Champion", 1),
                ("Nani - Stage Manager", 1),
                ("Rapunzel - Tower Defender", 1),
            ],
            [
                ("Mickey Mouse - Best in Town", 1),
                ("Mickey Mouse - Expedition Leader", 1),
                ("Lionheart - Incumbent Mayor", 1),
            ],
            "WeekLike",
        ),
        "finick2_bit1": swap(
            USER,
            [("Mickey Mouse - Amber Champion", 2), ("Rapunzel - Tower Defender", 1)],
            [("Finnick - Tiny Terror", 2), ("Mickey Mouse - Best in Town", 1)],
            "Finnick2+BiT",
        ),
        "recommended": swap(
            USER,
            [
                ("Mickey Mouse - Amber Champion", 1),
                ("Rapunzel - Tower Defender", 1),
                ("Nani - Stage Manager", 1),
            ],
            [
                ("Finnick - Tiny Terror", 2),
                ("Mickey Mouse - Best in Town", 1),
            ],
            "Recommended",
        ),
    }
    out = {}
    for name, deck in candidates.items():
        print(f"H2H {name}...")
        # Average two seeds for stability
        a = eval_all(deck, meta, shares, games=40, seed=111)
        b = eval_all(deck, meta, shares, games=40, seed=777)
        meta_wr = 0.5 * (a["meta_wr"] + b["meta_wr"])
        merge = {}
        for opp in a["matchups"]:
            merge[opp] = {
                "wr": 0.5 * (a["matchups"][opp]["wr"] + b["matchups"][opp]["wr"]),
                "play": 0.5 * (a["matchups"][opp]["play"] + b["matchups"][opp]["play"]),
                "draw": 0.5 * (a["matchups"][opp]["draw"] + b["matchups"][opp]["draw"]),
                "share": a["matchups"][opp]["share"],
            }
        out[name] = {"meta_wr": meta_wr, "matchups": merge}
        print(f"  meta_wr={meta_wr*100:.1f}%")
    Path("scratch/princess_h2h.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ranked = sorted(out.items(), key=lambda kv: kv[1]["meta_wr"], reverse=True)
    print("FINAL RANKING")
    for name, v in ranked:
        print(f"  {v['meta_wr']*100:5.1f}%  {name}")


if __name__ == "__main__":
    main()
