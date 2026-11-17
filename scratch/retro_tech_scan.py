"""
Targeted tech scan for user Emerald/Ruby Retro Device vs week top matchups.
Forces candidate pool toward Action Retro centroid + common Emerald/Ruby techs.
Python DSL only (use_full_dsl_sim=True).
"""
from __future__ import annotations

import json
from pathlib import Path

from lorcana_ai.config import RESULTS_DIR
from lorcana_ai.data.card_db import GLOBAL_CARD_DB as db
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import get_metagame_shares, get_reference_meta_decks
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator
from lorcana_ai.deckbuilder.tech_scanner import TechCardScanner

FORCE_TECHS = [
    "Piercing Attack",
    "Mother Knows Best",
    "Max Goof - Rebellious Teen",
    "The Leviathan - Guardian of Atlantis",
    "Prince Phillip - Vanquisher of Foes",
    "Strange Things",
    "Rivera Family Photo",
    "Roller Bob - Sid's Toy",
    "White Rabbit - Late Again",
    "Sid Phillips - Toy Surgeon",
    "To Wither a Flower",
    "Everyone Knows Juanita",
    "Under the Sea",
    "Second Star to the Right",
    "Don Karnage - Khan's Courier",
    "Aladdin - Prince Ali",
    "Creative Inspiration",
]


def swap_copies(deck: Deck, out_name: str, in_name: str, copies: int) -> Deck | None:
    out_c = db.get_by_name(out_name)
    in_c = db.get_by_name(in_name)
    if not out_c or not in_c:
        return None
    counts = dict(deck.card_counts)
    if counts.get(out_c.card_id, 0) < copies:
        return None
    if counts.get(in_c.card_id, 0) + copies > 4:
        copies = 4 - counts.get(in_c.card_id, 0)
        if copies <= 0:
            return None
        if counts.get(out_c.card_id, 0) < copies:
            return None
    counts[out_c.card_id] -= copies
    if counts[out_c.card_id] <= 0:
        del counts[out_c.card_id]
    counts[in_c.card_id] = counts.get(in_c.card_id, 0) + copies
    out = deck.clone(new_name=f"{deck.name} +{copies} {in_name} -{copies} {out_name}")
    out.card_counts = counts
    return out


def main() -> None:
    base = Deck.from_dreamborn_text(
        Path("decks/user_emerald_ruby_retro_device.txt").read_text(encoding="utf-8"),
        name="User Emerald/Ruby Retro",
    )
    core = Deck.from_dreamborn_text(
        Path("decks/user_emerald_ruby_retro_device_core48.txt").read_text(encoding="utf-8"),
        name="Core48",
    )
    locked = set(core.card_counts.keys())

    meta = get_reference_meta_decks(top_n=8)
    shares = get_metagame_shares(top_n=8)
    ev = MatchupEvaluator(use_full_dsl_sim=True, lookahead_depth=1, workers=1)
    scanner = TechCardScanner(evaluator=ev, lookahead_depth=1, workers=1)

    # --- UCB scan vs top 3 MUs ---
    top_ops = list(meta.items())[:3]
    scan_rows = []
    for opp_name, opp in top_ops:
        print(f"\n=== scan-tech vs {opp_name} ===")
        recs = scanner.scan_for_matchup(
            base_deck=base,
            opponent_deck=opp,
            copies_to_swap=2,
            total_simulation_budget=64,
            top_k=5,
            locked_core_ids=locked,
        )
        for i, r in enumerate(recs, 1):
            print(
                f"{i}. +{r.copies_swapped}x {r.card_in_name} (cut {r.card_out_name}) "
                f"BO3={r.new_bo3_winrate*100:.1f}% d={r.delta_bo3_winrate*100:+.1f}%"
            )
            scan_rows.append(
                {
                    "opp": opp_name,
                    "card_in": r.card_in_name,
                    "card_out": r.card_out_name,
                    "copies": r.copies_swapped,
                    "bo3": r.new_bo3_winrate,
                    "delta_bo3": r.delta_bo3_winrate,
                    "rationale": r.tactical_rationale,
                }
            )

    # --- Forced package probes (meta Action Retro deltas) ---
    cuts_priority = [
        "Sid Phillips - Toy Surgeon",
        "Strange Things",
        "The Torn Corner",
        "White Rabbit - Late Again",
        "Prince Phillip - Vanquisher of Foes",
        "Rivera Family Photo",
        "Strike a Good Match",
    ]
    packages = [
        ("+2 Piercing Attack", [("The Torn Corner", "Piercing Attack", 2)]),
        ("+2 Mother Knows Best", [("Strange Things", "Mother Knows Best", 2)]),
        ("+1 Rebellious Teen", [("Sid Phillips - Toy Surgeon", "Max Goof - Rebellious Teen", 1)]),
        ("+1 Leviathan", [("Sid Phillips - Toy Surgeon", "The Leviathan - Guardian of Atlantis", 1)]),
        ("+1 Phillip", [("Strange Things", "Prince Phillip - Vanquisher of Foes", 1)]),
        (
            "Action-like flex",
            [
                ("White Rabbit - Late Again", "Piercing Attack", 2),
                ("Sid Phillips - Toy Surgeon", "Max Goof - Rebellious Teen", 1),
                ("Strange Things", "Mother Knows Best", 1),
                ("Rivera Family Photo", "The Leviathan - Guardian of Atlantis", 1),
                ("The Torn Corner", "Prince Phillip - Vanquisher of Foes", 1),
            ],
        ),
        (
            "GS hate keep Rabbit",
            [
                ("Sid Phillips - Toy Surgeon", "Piercing Attack", 1),
                ("Strange Things", "Piercing Attack", 1),
                ("The Torn Corner", "Mother Knows Best", 2),
            ],
        ),
    ]

    print("\n=== baseline meta gauntlet ===")
    base_bo1, base_bo3 = _meta_wr(ev, base, meta, shares, games=20, seed=2137)
    print(f"Baseline BO1={base_bo1*100:.1f}% BO3={base_bo3*100:.1f}%")

    pkg_rows = []
    for label, swaps in packages:
        d = base
        ok = True
        for out_n, in_n, n in swaps:
            nd = swap_copies(d, out_n, in_n, n)
            if nd is None:
                ok = False
                break
            d = nd
        if not ok:
            print(f"SKIP {label}")
            continue
        bo1, bo3 = _meta_wr(ev, d, meta, shares, games=20, seed=2137)
        print(
            f"{label}: BO1={bo1*100:.1f}% ({(bo1-base_bo1)*100:+.1f}) "
            f"BO3={bo3*100:.1f}% ({(bo3-base_bo3)*100:+.1f})"
        )
        pkg_rows.append(
            {
                "label": label,
                "swaps": [{"out": a, "in": b, "n": c} for a, b, c in swaps],
                "bo1": bo1,
                "bo3": bo3,
                "d_bo1": bo1 - base_bo1,
                "d_bo3": bo3 - base_bo3,
            }
        )

    out = {
        "baseline_bo1": base_bo1,
        "baseline_bo3": base_bo3,
        "scan_rows": scan_rows,
        "packages": pkg_rows,
        "note": "Absolute WRs low: Retro Device/song effects mostly empty in CompiledAbilities; use relative Δ + meta centroid.",
    }
    path = RESULTS_DIR / "retro_device_week_tech.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {path}")


def _meta_wr(ev, deck, meta, shares, games=20, seed=2137):
    total_share = sum(max(0.0, shares.get(n, 0.0)) for n in meta) or 1.0
    bo1 = bo3 = 0.0
    for name, opp in meta.items():
        st = ev.evaluate_matchup(deck, opp, num_games=games, seed=seed)
        w = shares.get(name, 0.0) / total_share
        bo1 += w * st.overall_winrate
        bo3 += w * st.bo3_winrate
    return bo1, bo3


if __name__ == "__main__":
    main()
