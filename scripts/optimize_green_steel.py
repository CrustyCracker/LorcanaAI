#!/usr/bin/env python3
"""
Skrypt do optymalizacji i testowania talii Emerald/Steel (Green Steel)
z 40-kartowym zablokowanym rdzeniem (locked core) i 20 slotami flex/tech.

Porównuje wersję wyjściową (najbardziej consistent baseline 60 kart)
z kandydatami techowymi oraz generuje i ocenia pakiety kombinacji techów
przeciwko reprezentatywnej gauntletce metagry Core Constructed (Sets 9-14) z Duels.ink.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lorcana_ai.config import LOG_DIR, RESULTS_DIR
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import (
    get_green_steel_core_counts,
    get_green_steel_baseline_deck,
    get_metagame_shares,
    get_reference_meta_decks,
)
from lorcana_ai.deckbuilder.tech_scanner import TechCardScanner
from lorcana_ai.engine.game_replay import simulate_and_format_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optymalizator slotów flex/tech dla Emerald/Steel Green Steel vs Duels.ink Meta"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=12,
        help="Liczba gier na każdy matchup w gauntletce (domyślnie: 12)",
    )
    parser.add_argument(
        "--packages",
        type=int,
        default=10,
        help="Liczba kombinacji pakietów flex do przetestowania (domyślnie: 10)",
    )
    parser.add_argument(
        "--top-meta",
        type=int,
        default=8,
        help="Liczba metowych talii z Duels.ink w gauntletce (domyślnie: 8)",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=1,
        help="Głębokość analizy taktycznej agenta heurystycznego (domyślnie: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Liczba procesów roboczych CPU (0 = auto-detekcja)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Ziarno losowości dla Common Random Numbers (domyślnie: 2026)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    core_counts = get_green_steel_core_counts(GLOBAL_CARD_DB)
    baseline_deck = get_green_steel_baseline_deck(GLOBAL_CARD_DB)
    meta_decks = get_reference_meta_decks(top_n=args.top_meta)
    meta_shares = get_metagame_shares(top_n=args.top_meta)

    locked_total = sum(core_counts.values())
    flex_needed = max(0, 60 - locked_total)
    workers_arg = args.workers if args.workers > 0 else None

    print("=" * 105)
    print("LORCANA ALPHA — TESTOWANIE I OPTYMALIZACJA GREEN STEEL (EMERALD / STEEL)")
    print("=" * 105)
    print(f"Rdzeń talii (Locked Core): {locked_total} kart (zablokowane)")
    print(f"Sloty elastyczne (Flex/Tech Slots): {flex_needed} kart")
    print(f"Talia bazowa (Most Consistent Baseline): {baseline_deck.name} ({baseline_deck.total_cards} kart, Uninkables: {baseline_deck.uninkable_count})")
    print(f"Meta gauntlet: Top {len(meta_decks)} talii z Duels.ink (Core Set 14)")
    print(f"Parametry: games_per_matchup={args.games}, packages={args.packages}, depth={args.lookahead_depth}")
    print("=" * 105)

    scanner = TechCardScanner(
        workers=workers_arg,
        lookahead_depth=args.lookahead_depth,
    )

    print("\n[1/3] Ewaluacja kandydatów na karty techowe vs Meta Gauntlet...")
    report = scanner.optimize_core_for_duels_ink_meta(
        core_card_counts=core_counts,
        meta_decks=meta_decks,
        meta_shares=meta_shares,
        archetype_name="Emerald/Steel Green Steel",
        games_per_matchup=args.games,
        package_candidates=args.packages,
        seed=args.seed,
        baseline_deck=baseline_deck,
    )

    print("\n" + "=" * 105)
    print(f"RANKING INDYWIDUALNYCH KART TECHOWYCH (WPŁYW NA WINRATE VS BAZA)")
    print("=" * 105)
    for idx, te in enumerate(report.top_tech_cards[:12], 1):
        ink_str = "Inkable" if te.inkable else "Uninkable"
        s_bo1 = "+" if te.delta_bo1_winrate >= 0 else ""
        s_bo3 = "+" if te.delta_bo3_winrate >= 0 else ""
        deltas_parts = []
        for opp_nm, d_val in te.matchup_deltas.items():
            short_opp = opp_nm.replace("Midrange", "Mid").replace("Control", "Ctrl").replace("Good Stuff", "GS")
            d_sign = "+" if d_val >= 0 else ""
            w_val = te.matchup_winrates.get(opp_nm, 0.0)
            deltas_parts.append(f"{short_opp}: {w_val*100:.0f}% ({d_sign}{d_val*100:.1f}%)")
        deltas_line = " | ".join(deltas_parts)
        print(
            f"{idx:2d}. {te.card_name} (Koszt: {te.cost}, {ink_str}, {te.card_type}) — Zalecane: {te.recommended_copies}x\n"
            f"    Meta BO1 WR: {te.expected_bo1_winrate*100:.1f}% ({s_bo1}{te.delta_bo1_winrate*100:.1f}% vs baza) | "
            f"BO3 WR: {te.expected_bo3_winrate*100:.1f}% ({s_bo3}{te.delta_bo3_winrate*100:.1f}% vs baza) | "
            f"Play: {te.play_winrate*100:.1f}% | Draw: {te.draw_winrate*100:.1f}%\n"
            f"    Wpływ % na matchupy: {deltas_line}\n"
            f"    Rola: {te.tactical_role}"
        )

    if report.top_flex_packages:
        print("\n" + "=" * 105)
        print(f"TOP {len(report.top_flex_packages)} NAJLEPSZYCH KOMBINACJI TECHÓW (PAKIETY FLEX) DLA GREEN STEEL")
        print("=" * 105)
        for pkg in report.top_flex_packages:
            ps1 = "+" if pkg.delta_bo1_winrate >= 0 else ""
            ps3 = "+" if pkg.delta_bo3_winrate >= 0 else ""
            best_tag = " ★ [WYBÓR #1 - NAJLEPSZY PAKIET]" if pkg.rank == 1 else ""
            print(
                f"\n[KOMBINACJA #{pkg.rank}]{best_tag} {pkg.package_name}\n"
                f"  Meta BO1 WR: {pkg.expected_bo1_winrate*100:5.1f}% ({ps1}{pkg.delta_bo1_winrate*100:5.1f}% vs baza) | "
                f"BO3 WR: {pkg.expected_bo3_winrate*100:5.1f}% ({ps3}{pkg.delta_bo3_winrate*100:5.1f}% vs baza)\n"
                f"  Play WR: {pkg.play_winrate*100:5.1f}% | Draw WR: {pkg.draw_winrate*100:5.1f}% | "
                f"Uninkables: {pkg.uninkable_count:2d} | Średni koszt: {pkg.average_cost:.2f}\n"
                f"  Karty pakietu ({sum(pkg.flex_cards.values())} kart): "
                f"{', '.join(f'+{cnt}x {cname}' for cname, cnt in pkg.flex_cards.items())}\n"
                f"  Rola taktyczna: {pkg.tactical_summary}"
            )

    tot_d_bo1 = report.optimized_meta_bo1_wr - report.baseline_meta_bo1_wr
    tot_d_bo3 = report.optimized_meta_bo3_wr - report.baseline_meta_bo3_wr
    s_tot1 = "+" if tot_d_bo1 >= 0 else ""
    s_tot3 = "+" if tot_d_bo3 >= 0 else ""

    print("\n" + "=" * 105)
    print(
        f"SZCZEGÓŁY NAJLEPSZEJ KOMBINACJI (#1) — PAKIET {report.flex_slots_count} KART TECH/FLEX\n"
        f"(BO1 Meta WR: {report.optimized_meta_bo1_wr*100:.1f}% [{s_tot1}{tot_d_bo1*100:.1f}% vs baza {report.baseline_meta_bo1_wr*100:.1f}%] | "
        f"BO3 Meta WR: {report.optimized_meta_bo3_wr*100:.1f}% [{s_tot3}{tot_d_bo3*100:.1f}% vs baza {report.baseline_meta_bo3_wr*100:.1f}%])"
    )
    print("=" * 105)
    for cname, cnt in report.selected_flex_package.items():
        print(f"  + {cnt}x {cname}")

    print("-" * 105)
    print("SZCZEGÓŁOWY WPŁYW PROCENTOWY NA WINRATE W KAŻDYM MATCHUPIE NA DUELS.INK (CORE SET 14):")
    for opp_name, m_info in report.matchup_breakdown.items():
        d1 = m_info.get("delta_bo1_winrate", 0.0)
        d3 = m_info.get("delta_bo3_winrate", 0.0)
        b1 = m_info.get("baseline_bo1_winrate", m_info["bo1_winrate"])
        s1 = "+" if d1 >= 0 else ""
        s3 = "+" if d3 >= 0 else ""
        print(
            f"  vs {opp_name:<33} (Share: {m_info['meta_share']*100:4.1f}%) -> "
            f"BO1: {m_info['bo1_winrate']*100:5.1f}% ({s1}{d1*100:5.1f}% vs baza {b1*100:4.1f}%) | "
            f"BO3: {m_info['bo3_winrate']*100:5.1f}% ({s3}{d3*100:5.1f}%) | "
            f"Play: {m_info['play_winrate']*100:5.1f}% | Draw: {m_info['draw_winrate']*100:5.1f}%"
        )

    print("-" * 105)
    print(
        f"GOTOWA TALIA 60 KART ({report.optimized_deck.name} | "
        f"Uninkables: {report.optimized_deck.uninkable_count} | Avg Cost: {report.optimized_deck.average_cost:.2f}):"
    )
    print(report.optimized_deck.to_dreamborn_text())
    print("=" * 105)

    deck_out_path = PROJECT_ROOT / "decks" / "emerald_steel_green_steel_optimized.txt"
    deck_out_path.parent.mkdir(parents=True, exist_ok=True)
    deck_out_path.write_text(report.optimized_deck.to_dreamborn_text() + "\n", encoding="utf-8")

    json_out_path = RESULTS_DIR / "green_steel_optimized_report.json"
    json_payload = {
        "archetype_name": report.archetype_name,
        "locked_core_count": report.locked_core_count,
        "flex_slots_count": report.flex_slots_count,
        "baseline_meta_bo1_wr": report.baseline_meta_bo1_wr,
        "baseline_meta_bo3_wr": report.baseline_meta_bo3_wr,
        "optimized_meta_bo1_wr": report.optimized_meta_bo1_wr,
        "optimized_meta_bo3_wr": report.optimized_meta_bo3_wr,
        "delta_meta_bo1_wr": round(tot_d_bo1, 4),
        "delta_meta_bo3_wr": round(tot_d_bo3, 4),
        "optimized_play_wr": report.optimized_play_wr,
        "optimized_draw_wr": report.optimized_draw_wr,
        "selected_flex_package": report.selected_flex_package,
        "top_flex_packages": [
            {
                "rank": p.rank,
                "package_name": p.package_name,
                "flex_cards": p.flex_cards,
                "expected_bo1_winrate": p.expected_bo1_winrate,
                "expected_bo3_winrate": p.expected_bo3_winrate,
                "delta_bo1_winrate": p.delta_bo1_winrate,
                "delta_bo3_winrate": p.delta_bo3_winrate,
                "play_winrate": p.play_winrate,
                "draw_winrate": p.draw_winrate,
                "uninkable_count": p.uninkable_count,
                "average_cost": p.average_cost,
                "tactical_summary": p.tactical_summary,
                "dreamborn_decklist": p.deck.to_dreamborn_text(),
            }
            for p in getattr(report, "top_flex_packages", [])
        ],
        "top_tech_cards": [te.__dict__ for te in report.top_tech_cards],
        "matchup_breakdown": report.matchup_breakdown,
        "dreamborn_decklist": report.optimized_deck.to_dreamborn_text(),
    }
    json_out_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_replays: list[str] = []
    top_opps = list(meta_decks.items())[:2]
    for o_idx, (_opp_nm, opp_d) in enumerate(top_opps):
        for sp in (0, 1):
            sample_replays.append(
                simulate_and_format_replay(
                    deck_a=report.optimized_deck,
                    deck_b=opp_d,
                    starting_player=sp,
                    seed=args.seed + o_idx * 101 + sp * 37,
                    lookahead_depth=max(1, args.lookahead_depth),
                )
            )
    replay_path = LOG_DIR / "green_steel_sample_games.log"
    replay_path.write_text("\n\n".join(sample_replays) + "\n", encoding="utf-8")

    print(f"\n[OK] Zapisano zoptymalizowaną talię do: {deck_out_path}")
    print(f"[OK] Zapisano pełny raport JSON do: {json_out_path}")
    print(f"[OK] Zapisano 4 pełne zapisy rozgrywek (Play-by-Play Replay) do: {replay_path}")


if __name__ == "__main__":
    main()
