#!/usr/bin/env python3
"""
Skrypt do optymalizacji i testowania talii Emerald/Steel (Green Steel)
wyłącznie w mirrorze (Green Steel Mirror Match).

Gracz 2 (przeciwnik) zawsze gra stałą, bazową listą 60 kart (15 kart po 4 kopie).
Gracz 1 zachowuje zablokowany 40-kartowy rdzeń i modyfikuje 20 slotów flex/tech.
Faza 1: Testowanie pojedynczych kart techowych w mirrorze.
Faza 2: Testowanie pełnych 20-kartowych pakietów flex w mirrorze.
Faza 3: Generowanie szczegółowych zapisów rozgrywki turn-by-turn.
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
from lorcana_ai.deckbuilder.mirror_optimizer import GreenSteelMirrorOptimizer
from lorcana_ai.engine.game_replay import simulate_and_format_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optymalizator kart techowych i pakietów flex w mirrorze Green/Steel (Emerald/Steel)"
    )
    parser.add_argument(
        "--games-tech",
        type=int,
        default=60,
        help="Liczba gier na każdego kandydata techowego w Fazie 1 (domyślnie: 60)",
    )
    parser.add_argument(
        "--games-packages",
        type=int,
        default=100,
        help="Liczba gier na każdy 20-kartowy pakiet flex w Fazie 2 (domyślnie: 100)",
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
        default=2137,
        help="Ziarno losowości dla Common Random Numbers (domyślnie: 2137)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers_arg = args.workers if args.workers > 0 else None

    print("=" * 105)
    print("LORCANA ALPHA — TESTOWANIE I OPTYMALIZACJA GREEN STEEL W MIRRORZE (EMERALD / STEEL)")
    print("=" * 105)
    print("Konfiguracja rozgrywki:")
    print("  Gracz 2 (Rywal):  Zablokowana lista bazowa 60 kart (15 kart po 4 sztuki)")
    print("  Gracz 1 (Test):   Zablokowany 40-kartowy rdzeń + 20 modyfikowanych slotów flex/tech")
    print(f"  Gry Faza 1 (Tech): {args.games_tech} gier na kartę techową")
    print(f"  Gry Faza 2 (Pkg):  {args.games_packages} gier na pakiet flex")
    print(f"  Głębokość agenta:  depth={args.lookahead_depth} | Ziarno CRN: {args.seed}")
    print("=" * 105)

    optimizer = GreenSteelMirrorOptimizer(
        db=GLOBAL_CARD_DB,
        workers=workers_arg,
        lookahead_depth=args.lookahead_depth,
    )

    report = optimizer.run_full_mirror_optimization(
        num_games_tech=args.games_tech,
        num_games_packages=args.games_packages,
        seed=args.seed,
    )

    print("\n" + "=" * 105)
    print("BAZOWY BENCHMARK W MIRRORZE (Wersja Wyjściowa vs Wersja Wyjściowa):")
    print(
        f"  BO1 Winrate: {report.baseline_bo1_winrate*100:5.1f}% | "
        f"BO3 Winrate: {report.baseline_bo3_winrate*100:5.1f}% | "
        f"Play WR: {report.baseline_play_winrate*100:5.1f}% | "
        f"Draw WR: {report.baseline_draw_winrate*100:5.1f}%"
    )
    print("=" * 105)

    print("\n" + "=" * 105)
    print("FAZA 1: RANKING INDYWIDUALNYCH KART TECHOWYCH W MIRRORZE GREEN/STEEL")
    print("=" * 105)
    for idx, te in enumerate(report.top_tech_cards, 1):
        ink_str = "Inkable" if te.inkable else "Uninkable"
        s1 = "+" if te.delta_bo1_winrate >= 0 else ""
        s3 = "+" if te.delta_bo3_winrate >= 0 else ""
        tag = "[Rdzeń -> 4x]" if te.is_core_tech else "[Tech Zewn.]"
        print(
            f"{idx:2d}. {tag} {te.card_name} ({te.cost} ink, {ink_str}, {te.card_type}) — Testowano: {te.copies_tested}x\n"
            f"    Mirror BO1 WR: {te.bo1_winrate*100:5.1f}% ({s1}{te.delta_bo1_winrate*100:5.1f}% vs baza) | "
            f"BO3 WR: {te.bo3_winrate*100:5.1f}% ({s3}{te.delta_bo3_winrate*100:5.1f}%) | "
            f"Play: {te.play_winrate*100:5.1f}% | Draw: {te.draw_winrate*100:5.1f}%\n"
            f"    Rola w mirrorze: {te.mirror_tactical_role}"
        )

    print("\n" + "=" * 105)
    print(f"FAZA 2: RANKING {len(report.top_flex_packages)} PAKIETÓW FLEX (20 KART) W MIRRORZE GREEN/STEEL")
    print("=" * 105)
    for pkg in report.top_flex_packages:
        ps1 = "+" if pkg.delta_bo1_winrate >= 0 else ""
        ps3 = "+" if pkg.delta_bo3_winrate >= 0 else ""
        best_tag = " ★ [WYBÓR #1 - NAJLEPSZY PAKIET MIRROROWY]" if pkg.rank == 1 else ""
        print(
            f"\n[PAKIET #{pkg.rank:2d}]{best_tag} {pkg.package_name}\n"
            f"  Mirror BO1 WR: {pkg.bo1_winrate*100:5.1f}% ({ps1}{pkg.delta_bo1_winrate*100:5.1f}% vs baza) | "
            f"BO3 WR: {pkg.bo3_winrate*100:5.1f}% ({ps3}{pkg.delta_bo3_winrate*100:5.1f}%)\n"
            f"  Play WR: {pkg.play_winrate*100:5.1f}% | Draw WR: {pkg.draw_winrate*100:5.1f}% | "
            f"Uninkables: {pkg.uninkable_count:2d} | Średni koszt: {pkg.average_cost:.2f}\n"
            f"  Karty pakietu ({sum(pkg.flex_cards.values())} kart): "
            f"{', '.join(f'+{cnt}x {cname}' for cname, cnt in pkg.flex_cards.items())}\n"
            f"  Rola taktyczna: {pkg.tactical_summary}"
        )

    tot_d1 = report.optimized_bo1_winrate - report.baseline_bo1_winrate
    tot_d3 = report.optimized_bo3_winrate - report.baseline_bo3_winrate
    s_tot1 = "+" if tot_d1 >= 0 else ""
    s_tot3 = "+" if tot_d3 >= 0 else ""

    print("\n" + "=" * 105)
    print(
        f"SZCZEGÓŁY ZWYCIĘSKIEJ TALII (#1) W MIRRORZE GREEN/STEEL\n"
        f"(BO1 Mirror WR: {report.optimized_bo1_winrate*100:.1f}% [{s_tot1}{tot_d1*100:.1f}% vs baza {report.baseline_bo1_winrate*100:.1f}%] | "
        f"BO3 Mirror WR: {report.optimized_bo3_winrate*100:.1f}% [{s_tot3}{tot_d3*100:.1f}% vs baza {report.baseline_bo3_winrate*100:.1f}%])"
    )
    print("=" * 105)
    for cname, cnt in report.selected_flex_package.items():
        print(f"  + {cnt}x {cname}")

    print("-" * 105)
    print(
        f"GOTOWA TALIA 60 KART ({report.optimized_deck.name} | "
        f"Uninkables: {report.optimized_deck.uninkable_count} | Avg Cost: {report.optimized_deck.average_cost:.2f}):"
    )
    print(report.optimized_deck.to_dreamborn_text())
    print("=" * 105)

    # File exports
    deck_out_path = PROJECT_ROOT / "decks" / "emerald_steel_green_steel_mirror_optimized.txt"
    deck_out_path.parent.mkdir(parents=True, exist_ok=True)
    deck_out_path.write_text(report.optimized_deck.to_dreamborn_text() + "\n", encoding="utf-8")

    json_out_path = RESULTS_DIR / "green_steel_mirror_optimized_report.json"
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "archetype_name": report.archetype_name,
        "locked_core_count": report.locked_core_count,
        "flex_slots_count": report.flex_slots_count,
        "baseline_bo1_winrate": report.baseline_bo1_winrate,
        "baseline_bo3_winrate": report.baseline_bo3_winrate,
        "baseline_play_winrate": report.baseline_play_winrate,
        "baseline_draw_winrate": report.baseline_draw_winrate,
        "optimized_bo1_winrate": report.optimized_bo1_winrate,
        "optimized_bo3_winrate": report.optimized_bo3_winrate,
        "optimized_play_winrate": report.optimized_play_winrate,
        "optimized_draw_winrate": report.optimized_draw_winrate,
        "delta_bo1_winrate": report.delta_bo1_winrate,
        "delta_bo3_winrate": report.delta_bo3_winrate,
        "selected_flex_package": report.selected_flex_package,
        "top_flex_packages": [
            {
                "rank": p.rank,
                "package_name": p.package_name,
                "flex_cards": p.flex_cards,
                "bo1_winrate": p.bo1_winrate,
                "bo3_winrate": p.bo3_winrate,
                "delta_bo1_winrate": p.delta_bo1_winrate,
                "delta_bo3_winrate": p.delta_bo3_winrate,
                "play_winrate": p.play_winrate,
                "draw_winrate": p.draw_winrate,
                "uninkable_count": p.uninkable_count,
                "average_cost": p.average_cost,
                "tactical_summary": p.tactical_summary,
                "dreamborn_decklist": p.deck.to_dreamborn_text(),
            }
            for p in report.top_flex_packages
        ],
        "top_tech_cards": [te.__dict__ for te in report.top_tech_cards],
        "dreamborn_decklist": report.optimized_deck.to_dreamborn_text(),
    }
    json_out_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate sample replays: 1 on the Play, 1 on the Draw
    replays: list[str] = []
    for sp, sp_label in ((0, "P1 ON THE PLAY"), (1, "P1 ON THE DRAW")):
        replays.append(
            f"=== MIRROR MATCH REPLAY ({sp_label}) ===\n"
            + simulate_and_format_replay(
                deck_a=report.optimized_deck,
                deck_b=report.baseline_deck,
                starting_player=sp,
                seed=args.seed + sp * 777,
                lookahead_depth=max(1, args.lookahead_depth),
            )
        )
    replay_path = LOG_DIR / "green_steel_mirror_sample_games.log"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text("\n\n" + ("=" * 80) + "\n\n".join(replays) + "\n", encoding="utf-8")

    print(f"\n[OK] Zapisano zoptymalizowaną talię do: {deck_out_path}")
    print(f"[OK] Zapisano raport JSON do: {json_out_path}")
    print(f"[OK] Zapisano zapisy rozgrywek (Play-by-Play Replay) do: {replay_path}")


if __name__ == "__main__":
    main()
