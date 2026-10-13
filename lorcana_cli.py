#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for LorcanaAlpha.
Used both locally on Windows/Linux and inside SLURM batch jobs on PLGrid Athena.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

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


from lorcana_ai.config import CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT, RESULTS_DIR, MCTSConfig
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import (
    DEFAULT_METAGAME_SHARES,
    EXPORTED_DECKS_DIR,
    get_archetype_core_and_baseline,
    get_metagame_shares,
    get_reference_meta_decks,
    sync_duels_ink_meta,
)
from lorcana_ai.deckbuilder.evolutionary_map import MAPElitesDeckbuilder
from lorcana_ai.deckbuilder.nash_solver import NashMetaSolver
from lorcana_ai.deckbuilder.tech_scanner import TechCardScanner
from lorcana_ai.engine.simulator import HAS_RUST_ENGINE


class _TeeLogger:
    """Mirrors stdout/stderr to both console and a file inside logs/."""

    def __init__(self, stream, log_file_path: Path) -> None:
        self._stream = stream
        self._log_file = log_file_path.open("a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, attr: str):
        return getattr(self._stream, attr)


def cmd_sync_cards(_args: argparse.Namespace) -> None:
    print("[LorcanaAlpha] Synchronizing card database from LorcanaJSON...")
    try:
        added = GLOBAL_CARD_DB.sync_from_lorcana_json()
        print(f"[OK] Added {added} new cards from LorcanaJSON. Total registered cards: {len(GLOBAL_CARD_DB.by_id)}")
    except Exception as exc:
        print(f"[WARN] Could not reach LorcanaJSON ({exc}). Using built-in Core Constructed card database ({len(GLOBAL_CARD_DB.by_id)} cards).")


def cmd_sync_meta(args: argparse.Namespace) -> None:
    print(f"[LorcanaAlpha] Scraping Core Set 14 metagame from https://duels.ink/stats/{args.queue} ...")
    profiles = sync_duels_ink_meta(queue=args.queue, period=args.period, era=args.era, export_txt_files=True)
    print("\n" + "=" * 95)
    print(f"SCRAPED DUELS.INK META ARCHETYPES ({args.queue.upper()} | Era: {args.era} | Period: {args.period})")
    print("=" * 95)
    for idx, prof in enumerate(profiles[:15], 1):
        sigs = ", ".join(prof.signature_cards[:3])
        print(
            f"{idx:2d}. {prof.name:<34} | Games: {prof.games_played:5d} | "
            f"Winrate: {prof.win_rate:5.2f}% | Share: {prof.meta_share*100:5.1f}% | Key: {sigs}"
        )
    print("=" * 95)
    print(f"[OK] Exported {len(profiles)} Dreamborn-format .txt decklists to {EXPORTED_DECKS_DIR}")


def cmd_optimize_core(args: argparse.Namespace) -> None:
    from lorcana_ai.engine.game_replay import simulate_and_format_replay

    core_counts, baseline_deck = get_archetype_core_and_baseline(args.archetype)
    if args.core_file and Path(args.core_file).exists():
        raw_text = Path(args.core_file).read_text(encoding="utf-8")
        core_deck = Deck.from_dreamborn_text(raw_text, name=args.archetype)
        core_counts = dict(core_deck.card_counts)

    if getattr(args, "baseline_file", "") and Path(args.baseline_file).exists():
        raw_base = Path(args.baseline_file).read_text(encoding="utf-8")
        baseline_deck = Deck.from_dreamborn_text(raw_base, name=f"{args.archetype} (Baseline)")

    meta_decks = get_reference_meta_decks(top_n=args.top_meta)
    meta_shares = get_metagame_shares(top_n=args.top_meta)

    locked_total = sum(core_counts.values())
    flex_slots = max(0, 60 - locked_total)
    workers_arg = getattr(args, "workers", 0)
    lookahead_depth = getattr(args, "lookahead_depth", 1)
    scanner = TechCardScanner(
        workers=workers_arg if workers_arg > 0 else None,
        lookahead_depth=lookahead_depth,
    )
    est_decks = (15 if "princess" in args.archetype.lower() else 13) + args.packages
    est_total_games = est_decks * len(meta_decks) * args.games
    print(
        f"[LorcanaAlpha] Optimizing {flex_slots} Flex/Tech Slots around {locked_total}-Card Locked Core "
        f"for '{args.archetype}' vs Top {len(meta_decks)} Duels.ink Meta Decks "
        f"(games_per_matchup={args.games}, packages={args.packages}, lookahead_depth={lookahead_depth}, "
        f"workers={scanner.evaluator.workers}, ~{est_total_games:,} total games)..."
    )

    report = scanner.optimize_core_for_duels_ink_meta(
        core_card_counts=core_counts,
        meta_decks=meta_decks,
        meta_shares=meta_shares,
        archetype_name=args.archetype,
        games_per_matchup=args.games,
        package_candidates=args.packages,
        seed=args.seed,
        baseline_deck=baseline_deck,
    )

    print("\n" + "=" * 105)
    print(f"RANKING KART TECHOWYCH I WPŁYW % NA MATCHUPY DLA: {report.archetype_name}")
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

    tot_d_bo1 = report.optimized_meta_bo1_wr - report.baseline_meta_bo1_wr
    tot_d_bo3 = report.optimized_meta_bo3_wr - report.baseline_meta_bo3_wr
    s_tot1 = "+" if tot_d_bo1 >= 0 else ""
    s_tot3 = "+" if tot_d_bo3 >= 0 else ""

    if report.top_flex_packages:
        print("\n" + "=" * 105)
        print(f"TOP {len(report.top_flex_packages)} NAJLEPSZYCH KOMBINACJI TECHÓW (PAKIETY FLEX) DLA: {report.archetype_name}")
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

    name_low = report.archetype_name.lower()
    if "princess" in name_low:
        slug = "princess_aggro"
        out_filename = "amber_emerald_princess_aggro_optimized.txt"
    elif "steel" in name_low or "green" in name_low:
        slug = "green_steel"
        out_filename = "emerald_steel_green_steel_optimized.txt"
    else:
        slug = "circle_of_life"
        out_filename = "amber_emerald_circle_of_life_optimized.txt"

    deck_out_path = PROJECT_ROOT / "decks" / out_filename
    deck_out_path.parent.mkdir(parents=True, exist_ok=True)
    deck_out_path.write_text(report.optimized_deck.to_dreamborn_text() + "\n", encoding="utf-8")

    json_out_path = RESULTS_DIR / f"{slug}_optimized_report.json"
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

    # Automatically save 4 complete play-by-play sample game transcripts to logs/
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
                    lookahead_depth=max(1, lookahead_depth),
                )
            )
    replay_path = LOG_DIR / f"{slug}_sample_games.log"
    replay_path.write_text("\n\n".join(sample_replays) + "\n", encoding="utf-8")

    print(f"[OK] Zapisano gotową talię do: {deck_out_path}")
    print(f"[OK] Zapisano pełny raport JSON do: {json_out_path}")
    print(f"[OK] Zapisano 4 pełne zapisy rozgrywek (Play-by-Play Replay) do: {replay_path}")



def cmd_train(args: argparse.Namespace) -> None:
    from lorcana_ai.agents.trainer import SelfPlayTrainer

    print(f"[LorcanaAlpha] Starting Self-Play RL Trainer (Rust Engine Active: {HAS_RUST_ENGINE})")
    mcts_cfg = MCTSConfig(
        num_determinizations=args.determinizations,
        simulations_per_determinization=args.simulations,
    )
    trainer = SelfPlayTrainer(mcts_cfg=mcts_cfg, lr=args.lr)
    print(f"[LorcanaAlpha] Device: {trainer.device} | Starting from completed_epochs={trainer.completed_epochs}")

    for ep in range(args.epochs):
        metrics = trainer.train_epoch(
            games_per_epoch=args.games_per_epoch,
            batch_size=args.batch_size,
            use_mcts=args.use_mcts,
        )
        print(
            f"Epoch {metrics['epoch']:04d} | "
            f"Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Value Loss: {metrics['value_loss']:.4f} | "
            f"Belief Loss: {metrics['belief_loss']:.4f} | "
            f"Avg Turns: {metrics['avg_game_turns']:.1f} | "
            f"Buffer: {metrics['buffer_size']} | "
            f"Time: {metrics['elapsed_sec']}s"
        )
    print(f"[OK] Checkpoint saved to {CHECKPOINT_DIR / 'latest_agent.pt'}")


def cmd_scan_tech(args: argparse.Namespace) -> None:
    decks = get_reference_meta_decks()
    core_counts, baseline_deck = get_archetype_core_and_baseline(args.my_deck)
    locked_ids = set(core_counts.keys()) if core_counts else set()

    opt_slug = "princess_aggro" if "princess" in args.my_deck.lower() else ("circle_of_life" if "circle" in args.my_deck.lower() else "")
    opt_file = PROJECT_ROOT / "decks" / f"amber_emerald_{opt_slug}_optimized.txt" if opt_slug else None
    if opt_file and opt_file.exists():
        my_deck = Deck.from_dreamborn_text(opt_file.read_text(encoding="utf-8"), name=f"{args.my_deck} (Optimized)")
    elif baseline_deck is not None:
        my_deck = baseline_deck
    else:
        my_deck = decks.get(args.my_deck) or next(
            (d for k, d in decks.items() if args.my_deck.lower() in k.lower()),
            list(decks.values())[0],
        )
    opp_deck = decks.get(args.opp_deck) or next(
        (d for k, d in decks.items() if args.opp_deck.lower() in k.lower()),
        list(decks.values())[1],
    )

    print(f"[LorcanaAlpha] Scanning Tech-Cards for '{my_deck.name}' vs '{opp_deck.name}'...")
    lookahead_depth = getattr(args, "lookahead_depth", 1)
    scanner = TechCardScanner(lookahead_depth=lookahead_depth)
    recs = scanner.scan_for_matchup(
        base_deck=my_deck,
        opponent_deck=opp_deck,
        copies_to_swap=args.copies,
        total_simulation_budget=args.budget,
        top_k=args.top_k,
        locked_core_ids=locked_ids,
    )

    print("\n" + "=" * 95)
    print(f"TOP TECH-CARD RECOMMENDATIONS: {my_deck.name}  vs  {opp_deck.name}")
    print("=" * 95)
    for idx, r in enumerate(recs, 1):
        sign = "+" if r.delta_bo3_winrate >= 0 else ""
        print(
            f"{idx}. +{r.copies_swapped}x {r.card_in_name} (zamiast {r.card_out_name})\n"
            f"   Bo3 Winrate: {r.new_bo3_winrate*100:.1f}% ({sign}{r.delta_bo3_winrate*100:.1f}% vs baza {r.baseline_bo3_winrate*100:.1f}%) | "
            f"Play: {r.play_winrate*100:.1f}% | Draw: {r.draw_winrate*100:.1f}% | Uninkables: {r.uninkable_count}\n"
            f"   Uzasadnienie: {r.tactical_rationale}"
        )

    solver = NashMetaSolver()
    mull = solver.generate_mulligan_guide(my_deck, opp_deck, games_per_strategy=12)
    print("-" * 95)
    print(f"REKOMENDACJA MULLIGANU ({my_deck.name} vs {opp_deck.name}):")
    print(f"  * Gdy zaczynasz (On the Play): {mull.best_strategy_on_play}")
    print(f"  * Gdy drugi (On the Draw):     {mull.best_strategy_on_draw}")
    print(f"  * Priorytetowe karty na rękę:  {', '.join(mull.priority_keep_cards)}")
    print("=" * 95)

    out_path = RESULTS_DIR / "latest_tech_scan.json"
    out_path.write_text(
        json.dumps([r.__dict__ for r in recs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] Saved detailed tech scan report to {out_path}")


def cmd_watch_game(args: argparse.Namespace) -> None:
    from lorcana_ai.engine.game_replay import simulate_and_format_replay

    decks = get_reference_meta_decks()

    def _resolve_deck(spec: str) -> Deck:
        if Path(spec).exists():
            return Deck.from_dreamborn_text(Path(spec).read_text(encoding="utf-8"), name=Path(spec).stem)
        opt_slug = "princess_aggro" if "princess" in spec.lower() else ("circle_of_life" if "circle" in spec.lower() else "")
        opt_file = PROJECT_ROOT / "decks" / f"amber_emerald_{opt_slug}_optimized.txt" if opt_slug else None
        if opt_file and opt_file.exists():
            return Deck.from_dreamborn_text(opt_file.read_text(encoding="utf-8"), name=f"{spec} (Optimized)")
        _, base_d = get_archetype_core_and_baseline(spec)
        if base_d is not None and spec.lower() in base_d.name.lower():
            return base_d
        return decks.get(spec) or next(
            (d for k, d in decks.items() if spec.lower() in k.lower()),
            list(decks.values())[0],
        )

    deck_a = _resolve_deck(args.deck_a)
    deck_b = _resolve_deck(args.deck_b)
    replays: list[str] = []
    for g_idx in range(max(1, args.games)):
        sp = g_idx % 2
        g_seed = args.seed + g_idx * 97
        txt = simulate_and_format_replay(
            deck_a=deck_a,
            deck_b=deck_b,
            starting_player=sp,
            seed=g_seed,
            lookahead_depth=args.lookahead_depth,
            agent_type=getattr(args, "agent", "heuristic"),
            mcts_simulations=getattr(args, "simulations", 16),
            mcts_determinizations=getattr(args, "determinizations", 4),
        )
        print(txt)
        replays.append(txt)


    replay_out = LOG_DIR / "sample_games_replay.log"
    replay_out.write_text("\n\n".join(replays) + "\n", encoding="utf-8")
    print(f"\n[OK] Pełny zapis {len(replays)} gier (Play-by-Play) zapisano również do: {replay_out}")


def cmd_evolve(args: argparse.Namespace) -> None:
    print(f"[LorcanaAlpha] Running MAP-Elites Coevolution Deckbuilder (Generations={args.generations})...")
    builder = MAPElitesDeckbuilder()
    cells = builder.run_evolution(
        generations=args.generations,
        offspring_per_gen=args.offspring,
        games_per_matchup=args.games,
    )

    print("\n" + "=" * 95)
    print("TOP DISCOVERED / EVOLVED ARCHETYPES (MAP-Elites Archive)")
    print("=" * 95)
    for idx, cell in enumerate(cells[:8], 1):
        print(
            f"{idx}. {cell.deck.name} [{cell.ink_pair} - {cell.style_profile}]\n"
            f"   Expected Meta Bo3 Winrate: {cell.expected_meta_winrate*100:.1f}% | "
            f"Play: {cell.play_winrate*100:.1f}% | Draw: {cell.draw_winrate*100:.1f}% | "
            f"Avg Cost: {cell.deck.average_cost:.2f} | Uninkables: {cell.deck.uninkable_count}"
        )
    print(f"\n[OK] Saved full MAP-Elites deck archive to {RESULTS_DIR / 'map_elites_archive.json'}")


def cmd_nash_meta(args: argparse.Namespace) -> None:
    print("[LorcanaAlpha] Computing Bo3 Metagame Payoff Matrix & Nash Equilibrium...")
    decks = list(get_reference_meta_decks().values())
    solver = NashMetaSolver()
    report = solver.solve_metagame(
        decks=decks,
        current_meta_shares=DEFAULT_METAGAME_SHARES,
        games_per_pair=args.games,
    )

    print("\n" + "=" * 95)
    print("NASH EQUILIBRIUM METAGAME SOLVER REPORT (Best-of-3 Tournament Rules)")
    print("=" * 95)
    for nm in report.deck_names:
        ev = report.expected_winrate_vs_current_meta[nm] * 100
        nw = report.nash_weights[nm] * 100
        cur = DEFAULT_METAGAME_SHARES.get(nm, 0.0) * 100
        print(f"  {nm:<34} | Meta Share: {cur:5.1f}% | EV Bo3 Winrate: {ev:5.1f}% | Nash Weight: {nw:5.1f}%")
    print("-" * 95)
    print(f"Najlepsza talia na obecny turniej (Best Counter Deck): {report.best_counter_deck}")
    print("=" * 95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LorcanaAlpha CLI: AI Agent & Evolutionary Deckbuilder")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sync-cards", help="Download and cache latest LorcanaJSON card database")

    p_meta = sub.add_parser("sync-meta", help="Scrape Core Set 14 meta decks and stats from https://duels.ink/stats/core-bo1")
    p_meta.add_argument("--queue", type=str, default="core-bo1", help="Duels.ink queue ID (e.g. core-bo1, core-bo3)")
    p_meta.add_argument("--period", type=str, default="all_time", help="Time period (all_time or week)")
    p_meta.add_argument("--era", type=str, default="set14", help="Era identifier (default: set14)")

    p_opt = sub.add_parser(
        "optimize-core",
        help="Find the best tech cards, per-matchup percentage winrate impact, and 16-card flex package vs Duels.ink meta",
    )
    p_opt.add_argument("--archetype", type=str, default="Amber/Emerald Circle of Life")
    p_opt.add_argument("--core-file", type=str, default="", help="Optional path to custom 44-card core .txt file")
    p_opt.add_argument("--baseline-file", type=str, default="", help="Optional path to starting 60-card baseline .txt file")
    p_opt.add_argument("--top-meta", type=int, default=8, help="Number of top Duels.ink meta decks in gauntlet")
    p_opt.add_argument("--games", type=int, default=16, help="Games simulated per matchup")
    p_opt.add_argument("--packages", type=int, default=8, help="Number of candidate 16-card flex packages to evaluate")
    p_opt.add_argument("--workers", type=int, default=0, help="Number of parallel CPU worker processes (0 = auto)")
    p_opt.add_argument("--lookahead-depth", type=int, default=1, help="Turn lookahead depth (0=greedy, 1=turn rollout, 2=turn+crackback)")
    p_opt.add_argument("--seed", type=int, default=2026)

    p_watch = sub.add_parser(
        "watch-game",
        help="Simulate and display a complete turn-by-turn, action-by-action play-by-play transcript of a Lorcana match",
    )
    p_watch.add_argument("--deck-a", type=str, default="Amber/Emerald Princess Aggro", help="Deck A name or .txt path")
    p_watch.add_argument("--deck-b", type=str, default="Amber/Sapphire Detectives", help="Deck B name or .txt path")
    p_watch.add_argument("--games", type=int, default=2, help="Number of sample games to play and log")
    p_watch.add_argument("--lookahead-depth", type=int, default=2, help="Turn lookahead depth (1 or 2)")
    p_watch.add_argument("--agent", type=str, choices=["heuristic", "is-mcts"], default="heuristic", help="Action selection algorithm (heuristic or is-mcts)")
    p_watch.add_argument("--simulations", type=int, default=16, help="IS-MCTS simulations per determinization")
    p_watch.add_argument("--determinizations", type=int, default=4, help="IS-MCTS number of determinizations")
    p_watch.add_argument("--seed", type=int, default=2026)


    p_train = sub.add_parser("train", help="Run Self-Play RL & Opponent Belief Network training")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--games-per-epoch", type=int, default=6)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--determinizations", type=int, default=4)
    p_train.add_argument("--simulations", type=int, default=12)
    p_train.add_argument("--use-mcts", action="store_true", help="Enable full IS-MCTS during self-play")

    p_scan = sub.add_parser("scan-tech", help="Run UCB1 Tech-Card Scanner for a specific matchup")
    p_scan.add_argument("--my-deck", type=str, default="Amber/Emerald Circle of Life")
    p_scan.add_argument("--opp-deck", type=str, default="Emerald/Steel Midrange")
    p_scan.add_argument("--copies", type=int, default=3)
    p_scan.add_argument("--budget", type=int, default=96)
    p_scan.add_argument("--top-k", type=int, default=5)
    p_scan.add_argument("--lookahead-depth", type=int, default=1)

    p_evol = sub.add_parser("evolve", help="Discover and evolve new decks using MAP-Elites")
    p_evol.add_argument("--generations", type=int, default=3)
    p_evol.add_argument("--offspring", type=int, default=4)
    p_evol.add_argument("--games", type=int, default=6)

    p_nash = sub.add_parser("nash-meta", help="Solve Bo3 metagame Nash Equilibrium")
    p_nash.add_argument("--games", type=int, default=12)

    return parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.command.replace('-', '_')}.log"
    sys.stdout = _TeeLogger(sys.stdout, log_path)
    print(f"[Log] Zapisywanie logów do: {log_path}")
    if args.command == "sync-cards":
        cmd_sync_cards(args)
    elif args.command == "sync-meta":
        cmd_sync_meta(args)
    elif args.command == "optimize-core":
        cmd_optimize_core(args)
    elif args.command == "watch-game":
        cmd_watch_game(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "scan-tech":
        cmd_scan_tech(args)
    elif args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "nash-meta":
        cmd_nash_meta(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
