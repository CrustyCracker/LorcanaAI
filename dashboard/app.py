"""
LorcanaAlpha Interactive Web Dashboard (Streamlit + Plotly).
Launch locally with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.agents.is_mcts import ISMCTSAgent
from lorcana_ai.config import CHECKPOINT_DIR, RESULTS_DIR, MCTSConfig
from lorcana_ai.data.card_db import GLOBAL_CARD_DB, INK_COLORS
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import DEFAULT_METAGAME_SHARES, get_reference_meta_decks
from lorcana_ai.deckbuilder.evolutionary_map import MAPElitesDeckbuilder
from lorcana_ai.deckbuilder.nash_solver import NashMetaSolver
from lorcana_ai.deckbuilder.tech_scanner import TechCardScanner
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import HAS_RUST_ENGINE, LorcanaSimulator

st.set_page_config(
    page_title="LorcanaAlpha AI & Deckbuilder",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LorcanaAlpha: Agent AI (IS-MCTS) & Optymalizator Talii (PLGrid Athena)")
st.caption(
    f"Baza kart: {len(GLOBAL_CARD_DB.by_id)} kart Core Constructed | "
    f"Natywny akcelerator Rust (lorcana_rs): {'AKTYWNY' if HAS_RUST_ENGINE else 'Python Fallback'}"
)

meta_decks = get_reference_meta_decks()
deck_names = list(meta_decks.keys())

tab_tech, tab_evo, tab_nash, tab_play = st.tabs(
    [
        "1. Skaner Tech-Kart & Mulligan",
        "2. Odkrywca Nowych Talii (MAP-Elites)",
        "3. Macierz Metagame & Równowaga Nasha",
        "4. Sparing Człowiek vs Agent AI",
    ]
)

# ==============================================================================
# TAB 1: TECH-CARD SCANNER & MULLIGAN ADVISOR
# ==============================================================================
with tab_tech:
    st.subheader("Wyszukiwanie Tech-Kart na konkretny Matchup (Algorytm UCB1 Bandit)")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        my_deck_choice = st.selectbox("Twoja talia bazowa:", deck_names, index=1)
        custom_list = st.text_area(
            "Lub wklej własną talię (format Dreamborn.ink):",
            value=meta_decks[my_deck_choice].to_dreamborn_text(),
            height=220,
        )
    with col2:
        opp_deck_choice = st.selectbox("Talia przeciwnika (Target Matchup):", deck_names, index=0)
        copies_swap = st.slider("Liczba podmienianych kopii (Flex Slots):", 1, 4, 3)
        sim_budget = st.slider("Budżet symulacji gier UCB1:", 48, 480, 120, step=24)

    with col3:
        st.markdown("### Reguły walidacji")
        parsed_my_deck = Deck.from_dreamborn_text(custom_list, name=my_deck_choice)
        is_valid, errs = parsed_my_deck.validate()
        st.metric("Liczba kart", parsed_my_deck.total_cards)
        st.metric("Karty Uninkable", parsed_my_deck.uninkable_count)
        st.metric("Średni koszt Inka", f"{parsed_my_deck.average_cost:.2f}")
        if not is_valid:
            for e in errs:
                st.error(e)

    if st.button("Uruchom Skaner Tech-Kart i Analizę Mulliganu", type="primary"):
        opp_deck = meta_decks[opp_deck_choice]
        with st.spinner("Trwa symulacja gier i optymalizacja UCB1..."):
            scanner = TechCardScanner()
            recs = scanner.scan_for_matchup(
                base_deck=parsed_my_deck,
                opponent_deck=opp_deck,
                copies_to_swap=copies_swap,
                total_simulation_budget=sim_budget,
                top_k=6,
            )
            solver = NashMetaSolver()
            mull_guide = solver.generate_mulligan_guide(parsed_my_deck, opp_deck, games_per_strategy=12)

        if recs:
            df_recs = pd.DataFrame(
                [
                    {
                        "Dodaj kartę (Tech In)": f"+{r.copies_swapped}x {r.card_in_name}",
                        "Wyjmij kartę (Flex Out)": f"-{r.copies_swapped}x {r.card_out_name}",
                        "Nowy Winrate Bo3": f"{r.new_bo3_winrate*100:.1f}%",
                        "Zmiana Bo3": f"{r.delta_bo3_winrate*100:+.1f}%",
                        "Winrate Play": f"{r.play_winrate*100:.1f}%",
                        "Winrate Draw": f"{r.draw_winrate*100:.1f}%",
                        "Uninkables": r.uninkable_count,
                        "Uzasadnienie taktyczne": r.tactical_rationale,
                    }
                    for r in recs
                ]
            )
            st.success(f"Bazowy Winrate Bo3 w tym matchupie: {recs[0].baseline_bo3_winrate*100:.1f}%")
            st.dataframe(df_recs, use_container_width=True)

        st.markdown("#### Poradnik Mulliganu dla tego matchupu")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.info(f"**Gdy zaczynasz (On the Play):** {mull_guide.best_strategy_on_play}")
            st.write("Winrate poszczególnych strategii (Play):", mull_guide.strategy_winrates_play)
        with mc2:
            st.warning(f"**Gdy jesteś drugi (On the Draw):** {mull_guide.best_strategy_on_draw}")
            st.write("Winrate poszczególnych strategii (Draw):", mull_guide.strategy_winrates_draw)
        st.markdown(f"**Priorytetowe karty do zatrzymania na ręce startowej:** `{', '.join(mull_guide.priority_keep_cards)}`")

# ==============================================================================
# TAB 2: MAP-ELITES EVOLUTIONARY DECK DISCOVERY
# ==============================================================================
with tab_evo:
    st.subheader("Odkrywanie Nowych Archetypów Talii (Algorytm MAP-Elites + Koewolucja)")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        sel_inks = st.multiselect(
            "Wybierz kolory Ink do eksploracji:",
            options=list(INK_COLORS),
            default=["Amethyst", "Ruby", "Sapphire", "Steel"],
        )
    with ec2:
        evo_gens = st.slider("Liczba generacji ewolucyjnych:", 1, 20, 3)
    with ec3:
        evo_offspring = st.slider("Potomków na generację:", 2, 16, 4)

    if st.button("Ewoluuj Nowe Talie", type="primary"):
        pairs = [
            (sel_inks[i], sel_inks[j])
            for i in range(len(sel_inks))
            for j in range(i + 1, len(sel_inks))
        ]
        if not pairs:
            st.error("Wybierz co najmniej 2 kolory Ink.")
        else:
            with st.spinner("Ewolucja talii MAP-Elites przeciwko metagame..."):
                builder = MAPElitesDeckbuilder()
                builder.run_evolution(
                    generations=evo_gens,
                    offspring_per_gen=evo_offspring,
                    games_per_matchup=6,
                    target_ink_pairs=pairs,
                )

    archive_file = RESULTS_DIR / "map_elites_archive.json"
    if archive_file.exists():
        archive_data = json.loads(archive_file.read_text(encoding="utf-8"))
        if archive_data:
            df_arch = pd.DataFrame(
                [
                    {
                        "Nazwa Talii": item["deck_name"],
                        "Para Kolorów": item["ink_pair"],
                        "Styl (Nisza)": item["style_profile"],
                        "EV Winrate Bo3": f"{item['expected_meta_winrate']*100:.1f}%",
                        "Winrate Play": f"{item['play_winrate']*100:.1f}%",
                        "Winrate Draw": f"{item['draw_winrate']*100:.1f}%",
                        "Śr. Koszt": item["average_cost"],
                        "Uninkables": item["uninkable_count"],
                    }
                    for item in archive_data
                ]
            )
            st.dataframe(df_arch, use_container_width=True)
            chosen_deck_idx = st.selectbox(
                "Podejrzyj wygenerowaną listę 60 kart:",
                range(len(archive_data)),
                format_func=lambda i: f"{archive_data[i]['deck_name']} ({archive_data[i]['expected_meta_winrate']*100:.1f}% EV)",
            )
            st.code(archive_data[chosen_deck_idx]["decklist"], language="text")

# ==============================================================================
# TAB 3: METAGAME MATRIX & NASH EQUILIBRIUM SOLVER
# ==============================================================================
with tab_nash:
    st.subheader("Analiza Metagame Turniejowego (Bo3) i Równowaga Nasha")
    games_per_pair = st.slider("Liczba gier na każdą parę talii:", 4, 48, 8, step=4)
    if st.button("Oblicz Macierz Matchupów i Równowagę Nasha", type="primary"):
        with st.spinner("Obliczanie pełnej macierzy matchupów Play/Draw/Bo3..."):
            solver = NashMetaSolver()
            report = solver.solve_metagame(
                decks=list(meta_decks.values()),
                current_meta_shares=DEFAULT_METAGAME_SHARES,
                games_per_pair=games_per_pair,
            )

        st.success(f"Najlepsza talia kontrująca obecne metagame: **{report.best_counter_deck}**")
        fig = px.imshow(
            report.payoff_matrix_bo3 * 100.0,
            x=report.deck_names,
            y=report.deck_names,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            labels={"x": "Przeciwnik", "y": "Twoja Talia", "color": "Bo3 Winrate %"},
            title="Macierz Winrate Best-of-3 (Bo3 %)",
        )
        st.plotly_chart(fig, use_container_width=True)

        df_nash = pd.DataFrame(
            [
                {
                    "Talia": nm,
                    "Obecny udział w mecie (%)": round(DEFAULT_METAGAME_SHARES.get(nm, 0.0) * 100, 1),
                    "Oczekiwany Winrate Bo3 (%)": round(report.expected_winrate_vs_current_meta[nm] * 100, 1),
                    "Waga w Równowadze Nasha (%)": round(report.nash_weights[nm] * 100, 1),
                }
                for nm in report.deck_names
            ]
        )
        st.dataframe(df_nash, use_container_width=True)

# ==============================================================================
# TAB 4: HUMAN VS AI AGENT SPARRING & REPLAY LOG
# ==============================================================================
with tab_play:
    st.subheader("⚔️ Sparing Człowiek vs Agent AI & Rejestr Ruchów")

    if "match_log" not in st.session_state:
        st.session_state.match_log = []

    if "interactive_game" not in st.session_state:
        d_human = meta_decks[deck_names[1]]
        d_ai = meta_decks[deck_names[0]]
        st.session_state.interactive_game = GameState.create_new_game(
            d_human.expand_cards(), d_ai.expand_cards(), starting_player=0, know_opponent_inks=True, seed=42
        )
        st.session_state.match_log = [
            {"turn": 1, "player": "System", "action": "Rozpoczęto nową partię", "detail": f"Gracz 0: {deck_names[1]} | AI (Gracz 1): {deck_names[0]}"}
        ]

    col_setup, col_board = st.columns([1, 2.8])
    with col_setup:
        st.markdown("### ⚙️ Konfiguracja Meczu")
        human_deck_sel = st.selectbox("Twoja talia (Gracz 0):", deck_names, index=1, key="sp_h")
        ai_deck_sel = st.selectbox("Talia AI (Gracz 1):", deck_names, index=0, key="sp_ai")

        ai_engine_type = st.radio(
            "Silnik Agenta AI:",
            [
                "🏆 Tactical Heuristic (Grandmaster Lookahead & Zestawy 9-14) [Zalecany]",
                "🧠 IS-MCTS (Information-Set Tree Search + Heuristic Prior)",
            ],
            index=0,
            key="sp_ai_type",
        )

        ai_lookahead = 1
        if "Tactical" in ai_engine_type:
            ai_lookahead = st.slider("Głębokość analizy taktycznej (Lookahead Depth):", min_value=1, max_value=2, value=1, key="sp_la")

        if st.button("🔄 Rozpocznij Nową Grę", type="secondary"):
            seed_val = random.randint(1, 999999)
            st.session_state.interactive_game = GameState.create_new_game(
                meta_decks[human_deck_sel].expand_cards(),
                meta_decks[ai_deck_sel].expand_cards(),
                starting_player=0,
                know_opponent_inks=True,
                seed=seed_val,
            )
            st.session_state.match_log = [
                {"turn": 1, "player": "System", "action": "Rozpoczęto nową partię", "detail": f"Seed: {seed_val} | Ty: {human_deck_sel} | AI: {ai_deck_sel}"}
            ]
            st.rerun()

    g_state: GameState = st.session_state.interactive_game
    p0 = g_state.players[0]
    p1 = g_state.players[1]

    def get_ai_agent():
        if "Tactical" in ai_engine_type:
            return TacticalHeuristicAgent(lookahead_depth=ai_lookahead)
        return ISMCTSAgent(cfg=MCTSConfig(num_determinizations=3, simulations_per_determinization=12))

    with col_board:
        # Status header
        c_status1, c_status2 = st.columns(2)
        with c_status1:
            st.markdown(
                f"### 🧑 Ty (Gracz 0)\n"
                f"**Lore:** `{p0.lore}/20` | **Ink:** `{p0.inkwell_available}/{p0.inkwell_total}`"
                f"{f' | Ink Drops: `{p0.ink_drops}`' if p0.ink_drops > 0 else ''}\n\n"
                f"**Karty w ręce:** {len(p0.hand)} | **Talia:** {len(p0.deck)} | **Cmentarz:** {len(p0.discard)}"
            )
        with c_status2:
            st.markdown(
                f"### 🤖 Agent AI (Gracz 1)\n"
                f"**Lore:** `{p1.lore}/20` | **Ink:** `{p1.inkwell_available}/{p1.inkwell_total}`"
                f"{f' | Ink Drops: `{p1.ink_drops}`' if p1.ink_drops > 0 else ''}\n\n"
                f"**Karty w ręce:** {len(p1.hand)} | **Talia:** {len(p1.deck)} | **Cmentarz:** {len(p1.discard)}"
            )

        st.divider()

        st.markdown(
            f"**Tura gry:** {g_state.turn_number} | **Faza:** `{g_state.phase}` | **Aktywny gracz:** {'🧑 Ty' if g_state.active_player == 0 else '🤖 AI'}"
        )

        st.markdown("#### 🤖 Stół AI:")
        if p1.board:
            st.write(
                ", ".join(
                    f"**{e.card.full_name}** `[{e.effective_strength}/{e.effective_willpower - e.damage}, {'Exerted' if e.exerted else 'Ready'}{', Drying' if e.drying else ''}]`"
                    for e in p1.board
                )
            )
        else:
            st.caption("(Pusty stół AI)")

        st.markdown("#### 🧑 Twój Stół:")
        if p0.board:
            st.write(
                ", ".join(
                    f"**{e.card.full_name}** `[{e.effective_strength}/{e.effective_willpower - e.damage}, {'Exerted' if e.exerted else 'Ready'}{', Drying' if e.drying else ''}]`"
                    for e in p0.board
                )
            )
        else:
            st.caption("(Twój stół jest pusty)")

        # Revealed cards displays
        rev_ai_cards = [c for c in p1.hand if c.card_id in p1.revealed_hand_ids]
        if rev_ai_cards:
            st.warning(
                "👁️ **Karty na ręce AI ujawnione dla Ciebie:** "
                + ", ".join(f"**{c.full_name}** ({c.cost} ink)" for c in rev_ai_cards)
            )

        rev_human_cards = [c for c in p0.hand if c.card_id in p0.revealed_hand_ids]
        if rev_human_cards:
            st.info(
                "👁️ **Twoje karty znane AI:** "
                + ", ".join(f"**{c.full_name}** ({c.cost} ink)" for c in rev_human_cards)
            )

        # Discards preview if non-empty
        if p0.discard or p1.discard:
            with st.expander("🪦 Cmentarze (Discards) obu graczy"):
                col_d0, col_d1 = st.columns(2)
                with col_d0:
                    st.markdown("**Twój Cmentarz:** " + (", ".join(c.full_name for c in p0.discard) or "(pusty)"))
                with col_d1:
                    st.markdown("**Cmentarz AI:** " + (", ".join(c.full_name for c in p1.discard) or "(pusty)"))

        st.markdown("#### 🃏 Twoja Ręka:")
        st.write(
            ", ".join(
                f"**{c.full_name}** `({c.cost} ink{'' if c.inkable else ', Uninkable'})`"
                for c in p0.hand
            ) or "(Pusta ręka)"
        )

        st.divider()

        # Phase logic
        if g_state.phase == "GAME_OVER":
            winner_str = "🏆 Ty (Gracz 0)" if g_state.winner == 0 else "🤖 Agent AI (Gracz 1)"
            st.success(f"🎉 **Koniec Gry! Zwycięzca: {winner_str}** | Powód: `{g_state.victory_reason}`")

        elif g_state.phase == "MULLIGAN_P0":
            st.markdown("### 🔄 Faza Mulliganu — Twoja decyzja:")
            st.info("Zaznacz karty z ręki początkowej, które chcesz **odrzucić i wymienić** na nowe:")
            mulligan_discard = []
            cols = st.columns(min(7, max(1, len(p0.hand))))
            for idx, c in enumerate(p0.hand):
                with cols[idx]:
                    chk = st.checkbox(f"{c.full_name}\n({c.cost} ink)", key=f"mull_chk_{idx}")
                    if chk:
                        mulligan_discard.append(idx)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if st.button("✅ Zatwierdź Mulligan (Wymień zaznaczone)", type="primary"):
                    mask = tuple(i not in mulligan_discard for i in range(len(p0.hand)))
                    mull_act = GameAction(ActionType.MULLIGAN, mulligan_keep_mask=mask, description=f"Mulligan: wymieniono {len(mulligan_discard)} kart")
                    LorcanaSimulator.step(g_state, mull_act)
                    st.session_state.match_log.append({
                        "turn": 1,
                        "player": "Ty",
                        "action": f"Mulligan ({len(mulligan_discard)} wymienionych)",
                        "detail": f"Odrzucono: {', '.join(p0.hand[i].full_name for i in mulligan_discard) or 'brak'}",
                    })
                    st.rerun()
            with col_m2:
                if st.button("⏩ Zatrzymaj wszystkie karty (Keep 7)"):
                    mask = (True,) * len(p0.hand)
                    mull_act = GameAction(ActionType.MULLIGAN, mulligan_keep_mask=mask, description="Mulligan: zachowano całą rękę")
                    LorcanaSimulator.step(g_state, mull_act)
                    st.session_state.match_log.append({
                        "turn": 1,
                        "player": "Ty",
                        "action": "Mulligan: zachowano całą rękę",
                        "detail": "Keep 7",
                    })
                    st.rerun()

        elif g_state.phase == "MULLIGAN_P1":
            st.markdown("### 🤖 Faza Mulliganu — Decyzja Agenta AI:")
            if st.button("▶ Wykonaj Mulligan Agenta AI", type="primary"):
                agent = TacticalHeuristicAgent(lookahead_depth=0)
                mull_actions = MoveGenerator.generate_legal_actions(g_state)
                chosen_mull = agent.select_action(g_state, mull_actions)
                kept_cnt = sum(1 for k in chosen_mull.mulligan_keep_mask if k)
                LorcanaSimulator.step(g_state, chosen_mull)
                st.session_state.match_log.append({
                    "turn": 1,
                    "player": "Agent AI",
                    "action": f"Mulligan AI: zachowano {kept_cnt}, wymieniono {len(chosen_mull.mulligan_keep_mask) - kept_cnt} kart",
                    "detail": chosen_mull.description,
                })
                st.rerun()

        elif g_state.active_player == 1:
            st.markdown("### 🤖 Tura Agenta AI:")
            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                if st.button("▶ Wykonaj PEŁNĄ turę Agenta AI", type="primary"):
                    agent = get_ai_agent()
                    step_cnt = 0
                    while g_state.active_player == 1 and g_state.phase != "GAME_OVER" and step_cnt < 12:
                        step_cnt += 1
                        legal = MoveGenerator.generate_legal_actions(g_state)
                        if not legal:
                            break
                        if hasattr(agent, "select_action_with_diagnostics"):
                            act, sc, alts = agent.select_action_with_diagnostics(g_state, legal)
                            alt_str = ", ".join(f"{a.description}: {s:+.1f}" for a, s in alts[:2])
                        else:
                            act, _ = agent.select_action(g_state, use_mcts=True)
                            sc = 0.0
                            alt_str = ""
                        LorcanaSimulator.step(g_state, act)
                        st.session_state.match_log.append({
                            "turn": g_state.turn_number,
                            "player": "Agent AI",
                            "action": act.description,
                            "score": f"{sc:+.1f}" if sc != 0.0 else "",
                            "detail": f"Alts: [{alt_str}]" if alt_str else "",
                        })
                        if act.action_type == ActionType.PASS_TURN:
                            break
                    st.rerun()

            with col_ai2:
                if st.button("⏭ Wykonaj 1 akcję Agenta AI"):
                    agent = get_ai_agent()
                    legal = MoveGenerator.generate_legal_actions(g_state)
                    if legal:
                        if hasattr(agent, "select_action_with_diagnostics"):
                            act, sc, alts = agent.select_action_with_diagnostics(g_state, legal)
                            alt_str = ", ".join(f"{a.description}: {s:+.1f}" for a, s in alts[:2])
                        else:
                            act, _ = agent.select_action(g_state, use_mcts=True)
                            sc = 0.0
                            alt_str = ""
                        LorcanaSimulator.step(g_state, act)
                        st.session_state.match_log.append({
                            "turn": g_state.turn_number,
                            "player": "Agent AI",
                            "action": act.description,
                            "score": f"{sc:+.1f}" if sc != 0.0 else "",
                            "detail": f"Alts: [{alt_str}]" if alt_str else "",
                        })
                    st.rerun()

        else:
            # Human Player Turn
            st.markdown("### 🧑 Twoja Tura:")
            legal = MoveGenerator.generate_legal_actions(g_state)
            advisor = TacticalHeuristicAgent(lookahead_depth=1)
            rec_act, rec_sc, alts = advisor.select_action_with_diagnostics(g_state, legal)
            alt_preview = ", ".join(f"{a.description} ({s:+.1f})" for a, s in alts[:2])
            st.info(f"💡 **Rekomendacja Agenta AI dla Ciebie:** **{rec_act.description}** *(Ocena: {rec_sc:+.1f} pkt | Alternatywy: {alt_preview})*")

            chosen_act = st.selectbox(
                "Wybierz akcję do wykonania:",
                legal,
                format_func=lambda a: a.description,
                key=f"act_sel_{g_state.turn_number}_{len(p0.hand)}",
            )
            if st.button("✅ Wykonaj Ruch", type="primary"):
                LorcanaSimulator.step(g_state, chosen_act)
                st.session_state.match_log.append({
                    "turn": g_state.turn_number,
                    "player": "Ty",
                    "action": chosen_act.description,
                    "detail": "",
                })
                st.rerun()

    # Replay / Match Log viewer
    st.divider()
    st.markdown("### 📜 Dziennik Przebiegu Meczu & Diagnostyka Ruchów")
    if st.session_state.match_log:
        with st.expander("Kliknij, aby rozwinąć pełną historię ruchów w tej grze", expanded=True):
            for entry in reversed(st.session_state.match_log[-30:]):
                p_badge = "🤖 [AI]" if entry["player"] == "Agent AI" else ("🧑 [Ty]" if entry["player"] == "Ty" else "ℹ️ [System]")
                sc_txt = f" *(Wynik AI: `{entry.get('score', '')}`)*" if entry.get("score") else ""
                dt_txt = f" — `{entry['detail']}`" if entry.get("detail") else ""
                st.markdown(f"- **Tura {entry['turn']}** {p_badge}: **{entry['action']}**{sc_txt}{dt_txt}")
