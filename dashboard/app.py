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

from lorcana_ai.agents.is_mcts import ISMCTSAgent
from lorcana_ai.config import CHECKPOINT_DIR, RESULTS_DIR, MCTSConfig
from lorcana_ai.data.card_db import GLOBAL_CARD_DB, INK_COLORS
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.meta_decks import DEFAULT_METAGAME_SHARES, get_reference_meta_decks
from lorcana_ai.deckbuilder.evolutionary_map import MAPElitesDeckbuilder
from lorcana_ai.deckbuilder.nash_solver import NashMetaSolver
from lorcana_ai.deckbuilder.tech_scanner import TechCardScanner
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import MoveGenerator
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
# TAB 4: HUMAN VS AI AGENT SPARRING
# ==============================================================================
with tab_play:
    st.subheader("Sparing Człowiek vs Agent AI (IS-MCTS + Transformer Policy-Value)")
    if "interactive_game" not in st.session_state:
        d_human = meta_decks[deck_names[1]]
        d_ai = meta_decks[deck_names[0]]
        st.session_state.interactive_game = GameState.create_new_game(
            d_human.expand_cards(), d_ai.expand_cards(), starting_player=0, know_opponent_inks=True, seed=42
        )

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        human_deck_sel = st.selectbox("Twoja talia (Gracz 0):", deck_names, index=1, key="sp_h")
        ai_deck_sel = st.selectbox("Talia AI (Gracz 1):", deck_names, index=0, key="sp_ai")
        if st.button("Nowa Gra"):
            st.session_state.interactive_game = GameState.create_new_game(
                meta_decks[human_deck_sel].expand_cards(),
                meta_decks[ai_deck_sel].expand_cards(),
                starting_player=0,
                know_opponent_inks=True,
                seed=random.randint(1, 999999),
            )
            st.rerun()

    g_state: GameState = st.session_state.interactive_game
    p0 = g_state.players[0]
    p1 = g_state.players[1]

    with col_p2:
        st.markdown(
            f"**Tura {g_state.turn_number}** | Faza: `{g_state.phase}` | "
            f"Twój Lore: **{p0.lore}/20** (Ink: {p0.inkwell_available}/{p0.inkwell_total}) | "
            f"Lore AI: **{p1.lore}/20** (Ink: {p1.inkwell_available}/{p1.inkwell_total}, Kart na ręce: {len(p1.hand)})"
        )

        st.markdown("**Stół AI:** " + (", ".join(
            f"{e.card.full_name} [{e.effective_strength}/{e.effective_willpower - e.damage}, {'Exerted' if e.exerted else 'Ready'}]"
            for e in p1.board
        ) or "(pusty)"))

        st.markdown("**Twój Stół:** " + (", ".join(
            f"{e.card.full_name} [{e.effective_strength}/{e.effective_willpower - e.damage}, {'Exerted' if e.exerted else 'Ready'}{', Drying' if e.drying else ''}]"
            for e in p0.board
        ) or "(pusty)"))

        st.markdown("**Twoja Ręka:** " + (", ".join(
            f"{c.full_name} ({c.cost}{'' if c.inkable else ' [Unink]'})" for c in p0.hand
        ) or "(pusta)"))

        if g_state.phase == "GAME_OVER":
            st.success(f"Gra zakończona! Zwycięzca: {'Ty (Gracz 0)' if g_state.winner == 0 else 'Agent AI (Gracz 1)'} ({g_state.victory_reason})")
        elif g_state.active_player == 1:
            if st.button("Wykonaj ruch Agenta AI (IS-MCTS)", type="primary"):
                ai_agent = ISMCTSAgent(cfg=MCTSConfig(num_determinizations=3, simulations_per_determinization=10))
                ckpt_file = CHECKPOINT_DIR / "latest_agent.pt"
                if ckpt_file.exists():
                    try:
                        ckpt = json.loads(ckpt_file.read_text()) if ckpt_file.suffix == ".json" else None
                    except Exception:
                        pass
                ai_act, _ = ai_agent.select_action(g_state, use_mcts=True)
                LorcanaSimulator.step(g_state, ai_act)
                st.info(f"AI wykonało ruch: {ai_act.description}")
                st.rerun()
        else:
            legal = MoveGenerator.generate_legal_actions(g_state)
            ai_helper = ISMCTSAgent(cfg=MCTSConfig(num_determinizations=2, simulations_per_determinization=6))
            rec_act, _ = ai_helper.select_action(g_state, use_mcts=False)
            st.caption(f"Podpowiedź Agenta AI dla Ciebie: **{rec_act.description}**")
            chosen_act = st.selectbox(
                "Wybierz swój ruch:",
                legal,
                format_func=lambda a: a.description,
            )
            if st.button("Zatwierdź swój ruch", type="primary"):
                LorcanaSimulator.step(g_state, chosen_act)
                st.rerun()
