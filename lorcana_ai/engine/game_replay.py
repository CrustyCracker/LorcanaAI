"""
Turn-by-Turn & Action-by-Action Game Replay Recorder and Inspector for LorcanaAlpha.
Allows inspecting the exact play-by-play of AI games (Mulligan, Draw, Inkwell, Card Plays,
Triggered Abilities, Quests, Challenges, Board States, and AI Decision Scores).
"""

from __future__ import annotations
import random

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.effect_dsl import KeywordFlag
from lorcana_ai.engine.game_state import BoardEntity, GameState, PlayerState
from lorcana_ai.engine.move_generator import ActionType, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


def _format_entity(ent: BoardEntity, owner: PlayerState, opp: PlayerState | None = None) -> str:
    status_parts: list[str] = []
    if ent.exerted:
        status_parts.append("⟳ Wyczerpana")
    else:
        status_parts.append("Gotowa")
    if ent.drying and ent.card.card_type == "Character":
        status_parts.append("Mokra")
    if ent.has_keyword(KeywordFlag.BODYGUARD):
        status_parts.append("Bodyguard")
    if ent.has_keyword(KeywordFlag.EVASIVE):
        status_parts.append("Evasive")
    if ent.has_keyword(KeywordFlag.WARD):
        status_parts.append("Ward")
    status_str = ", ".join(status_parts)
    rem_wp = max(0, ent.effective_willpower - ent.damage)
    dmg_str = f" [Dmg:{ent.damage}]" if ent.damage > 0 else ""
    lore_val = ent.effective_lore(owner, opp) if ent.card.card_type == "Character" else ent.card.lore
    return (
        f"{ent.card.full_name} ({ent.effective_strength}¤/{rem_wp}⛨{dmg_str}, "
        f"{lore_val}◊ | {status_str})"
    )


def _format_board(p: PlayerState, opp: PlayerState | None = None) -> str:
    if not p.board:
        return "(pusty stół)"
    return " ; ".join(_format_entity(e, p, opp) for e in p.board)


def _format_hand(p: PlayerState) -> str:
    if not p.hand:
        return "(pusta ręka)"
    counts: dict[str, int] = {}
    cost_map: dict[str, str] = {}
    for c in p.hand:
        counts[c.full_name] = counts.get(c.full_name, 0) + 1
        ink_tag = "Ink" if c.inkable else "Unink"
        cost_map[c.full_name] = f"{c.cost}⬡/{ink_tag}"
    return ", ".join(
        f"{ (f'{cnt}x ' if cnt > 1 else '') }{name} [{cost_map[name]}]"
        for name, cnt in counts.items()
    )


def _format_revealed_hand(p: PlayerState) -> str:
    """Formats only the cards in `p.hand` that have been revealed to the opponent (`p.revealed_hand_ids`)."""
    if not p.revealed_hand_ids or not p.hand:
        return ""
    rev_pool = list(p.revealed_hand_ids)
    rev_cards = []
    for c in p.hand:
        if c.card_id in rev_pool:
            rev_cards.append(c)
            rev_pool.remove(c.card_id)
    if not rev_cards:
        return ""
    counts: dict[str, int] = {}
    cost_map: dict[str, str] = {}
    for c in rev_cards:
        counts[c.full_name] = counts.get(c.full_name, 0) + 1
        tag = "Song" if c.is_song else c.card_type
        cost_map[c.full_name] = f"{c.cost}⬡/{tag}"
    return ", ".join(
        f"{ (f'{cnt}x ' if cnt > 1 else '') }{name} [{cost_map[name]}]"
        for name, cnt in counts.items()
    )


def simulate_and_format_replay(
    deck_a: Deck,
    deck_b: Deck,
    starting_player: int = 0,
    seed: int = 2026,
    lookahead_depth: int = 1,
    know_opponent_inks: bool = True,
) -> str:
    """
    Simulates a single 1v1 game between `deck_a` (P0) and `deck_b` (P1)
    and returns a rich turn-by-turn Polish play-by-play transcript.
    """
    rng = random.Random(seed)
    agent = TacticalHeuristicAgent(lookahead_depth=lookahead_depth)
    cards_a = deck_a.expand_cards()
    cards_b = deck_b.expand_cards()

    state = GameState.create_new_game(
        cards_a,
        cards_b,
        starting_player=starting_player,
        know_opponent_inks=know_opponent_inks,
        seed=seed,
    )

    p_names = (
        f"P0 [{deck_a.name}]",
        f"P1 [{deck_b.name}]",
    )

    lines: list[str] = []
    lines.append("=" * 105)
    lines.append(
        f"PEŁNY ZAPIS ROZGRYWKI (PLAY-BY-PLAY REPLAY) | Seed: {seed} | Lookahead Depth: {lookahead_depth}"
    )
    lines.append(
        f"  Gracz 0 (P0): {deck_a.name} ({'ON THE PLAY - Zaczyna' if starting_player == 0 else 'ON THE DRAW - Drugi'})"
    )
    lines.append(
        f"  Gracz 1 (P1): {deck_b.name} ({'ON THE PLAY - Zaczyna' if starting_player == 1 else 'ON THE DRAW - Drugi'})"
    )
    lines.append("=" * 105)

    # 1. Mulligan Phase
    lines.append("\n--- FAZA MULLIGANU (Wymiana kart startowych) ---")
    while state.phase.startswith("MULLIGAN"):
        act_p = state.active_player
        p = state.players[act_p]
        before_hand = [c.full_name for c in p.hand]
        legal = MoveGenerator.generate_legal_actions(state)
        chosen, sc, alts = agent.select_action_with_diagnostics(state, legal)
        mask = chosen.mulligan_keep_mask
        kept_names = [p.hand[i].full_name for i, k in enumerate(mask) if i < len(p.hand) and k]
        mull_names = [p.hand[i].full_name for i, k in enumerate(mask) if i < len(p.hand) and not k]

        LorcanaSimulator.step(state, chosen, rng=rng)
        after_hand = _format_hand(state.players[act_p])

        lines.append(f"  * {p_names[act_p]}:")
        lines.append(f"    Ręka początkowa (7): {', '.join(before_hand)}")
        lines.append(f"    Decyzja AI: {chosen.description} [Ocena: {sc:+.1f}]")
        lines.append(f"    Zatrzymano ({len(kept_names)}): {', '.join(kept_names) if kept_names else '(nic)'}")
        lines.append(f"    Wymieniono ({len(mull_names)}): {', '.join(mull_names) if mull_names else '(nic)'}")
        lines.append(f"    Ręka po Mulliganie:  {after_hand}")

    # 2. Main Game Loop (Turn by Turn)
    current_logged_turn = -1
    action_step_in_turn = 0
    total_steps = 0

    while state.phase != "GAME_OVER" and total_steps < 200:
        total_steps += 1
        act_p = state.active_player
        opp_p = 1 - act_p
        p = state.players[act_p]
        opp = state.players[opp_p]

        if state.turn_number != current_logged_turn:
            current_logged_turn = state.turn_number
            action_step_in_turn = 0
            round_num = (state.turn_number + 1) // 2
            discount_tags: list[str] = []
            if p.ink_drops > 0:
                discount_tags.append(f"+{p.ink_drops} Ink Drop = {p.total_payable_ink}⬡ dostępnych")
            if p.next_char_discount > 0:
                discount_tags.append(f"Zniżka na postać: -{p.next_char_discount}⬡")
            if p.next_princess_queen_discount > 0:
                discount_tags.append(f"Zniżka na Princess/Queen: -{p.next_princess_queen_discount}⬡")
            disc_str = f" ({', '.join(discount_tags)})" if discount_tags else ""
            lines.append("\n" + "-" * 105)
            lines.append(
                f"TURA GRY {state.turn_number:2d} (Runda {round_num} — Tura {round_num} gracza {p_names[act_p]}) | "
                f"Wynik Lore: P0={state.players[0].lore}/20 vs P1={state.players[1].lore}/20 | "
                f"Inkwell: {p.inkwell_available}/{p.inkwell_total}{disc_str}"
            )
            lines.append(f"  Ręka ({len(p.hand)}): {_format_hand(p)}")
            rev_opp_str = _format_revealed_hand(opp)
            if rev_opp_str:
                lines.append(
                    f"  Znane karty na ręce przeciwnika ({len(opp.revealed_hand_ids)}/{len(opp.hand)} widoczne dla AI): {rev_opp_str}"
                )
            lines.append(f"  Stół {p_names[act_p]}: {_format_board(p, opp)}")
            lines.append(f"  Stół {p_names[opp_p]}: {_format_board(opp, p)}")
            lines.append("-" * 105)

        legal = MoveGenerator.generate_legal_actions(state)
        if not legal:
            break

        chosen, sc, alts = agent.select_action_with_diagnostics(state, legal)
        action_step_in_turn += 1

        # Snapshot pre-step state to explain triggered effects
        prev_p_lore = p.lore
        prev_opp_lore = opp.lore
        prev_discount = p.next_char_discount
        prev_pq_discount = p.next_princess_queen_discount
        prev_p_drops = p.ink_drops
        prev_opp_drops = opp.ink_drops
        prev_p_board_ids = {e.instance_id: e.card.full_name for e in p.board}
        prev_opp_board_ids = {e.instance_id: e.card.full_name for e in opp.board}
        prev_p_hand_len = len(p.hand)
        prev_opp_hand_len = len(opp.hand)
        prev_opp_hand_names = [
            f"{c.full_name} [{'Song' if c.is_song else c.card_type}]" for c in opp.hand
        ]
        prev_opp_discard_len = len(opp.discard)
        prev_p_ink = p.inkwell_total
        prev_turn = state.turn_number
        prev_top_deck_opp = state.players[opp_p].deck[-1].full_name if state.players[opp_p].deck else None
        played_card = (
            p.hand[chosen.hand_index]
            if chosen.action_type in (ActionType.PLAY_CARD, ActionType.SHIFT_CARD, ActionType.SING_SONG)
            and 0 <= chosen.hand_index < len(p.hand)
            else None
        )

        LorcanaSimulator.step(state, chosen, rng=rng)

        alt_str = ""
        if alts:
            top_alts = ", ".join(f"{a.description} ({s:+.1f})" for a, s in alts[:2])
            alt_str = f" | Alternatywy: {top_alts}"

        if chosen.action_type == ActionType.PASS_TURN:
            # Check if END_OF_TURN triggers fired during PASS_TURN (before turn flipped)
            effects: list[str] = []
            if p.lore != prev_p_lore:
                effects.append(f"Efekt END_OF_TURN Lore: {prev_p_lore} -> {p.lore} ({p.lore - prev_p_lore:+d}◊)")
            if p.ink_drops > prev_p_drops or opp.ink_drops > prev_opp_drops:
                effects.append(
                    f"Efekt END_OF_TURN Ink Drop: P{act_p} (+{p.ink_drops - prev_p_drops}), P{opp_p} (+{opp.ink_drops - prev_opp_drops})"
                )
            if len(p.hand) != prev_p_hand_len:
                effects.append(f"Efekt END_OF_TURN Ręka: {prev_p_hand_len} -> {len(p.hand)} kart (np. Elinor / Cinderella / Temporary Shift)")
            effect_suffix = f" ==> [{' ; '.join(effects)}]" if effects else ""
            lines.append(f"  [{action_step_in_turn}] PASS_TURN (Koniec tury){effect_suffix}")
            if state.phase != "GAME_OVER" and state.turn_number != prev_turn and prev_top_deck_opp:
                lines.append(f"      -> {p_names[opp_p]} dobiera na początek swojej tury: {prev_top_deck_opp}")
        else:
            effects = []
            if p.lore != prev_p_lore:
                effects.append(f"Lore: {prev_p_lore} -> {p.lore} ({p.lore - prev_p_lore:+d}◊)")
            if opp.lore != prev_opp_lore:
                effects.append(f"Lore Przeciwnika: {prev_opp_lore} -> {opp.lore} ({opp.lore - prev_opp_lore:+d}◊)")
            if p.next_char_discount != prev_discount and p.next_char_discount > 0:
                effects.append(f"Aktywna zniżka na następną postać: -{p.next_char_discount}⬡")
            if p.next_princess_queen_discount != prev_pq_discount and p.next_princess_queen_discount > 0:
                effects.append(
                    f"Aktywna zniżka na następną postać Princess/Queen (Aurora): -{p.next_princess_queen_discount}⬡"
                )
            if p.inkwell_total != prev_p_ink:
                effects.append(f"Inkwell: {prev_p_ink} -> {p.inkwell_total} (Dostępne: {p.inkwell_available}⬡)")

            # Explicitly log opponent's revealed hand when Ursula - Deceiver / Mowgli / Elisa Maza / Lenny is played!
            if played_card is not None and "reveals their hand" in played_card.full_text.lower():
                effects.append(
                    f"Ujawniona ręka przeciwnika ({len(prev_opp_hand_names)}): "
                    f"{', '.join(prev_opp_hand_names) if prev_opp_hand_names else '(pusta ręka)'}"
                )
                if len(opp.discard) > prev_opp_discard_len:
                    disc_card = opp.discard[-1]
                    effects.append(f"Odrzucono z ręki przeciwnika: {disc_card.full_name} ({prev_opp_hand_len} -> {len(opp.hand)})")
                else:
                    effects.append("Brak pasującej karty (np. Song) do odrzucenia — ręka przeciwnika pozostaje jawna dla AI!")

            new_p_board_ids = {e.instance_id: e.card.full_name for e in p.board}
            new_opp_board_ids = {e.instance_id: e.card.full_name for e in opp.board}
            added_allies = [nm for iid, nm in new_p_board_ids.items() if iid not in prev_p_board_ids]
            removed_allies = [nm for iid, nm in prev_p_board_ids.items() if iid not in new_p_board_ids]
            removed_opps = [nm for iid, nm in prev_opp_board_ids.items() if iid not in new_opp_board_ids]

            if len(added_allies) > 1:
                effects.append(f"Na stół weszły: {', '.join(added_allies)} (Bonusowe darmowe dogranie!)")
            if removed_allies and chosen.action_type != ActionType.SHIFT_CARD:
                effects.append(f"Opuszcza nasz stół: {', '.join(removed_allies)}")
            if removed_opps:
                effects.append(f"Usunięto ze stołu przeciwnika: {', '.join(removed_opps)}")
            expected_p_hand = prev_p_hand_len - (
                1 if chosen.action_type in (ActionType.INK_CARD, ActionType.PLAY_CARD, ActionType.SHIFT_CARD, ActionType.SING_SONG) else 0
            ) - (1 if len(added_allies) > 1 else 0)
            if len(p.hand) > expected_p_hand and p.hand:
                effects.append(f"Dobrano na rękę: {p.hand[-1].full_name} ({prev_p_hand_len} -> {len(p.hand)})")
            elif len(p.hand) < expected_p_hand:
                effects.append(f"Odrzucono kartę z ręki ({prev_p_hand_len} -> {len(p.hand)})")
            if not (played_card is not None and "reveals their hand" in played_card.full_text.lower()):
                if len(opp.hand) < prev_opp_hand_len:
                    disc_name = opp.discard[-1].full_name if len(opp.discard) > prev_opp_discard_len else "kartę"
                    effects.append(f"Przeciwnik odrzucił z ręki: {disc_name} ({prev_opp_hand_len} -> {len(opp.hand)})")
                elif len(opp.hand) > prev_opp_hand_len:
                    if removed_opps:
                        effects.append(f"Karta wróciła na rękę przeciwnika ({prev_opp_hand_len} -> {len(opp.hand)})")
                    else:
                        effects.append(f"Przeciwnik dobrał na rękę: {opp.hand[-1].full_name} ({prev_opp_hand_len} -> {len(opp.hand)})")

            effect_suffix = f" ==> [{' | '.join(effects)}]" if effects else ""
            lines.append(
                f"  [{action_step_in_turn}] {chosen.action_type.value:<10}: {chosen.description} "
                f"[Ocena AI: {sc:+.1f}{alt_str}]{effect_suffix}"
            )

    # 3. Game Over Summary
    winner_id = (
        state.winner
        if state.winner is not None
        else (0 if state.players[0].lore >= state.players[1].lore else 1)
    )
    lines.append("\n" + "=" * 105)
    lines.append(
        f"KONIEC GRY (Tura gry {state.turn_number}, Runda {(state.turn_number + 1) // 2}) | ZWYCIĘZCA: {p_names[winner_id]} | "
        f"Powód: {state.victory_reason or 'lore_20'}"
    )
    lines.append(
        f"Końcowy wynik Lore: {p_names[0]} = {state.players[0].lore}/20  vs  "
        f"{p_names[1]} = {state.players[1].lore}/20"
    )
    lines.append(f"Końcowy stół P0: {_format_board(state.players[0], state.players[1])}")
    lines.append(f"Końcowy stół P1: {_format_board(state.players[1], state.players[0])}")
    lines.append("=" * 105)
    return "\n".join(lines)
