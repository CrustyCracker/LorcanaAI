"""
Game Simulator & DSL Effect Execution Engine for Disney Lorcana (Season 14 Core Constructed: Sets 9-14).
Compliant with Official Disney Lorcana Comprehensive Rules v2.3.0 (Effective October 16, 2026):
- Rule 1.8.1.2 & 2.3.3.2 (Deck-out loss checked when a player's turn ends with 0 cards in deck)
- Rule 1.13.1.2 (Ink Drop Counters: GAIN_INK_DROP / EACH_PLAYER_INK_DROP and paying ink costs)
- Rule 4.6.8 & 5.6.6 (Challenging Locations, 0 counter-damage, clearing location_instance_id when banished)
- Rule 4.7 (MOVE_TO_LOCATION)
- Rule 8.3.2 (Bodyguard optional enter-exerted)
- Rule 8.10.4.2 & 8.10.8 (All 6 Shift Variants: Standard, Classification, Universal, Duo, Combo, Temporary, Potato)
- Rule 8.14 (Vanish: banished when chosen by an opponent's Action effect)
- Rule 8.16 (Adventurous) & Rule 8.17 (Boost X)
"""

from __future__ import annotations
import random

from lorcana_ai.data.card_db import CardTemplate
from lorcana_ai.data.effect_dsl import (
    EffectClause,
    EffectOpType,
    KeywordFlag,
    TriggerType,
)
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction

try:
    import lorcana_rs  # type: ignore
    HAS_RUST_ENGINE = True
except ImportError:
    HAS_RUST_ENGINE = False


class LorcanaSimulator:
    """Executes legal GameActions and resolves DSL card triggers on a GameState."""

    @classmethod
    def step(cls, state: GameState, action: GameAction, rng: random.Random | None = None) -> GameState:
        if state.phase == "GAME_OVER":
            return state

        rng = rng or random.Random()
        p_id = state.active_player
        opp_id = 1 - p_id
        p = state.players[p_id]
        opp = state.players[opp_id]

        # 1. MULLIGAN PHASE (CR Rule 2.2.4)
        if action.action_type == ActionType.MULLIGAN:
            kept: list[CardTemplate] = []
            put_back: list[CardTemplate] = []
            mask = action.mulligan_keep_mask
            for idx, card in enumerate(p.hand):
                if idx < len(mask) and mask[idx]:
                    kept.append(card)
                else:
                    put_back.append(card)
            draw_needed = len(put_back)
            for _ in range(draw_needed):
                if p.deck:
                    kept.append(p.deck.pop())
            p.deck.extend(put_back)
            rng.shuffle(p.deck)
            p.hand = kept
            p.revealed_hand_ids.clear()

            other_id = 1 - state.starting_player
            if state.phase == f"MULLIGAN_P{state.starting_player}":
                state.phase = f"MULLIGAN_P{other_id}"
                state.active_player = other_id
            else:
                state.phase = "MAIN"
                state.active_player = state.starting_player
                state.turn_number = 1
                cls._begin_turn(state, draw_card=False)
            return state

        # 2. INK CARD (CR Rule 4.3)
        if action.action_type == ActionType.INK_CARD:
            if 0 <= action.hand_index < len(p.hand):
                inked_card = p.hand.pop(action.hand_index)
                p.inkwell_cards.append(inked_card)
                p.inkwell_total += 1
                p.inkwell_available += 1
                p.inked_this_turn = True
                if inked_card.card_id in p.revealed_hand_ids:
                    p.revealed_hand_ids.remove(inked_card.card_id)
            return state

        # 3. PLAY CARD / SHIFT CARD / SING SONG (CR Rule 4.4, 8.3.2, 8.10.4.2, 8.10.8, 8.11, 8.12)
        if action.action_type in (ActionType.PLAY_CARD, ActionType.SHIFT_CARD, ActionType.SING_SONG):
            if not (0 <= action.hand_index < len(p.hand)):
                return state
            card = p.hand.pop(action.hand_index)
            if card.card_id in p.revealed_hand_ids:
                p.revealed_hand_ids.remove(card.card_id)

            is_p2_first_turn = (state.turn_number == 2 and state.active_player != state.starting_player)
            if action.action_type == ActionType.PLAY_CARD:
                eff_cost = p.effective_play_cost(card, is_second_player_first_turn=is_p2_first_turn, opp=opp)
                p.pay_ink(eff_cost)
                p.consume_character_discounts(card)
            elif action.action_type == ActionType.SHIFT_CARD:
                eff_shift_cost = p.effective_shift_cost(card)
                p.pay_ink(eff_shift_cost)
                p.consume_character_discounts(card)
            elif action.action_type == ActionType.SING_SONG:
                if action.source_instance_id == -99:
                    # Sing Together X (CR Rule 8.11): exert ready dry characters until total singer cost >= X
                    needed = card.abilities.sing_together_cost or card.cost
                    accum = 0
                    ready_singers = sorted(
                        [e for e in p.characters_in_play if not e.exerted and not e.drying],
                        key=lambda e: (e.effective_lore(p, opp), -(e.card.abilities.singer_cost or e.card.cost)),
                    )
                    for ent in ready_singers:
                        ent.exerted = True
                        accum += ent.card.abilities.singer_cost or ent.card.cost
                        if accum >= needed:
                            break
                else:
                    for ent in p.characters_in_play:
                        if ent.instance_id == action.source_instance_id:
                            ent.exerted = True
                            break

            if card.is_song:
                p.songs_played_this_turn += 1

            # Place on board or discard if Action
            if card.card_type in ("Character", "Item", "Location"):
                # CR Rule 8.3.2: Bodyguard may enter play exerted
                inherited_exerted = bool(action.enter_exerted and card.card_type == "Character")
                inherited_drying = True
                inherited_damage = 0
                inherited_under_count = 0
                inherited_location_id = -1
                under_card: CardTemplate | None = None
                under_cards: list[CardTemplate] = []

                if action.action_type == ActionType.SHIFT_CARD:
                    if action.secondary_target_id != -1:
                        # CR Rule 8.10.4.2 & 8.10.8.3/8.10.8.4: Duo Shift / Combo Shift onto TWO characters
                        t_ids = {action.target_instance_id, action.secondary_target_id}
                        matched_ents = [e for e in p.board if e.instance_id in t_ids]
                        p.board = [e for e in p.board if e.instance_id not in t_ids]
                        if matched_ents:
                            inherited_exerted = any(e.exerted for e in matched_ents)
                            inherited_drying = any(e.drying for e in matched_ents)
                            inherited_damage = sum(e.damage for e in matched_ents)
                            inherited_under_count = sum(e.cards_under_count + 1 for e in matched_ents)
                            for e in matched_ents:
                                under_cards.extend(e.shifted_under_cards)
                                under_cards.append(e.card)
                            under_card = under_cards[-1] if under_cards else None
                    else:
                        # Single-target Shift (Standard, Classification, Universal, Combo-1, Temporary, Potato)
                        for idx_b, ent in enumerate(p.board):
                            if ent.instance_id == action.target_instance_id:
                                inherited_exerted = ent.exerted
                                inherited_drying = ent.drying if ent.card.card_type == "Character" else False
                                inherited_damage = ent.damage
                                inherited_under_count = ent.cards_under_count + 1
                                inherited_location_id = ent.location_instance_id
                                under_cards = list(ent.shifted_under_cards) + [ent.card]
                                under_card = ent.card
                                p.board.pop(idx_b)
                                break

                new_ent = BoardEntity(
                    instance_id=state.next_instance_id,
                    card=card,
                    exerted=inherited_exerted,
                    drying=inherited_drying if action.action_type == ActionType.SHIFT_CARD else True,
                    damage=inherited_damage,
                    cards_under_count=inherited_under_count,
                    shifted_under_card=under_card,
                    shifted_under_cards=under_cards,
                    location_instance_id=inherited_location_id,
                )
                state.next_instance_id += 1
                p.board.append(new_ent)
                if card.card_type == "Character":
                    p.chars_played_this_turn += 1
                    if "Princess" in card.subtypes:
                        p.princess_played_this_turn = True
                    if "opponents can't play actions until the start of your next turn" in card.full_text.lower():
                        opp.cant_play_actions_this_turn = True
                cls._resolve_clauses(
                    state, p_id, card, TriggerType.ON_PLAY, source_id=new_ent.instance_id, is_action_source=False
                )
            else:
                # Action / Song card (CR Rule 5.3)
                cls._resolve_clauses(
                    state, p_id, card, TriggerType.ON_PLAY, source_id=-1, is_action_source=True
                )
                p.discard.append(card)

            cls._check_victory(state)
            return state

        # 4. QUEST (CR Rule 4.5)
        if action.action_type == ActionType.QUEST:
            for ent in p.characters_in_play:
                if ent.instance_id == action.source_instance_id and ent.can_quest():
                    ent.exerted = True
                    p.lore += ent.effective_lore(p, opp)
                    if ent.has_keyword(KeywordFlag.SUPPORT):
                        for ally in p.characters_in_play:
                            if ally.instance_id != ent.instance_id:
                                ally.temp_strength_mod += ent.effective_strength
                                break
                    # Mickey Mouse - Bob Cratchit (Set 11): "Whenever this character quests, put the top card of your deck facedown under him."
                    if "whenever this character quests, put the top card of your deck facedown under" in ent.card.full_text.lower():
                        if p.deck:
                            p.deck.pop()
                            ent.cards_under_count += 1
                    # Mickey Mouse - Expedition Leader: "While this character is exerted, whenever one of your other characters quests, chosen opposing character gets -2 ¤ until the start of your next turn."
                    for ally in p.characters_in_play:
                        if (
                            ally.instance_id != ent.instance_id
                            and ally.exerted
                            and "whenever one of your other characters quests" in ally.card.full_text.lower()
                        ):
                            opp_targets = [
                                e for e in opp.characters_in_play
                                if e.effective_strength > 0 and not e.has_keyword(KeywordFlag.WARD)
                            ]
                            if opp_targets:
                                best_opp = max(opp_targets, key=lambda e: (e.effective_strength, e.card.lore))
                                best_opp.temp_strength_mod -= 2
                    cls._resolve_clauses(state, p_id, ent.card, TriggerType.ON_QUEST, source_id=ent.instance_id)
                    break
            cls._check_victory(state)
            return state

        # 5. CHALLENGE CHARACTER OR LOCATION (CR Rule 4.6 & 4.6.8)
        if action.action_type == ActionType.CHALLENGE:
            attacker = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            defender = next(
                (e for e in opp.board if e.instance_id == action.target_instance_id and e.card.card_type in ("Character", "Location")),
                None,
            )
            if attacker is not None and defender is not None:
                attacker.exerted = True
                p.challenged_this_turn = True
                p_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in p.characters_in_play)
                opp_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in opp.characters_in_play)
                has_lead_detective = any(
                    "your detective characters gain alert and resist +2" in e.card.full_text.lower()
                    for e in p.characters_in_play
                )

                base_atk = attacker.effective_willpower if p_uses_wp else attacker.effective_strength
                if "while you have a villain character in play" in attacker.card.full_text.lower():
                    if any("Villain" in e.card.subtypes for e in p.characters_in_play):
                        base_atk += 2
                atk_power = (
                    base_atk
                    + attacker.card.abilities.challenger_bonus
                    + attacker.temp_challenger_mod
                )
                def_resist = defender.effective_resist(opp)
                dmg_to_def = max(0, atk_power - def_resist)
                # Rafiki - Mystical Fighter (Set 9): "Whenever he challenges a Hyena character, this character takes no damage"
                takes_no_dmg = "challenges a hyena character" in attacker.card.full_text.lower() and "Hyena" in defender.card.subtypes
                defender.damage += dmg_to_def

                # CR Rule 4.6.8.2: Locations do not have Strength and do not deal damage when challenged
                if defender.card.card_type == "Character" and not takes_no_dmg:
                    base_def = defender.effective_willpower if opp_uses_wp else defender.effective_strength
                    if "while you have a villain character in play" in defender.card.full_text.lower():
                        if any("Villain" in e.card.subtypes for e in opp.characters_in_play):
                            base_def += 2
                    atk_resist = attacker.effective_resist(p)
                    dmg_to_atk = max(0, base_def - atk_resist)
                    attacker.damage += dmg_to_atk
                    # The Prince - Searching for His Love: "During your turn, whenever this character banishes another character in a challenge, draw a card."
                    if defender.is_banished and "banishes another character in a challenge, draw a card" in attacker.card.full_text.lower():
                        if p.deck:
                            p.hand.append(p.deck.pop())

                cls._cleanup_banished(state)
                cls._check_victory(state)
            return state

        # 6. MOVE CHARACTER TO LOCATION (CR Rule 4.7)
        if action.action_type == ActionType.MOVE_TO_LOCATION:
            my_char = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            loc = next((e for e in p.locations_in_play if e.instance_id == action.target_instance_id), None)
            if my_char is not None and loc is not None and p.total_payable_ink >= loc.card.move_cost:
                p.pay_ink(loc.card.move_cost)
                my_char.location_instance_id = loc.instance_id
            cls._check_victory(state)
            return state

        # 7. USE ACTIVATED ABILITY OR BOOST X (CR Rule 6.2 & 8.17)
        if action.action_type == ActionType.USE_ABILITY:
            ent = next((e for e in p.board if e.instance_id == action.source_instance_id), None)
            if ent is not None:
                if action.target_instance_id == -88:
                    # Boost X mechanic (CR Rule 8.17): pay boost_cost ink, put top card of deck facedown under character
                    b_cost = ent.card.abilities.boost_cost or 1
                    if not ent.boosted_this_turn and p.total_payable_ink >= b_cost and p.deck:
                        p.pay_ink(b_cost)
                        p.deck.pop()
                        ent.cards_under_count += 1
                        ent.boosted_this_turn = True
                else:
                    for cl in ent.card.abilities.clauses:
                        if cl.trigger == TriggerType.ACTIVATED:
                            if cl.requires_exert and (ent.exerted or (ent.card.card_type == "Character" and ent.drying)):
                                continue
                            if cl.cost_ink > p.total_payable_ink:
                                continue
                            if cl.requires_exert:
                                ent.exerted = True
                            p.pay_ink(cl.cost_ink)
                            cls._execute_single_clause(state, p_id, cl, source_id=ent.instance_id, card_text=ent.card.full_text.lower())
                            # Junior Woodchuck Guidebook (Set 10): "⟳, 1 ⬡, Banish this item — Draw 2 cards."
                            if "banish this item" in ent.card.full_text.lower() and ent in p.board:
                                p.board.remove(ent)
                                p.discard.append(ent.card)
                            break
            cls._check_victory(state)
            return state

        # 8. PASS TURN (End of Turn Triggers + Temporary Shift Return + Rule 1.8.1.2 Deck-out Check)
        if action.action_type == ActionType.PASS_TURN:
            cls._end_turn(state)
            if state.phase == "GAME_OVER" or cls._check_victory(state):
                return state
            state.active_player = opp_id
            state.turn_number += 1
            if state.turn_number > 65:
                state.phase = "GAME_OVER"
                state.winner = 0 if state.players[0].lore >= state.players[1].lore else 1
                state.victory_reason = "turn_limit_lore_tiebreak"
                return state
            cls._begin_turn(state, draw_card=True)
            return state

        return state

    @classmethod
    def _end_turn(cls, state: GameState) -> None:
        p_id = state.active_player
        p = state.players[p_id]

        # 1. Resolve END_OF_TURN triggers
        for ent in list(p.board):
            cls._resolve_clauses(state, p_id, ent.card, TriggerType.END_OF_TURN, source_id=ent.instance_id)

        # Clear cant_challenge_this_turn, cant_play_actions_this_turn, and same-turn buffs for the player whose turn just ended
        p.cant_challenge_this_turn = False
        p.cant_play_actions_this_turn = False
        for ent in p.board:
            if ent.temp_strength_mod > 0:
                ent.temp_strength_mod = 0
            ent.temp_challenger_mod = 0
            ent.temp_lore_mod = 0
            ent.temp_evasive = False
            ent.temp_adventurous = False

        # 2. Resolve Temporary Shift (CR Rule 8.10.8.5):
        # "At the end of the turn you play a character with Temporary Shift, remove all damage from it and return it to your hand."
        # Only the top card returns to hand; under-card(s) remain in play!
        updated_board: list[BoardEntity] = []
        for ent in p.board:
            if ent.card.abilities.is_temporary_shift:
                p.hand.append(ent.card)
                p.revealed_hand_ids.append(ent.card.card_id)
                under_list = ent.shifted_under_cards or ([ent.shifted_under_card] if ent.shifted_under_card else [])
                for u_idx, u_card in enumerate(under_list):
                    if u_card is not None:
                        updated_board.append(
                            BoardEntity(
                                instance_id=ent.instance_id if u_idx == 0 else state.next_instance_id,
                                card=u_card,
                                exerted=ent.exerted,
                                drying=False,
                                damage=0,
                                cards_under_count=max(0, ent.cards_under_count - len(under_list)),
                                location_instance_id=ent.location_instance_id,
                            )
                        )
                        if u_idx > 0:
                            state.next_instance_id += 1
            else:
                updated_board.append(ent)
        p.board = updated_board

        # 3. CR Rule 1.8.1.2 & 2.3.3.2: "If a player's turn ends with no cards in their deck, that player loses the game."
        if len(p.deck) == 0 and state.phase != "GAME_OVER":
            state.phase = "GAME_OVER"
            state.winner = 1 - p_id
            state.victory_reason = "deck_out"

    @classmethod
    def _begin_turn(cls, state: GameState, draw_card: bool) -> None:
        p_id = state.active_player
        p = state.players[p_id]
        opp = state.players[1 - p_id]

        # Ready step (CR Rule 2.3.1)
        p.inkwell_available = p.inkwell_total
        p.next_char_discount = 0
        p.next_princess_queen_discount = 0
        p.inked_this_turn = False
        p.challenged_this_turn = False
        p.chars_played_this_turn = 0
        p.princess_played_this_turn = False
        p.songs_played_this_turn = 0
        p.cards_discarded_this_turn = 0
        has_angela_night_warrior = any(
            "lose the stone by day" in e.card.full_text.lower() for e in p.characters_in_play
        )
        for ent in p.board:
            # STONE BY DAY (Sets 10, 11, 14 Gargoyles): "If you have 3 or more cards in your hand, this character can't ready."
            stone_locked = (
                "if you have 3 or more cards in your hand, this character can't ready" in ent.card.full_text.lower()
                and len(p.hand) >= 3
                and not has_angela_night_warrior
            )
            if not stone_locked:
                ent.exerted = False
            ent.drying = False
            ent.boosted_this_turn = False
            ent.temp_challenger_mod = 0
            ent.temp_lore_mod = 0
        # Clear "until the start of your next turn" strength debuffs placed on opp's characters
        for ent in opp.board:
            if ent.temp_strength_mod < 0:
                ent.temp_strength_mod = 0

        # Set step (CR Rule 2.3.2): passive location lore & start-of-turn triggers
        for loc in p.locations_in_play:
            loc_lore = loc.effective_lore(p, opp)
            if loc_lore > 0:
                p.lore += loc_lore

        for ent in list(p.board):
            cls._resolve_clauses(state, p_id, ent.card, TriggerType.START_OF_TURN, source_id=ent.instance_id)

        if cls._check_victory(state):
            return

        # Draw step (CR Rule 2.3.3 & 1.8.1.2):
        # Drawing from an empty deck does NOT immediately end the game; per Rule 1.8.1.2 & 2.3.3.2,
        # the player loses if their turn ends with no cards in their deck.
        if draw_card and p.deck:
            p.hand.append(p.deck.pop())

    @classmethod
    def _resolve_clauses(
        cls,
        state: GameState,
        controller_id: int,
        card: CardTemplate,
        trigger: TriggerType,
        source_id: int,
        is_action_source: bool = False,
    ) -> None:
        p = state.players[controller_id]
        opp = state.players[1 - controller_id]
        source_ent = next((e for e in p.board if e.instance_id == source_id), None)
        card_lower = card.full_text.lower()
        if trigger == TriggerType.END_OF_TURN:
            if "3 or more exerted characters" in card_lower and sum(1 for e in p.characters_in_play if e.exerted) < 3:
                return
            if "if this character is exerted" in card_lower and (source_ent is None or not source_ent.exerted):
                return
            if "if you played 2 or more characters this turn" in card_lower and p.chars_played_this_turn < 2:
                return
            if "if you played a princess character this turn" in card_lower and (not p.princess_played_this_turn or len(p.hand) == 0):
                return
            if "if 2 or more cards were put into your discard this turn" in card_lower and p.cards_discarded_this_turn < 2:
                return
        if trigger == TriggerType.ON_PLAY:
            if "if you played another character this turn" in card_lower and p.chars_played_this_turn < 2:
                return
            if "if you have another detective character in play" in card_lower:
                has_other_det = any(
                    "Detective" in e.card.subtypes and e.instance_id != source_id
                    for e in p.characters_in_play
                )
                if not has_other_det:
                    return
        if trigger == TriggerType.ON_QUEST:
            if "if you played another character this turn" in card_lower and p.chars_played_this_turn < 1:
                return
            if "if an opponent has more cards in their hand than you" in card_lower and len(opp.hand) <= len(p.hand):
                return
        if trigger == TriggerType.ACTIVATED:
            if "if you have 10 or more cards in your discard" in card_lower and len(p.discard) < 10:
                return

        for cl in card.abilities.clauses:
            if cl.trigger == trigger:
                if cl.requires_card_under and (source_ent is None or source_ent.cards_under_count <= 0):
                    continue
                cls._execute_single_clause(
                    state,
                    controller_id,
                    cl,
                    source_id=source_id,
                    is_action_source=is_action_source,
                    card_text=card_lower,
                )

    @classmethod
    def _apply_vanish_if_action(cls, state: GameState, target: BoardEntity, is_action_source: bool) -> None:
        """
        CR Rule 8.14.1 (Vanish): "When this character is chosen by an opponent as part of resolving
        an action's effect, banish it."
        """
        if is_action_source and target.has_keyword(KeywordFlag.VANISH):
            target.damage = max(target.damage, target.effective_willpower + 10)

    @classmethod
    def _execute_single_clause(
        cls,
        state: GameState,
        controller_id: int,
        cl: EffectClause,
        source_id: int,
        is_action_source: bool = False,
        card_text: str = "",
    ) -> None:
        p = state.players[controller_id]
        opp_id = 1 - controller_id
        opp = state.players[opp_id]
        op = cl.op
        source_ent = next((e for e in p.board if e.instance_id == source_id), None)
        if not card_text and source_ent is not None:
            card_text = source_ent.card.full_text.lower()

        if op == EffectOpType.DRAW:
            if "for each princess character you have in play" in card_text:
                # Besties, Assemble! (Set 14): "Draw a card for each Princess character you have in play."
                p_cnt = sum(1 for e in p.characters_in_play if "Princess" in e.card.subtypes)
                for _ in range(p_cnt):
                    if p.deck:
                        p.hand.append(p.deck.pop())
            elif "look at the top 3 cards of your deck" in card_text and "detective" in card_text:
                # Judy Hopps - Uncovering Clues (Set 10): "look at the top 3 cards of your deck. You may reveal a Detective character card and put it into your hand. Put the rest on the bottom"
                looked: list[CardTemplate] = [p.deck.pop() for _ in range(min(3, len(p.deck)))]
                valid_det = [(i, c) for i, c in enumerate(looked) if c.card_type == "Character" and "Detective" in c.subtypes]
                if valid_det:
                    best_i, chosen_c = max(valid_det, key=lambda item: (len(item[1].abilities.clauses), item[1].lore * 2 + item[1].cost))
                    looked.pop(best_i)
                    p.hand.append(chosen_c)
                    p.revealed_hand_ids.append(chosen_c.card_id)
                for rem_c in looked:
                    p.deck.insert(0, rem_c)
            elif "look at the top 3 cards of your deck" in card_text and "action card" in card_text:
                # Meilin Lee - Poised and Ready (Set 14): "look at the top 3 cards of your deck. You may reveal an action card and put it into your hand. Put the rest on the bottom"
                looked = [p.deck.pop() for _ in range(min(3, len(p.deck)))]
                valid_act = [(i, c) for i, c in enumerate(looked) if c.card_type == "Action"]
                if valid_act:
                    best_i, chosen_a = max(valid_act, key=lambda item: (item[1].is_song, item[1].cost))
                    looked.pop(best_i)
                    p.hand.append(chosen_a)
                    p.revealed_hand_ids.append(chosen_a.card_id)
                for rem_c in looked:
                    p.deck.insert(0, rem_c)
            elif (
                "look at the top 4 cards of your deck" in card_text
                and "cost 2 or less" in card_text
            ):
                # Nani - Stage Manager: look at top 4, reveal a character with cost <= 2 and put into hand, rest on bottom
                looked = [p.deck.pop() for _ in range(min(4, len(p.deck)))]
                valid_c2 = [
                    (i, c) for i, c in enumerate(looked) if c.card_type == "Character" and c.cost <= 2
                ]
                if valid_c2:
                    best_i, chosen_c = max(
                        valid_c2,
                        key=lambda item: (
                            len(item[1].abilities.clauses),
                            item[1].lore * 2 + item[1].willpower,
                        ),
                    )
                    looked.pop(best_i)
                    p.hand.append(chosen_c)
                    p.revealed_hand_ids.append(chosen_c.card_id)
                for rem_c in looked:
                    p.deck.insert(0, rem_c)
            elif "look at the top 2 cards of your deck. put one into your hand" in card_text:
                # Develop Your Brain (Set 9): look at top 2, put 1 into hand, 1 on bottom
                looked = [p.deck.pop() for _ in range(min(2, len(p.deck)))]
                if looked:
                    best_i, chosen_c = max(
                        enumerate(looked),
                        key=lambda item: (
                            1 <= item[1].cost <= p.inkwell_total + 1,
                            len(item[1].abilities.clauses),
                            item[1].lore,
                        ),
                    )
                    looked.pop(best_i)
                    p.hand.append(chosen_c)
                    for rem_c in looked:
                        p.deck.insert(0, rem_c)
            elif "return" in card_text and "character card" in card_text and "from your discard to your hand" in card_text:
                max_c = 2 if "cost 2 or less" in card_text else 99
                req_det = "detective" in card_text and "chosen detective character card" in card_text
                char_grave = [
                    (i, c) for i, c in enumerate(p.discard)
                    if c.card_type == "Character" and c.cost <= max_c and (not req_det or "Detective" in c.subtypes)
                ]
                if char_grave:
                    best_i, chosen_c = max(char_grave, key=lambda item: (item[1].cost + item[1].lore * 2, item[1].willpower))
                    p.discard.pop(best_i)
                    p.hand.append(chosen_c)
                    p.revealed_hand_ids.append(chosen_c.card_id)
            elif "return" in card_text and "song card" in card_text and "from your discard to your hand" in card_text:
                if "you may pay 1" in card_text:
                    if p.total_payable_ink < 1:
                        return
                    song_grave = [(i, c) for i, c in enumerate(p.discard) if c.is_song]
                    if not song_grave:
                        return
                    p.pay_ink(1)
                else:
                    song_grave = [(i, c) for i, c in enumerate(p.discard) if c.is_song]
                if song_grave:
                    best_i, chosen_s = max(song_grave, key=lambda item: item[1].cost)
                    p.discard.pop(best_i)
                    p.hand.append(chosen_s)
                    p.revealed_hand_ids.append(chosen_s.card_id)
            elif "each player draws until they have 3 cards in their hand" in card_text:
                # Demona - Scourge of the Wyvern Clan (Set 10)
                for pl in (p, opp):
                    while len(pl.hand) < 3 and pl.deck:
                        pl.hand.append(pl.deck.pop())
            else:
                for _ in range(cl.amount):
                    if p.deck:
                        p.hand.append(p.deck.pop())

        elif op == EffectOpType.LOOT_SELF:
            # Max Goof - Karaoke Star: "you may choose and discard a song card. If you do, draw 2 cards."
            if "choose and discard a song card" in card_text:
                song_indices = [i for i, c in enumerate(p.hand) if c.is_song]
                if not song_indices:
                    return
                best_s_idx = max(
                    song_indices,
                    key=lambda i: (sum(1 for h in p.hand if h.card_id == p.hand[i].card_id) > 1, -p.hand[i].cost),
                )
                discarded = p.hand.pop(best_s_idx)
                p.discard.append(discarded)
                p.cards_discarded_this_turn += 1
                if discarded.card_id in p.revealed_hand_ids:
                    p.revealed_hand_ids.remove(discarded.card_id)
                for _ in range(cl.amount):
                    if p.deck:
                        p.hand.append(p.deck.pop())
                return

            has_reanimator = any(
                any(c.op == EffectOpType.REANIMATE_CHAR for c in h.abilities.clauses)
                for h in (p.hand + p.deck)
            )
            # Optional loot (e.g. Bobby Zimuruski): in non-reanimator aggro, skip looting if hand is small and every card is a unique high-value curve play!
            if "you may" in card_text and not has_reanimator and len(p.hand) <= 2:
                has_dup_or_brick = any(
                    sum(1 for h in p.hand if h.card_id == c.card_id) > 1 or c.cost > p.inkwell_total + 2
                    for c in p.hand
                )
                if not has_dup_or_brick:
                    return

            for _ in range(cl.amount):
                if p.deck:
                    p.hand.append(p.deck.pop())
            if p.hand:
                def _loot_discard_priority(idx: int) -> tuple[int, int, int]:
                    cand = p.hand[idx]
                    copies = sum(1 for h in p.hand if h.card_id == cand.card_id)
                    # 1. If we play Circle of Life, pitching a 6+ cost Character (Prince Phillip, Milo Thatch, etc.) is top priority!
                    if has_reanimator and cand.card_type == "Character" and cand.cost >= 6:
                        return (100, cand.cost, cand.lore)
                    # 2. Do NOT pitch Circle of Life itself unless we have multiple copies
                    if any(c.op == EffectOpType.REANIMATE_CHAR for c in cand.abilities.clauses):
                        return (10 if copies > 1 else -50, 0, 0)
                    # 3. Pitch extra songs to fuel Max Goof - Karaoke Star (+3 Lore at 5 songs in discard) or Chart Topper
                    if cand.is_song and copies > 1:
                        return (45, cand.cost, 0)
                    # 4. In non-reanimator decks (Princess Aggro), protect key engine/payoff cards (Elinor, Peacekeeper, Dale, Guiding the Tribe)!
                    if not has_reanimator:
                        is_key_engine = (
                            "3 or more exerted characters" in cand.full_text.lower()
                            or cand.abilities.shift_cost is not None
                            or cand.abilities.deals_willpower_damage_in_challenges
                            or any(c.op == EffectOpType.PLAY_COST1_CHAR_FREE for c in cand.abilities.clauses)
                        )
                        if is_key_engine and copies == 1 and cand.cost <= p.inkwell_total + 3:
                            return (-30, -cand.cost, -cand.lore)
                        if copies > 1:
                            return (40, -cand.lore, cand.cost)
                        if cand.cost > p.inkwell_total + 3:
                            return (30, cand.cost, -cand.lore)
                        # Prefer pitching a spare 1-drop or vanilla card over mid-curve engines
                        return (15 if cand.cost == 1 else 5, -cand.lore, -cand.cost)
                    # 5. Otherwise pitch expensive unplayables or duplicates
                    return (20 if cand.cost > p.inkwell_total + 2 else 5, cand.cost, -cand.lore)

                best_disc_idx = max(range(len(p.hand)), key=_loot_discard_priority)
                discarded = p.hand.pop(best_disc_idx)
                p.discard.append(discarded)
                p.cards_discarded_this_turn += 1
                if discarded.card_id in p.revealed_hand_ids:
                    p.revealed_hand_ids.remove(discarded.card_id)

        elif op == EffectOpType.DRAW_PER_INK_TYPE:
            # With a Few Good Friends: draw a card for each different ink type of characters you have in play
            inks_in_play = {ink for e in p.characters_in_play for ink in e.card.inks}
            draw_cnt = max(1, len(inks_in_play)) if p.characters_in_play else 0
            for _ in range(draw_cnt):
                if p.deck:
                    p.hand.append(p.deck.pop())

        elif op == EffectOpType.REANIMATE_CHAR:
            # Circle of Life: Play a character from your discard for free!
            char_candidates = [
                (idx, c) for idx, c in enumerate(p.discard) if c.card_type == "Character"
            ]
            if char_candidates:
                opp_has_damaged = any(e.damage > 0 for e in opp.characters_in_play)
                def _reanimate_score(item: tuple[int, CardTemplate]) -> tuple[int, int, int]:
                    _, c = item
                    wipes_damaged = opp_has_damaged and any(
                        cl_x.op == EffectOpType.BANISH_ALL_DAMAGED_OPP for cl_x in c.abilities.clauses
                    )
                    return (1 if wipes_damaged else 0, c.cost + c.lore * 2, c.strength + c.willpower)

                best_idx, chosen_char = max(char_candidates, key=_reanimate_score)
                p.discard.pop(best_idx)
                new_ent = BoardEntity(
                    instance_id=state.next_instance_id,
                    card=chosen_char,
                    exerted=False,
                    drying=True,
                )
                state.next_instance_id += 1
                p.board.append(new_ent)
                p.chars_played_this_turn += 1
                if p.is_princess_or_queen(chosen_char):
                    p.princess_played_this_turn += 1
                cls._resolve_clauses(
                    state, controller_id, chosen_char, TriggerType.ON_PLAY, source_id=new_ent.instance_id
                )

        elif op == EffectOpType.REPLAY_SONG_FROM_DISCARD:
            # Max Goof - Chart Topper: play a song card with cost <= threshold (4) from discard for free, then put on bottom of deck
            max_cost = cl.threshold if cl.threshold > 0 else 4
            song_candidates = [
                (idx, c) for idx, c in enumerate(p.discard) if c.is_song and c.cost <= max_cost
            ]
            if song_candidates:
                opp_has_damaged = any(e.damage > 0 for e in opp.characters_in_play)
                def _song_score(item: tuple[int, CardTemplate]) -> tuple[int, int]:
                    _, c = item
                    has_dmg_aoe = opp_has_damaged and any(
                        cl_x.op == EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP for cl_x in c.abilities.clauses
                    )
                    return (1 if has_dmg_aoe else 0, c.cost)

                s_idx, chosen_song = max(song_candidates, key=_song_score)
                p.discard.pop(s_idx)
                p.songs_played_this_turn += 1
                cls._resolve_clauses(
                    state, controller_id, chosen_song, TriggerType.ON_PLAY, source_id=-1, is_action_source=True
                )
                p.deck.insert(0, chosen_song)

        elif op == EffectOpType.PLAY_COST1_CHAR_FREE:
            # Pocahontas - Guiding the Tribe: "When you play this character, you may play a character with cost 1 for free."
            cost1_candidates = [
                (idx, c) for idx, c in enumerate(p.hand) if c.card_type == "Character" and c.cost <= 1
            ]
            if cost1_candidates:
                best_idx, chosen_c1 = max(
                    cost1_candidates,
                    key=lambda item: (
                        len(item[1].abilities.clauses),
                        item[1].lore * 2 + item[1].willpower + item[1].strength,
                    ),
                )
                p.hand.pop(best_idx)
                if chosen_c1.card_id in p.revealed_hand_ids:
                    p.revealed_hand_ids.remove(chosen_c1.card_id)
                new_ent = BoardEntity(
                    instance_id=state.next_instance_id,
                    card=chosen_c1,
                    exerted=False,
                    drying=True,
                )
                state.next_instance_id += 1
                p.board.append(new_ent)
                p.chars_played_this_turn += 1
                if p.is_princess_or_queen(chosen_c1):
                    p.princess_played_this_turn += 1
                cls._resolve_clauses(
                    state, controller_id, chosen_c1, TriggerType.ON_PLAY, source_id=new_ent.instance_id
                )

        elif op == EffectOpType.FOG_CHALLENGES:
            # Pocahontas - Peacekeeper: "When you play this character, if you used Shift to play her and none of your characters challenged this turn, characters can't challenge until the start of your next turn."
            source_ent = next((e for e in p.board if e.instance_id == source_id), None)
            if source_ent is not None and source_ent.cards_under_count > 0 and not p.challenged_this_turn:
                p.cant_challenge_this_turn = True
                opp.cant_challenge_this_turn = True

        elif op == EffectOpType.REVEAL_TOP_CHAR_BOTH:
            # Mike Wazowski - Heroic Climber: "When you play this character and whenever he quests, each player reveals the top card of their deck. If it's a character card, they put it into their hand. Otherwise, they put it on the bottom of their deck."
            for pl in (p, opp):
                if pl.deck:
                    top_c = pl.deck.pop()
                    if top_c.card_type == "Character":
                        pl.hand.append(top_c)
                        pl.revealed_hand_ids.append(top_c.card_id)
                    else:
                        pl.deck.insert(0, top_c)

        elif op == EffectOpType.DISCOUNT_NEXT_CHAR:
            p.next_char_discount += cl.amount

        elif op == EffectOpType.DISCOUNT_NEXT_PRINCESS_QUEEN:
            # Aurora - Holding Court (Set 9): "Whenever this character quests, you pay 1 ⬡ less for the next Princess or Queen character you play this turn."
            p.next_princess_queen_discount += cl.amount

        elif op == EffectOpType.EXERT_OPP_CHAR:
            # Isis Vanderchill / Peter Pan's Shadow / Demona - Scourge of the Wyvern Clan
            if "all opposing characters" in card_text:
                for e in opp.characters_in_play:
                    e.exerted = True
            else:
                ready_opps = [e for e in opp.characters_in_play if not e.exerted and not e.has_keyword(KeywordFlag.WARD)]
                if not ready_opps:
                    ready_opps = [e for e in opp.characters_in_play if not e.has_keyword(KeywordFlag.WARD)]
                if ready_opps:
                    best_opp = max(ready_opps, key=lambda e: (e.card.lore * 2 + e.effective_strength, e.card.cost))
                    best_opp.exerted = True
                    if "can't ready at the start of their next turn" in card_text:
                        best_opp.cant_ready_next_turn = True
                    cls._apply_vanish_if_action(state, best_opp, is_action_source)
                    cls._cleanup_banished(state)

        elif op == EffectOpType.INK_OPP_CHAR:
            # Hades - Infernal Schemer (Set 9) / Bellwether - Highly Qualified (Set 14) / Scram! (Set 14)
            requires_discard = "choose and discard a card" in card_text
            if requires_discard and not p.hand:
                return
            max_cost_ink = cl.threshold if cl.threshold > 0 else 99
            targets = [
                e for e in opp.characters_in_play
                if e.card.cost <= max_cost_ink and not e.has_keyword(KeywordFlag.WARD)
            ]
            if targets:
                best_t = max(targets, key=lambda e: (e.card.lore * 3 + e.effective_strength + e.card.cost))
                if requires_discard:
                    disc_idx = min(
                        range(len(p.hand)),
                        key=lambda i: (p.hand[i].lore, len(p.hand[i].abilities.clauses), p.hand[i].cost),
                    )
                    discarded = p.hand.pop(disc_idx)
                    p.discard.append(discarded)
                    p.cards_discarded_this_turn += 1
                    if discarded.card_id in p.revealed_hand_ids:
                        p.revealed_hand_ids.remove(discarded.card_id)
                if is_action_source and best_t.has_keyword(KeywordFlag.VANISH):
                    best_t.damage = best_t.effective_willpower + 10
                    cls._cleanup_banished(state)
                else:
                    opp.board.remove(best_t)
                    cls._resolve_clauses(
                        state, opp_id, best_t.card, TriggerType.ON_LEAVE_PLAY, source_id=best_t.instance_id
                    )
                    opp.inkwell_cards.append(best_t.card)
                    opp.inkwell_total += 1
                    for u_c in best_t.shifted_under_cards:
                        opp.inkwell_cards.append(u_c)
                        opp.inkwell_total += 1

        elif op == EffectOpType.HEAL_DRAW:
            damaged = [c for c in p.characters_in_play if c.damage > 0]
            if cl.amount >= 99:
                # Ohana Means Family (Set 14): "Remove all damage from all of your damaged characters. Draw a card for each character that had damage removed this way."
                healed_chars = 0
                for c in damaged:
                    c.damage = 0
                    healed_chars += 1
                for _ in range(healed_chars):
                    if p.deck:
                        p.hand.append(p.deck.pop())
            elif cl.amount == 0:
                # Judy Hopps - Helpful Officer (Set 12): "Remove up to 2 damage from chosen character." (no card draw)
                if damaged:
                    target = max(damaged, key=lambda e: (e.card.lore * 2 + e.card.cost, e.damage))
                    target.damage = max(0, target.damage - 2)
            else:
                # Rapunzel - Gifted with Healing (Set 9): "Remove up to 3 damage from one of your characters. Draw a card for each 1 damage removed this way."
                if damaged:
                    target = max(damaged, key=lambda e: e.damage)
                    healed = min(cl.amount, target.damage)
                    target.damage -= healed
                    for _ in range(healed):
                        if p.deck:
                            p.hand.append(p.deck.pop())

        elif op == EffectOpType.GAIN_INK_DROP:
            # CR Rule 1.13.1.2: Active player gets Ink Drop counter(s)
            p.ink_drops += cl.amount

        elif op == EffectOpType.EACH_PLAYER_INK_DROP:
            # CR Rule 1.13.1.2: Each player gets Ink Drop counter(s)
            p.ink_drops += cl.amount
            opp.ink_drops += cl.amount

        elif op == EffectOpType.RAMP_INK:
            if cl.requires_exert:
                if len(p.hand) >= 2 and p.inkwell_total < 9:
                    best_idx = max(range(len(p.hand)), key=lambda i: (not p.hand[i].inkable, p.hand[i].cost))
                    c = p.hand.pop(best_idx)
                    p.inkwell_cards.append(c)
                    p.inkwell_total += 1
            elif p.deck:
                c = p.deck.pop()
                p.inkwell_cards.append(c)
                p.inkwell_total += 1

        elif op == EffectOpType.GAIN_LORE:
            p.lore += cl.amount

        elif op == EffectOpType.LOSE_OPP_LORE:
            opp.lore = max(0, opp.lore - cl.amount)

        elif op == EffectOpType.DEAL_DAMAGE:
            dmg_amt = cl.amount
            if "for each character you have in play" in card_text:
                # Strength of a Raging Fire (Set 9): "Deal damage to chosen character equal to the number of characters you have in play."
                dmg_amt = len(p.characters_in_play)
            elif "equal to the number of detective characters you have in play" in card_text:
                # The Terror That Flaps in the Night (Set 11)
                dmg_amt = sum(1 for e in p.characters_in_play if "Detective" in e.card.subtypes)
            if dmg_amt <= 0:
                return
            unpreventable = any(
                "damage can't be reduced" in e.card.full_text.lower()
                for e in p.characters_in_play
            )
            targets = [
                e for e in opp.characters_in_play if not e.has_keyword(KeywordFlag.WARD)
            ]
            if targets:
                best_t = max(
                    targets,
                    key=lambda e: (
                        (is_action_source and e.has_keyword(KeywordFlag.VANISH))
                        or (e.damage + dmg_amt >= e.effective_willpower),
                        e.card.lore,
                        e.effective_strength,
                    ),
                )
                resist = 0 if unpreventable else best_t.effective_resist(opp)
                dmg = max(0, dmg_amt - resist)
                best_t.damage += dmg
                cls._apply_vanish_if_action(state, best_t, is_action_source)
                cls._cleanup_banished(state)

        elif op == EffectOpType.DEAL_DAMAGE_ALL_OPP:
            unpreventable = any(
                "damage can't be reduced" in e.card.full_text.lower()
                for e in p.characters_in_play
            )
            for e in opp.characters_in_play:
                resist = 0 if unpreventable else e.effective_resist(opp)
                dmg = max(0, cl.amount - resist)
                e.damage += dmg
            cls._cleanup_banished(state)

        elif op == EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP:
            # To Wither a Flower: Deal 2 damage to each opposing damaged character
            unpreventable = any(
                "damage can't be reduced" in e.card.full_text.lower()
                for e in p.characters_in_play
            )
            for e in opp.characters_in_play:
                if e.damage > 0:
                    resist = 0 if unpreventable else e.effective_resist(opp)
                    dmg = max(0, cl.amount - resist)
                    e.damage += dmg
            cls._cleanup_banished(state)

        elif op == EffectOpType.BANISH_ALL_DAMAGED_OPP:
            # Prince Phillip - Vanquisher of Foes: When you play this character, banish all opposing damaged characters
            for e in opp.characters_in_play:
                if e.damage > 0:
                    e.damage = e.effective_willpower + 10
            cls._cleanup_banished(state)

        elif op == EffectOpType.BANISH_CHAR:
            min_str = abs(cl.threshold) if cl.threshold < 0 else 0
            only_evasive = "chosen character with evasive" in card_text
            targets = [
                e for e in opp.characters_in_play
                if e.effective_strength >= min_str
                and not e.has_keyword(KeywordFlag.WARD)
                and (not only_evasive or e.has_keyword(KeywordFlag.EVASIVE))
            ]
            if targets:
                best_t = max(targets, key=lambda e: (e.card.lore * 3 + e.effective_strength + e.card.cost))
                best_t.damage = best_t.effective_willpower + 10
                cls._cleanup_banished(state)

        elif op == EffectOpType.BANISH_LOW_STR_CHAR:
            thresh = cl.threshold if cl.threshold > 0 else 2
            targets = [
                e for e in opp.characters_in_play
                if e.effective_strength <= thresh and not e.has_keyword(KeywordFlag.WARD)
            ]
            if targets:
                best_t = max(targets, key=lambda e: (e.card.lore * 3 + e.card.cost))
                best_t.damage = best_t.effective_willpower + 10
                cls._cleanup_banished(state)

        elif op == EffectOpType.BANISH_ALL_CHARS:
            if cl.threshold > 0:
                for e in opp.characters_in_play:
                    if e.effective_strength <= cl.threshold:
                        e.damage = e.effective_willpower + 10
            else:
                for pl in state.players:
                    for e in pl.characters_in_play:
                        e.damage = e.effective_willpower + 10
            cls._cleanup_banished(state)

        elif op == EffectOpType.BOUNCE_OWN_CHAR:
            self_ent = next((e for e in p.characters_in_play if e.instance_id == source_id), None)
            is_optional = self_ent is not None and "you may" in self_ent.card.full_text.lower()
            others = [e for e in p.characters_in_play if e.instance_id != source_id]
            if others:
                best_own = max(
                    others,
                    key=lambda e: (
                        any(c.trigger == TriggerType.ON_LEAVE_PLAY for c in e.card.abilities.clauses),
                        any(c.trigger == TriggerType.ON_PLAY for c in e.card.abilities.clauses),
                        e.damage > 0,
                        e.exerted,
                        -e.card.cost,
                    ),
                )
                p.board.remove(best_own)
                cls._resolve_clauses(
                    state, controller_id, best_own.card, TriggerType.ON_LEAVE_PLAY, source_id=best_own.instance_id
                )
                p.hand.append(best_own.card)
                p.revealed_hand_ids.append(best_own.card.card_id)
            elif not is_optional and self_ent is not None:
                p.board.remove(self_ent)
                p.discard.append(self_ent.card)

        elif op == EffectOpType.BOUNCE_OPP_CHAR:
            if cl.cost_ink > 0 and p.total_payable_ink < cl.cost_ink:
                return
            requires_discard = "choose and discard a card" in card_text and "if you do" in card_text
            if requires_discard and not p.hand:
                return
            thresh = cl.threshold if cl.threshold > 0 else 99
            max_cost_b = 2 if "cost 2 or less" in card_text else 99
            targets = [
                e for e in opp.characters_in_play
                if e.effective_strength <= thresh and e.card.cost <= max_cost_b and not e.has_keyword(KeywordFlag.WARD)
            ]
            if targets:
                best_t = max(targets, key=lambda e: (not e.drying, e.card.cost + e.card.lore * 2, e.effective_willpower))
                if cl.cost_ink > 0:
                    # Finnick - Tiny Terror ("you may pay 2 ⬡"): don't burn 2 ink on a 1-cost 1-lore character if we still have a character in hand we can play with that ink (unless opponent is near lethal)!
                    rem_ink_after_bounce = p.total_payable_ink - cl.cost_ink
                    blocks_hand_char = any(
                        c.card_type == "Character" and rem_ink_after_bounce < p.effective_play_cost(c) <= p.total_payable_ink
                        for c in p.hand
                    )
                    if blocks_hand_char and opp.lore < 14 and best_t.card.cost <= 1 and best_t.card.lore <= 1:
                        return
                    p.pay_ink(cl.cost_ink)
                if requires_discard:
                    # Rapunzel - Tower Defender: don't discard our only high-value payoff to bounce a 1-drop unless opp is near lethal
                    if best_t.card.cost <= 1 and best_t.card.lore <= 1 and len(p.hand) == 1 and opp.lore < 15:
                        return
                    disc_idx = min(
                        range(len(p.hand)),
                        key=lambda i: (
                            sum(1 for h in p.hand if h.card_id == p.hand[i].card_id) == 1,
                            p.hand[i].lore,
                            len(p.hand[i].abilities.clauses),
                            p.hand[i].cost,
                        ),
                    )
                    discarded = p.hand.pop(disc_idx)
                    p.discard.append(discarded)
                    p.cards_discarded_this_turn += 1
                    if discarded.card_id in p.revealed_hand_ids:
                        p.revealed_hand_ids.remove(discarded.card_id)
                if is_action_source and best_t.has_keyword(KeywordFlag.VANISH):
                    # CR Rule 8.14.1: Chosen by opponent's action -> banished
                    best_t.damage = best_t.effective_willpower + 10
                    cls._cleanup_banished(state)
                else:
                    opp.board.remove(best_t)
                    cls._resolve_clauses(
                        state, opp_id, best_t.card, TriggerType.ON_LEAVE_PLAY, source_id=best_t.instance_id
                    )
                    opp.hand.append(best_t.card)
                    opp.revealed_hand_ids.append(best_t.card.card_id)

        elif op == EffectOpType.DISCARD_OPP:
            if "reveals their hand" in card_text:
                opp.revealed_hand_ids = [c.card_id for c in opp.hand]
            if "discards a song card" in card_text:
                # Ursula - Deceiver (Set 9): "chosen opponent reveals their hand and discards a song card of your choice."
                song_indices = [i for i, c in enumerate(opp.hand) if c.is_song]
                if song_indices:
                    best_song_idx = max(
                        song_indices,
                        key=lambda i: (
                            any(cl_x.op in (EffectOpType.REANIMATE_CHAR, EffectOpType.BANISH_ALL_CHARS, EffectOpType.DEAL_DAMAGE_ALL_OPP) for cl_x in opp.hand[i].abilities.clauses),
                            len(opp.hand[i].abilities.clauses),
                            opp.hand[i].cost,
                        ),
                    )
                    discarded = opp.hand.pop(best_song_idx)
                    opp.discard.append(discarded)
                    if discarded.card_id in opp.revealed_hand_ids:
                        opp.revealed_hand_ids.remove(discarded.card_id)
            elif "discards a non-character card" in card_text:
                # Mowgli - Man Cub (Set 9) / Elisa Maza - Turned to Stone (Set 10): "reveals their hand and discards a non-character card of your choice."
                non_char_indices = [i for i, c in enumerate(opp.hand) if c.card_type != "Character"]
                if non_char_indices:
                    best_nc_idx = max(
                        non_char_indices,
                        key=lambda i: (opp.hand[i].is_song, len(opp.hand[i].abilities.clauses), opp.hand[i].cost),
                    )
                    discarded = opp.hand.pop(best_nc_idx)
                    opp.discard.append(discarded)
                    if discarded.card_id in opp.revealed_hand_ids:
                        opp.revealed_hand_ids.remove(discarded.card_id)
            elif "discards an action card" in card_text:
                # Lenny - Toy Binoculars (Set 14): "reveals their hand and discards an action card of your choice."
                act_indices = [i for i, c in enumerate(opp.hand) if c.card_type == "Action"]
                if act_indices:
                    best_act_idx = max(
                        act_indices,
                        key=lambda i: (opp.hand[i].is_song, len(opp.hand[i].abilities.clauses), opp.hand[i].cost),
                    )
                    discarded = opp.hand.pop(best_act_idx)
                    opp.discard.append(discarded)
                    if discarded.card_id in opp.revealed_hand_ids:
                        opp.revealed_hand_ids.remove(discarded.card_id)
            else:
                for _ in range(cl.amount):
                    if opp.hand:
                        worst_idx = min(
                            range(len(opp.hand)),
                            key=lambda i: (opp.hand[i].inkable, opp.hand[i].lore, opp.hand[i].cost),
                        )
                        discarded = opp.hand.pop(worst_idx)
                        opp.discard.append(discarded)
                        if discarded.card_id in opp.revealed_hand_ids:
                            opp.revealed_hand_ids.remove(discarded.card_id)

        elif op == EffectOpType.BUFF_ALLY_STR:
            buff_val = cl.amount
            if "put the top 3 cards of your deck into your discard" in card_text:
                for _ in range(min(3, len(p.deck))):
                    p.discard.append(p.deck.pop())
                buff_val = sum(1 for c in p.discard if c.is_song)
            allies = [e for e in p.characters_in_play if e.instance_id != source_id and e.can_challenge()]
            if not allies:
                allies = [e for e in p.characters_in_play if e.instance_id != source_id]
            if allies and buff_val > 0:
                best_ally = max(allies, key=lambda e: e.effective_strength)
                best_ally.temp_challenger_mod += buff_val

        elif op == EffectOpType.BUFF_ALLY_LORE:
            # Lionheart - Incumbent Mayor ("up to 2 chosen characters get +1 ◊ this turn")
            # Pocahontas - Finding the Way ("chosen character gets +1 ◊ this turn")
            ready_questers = [
                e for e in p.characters_in_play
                if not e.exerted and not e.drying and not e.has_keyword(KeywordFlag.RECKLESS)
            ]
            ready_questers.sort(key=lambda e: (e.effective_willpower - e.damage, e.card.lore), reverse=True)
            for q_ent in ready_questers[: max(1, cl.amount)]:
                q_ent.temp_lore_mod += 1

        elif op == EffectOpType.DEBUFF_OPP_STR:
            targets = [e for e in opp.characters_in_play if not e.has_keyword(KeywordFlag.WARD)]
            if targets:
                best_opp = max(
                    targets,
                    key=lambda e: (1 <= e.effective_strength <= 4, e.card.lore),
                )
                best_opp.temp_strength_mod -= cl.amount
                cls._apply_vanish_if_action(state, best_opp, is_action_source)
                cls._cleanup_banished(state)

        elif op == EffectOpType.READY_CHAR:
            exerted_allies = [e for e in p.characters_in_play if e.exerted]
            if exerted_allies:
                best_ally = max(exerted_allies, key=lambda e: e.effective_strength)
                best_ally.exerted = False

        elif op == EffectOpType.RETRO_EVOLVE:
            # Retro Evolution Device (Set 14): "⟳, 1 ⬡, Banish chosen character of yours — Play a character with cost up to 2 more than the banished character for free."
            if p.characters_in_play:
                cand_banish = min(
                    p.characters_in_play,
                    key=lambda e: (e.damage > 0, e.effective_lore(p, opp), e.card.cost),
                )
                p.board.remove(cand_banish)
                p.discard.append(cand_banish.card)
                max_c = cand_banish.card.cost + 2
                valid_hand_chars = [
                    (idx, c) for idx, c in enumerate(p.hand)
                    if c.card_type == "Character" and c.cost <= max_c
                ]
                if valid_hand_chars:
                    best_idx, chosen_evolve = max(
                        valid_hand_chars,
                        key=lambda item: (item[1].cost, item[1].lore * 2 + item[1].willpower),
                    )
                    p.hand.pop(best_idx)
                    if chosen_evolve.card_id in p.revealed_hand_ids:
                        p.revealed_hand_ids.remove(chosen_evolve.card_id)
                    new_ent = BoardEntity(
                        instance_id=state.next_instance_id,
                        card=chosen_evolve,
                        exerted=False,
                        drying=True,
                    )
                    state.next_instance_id += 1
                    p.board.append(new_ent)
                    p.chars_played_this_turn += 1
                    if p.is_princess_or_queen(chosen_evolve):
                        p.princess_played_this_turn = True
                    cls._resolve_clauses(
                        state, controller_id, chosen_evolve, TriggerType.ON_PLAY, source_id=new_ent.instance_id
                    )

        elif op == EffectOpType.PUT_UNDER_FROM_DISCARD:
            # The Black Cauldron (Set 14): "⟳, 1 ⬡ — Put a character card from your discard under this item faceup."
            source_ent = next((e for e in p.board if e.instance_id == source_id), None)
            if source_ent is not None:
                char_grave = [(i, c) for i, c in enumerate(p.discard) if c.card_type == "Character"]
                if char_grave:
                    best_i, chosen_c = max(char_grave, key=lambda item: (item[1].cost, item[1].lore))
                    p.discard.pop(best_i)
                    source_ent.cards_under_count += 1
                    source_ent.shifted_under_cards.append(chosen_c)

        elif op == EffectOpType.GIVE_SELF_RUSH:
            # Roller Bob - Sid's Toy (Set 12): "When you play this character, you may put 2 character cards from your discard on the bottom of your deck to give this character Rush this turn."
            char_grave = [c for c in p.discard if c.card_type == "Character"]
            source_ent = next((e for e in p.board if e.instance_id == source_id), None)
            if len(char_grave) >= 2 and source_ent is not None:
                for _ in range(2):
                    idx = next(i for i, c in enumerate(p.discard) if c.card_type == "Character")
                    c = p.discard.pop(idx)
                    p.deck.insert(0, c)
                source_ent.drying = False

        elif op == EffectOpType.MOVE_DAMAGE:
            # Luisa Madrigal - Confident Climber (Set 13): "1 ⬡ — Move up to 1 damage from chosen character of yours to this character. Then, if this character has 3 or more damage, move all damage from this character to chosen opposing character."
            source_ent = next((e for e in p.board if e.instance_id == source_id), None)
            if source_ent is not None:
                damaged_allies = [e for e in p.characters_in_play if e.instance_id != source_id and e.damage > 0]
                if damaged_allies:
                    t_ally = max(damaged_allies, key=lambda e: e.damage)
                    t_ally.damage -= 1
                    source_ent.damage += 1
                if source_ent.damage >= 3 and opp.characters_in_play:
                    opp_t = max(opp.characters_in_play, key=lambda e: (e.card.lore * 2 + e.effective_strength))
                    opp_t.damage += source_ent.damage
                    source_ent.damage = 0
                    cls._cleanup_banished(state)

        elif op == EffectOpType.INK_ITEM:
            # Judy Hopps - On the Case (Set 13): "When you play this character, if you have another Detective character in play, you may put chosen item into its player's inkwell facedown and exerted."
            has_other_det = any("Detective" in e.card.subtypes and e.instance_id != source_id for e in p.characters_in_play)
            if has_other_det:
                target_items = [e for e in opp.items_in_play] or [e for e in p.items_in_play if e.instance_id != source_id]
                if target_items:
                    best_item = target_items[0]
                    owner_pl = opp if best_item in opp.board else p
                    owner_pl.board.remove(best_item)
                    owner_pl.inkwell_cards.append(best_item.card)
                    owner_pl.inkwell_total += 1

        elif op == EffectOpType.BANISH_ITEM:
            # Aladdin - Doing His Part (Set 12): "When you play this character, you may pay 1 ⬡ to banish chosen item."
            if cl.cost_ink > 0:
                if p.total_payable_ink < cl.cost_ink:
                    return
                p.pay_ink(cl.cost_ink)
            if opp.items_in_play:
                t_item = max(opp.items_in_play, key=lambda e: e.card.cost)
                opp.board.remove(t_item)
                opp.discard.append(t_item.card)

        elif op == EffectOpType.GIVE_ADVENTUROUS:
            # Nick Wilde - Inquisitive Harbormaster (Set 12): "Whenever this character quests, you may give chosen character Adventurous until the start of your next turn."
            if opp.characters_in_play:
                best_t = max(opp.characters_in_play, key=lambda e: e.effective_strength)
                best_t.temp_adventurous = True

        elif op == EffectOpType.GIVE_RUSH_EVASIVE:
            # Demona - Imperious Spellcaster (Set 10): "Choose and discard a card — Chosen Gargoyle character gains Rush and Evasive until the start of your next turn."
            if p.hand:
                disc_idx = min(range(len(p.hand)), key=lambda i: (p.hand[i].lore, p.hand[i].cost))
                discarded = p.hand.pop(disc_idx)
                p.discard.append(discarded)
                p.cards_discarded_this_turn += 1
                if discarded.card_id in p.revealed_hand_ids:
                    p.revealed_hand_ids.remove(discarded.card_id)
                gargoyles = [e for e in p.characters_in_play if "Gargoyle" in e.card.subtypes]
                if gargoyles:
                    g = gargoyles[0]
                    g.drying = False
                    g.temp_evasive = True

    @classmethod
    def _cleanup_banished(cls, state: GameState) -> None:
        for pid in (0, 1):
            pl = state.players[pid]
            remaining: list[BoardEntity] = []
            banished: list[BoardEntity] = []
            banished_location_ids: set[int] = set()
            for ent in pl.board:
                if ent.is_banished:
                    banished.append(ent)
                    if ent.card.card_type == "Location":
                        banished_location_ids.add(ent.instance_id)
                else:
                    remaining.append(ent)

            # CR Rule 5.6.6: If a location leaves play, characters at that location remain in play and are no longer at a location
            if banished_location_ids:
                for ent in remaining:
                    if ent.location_instance_id in banished_location_ids:
                        ent.location_instance_id = -1

            pl.board = remaining
            has_belle_snowfield = any(
                "one of your characters is banished" in r.card.full_text.lower()
                and "into your inkwell" in r.card.full_text.lower()
                for r in remaining
            )
            has_mickey_minnie_duo = any(
                "whenever another character of yours is banished, draw a card" in r.card.full_text.lower()
                for r in remaining
            )
            for ent in banished:
                cls._resolve_clauses(
                    state, pid, ent.card, TriggerType.ON_LEAVE_PLAY, source_id=ent.instance_id
                )
                if ent.card.card_type == "Character" and has_mickey_minnie_duo and pl.deck:
                    pl.hand.append(pl.deck.pop())
                if ent.card.card_type == "Character" and has_belle_snowfield and pl.inkwell_total < 12:
                    # Belle - Snowfield Strategist (Set 11): "When one of your characters is banished, you may put that card into your inkwell facedown and exerted."
                    pl.inkwell_cards.append(ent.card)
                    pl.inkwell_total += 1
                else:
                    pl.discard.append(ent.card)
                for u_card in ent.shifted_under_cards:
                    pl.discard.append(u_card)

    @classmethod
    def _check_victory(cls, state: GameState) -> bool:
        for pid in (0, 1):
            if state.players[pid].lore >= 20:
                state.phase = "GAME_OVER"
                state.winner = pid
                state.victory_reason = "lore_20"
                return True
        return False
