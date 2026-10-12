"""
Tactical Heuristic Agent for Disney Lorcana.
Serves as a fast baseline, rollout policy, and bootstrap opponent for Self-Play RL.
"""

from __future__ import annotations
from lorcana_ai.data.effect_dsl import EffectOpType, KeywordFlag
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator


class TacticalHeuristicAgent:
    """
    Evaluates legal Lorcana actions using domain-specific tactical heuristics:
    - Lethal Lore detection
    - Matchup-aware Mulligan (blind G1 vs known inks G2/G3)
    - Smart Inkwell sacrifice (preserving sweepers, key songs, and curve)
    - Singing songs before questing
    - Value trading vs Lore racing
    """

    def select_action(self, state: GameState, legal_actions: list[GameAction] | None = None) -> GameAction:
        actions = legal_actions if legal_actions is not None else MoveGenerator.generate_legal_actions(state)
        if not actions:
            return GameAction(ActionType.PASS_TURN, description="Pass Turn")
        if len(actions) == 1:
            return actions[0]

        best_act = actions[0]
        best_score = -1e9
        for act in actions:
            sc = self.score_action(state, act)
            if sc > best_score:
                best_score = sc
                best_act = act
        return best_act

    def score_action(self, state: GameState, action: GameAction) -> float:
        p_id = state.active_player
        p = state.players[p_id]
        opp = state.players[1 - p_id]

        # 1. Mulligan evaluation
        if action.action_type == ActionType.MULLIGAN:
            mask = action.mulligan_keep_mask
            kept_cards = [p.hand[i] for i, k in enumerate(mask) if i < len(p.hand) and k]
            uninkables = sum(1 for c in kept_cards if not c.inkable)
            early_drops = sum(1 for c in kept_cards if c.cost <= 2 and c.card_type == "Character")
            three_drops = sum(1 for c in kept_cards if c.cost == 3 and c.card_type == "Character")
            singers = sum(1 for c in kept_cards if c.cost <= 2 and (c.abilities.singer_cost or 0) >= 3)
            underdogs = sum(1 for c in kept_cards if c.abilities.has_underdog)
            looters = sum(
                1 for c in kept_cards
                if c.cost <= 3 and any(cl.op == EffectOpType.LOOT_SELF for cl in c.abilities.clauses)
            )
            heavy_bombs = sum(1 for c in kept_cards if c.cost >= 7)
            score = (
                early_drops * 4.0
                + singers * 2.5
                + underdogs * 2.0
                + min(2, three_drops) * 2.5
                + min(2, looters) * 3.0
                - max(0, uninkables - 2) * 5.0
            )
            # Only keep at most 1 heavy reanimation bomb in opening hand, and only if we have a looter
            if heavy_bombs > 0:
                if looters > 0 and heavy_bombs == 1:
                    score += 2.0
                else:
                    score -= heavy_bombs * 4.5
            opp_inks = state.known_opponent_inks[p_id]
            if "Emerald" in opp_inks or "Amber" in opp_inks:
                # Prioritize cheap removal / AoE ping vs aggro
                removal_cnt = sum(
                    1 for c in kept_cards
                    if c.cost <= 4 and any(
                        cl.op in (
                            EffectOpType.DEAL_DAMAGE,
                            EffectOpType.DEAL_DAMAGE_ALL_OPP,
                            EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP,
                            EffectOpType.BANISH_LOW_STR_CHAR,
                        )
                        for cl in c.abilities.clauses
                    )
                )
                score += removal_cnt * 3.0
            return score

        # 2. Inkwell selection — MUST ink before spending ink on PLAY_CARD so we hit on-curve drops (3 ink T3, 4 ink T4, etc.)!
        if action.action_type == ActionType.INK_CARD:
            if p.inkwell_total >= 8 and len(p.hand) <= 2:
                return -10.0  # Hold cards in late game if already at 8+ ink
            card = p.hand[action.hand_index]
            dup_count = sum(1 for c in p.hand if c.card_id == card.card_id)
            # Base score 36.0 ensures we ink BEFORE PLAY_CARD (16..34), right after any 0-ink SING_SONG draw (39+)
            score = 36.0 if p.inkwell_total < 7 else (28.0 if len(p.hand) >= 3 else 8.0)
            if dup_count > 1:
                score += 4.5
            # Protect Circle of Life from being inked if we only have 1 copy
            if any(cl.op == EffectOpType.REANIMATE_CHAR for cl in card.abilities.clauses) and dup_count == 1:
                score -= 9.0
            # Protect 2-drop Characters in the first two turns if it's our only 2-drop
            two_drop_chars = sum(1 for c in p.hand if c.cost <= 2 and c.card_type == "Character")
            if card.cost <= 2 and card.card_type == "Character" and p.inkwell_total < 2 and two_drop_chars <= 1:
                score -= 10.0
            # Protect on-curve character for THIS turn (cost == p.inkwell_total + 1) if we only have 1 copy
            on_curve_cost = p.inkwell_total + 1
            on_curve_chars = sum(1 for c in p.hand if c.cost == on_curve_cost and c.card_type == "Character")
            if card.cost == on_curve_cost and card.card_type == "Character" and on_curve_chars <= 1:
                score -= 6.0
            # If we have a looter (Strike a Good Match / Tod) in hand, hold onto 1 heavy Character (Prince Phillip) to pitch to discard!
            has_looter_in_hand = any(
                any(cl.op == EffectOpType.LOOT_SELF for cl in h.abilities.clauses)
                for i, h in enumerate(p.hand)
                if i != action.hand_index
            )
            if has_looter_in_hand and card.card_type == "Character" and card.cost >= 7 and dup_count == 1:
                score -= 7.5
            elif card.cost > p.inkwell_total + 2:
                score += 3.5
            return score

        # 3. Sing Song (free mana & Circle of Life / Malicious / Akood combos!)
        if action.action_type == ActionType.SING_SONG:
            card = p.hand[action.hand_index]
            singer = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            singer_lore_cost = singer.effective_lore(p) if singer else 1
            singer_pwr = (singer.card.abilities.singer_cost or singer.card.cost) if singer else card.cost
            overpay_penalty = 0.4 * max(0, singer_pwr - card.cost) if singer else 0.0
            # Penalize singing a cheap song with Max Goof - Chart Topper when he could quest to replay a song from discard!
            if singer and any(cl.op == EffectOpType.REPLAY_SONG_FROM_DISCARD for cl in singer.card.abilities.clauses):
                if any(c.is_song and c.cost <= 4 for c in p.discard):
                    overpay_penalty += 18.0
            ops = {cl.op for cl in card.abilities.clauses}

            # Do not sing symmetric board wipe if ahead
            if any(cl.op == EffectOpType.BANISH_ALL_CHARS and cl.threshold == 0 for cl in card.abilities.clauses):
                if len(opp.characters_in_play) <= len(p.characters_in_play):
                    return -25.0
                return 45.0

            # Circle of Life (REANIMATE_CHAR): score by best character in discard!
            if EffectOpType.REANIMATE_CHAR in ops:
                discard_chars = [c for c in p.discard if c.card_type == "Character"]
                if not discard_chars:
                    return -50.0
                best_reanim = max(discard_chars, key=lambda c: c.cost + c.lore * 2)
                bonus = best_reanim.cost * 3.5 + best_reanim.lore * 4.0
                if any(cl.op == EffectOpType.BANISH_ALL_DAMAGED_OPP for cl in best_reanim.abilities.clauses):
                    damaged_opp_cnt = sum(1 for e in opp.characters_in_play if e.damage > 0)
                    bonus += damaged_opp_cnt * 14.0
                return 42.0 + bonus

            # Malicious, Mean, and Scary (DEAL_DAMAGE_ALL_OPP): sing BEFORE Prince Phillip or To Wither a Flower!
            if EffectOpType.DEAL_DAMAGE_ALL_OPP in ops:
                if not opp.characters_in_play:
                    return -15.0
                kills_one_wp = sum(1 for e in opp.characters_in_play if (e.effective_willpower - e.damage) <= 1)
                has_followup = any(
                    any(cl.op in (EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP, EffectOpType.BANISH_ALL_DAMAGED_OPP) for cl in h.abilities.clauses)
                    for h in p.hand
                ) or (
                    any(any(cl.op == EffectOpType.REANIMATE_CHAR for cl in h.abilities.clauses) for h in p.hand)
                    and any(any(cl.op == EffectOpType.BANISH_ALL_DAMAGED_OPP for cl in d.abilities.clauses) for d in p.discard)
                )
                if kills_one_wp == 0 and not has_followup and singer_lore_cost >= 2:
                    return 12.0  # Don't waste a 2+ lore quester if 1 dmg neither kills nor enables a wipe
                return 41.0 + len(opp.characters_in_play) * 4.5 + kills_one_wp * 8.0 + (12.0 if has_followup else 0.0) - singer_lore_cost - overpay_penalty

            # To Wither a Flower (DEAL_DAMAGE_ALL_DAMAGED_OPP)
            if EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP in ops:
                damaged_cnt = sum(1 for e in opp.characters_in_play if e.damage > 0)
                if damaged_cnt == 0:
                    return -25.0
                return 40.0 + damaged_cnt * 7.0 - singer_lore_cost - overpay_penalty

            # Strike a Good Match (LOOT_SELF): sing before Circle of Life / before Inking!
            if EffectOpType.LOOT_SELF in ops:
                has_bomb_in_hand = any(c.card_type == "Character" and c.cost >= 6 for c in p.hand)
                return 40.0 + (12.0 if has_bomb_in_hand else 0.0) - singer_lore_cost - overpay_penalty

            # Akood et Emuti (DISCOUNT_NEXT_CHAR): sing BEFORE inking / playing a character!
            if EffectOpType.DISCOUNT_NEXT_CHAR in ops:
                has_char_in_hand = any(c.card_type == "Character" for c in p.hand)
                return 39.5 + (8.0 if has_char_in_hand else 0.0) - singer_lore_cost - overpay_penalty

            # With a Few Good Friends (DRAW_PER_INK_TYPE)
            if EffectOpType.DRAW_PER_INK_TYPE in ops:
                inks_in_play = {ink for e in p.characters_in_play for ink in e.card.inks}
                if not inks_in_play:
                    return -20.0
                return 38.0 + len(inks_in_play) * 5.0

            return 28.0 + card.cost * 2.5 - singer_lore_cost * 1.5 - overpay_penalty

        # 4. Shift Card
        if action.action_type == ActionType.SHIFT_CARD:
            card = p.hand[action.hand_index]
            bonus = 0.0
            if any(cl.op == EffectOpType.BANISH_ALL_DAMAGED_OPP for cl in card.abilities.clauses):
                bonus += sum(12.0 for e in opp.characters_in_play if e.damage > 0)
            if any(cl.op == EffectOpType.REPLAY_SONG_FROM_DISCARD for cl in card.abilities.clauses):
                songs_in_grave = sum(1 for c in p.discard if c.is_song and c.cost <= 4)
                bonus += min(2, songs_in_grave) * 5.0
            return 25.0 + card.cost * 2.0 + card.lore * 2.0 + bonus

        # 5. Play Card
        if action.action_type == ActionType.PLAY_CARD:
            card = p.hand[action.hand_index]
            ops = {cl.op for cl in card.abilities.clauses}
            if any(cl.op == EffectOpType.BANISH_ALL_CHARS and cl.threshold == 0 for cl in card.abilities.clauses):
                opp_threat = sum(e.card.lore * 2 + e.effective_strength for e in opp.characters_in_play)
                my_threat = sum(e.card.lore * 2 + e.effective_strength for e in p.characters_in_play)
                if opp_threat > my_threat + 4 or (opp.lore >= 14 and len(opp.characters_in_play) >= 2):
                    return 35.0
                return -30.0
            if EffectOpType.DEAL_DAMAGE_ALL_OPP in ops and not opp.characters_in_play:
                return -20.0
            if EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP in ops:
                damaged_cnt = sum(1 for e in opp.characters_in_play if e.damage > 0)
                return (28.0 + damaged_cnt * 5.0) if damaged_cnt > 0 else -20.0
            if EffectOpType.REANIMATE_CHAR in ops:
                discard_chars = [c for c in p.discard if c.card_type == "Character"]
                if not discard_chars:
                    return -40.0
                best_reanim = max(discard_chars, key=lambda c: c.cost + c.lore * 2)
                return 24.0 + best_reanim.cost * 2.5
            bonus = 0.0
            if EffectOpType.BANISH_ALL_DAMAGED_OPP in ops:
                bonus += sum(12.0 for e in opp.characters_in_play if e.damage > 0)
            # Strongly prefer deploying Characters over hard-casting Songs with ink in the early/mid game!
            if card.is_song:
                has_playable_char = any(
                    c.card_type == "Character" and c.cost <= p.total_payable_ink for c in p.hand
                )
                if has_playable_char:
                    bonus -= 10.0
                elif p.inkwell_total <= 3:
                    bonus -= 4.0
            elif EffectOpType.LOOT_SELF in ops and any(c.card_type == "Character" and c.cost >= 6 for c in p.hand):
                bonus += 5.0
            if card.card_type == "Character":
                bonus += 3.0  # Board presence bonus
                if card.cost <= 2 and (card.abilities.singer_cost or 0) >= 3:
                    bonus += 3.5
                if card.abilities.has_underdog and state.turn_number == 2:
                    bonus += 5.0
            return 14.0 + card.cost * 2.2 + card.lore * 2.5 + bonus

        # 6. Challenge vs Quest (Characters & Locations per CR Rule 4.6.8)
        if action.action_type == ActionType.CHALLENGE:
            attacker = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            defender = next((e for e in opp.board if e.instance_id == action.target_instance_id), None)
            if attacker is None or defender is None:
                return -5.0
            atk_pwr = (
                attacker.effective_strength
                + attacker.card.abilities.challenger_bonus
                + attacker.temp_challenger_mod
            )
            dmg_to_def = max(0, atk_pwr - defender.card.abilities.resist_bonus)
            if dmg_to_def <= 0:
                return -50.0
            dmg_to_atk = (
                0
                if defender.card.card_type == "Location"
                else max(0, defender.effective_strength - attacker.card.abilities.resist_bonus)
            )

            kills_def = (defender.damage + dmg_to_def) >= defender.effective_willpower
            survives = (attacker.damage + dmg_to_atk) < attacker.effective_willpower

            # Check if we have lethal lore on board right now
            potential_lore = sum(e.effective_lore(p) for e in p.characters_in_play if e.can_quest())
            if p.lore + potential_lore >= 20:
                return -15.0  # Don't trade when we can win right now by questing!

            opp_board_lore = sum(e.effective_lore(opp) for e in opp.characters_in_play)
            def_lore = defender.effective_lore(opp) if defender.card.card_type == "Character" else defender.card.lore
            atk_lore = attacker.effective_lore(p)

            score = 5.0
            if kills_def and survives:
                score += 22.0 + def_lore * 5.0 + defender.card.cost * 1.5
            elif kills_def and not survives:
                if (
                    def_lore > atk_lore
                    or opp.lore >= 14
                    or opp_board_lore >= 4
                    or defender.card.cost > attacker.card.cost
                ):
                    score += 15.0 + def_lore * 4.5 - atk_lore * 2.0
                else:
                    score -= 3.0
            else:
                if opp.lore >= 15 or opp_board_lore >= 6 or (defender.card.card_type == "Location" and defender.card.lore >= 2):
                    score += 11.0 + def_lore * 2.0
                else:
                    score -= 4.0
            return score

        if action.action_type == ActionType.QUEST:
            quester = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            if quester is None:
                return 0.0
            q_lore = quester.effective_lore(p)
            if p.lore + q_lore >= 20:
                return 1000.0  # Immediate victory!
            bonus = 0.0
            if any(cl.op == EffectOpType.REPLAY_SONG_FROM_DISCARD for cl in quester.card.abilities.clauses):
                if any(c.is_song and c.cost <= 4 for c in p.discard):
                    bonus += 18.0
                else:
                    bonus += 4.0
            return 10.0 + q_lore * 4.5 + bonus

        if action.action_type == ActionType.MOVE_TO_LOCATION:
            # Move to location if we have spare ink after playing cards
            loc = next((e for e in p.locations_in_play if e.instance_id == action.target_instance_id), None)
            if loc is None:
                return -5.0
            return 4.0 if p.total_payable_ink > loc.card.move_cost + 2 else -2.0

        if action.action_type == ActionType.USE_ABILITY:
            ent = next((e for e in p.board if e.instance_id == action.source_instance_id), None)
            if ent is None:
                return -5.0
            for cl in ent.card.abilities.clauses:
                if cl.op == EffectOpType.GAIN_LORE:
                    return 32.0
                if cl.op == EffectOpType.RAMP_INK:
                    if p.inkwell_total < 7 and len(p.hand) >= 3:
                        return 19.0
                    return -15.0
                if cl.op == EffectOpType.DEBUFF_OPP_STR:
                    return 16.0 if len(opp.characters_in_play) > 0 else -10.0
            return 11.0

        if action.action_type == ActionType.PASS_TURN:
            return 0.0

        return 0.0
