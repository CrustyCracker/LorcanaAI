"""
Tactical Heuristic & Turn-Lookahead Agent for Disney Lorcana.
Serves as a tournament-grade tactical player, rollout policy, and bootstrap opponent for Self-Play RL:
- Lethal Lore detection & Anti-Lethal Opponent Crackback prevention
- Multi-character Cooperative Team-Challenge math (ganging up on high-WP questers like Elinor/Cindy)
- Matchup-aware Mulligan (blind G1 vs known inks G2/G3)
- Smart Inkwell sacrifice (preserving sweepers, key songs, combos, and curve)
- Optimal Turn Sequencing (Aurora discount -> Willow/Pluto discount -> Lionheart/Pocahontas +Lore buffs -> Questing)
- Elinor (3+ exerted characters draw) & Mickey Mouse (Best in Town / Expedition Leader) trigger optimization
- Turn-Level Lookahead Rollout & Board State Evaluator (`lookahead_depth`)
"""

from __future__ import annotations
import random

from lorcana_ai.data.effect_dsl import EffectOpType, KeywordFlag, TriggerType
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


class TacticalHeuristicAgent:
    """
    Evaluates legal Lorcana actions using domain-specific tactical heuristics
    plus optional multi-step turn lookahead (`lookahead_depth`).
    """

    def __init__(self, lookahead_depth: int = 1, beam_width: int = 3) -> None:
        self.lookahead_depth = lookahead_depth
        self.beam_width = max(2, beam_width)

    def select_action(self, state: GameState, legal_actions: list[GameAction] | None = None) -> GameAction:
        actions = legal_actions if legal_actions is not None else MoveGenerator.generate_legal_actions(state)
        if not actions:
            return GameAction(ActionType.PASS_TURN, description="Pass Turn")
        if len(actions) == 1:
            return actions[0]

        scored = [(self.score_action(state, act), idx, act) for idx, act in enumerate(actions)]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        best_score, _, best_act = scored[0]
        # Immediate lethal or mulligan needs no lookahead
        if best_score >= 500.0 or state.phase.startswith("MULLIGAN") or self.lookahead_depth <= 0:
            return best_act

        # Filter competitive candidates within 14 points of the top heuristic action
        candidates = [item for item in scored[: self.beam_width] if item[0] >= best_score - 14.0]
        if len(candidates) <= 1:
            return best_act

        p_id = state.active_player
        best_combined = -1e9
        chosen_act = best_act
        det_rng = random.Random(state.turn_number * 131 + p_id * 17 + len(state.players[p_id].hand))

        for h_score, _, cand_act in candidates:
            sim_state = state.clone()
            LorcanaSimulator.step(sim_state, cand_act, rng=det_rng)

            # Fast greedy rollout to the end of active player's turn
            steps = 0
            while (
                sim_state.phase != "GAME_OVER"
                and sim_state.active_player == p_id
                and steps < 10
            ):
                steps += 1
                next_legal = MoveGenerator.generate_legal_actions(sim_state)
                if not next_legal:
                    break
                next_act = max(next_legal, key=lambda a: self.score_action(sim_state, a))
                LorcanaSimulator.step(sim_state, next_act, rng=det_rng)
                if next_act.action_type == ActionType.PASS_TURN:
                    break

            # Optional Depth-2: simulate opponent's best crackback turn
            if (
                self.lookahead_depth >= 2
                and sim_state.phase != "GAME_OVER"
                and sim_state.active_player == (1 - p_id)
            ):
                opp_steps = 0
                while (
                    sim_state.phase != "GAME_OVER"
                    and sim_state.active_player == (1 - p_id)
                    and opp_steps < 8
                ):
                    opp_steps += 1
                    opp_legal = MoveGenerator.generate_legal_actions(sim_state)
                    if not opp_legal:
                        break
                    opp_act = max(opp_legal, key=lambda a: self.score_action(sim_state, a))
                    LorcanaSimulator.step(sim_state, opp_act, rng=det_rng)
                    if opp_act.action_type == ActionType.PASS_TURN:
                        break

            state_val = self.evaluate_state(sim_state, p_id)
            combined = h_score + 0.35 * state_val
            if combined > best_combined:
                best_combined = combined
                chosen_act = cand_act

        return chosen_act

    def select_action_with_diagnostics(
        self, state: GameState, legal_actions: list[GameAction] | None = None
    ) -> tuple[GameAction, float, list[tuple[GameAction, float]]]:
        """Returns the chosen action, its score, and top alternative actions for replay logging."""
        actions = legal_actions if legal_actions is not None else MoveGenerator.generate_legal_actions(state)
        if not actions:
            fallback = GameAction(ActionType.PASS_TURN, description="Pass Turn")
            return fallback, 0.0, []
        scored = [(act, round(self.score_action(state, act), 2)) for act in actions]
        scored.sort(key=lambda item: item[1], reverse=True)
        chosen = self.select_action(state, actions)
        chosen_score = next((sc for act, sc in scored if act == chosen), scored[0][1])
        alts = [(act, sc) for act, sc in scored if act != chosen][:3]
        return chosen, chosen_score, alts

    @staticmethod
    def evaluate_state(state: GameState, player_id: int) -> float:
        """Evaluates a GameState from `player_id`'s perspective for turn-lookahead search."""
        if state.phase == "GAME_OVER":
            if state.winner == player_id:
                return 1000.0
            if state.winner == (1 - player_id):
                return -1000.0
            return 0.0

        p = state.players[player_id]
        opp = state.players[1 - player_id]

        # 1. Lore progress with exponential urgency near 20
        my_board_lore = sum(
            e.effective_lore(p, opp) for e in p.characters_in_play if not e.has_keyword(KeywordFlag.RECKLESS)
        ) + sum(loc.card.lore for loc in p.locations_in_play)
        opp_board_lore = sum(
            e.effective_lore(opp, p) for e in opp.characters_in_play if not e.has_keyword(KeywordFlag.RECKLESS)
        ) + sum(loc.card.lore for loc in opp.locations_in_play)

        # Check if opponent has unstoppable lethal on their turn
        if state.active_player == (1 - player_id) and not opp.cant_challenge_this_turn:
            opp_ready_lore = sum(e.effective_lore(opp, p) for e in opp.characters_in_play if e.can_quest())
            if opp.lore + opp_ready_lore >= 20:
                return -500.0
        elif state.active_player == player_id and opp.lore + opp_board_lore >= 20 and p.lore + my_board_lore < 20:
            # Opponent threatens lethal next turn if we leave their board alive!
            return -220.0 - opp_board_lore * 15.0

        lore_diff = (p.lore - opp.lore) * 4.5
        if p.lore >= 15:
            lore_diff += (p.lore - 14) * 3.5
        if opp.lore >= 15:
            lore_diff -= (opp.lore - 14) * 4.0

        # 2. Board presence & character quality
        p_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in p.characters_in_play)
        opp_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in opp.characters_in_play)

        def _entity_val(ent: BoardEntity, owner, enemy, uses_wp: bool) -> float:
            rem_wp = max(1, ent.effective_willpower - ent.damage)
            eff_atk = ent.effective_willpower if uses_wp else ent.effective_strength
            lore_v = ent.effective_lore(owner, enemy) * 4.2
            stat_v = eff_atk * 1.1 + rem_wp * 1.2
            kw_v = 0.0
            if ent.has_keyword(KeywordFlag.EVASIVE):
                kw_v += 2.0
            if ent.has_keyword(KeywordFlag.WARD):
                kw_v += 2.5
            if ent.has_keyword(KeywordFlag.BODYGUARD) and ent.exerted:
                kw_v += 2.0
            return lore_v + stat_v + kw_v

        my_board_val = sum(_entity_val(e, p, opp, p_uses_wp) for e in p.characters_in_play) + sum(
            loc.card.lore * 3.5 + max(1, loc.effective_willpower - loc.damage) for loc in p.locations_in_play
        )
        opp_board_val = sum(_entity_val(e, opp, p, opp_uses_wp) for e in opp.characters_in_play) + sum(
            loc.card.lore * 3.5 + max(1, loc.effective_willpower - loc.damage) for loc in opp.locations_in_play
        )

        # 3. Hand & Inkwell resources (In early turns 1-3, having ink to curve out is strictly higher value than holding 1 extra card)
        hand_val = (len(p.hand) - len(opp.hand)) * 2.8
        ink_factor = 3.8 if p.inkwell_total < 4 else (2.2 if p.inkwell_total < 7 else 0.8)
        opp_ink_factor = 3.8 if opp.inkwell_total < 4 else (2.2 if opp.inkwell_total < 7 else 0.8)
        ink_val = min(7, p.inkwell_total) * ink_factor - min(7, opp.inkwell_total) * opp_ink_factor

        return lore_diff + (my_board_val - opp_board_val) + hand_val + ink_val

    def score_action(self, state: GameState, action: GameAction) -> float:
        p_id = state.active_player
        p = state.players[p_id]
        opp = state.players[1 - p_id]

        # 1. Mulligan evaluation (CR Rule 2.2.4 & Matchup-Aware Tournament Logic)
        if action.action_type == ActionType.MULLIGAN:
            mask = action.mulligan_keep_mask
            kept_cards = [p.hand[i] for i, k in enumerate(mask) if i < len(p.hand) and k]
            returned_cnt = sum(1 for i, k in enumerate(mask) if i < len(p.hand) and not k)
            uninkables = sum(1 for c in kept_cards if not c.inkable)
            one_drops = sum(1 for c in kept_cards if c.cost == 1 and c.card_type == "Character")
            two_drops = sum(1 for c in kept_cards if c.cost == 2 and c.card_type == "Character")
            early_drops = one_drops + two_drops
            three_drops = sum(1 for c in kept_cards if c.cost == 3 and c.card_type == "Character")
            four_drops = sum(1 for c in kept_cards if c.cost == 4 and c.card_type == "Character")
            heavy_bombs = sum(1 for c in kept_cards if c.cost >= 6)

            # Matchup & Turn Initiative Context
            is_on_the_play = (state.active_player == state.starting_player)
            opp_inks = set(state.known_opponent_inks[p_id])

            # Detect player's own deck archetype from deck + hand
            all_known_deck = p.deck + p.hand
            deck_chars = [c for c in all_known_deck if c.card_type == "Character"]
            avg_deck_cost = (sum(c.cost for c in all_known_deck) / len(all_known_deck)) if all_known_deck else 3.5
            is_aggro_deck = (avg_deck_cost < 3.2) or (sum(1 for c in deck_chars if c.cost <= 2) >= 16)

            score = 0.0

            # 1. Early Curve Foundation
            score += one_drops * 5.0
            score += two_drops * 4.5
            if one_drops >= 1 and two_drops >= 1:
                score += 7.0  # Gold standard: guaranteed T1 into T2 curve
                if three_drops >= 1:
                    score += 4.5  # Platinum standard: perfect 1 -> 2 -> 3 curve
            elif early_drops == 0:
                # Fatal flaw: no 1-drop and no 2-drop character -> reward throwing cards back to dig!
                score -= 32.0 - (returned_cnt * 5.0)

            # 2. Play vs Draw Dynamics
            if is_on_the_play:
                # On the play: skips draw step on T1; having a 1-drop is mandatory to hold tempo
                if one_drops >= 1:
                    score += 6.0
                else:
                    score -= 8.0  # Harsh penalty for no 1-drop on the play
            else:
                # On the draw: draws on T1; Underdog cards trigger in T1
                underdogs = sum(1 for c in kept_cards if c.abilities.has_underdog)
                score += underdogs * 4.0
                # Cheap challengers / rush help contest opponent's T1 tempo opening
                cheap_challengers = sum(
                    1 for c in kept_cards
                    if c.cost <= 2 and (
                        (c.abilities.challenger_bonus or 0) >= 2
                        or bool(c.abilities.keywords & KeywordFlag.RUSH)
                    )
                )
                score += cheap_challengers * 3.5

            # 3. Archetype Alignment
            if is_aggro_deck:
                # Aggro must NEVER keep 4+ cost cards unless it already has 1-drop AND 2-drop
                if one_drops == 0 or two_drops == 0:
                    score -= (four_drops * 6.0 + heavy_bombs * 10.0)
                else:
                    score -= max(0, four_drops - 1) * 5.0
                # Swarming with multiple 1-drops is advantageous for aggro
                if one_drops >= 2:
                    score += 4.0
            else:
                # Midrange/Control rewards ramp (e.g. One Jump Ahead, Gramma Tala)
                ramp_cards = sum(
                    1 for c in kept_cards
                    if c.cost <= 3 and (
                        "into your inkwell" in c.full_text.lower()
                        or "one jump ahead" in c.full_name.lower()
                    )
                )
                score += ramp_cards * 5.0

            # Elinor - Renowned Diplomat competitive heuristics:
            has_elinor_in_kept = any("elinor - renowned diplomat" in c.full_name.lower() for c in kept_cards)
            has_ramp_or_discount = any(
                (c.cost <= 2 and any(cl.op == EffectOpType.DISCOUNT_NEXT_CHAR for cl in c.abilities.clauses))
                or "aurora - holding court" in c.full_name.lower()
                or "one jump ahead" in c.full_name.lower()
                or "gramma tala" in c.full_name.lower()
                or "sail the astral seas" in c.full_name.lower()
                for c in kept_cards
            )
            if has_elinor_in_kept:
                if is_on_the_play:
                    # OTP: Keep Elinor blind if we have ramp characters! She snowballs uncounterably on T3
                    if has_ramp_or_discount and early_drops >= 1:
                        score += 12.0
                    elif one_drops >= 1 and two_drops >= 1:
                        score += 6.5
                    else:
                        score -= 5.0
                else:
                    # OTD: "I don't like keeping Elinor OTD because a lot of the time you can't play her and she's just stuck in hand"
                    if not (has_ramp_or_discount and early_drops >= 2):
                        score -= 7.5
                    else:
                        score += 3.0

            # Princess Aggro On The Draw (OTD) interaction keeps: Ursula 2c & Dale
            if not is_on_the_play and is_aggro_deck:
                has_ursula_2c = any(c.cost == 2 and "ursula - deceiver" in c.full_name.lower() for c in kept_cards)
                has_dale = any("dale - ready for his shot" in c.full_name.lower() for c in kept_cards)
                if has_ursula_2c and (early_drops >= 1 or has_ramp_or_discount):
                    score += 8.5  # Song disruption stops blowouts
                if has_dale and (early_drops >= 1 or has_ramp_or_discount):
                    score += 8.0  # Counter-trades and threat equalizing

            # Purple Steel Droids combo assembly:
            is_droid_deck = any("omnidroid" in c.full_name.lower() or "syndrome" in c.full_name.lower() for c in all_known_deck)
            if is_droid_deck:
                droid_1c = sum(1 for c in kept_cards if c.cost == 1 and "omnidroid" in c.full_name.lower())
                shift_target_2c = sum(1 for c in kept_cards if c.cost == 2 and ("morph" in c.full_name.lower() or "incrediboy" in c.full_name.lower()))
                syndrome_shift = sum(1 for c in kept_cards if "syndrome - out for revenge" in c.full_name.lower())
                big_droid = sum(1 for c in kept_cards if c.cost >= 6 and "omnidroid" in c.full_name.lower())
                if shift_target_2c >= 1 and syndrome_shift >= 1:
                    score += 14.0
                    if big_droid >= 1:
                        score += 12.0  # Full Turn 4 combo blowout
                    if droid_1c >= 1:
                        score += 4.5
                elif syndrome_shift >= 1 and big_droid >= 1:
                    score += 6.0
                if not is_on_the_play:
                    defensive_pings = sum(
                        1 for c in kept_cards
                        if "look what you've done" in c.full_name.lower()
                        or "angel - experiment 624" in c.full_name.lower()
                        or ("omnidroid - v.9" in c.full_name.lower() and c.cost == 4)
                    )
                    score += defensive_pings * 6.5
                else:
                    if any("omnidroid - v.9" in c.full_name.lower() for c in kept_cards) and shift_target_2c == 0:
                        score -= 4.0

            # 4. Matchup Specifics (opp_inks)
            is_opp_aggro = ("Amber" in opp_inks and "Emerald" in opp_inks) or ("Amber" in opp_inks and "Amethyst" in opp_inks)
            is_opp_control = ("Ruby" in opp_inks and "Sapphire" in opp_inks) or ("Ruby" in opp_inks and "Amethyst" in opp_inks)
            is_opp_steel = "Steel" in opp_inks
            is_opp_discard = "Emerald" in opp_inks and "Steel" in opp_inks

            if is_opp_aggro:
                # Against Aggro: prioritize early removal, Bodyguards, and sturdy blockers
                early_removal = sum(
                    1 for c in kept_cards
                    if c.cost <= 3 and any(
                        cl.op in (
                            EffectOpType.DEAL_DAMAGE,
                            EffectOpType.DEAL_DAMAGE_ALL_OPP,
                            EffectOpType.BANISH_LOW_STR_CHAR,
                        )
                        for cl in c.abilities.clauses
                    )
                )
                bodyguards = sum(1 for c in kept_cards if c.cost <= 3 and bool(c.abilities.keywords & KeywordFlag.BODYGUARD))
                score += (early_removal * 4.5 + bodyguards * 3.5)
                score -= sum(1 for c in kept_cards if c.cost >= 5) * 6.0

            elif is_opp_control:
                # Against Control: prioritize card draw engines and recurring value
                draw_engines = sum(
                    1 for c in kept_cards
                    if any(cl.op == EffectOpType.DRAW for cl in c.abilities.clauses)
                )
                score += draw_engines * 4.0
                dead_removals = sum(
                    1 for c in kept_cards
                    if c.card_type == "Action" and not c.inkable and any(
                        cl.op in (EffectOpType.DEAL_DAMAGE, EffectOpType.BANISH_LOW_STR_CHAR)
                        for cl in c.abilities.clauses
                    )
                )
                score -= dead_removals * 5.0

            if is_opp_steel:
                # Against Steel: avoid fragile 1-willpower characters that die to 1-damage pings, keep 4 WP Thomas
                fragile_1_wp = sum(1 for c in kept_cards if c.card_type == "Character" and c.cost <= 2 and c.willpower <= 1)
                score -= fragile_1_wp * 3.5
                sturdy_2plus_wp = sum(1 for c in kept_cards if c.card_type == "Character" and c.cost <= 2 and c.willpower >= 2)
                score += sturdy_2plus_wp * 2.5
                thomas_cnt = sum(1 for c in kept_cards if "thomas - wide-eyed recruit" in c.full_name.lower())
                score += thomas_cnt * 4.0

            if is_opp_discard:
                # Against Discard: prioritize inkables and immediate cantrips
                score -= uninkables * 3.0
                cantrips = sum(
                    1 for c in kept_cards
                    if c.cost <= 3 and any(cl.op == EffectOpType.DRAW for cl in c.abilities.clauses)
                )
                score += cantrips * 4.0

            # 5. Synergies: Singer + Song & Shift Lines
            singers = [c for c in kept_cards if c.cost <= 3 and (c.abilities.singer_cost or 0) >= 3]
            songs = [c for c in kept_cards if c.is_song]
            for s in singers:
                singer_pwr = s.abilities.singer_cost or s.cost
                if any(song.cost <= singer_pwr for song in songs):
                    score += 5.0
                    break

            base_names = {c.base_name for c in kept_cards if c.abilities.shift_cost is None}
            shift_cards = [c for c in kept_cards if c.abilities.shift_cost is not None]
            for sc in shift_cards:
                if sc.base_name in base_names:
                    score += 4.0
                    break

            # 6. Penalties: Uninkables & Redundant Duplicates
            if uninkables == 0:
                score += 2.0
            elif uninkables > 2:
                score -= (uninkables - 2) * 8.0

            seen_cost_3plus: set[int] = set()
            for c in kept_cards:
                if c.cost >= 3:
                    if c.card_id in seen_cost_3plus:
                        score -= 5.0  # Redundant duplicate penalty
                    else:
                        seen_cost_3plus.add(c.card_id)

            if heavy_bombs > 0:
                looters = sum(1 for c in kept_cards if c.cost <= 3 and any(cl.op == EffectOpType.LOOT_SELF for cl in c.abilities.clauses))
                if looters > 0 and heavy_bombs == 1:
                    score += 1.0
                else:
                    score -= heavy_bombs * 6.0

            return score

        # Precompute race, lethal math, and revealed opponent hand knowledge for all MAIN phase actions
        my_ready_lore = sum(e.effective_lore(p, opp) for e in p.characters_in_play if e.can_quest())
        opp_next_turn_lore = sum(
            e.effective_lore(opp, p) for e in opp.characters_in_play if not e.has_keyword(KeywordFlag.RECKLESS)
        ) + sum(loc.card.lore for loc in opp.locations_in_play)
        opp_threatens_lethal = (opp.lore + opp_next_turn_lore >= 20) and (p.lore + my_ready_lore < 20)

        # Extract opponent's currently revealed cards in hand (visible to AI after Ursula - Deceiver, Mowgli, Mike Wazowski, etc.)
        rev_pool = list(opp.revealed_hand_ids)
        opp_revealed_cards = []
        for oc in opp.hand:
            if oc.card_id in rev_pool:
                opp_revealed_cards.append(oc)
                rev_pool.remove(oc.card_id)
        opp_revealed_songs = [oc for oc in opp_revealed_cards if oc.is_song]
        opp_revealed_non_chars = [oc for oc in opp_revealed_cards if oc.card_type != "Character"]
        opp_hand_fully_known = len(opp.hand) > 0 and len(opp_revealed_cards) == len(opp.hand)
        opp_has_revealed_sweeper = any(
            cl.op in (EffectOpType.BANISH_ALL_CHARS, EffectOpType.DEAL_DAMAGE_ALL_OPP, EffectOpType.BANISH_ALL_DAMAGED_OPP)
            for oc in opp_revealed_cards
            for cl in oc.abilities.clauses
        )

        # 2. Inkwell selection — MUST ink before spending ink on PLAY_CARD so we hit on-curve drops (3 ink T3, 4 ink T4, etc.)!
        if action.action_type == ActionType.INK_CARD:
            # NEVER ink the last card in hand — inking leaves 0 cards in hand to spend ink on!
            if len(p.hand) <= 1:
                return -30.0
            if p.inkwell_total >= 8 and len(p.hand) <= 2:
                return -10.0  # Hold cards in late game if already at 8+ ink

            is_p2_t1 = (state.turn_number == 2 and state.active_player != state.starting_player)

            def _eff_needed_ink(c) -> int:
                if c.abilities.shift_cost is not None and any(c.can_shift_onto_single(b.card) for b in p.board):
                    return p.effective_shift_cost(c)
                return p.effective_play_cost(c, is_second_player_first_turn=is_p2_t1)

            max_hand_cost = max((_eff_needed_ink(c) for c in p.hand), default=0)
            total_hand_cost = sum(_eff_needed_ink(c) for c in p.hand)
            # Never ink a card if we ALREADY have enough payable ink this turn to play EVERY card in our hand!
            if p.inkwell_total >= 2 and total_hand_cost <= p.total_payable_ink:
                return -20.0
            # Low-curve / Aggro rule: once at 4+ ink, don't ink away playable characters if hand is small and max cost <= p.inkwell_total
            if p.inkwell_total >= 4 and len(p.hand) <= 3 and max_hand_cost <= p.inkwell_total:
                return -15.0

            card = p.hand[action.hand_index]
            dup_count = sum(1 for c in p.hand if c.card_id == card.card_id)
            inkable_count = sum(1 for c in p.hand if c.inkable)
            score = 36.0 if p.inkwell_total < 6 else (28.0 if len(p.hand) >= 3 else 8.0)

            # Turn 1 / Empty Inkwell: Inking is mandatory to unlock playing cards!
            # Give massive unconditional bonus (+30.0) so inking on Turn 1 always dominates PASS_TURN
            if p.inkwell_total == 0:
                score += 30.0

            # Midgame curve management (Turns 2-3 with already >= 2 ink):
            # If inking leaves us with 0 playable cards this turn, penalize inking, BUT only when we ALREADY have ink!
            new_ink_after = p.total_payable_ink + 1
            rem_playable_after_ink = sum(
                1 for i, c in enumerate(p.hand)
                if i != action.hand_index and _eff_needed_ink(c) <= new_ink_after
            )
            if 2 <= p.inkwell_total <= 3 and rem_playable_after_ink == 0:
                score -= 15.0

            if dup_count > 1:
                score += 4.5
            # Revealed hand awareness for Ursula - Deceiver / Mowgli / Elisa Maza:
            card_lower = card.full_text.lower()
            if "discards a song card" in card_lower:
                if opp_revealed_songs and dup_count == 1:
                    # Opponent is KNOWN to hold a Song in hand! Protect Ursula so we can snipe it!
                    score -= 14.0
                elif opp_hand_fully_known and not opp_revealed_songs:
                    # We already revealed opponent's entire hand and they have 0 Songs — extra Ursula is ideal ink!
                    score += 6.5
            elif "discards a non-character card" in card_lower:
                if opp_revealed_non_chars and dup_count == 1:
                    score -= 12.0
                elif opp_hand_fully_known and not opp_revealed_non_chars:
                    score += 5.5

            # Prefer inking vanilla 1-lore characters (e.g. Thomas - Wide-Eyed Recruit) when we have multiple early plays
            if (
                card.card_type == "Character"
                and len(card.abilities.clauses) == 0
                and card.lore <= 1
                and not card.abilities.deals_willpower_damage_in_challenges
                and not (card.abilities.keywords & KeywordFlag.BODYGUARD)
            ):
                score += 1.5
            # Protect Dale - Ready for His Shot when we run 0-strength high-willpower characters!
            if card.abilities.deals_willpower_damage_in_challenges and dup_count == 1:
                has_low_str_allies = any(
                    e.card.strength <= 1 and e.card.willpower >= 3 for e in p.characters_in_play
                ) or any(
                    h.card_type == "Character" and h.strength <= 1 and h.willpower >= 3
                    for i, h in enumerate(p.hand) if i != action.hand_index
                )
                if has_low_str_allies:
                    score -= 7.5
            # Protect Shift cards (e.g. Pocahontas - Peacekeeper, Judy Hopps - Lead Detective) if we have a Shift target in play!
            if card.abilities.shift_cost is not None and dup_count == 1:
                if any(card.can_shift_onto_single(b.card) for b in p.board):
                    score -= 14.0
                elif any(card.can_shift_onto_single(h) for i, h in enumerate(p.hand) if i != action.hand_index):
                    score -= 7.0
            # Protect Circle of Life from being inked if we only have 1 copy
            if any(cl.op == EffectOpType.REANIMATE_CHAR for cl in card.abilities.clauses) and dup_count == 1:
                score -= 9.0
            # Protect Pocahontas - Guiding the Tribe on Turns 1-2 if we have a 1-drop in hand
            if any(cl.op == EffectOpType.PLAY_COST1_CHAR_FREE for cl in card.abilities.clauses) and dup_count == 1:
                score -= 10.0
            # Protect 2-drop Characters in the first two turns if it's our only 2-drop (unless it's our only inkable card!)
            two_drop_chars = sum(1 for c in p.hand if c.cost <= 2 and c.card_type == "Character")
            if card.cost <= 2 and card.card_type == "Character" and p.inkwell_total < 2 and two_drop_chars <= 1 and inkable_count > 1:
                score -= 10.0
            # Protect 1-drop Characters on Turn 1 if it's our only 1-drop (unless it's our only inkable card!)
            one_drop_chars = sum(1 for c in p.hand if c.cost == 1 and c.card_type == "Character")
            if card.cost == 1 and card.card_type == "Character" and p.inkwell_total == 0 and one_drop_chars <= 1 and inkable_count > 1:
                score -= 10.0
            # Protect on-curve character for THIS turn (cost == p.inkwell_total + 1) if we only have 1 copy (unless it's our only inkable card)
            on_curve_cost = p.inkwell_total + 1
            on_curve_chars = sum(1 for c in p.hand if c.cost == on_curve_cost and c.card_type == "Character")
            if card.cost == on_curve_cost and card.card_type == "Character" and on_curve_chars <= 1 and inkable_count > 1:
                score -= 5.0
            # If we have a looter (Strike a Good Match / Tod) in hand, hold onto 1 heavy Character (Prince Phillip) to pitch to discard!
            has_looter_in_hand = any(
                any(cl.op == EffectOpType.LOOT_SELF for cl in h.abilities.clauses)
                for i, h in enumerate(p.hand)
                if i != action.hand_index
            )
            if has_looter_in_hand and card.card_type == "Character" and card.cost >= 7 and dup_count == 1:
                score -= 7.5
            elif card.cost > p.inkwell_total + 3:
                score += 3.5
            return score

        # 3. Sing Song (free mana & Circle of Life / Malicious / Akood combos!)
        if action.action_type == ActionType.SING_SONG:
            card = p.hand[action.hand_index]
            singer = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            singer_lore_cost = singer.effective_lore(p, opp) if singer else 1
            singer_pwr = (singer.card.abilities.singer_cost or singer.card.cost) if singer else card.cost
            overpay_penalty = 0.4 * max(0, singer_pwr - card.cost) if singer else 0.0
            if singer and any(cl.op == EffectOpType.REPLAY_SONG_FROM_DISCARD for cl in singer.card.abilities.clauses):
                if any(c.is_song and c.cost <= 4 for c in p.discard):
                    overpay_penalty += 18.0
            ops = {cl.op for cl in card.abilities.clauses}

            # Do not sing symmetric board wipe if ahead
            if any(cl.op == EffectOpType.BANISH_ALL_CHARS and cl.threshold == 0 for cl in card.abilities.clauses):
                if len(opp.characters_in_play) <= len(p.characters_in_play) and not opp_threatens_lethal:
                    return -25.0
                return 55.0 if opp_threatens_lethal else 45.0

            # Under the Sea / Strength-threshold mass removal
            if any(cl.op == EffectOpType.BANISH_ALL_CHARS and cl.threshold > 0 for cl in card.abilities.clauses):
                thresh = max(cl.threshold for cl in card.abilities.clauses if cl.op == EffectOpType.BANISH_ALL_CHARS)
                victims = sum(1 for e in opp.characters_in_play if e.effective_strength <= thresh)
                if victims == 0:
                    return -25.0
                return 42.0 + victims * 9.0 + (20.0 if opp_threatens_lethal else 0.0)

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
                if kills_one_wp == 0 and not has_followup and singer_lore_cost >= 2 and not opp_threatens_lethal:
                    return 12.0
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

            # Targeted removal / bounce songs (Mother Knows Best, World's Greatest Criminal Mind, Brawl, etc.)
            if ops & {EffectOpType.BANISH_CHAR, EffectOpType.BANISH_LOW_STR_CHAR, EffectOpType.BOUNCE_OPP_CHAR, EffectOpType.DEAL_DAMAGE}:
                if not opp.characters_in_play:
                    return -20.0
                lethal_stop_bonus = 22.0 if opp_threatens_lethal else 0.0
                return 34.0 + card.cost * 2.5 + lethal_stop_bonus - singer_lore_cost * 1.2 - overpay_penalty

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
            if any(cl.op == EffectOpType.FOG_CHALLENGES for cl in card.abilities.clauses):
                # Pocahontas - Peacekeeper Shift: fogs all challenges until next turn!
                if not p.challenged_this_turn and len(opp.characters_in_play) >= 1 and len(p.characters_in_play) >= 2:
                    bonus += 16.0
            # Syndrome - Out for Revenge Shift: Turn 4 combo enabler
            if "syndrome" in card.full_name.lower():
                has_robot = any("Robot" in c.subtypes or "omnidroid" in c.full_name.lower() for c in p.discard + p.hand)
                if has_robot:
                    bonus += 18.0
            # Omnidroid - V.9 Shift (deals 2 damage on shift to snipe key early character)
            if any(cl.op == EffectOpType.DEAL_DAMAGE and cl.amount == 2 for cl in card.abilities.clauses):
                snipeable = [e for e in opp.characters_in_play if e.effective_willpower - e.damage <= 2]
                if snipeable:
                    bonus += 15.0
                    if any(any(k in e.card.full_name.lower() for k in ["willow", "pluto", "morph", "aurora"]) for e in snipeable):
                        bonus += 7.0
            return 25.0 + card.cost * 2.0 + card.lore * 2.0 + bonus

        # 5. Play Card
        if action.action_type == ActionType.PLAY_CARD:
            card = p.hand[action.hand_index]
            card_lower = card.full_text.lower()
            ops = {cl.op for cl in card.abilities.clauses}
            if any(cl.op == EffectOpType.BANISH_ALL_CHARS and cl.threshold == 0 for cl in card.abilities.clauses):
                opp_threat = sum(e.card.lore * 2 + e.effective_strength for e in opp.characters_in_play)
                my_threat = sum(e.card.lore * 2 + e.effective_strength for e in p.characters_in_play)
                if opp_threatens_lethal or opp_threat > my_threat + 4 or (opp.lore >= 14 and len(opp.characters_in_play) >= 2):
                    return 48.0 if opp_threatens_lethal else 35.0
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
            # Exploit revealed opponent hand when playing Ursula - Deceiver / Mowgli / Elisa Maza / Lenny!
            if EffectOpType.DISCARD_OPP in ops:
                if "discards a song card" in card_lower:
                    if opp_revealed_songs:
                        # Guaranteed Song snipe from opponent's revealed hand!
                        best_s_cost = max(s.cost for s in opp_revealed_songs)
                        bonus += 14.0 + best_s_cost * 2.0
                    elif opp_hand_fully_known and not opp_revealed_songs:
                        # We already know opponent's entire hand has 0 Songs — play other 2-drops first!
                        bonus -= 4.5
                    elif not opp_revealed_cards and len(opp.hand) >= 2:
                        # Unrevealed hand: scouting opponent's hand + sniping potential Song is high value!
                        bonus += 4.5
                elif "discards a non-character card" in card_lower:
                    if opp_revealed_non_chars:
                        bonus += 12.0
                    elif opp_hand_fully_known and not opp_revealed_non_chars:
                        bonus -= 4.0

            if EffectOpType.BANISH_ALL_DAMAGED_OPP in ops:
                bonus += sum(12.0 for e in opp.characters_in_play if e.damage > 0)
            if EffectOpType.PLAY_COST1_CHAR_FREE in ops:
                has_c1 = any(
                    c.card_type == "Character" and c.cost <= 1 and i != action.hand_index
                    for i, c in enumerate(p.hand)
                )
                if has_c1:
                    bonus += 10.5
            if EffectOpType.BUFF_ALLY_LORE in ops:
                # Lionheart - Incumbent Mayor / Pocahontas - Finding the Way:
                # MUST be played BEFORE ready characters quest so they receive +1 ◊ this turn!
                ready_questers = [
                    e for e in p.characters_in_play
                    if not e.exerted and not e.drying and not e.has_keyword(KeywordFlag.RECKLESS)
                ]
                buff_cap = max((cl.amount for cl in card.abilities.clauses if cl.op == EffectOpType.BUFF_ALLY_LORE), default=1)
                eff_buffed = min(len(ready_questers), buff_cap)
                if eff_buffed > 0:
                    if p.lore + my_ready_lore + eff_buffed >= 20:
                        return 950.0
                    return 33.5 + eff_buffed * 4.5 + card.cost * 1.5
                else:
                    other_c1 = any(
                        c.card_type == "Character" and c.cost == 1 and i != action.hand_index
                        and EffectOpType.BUFF_ALLY_LORE not in {cl.op for cl in c.abilities.clauses}
                        for i, c in enumerate(p.hand)
                    )
                    if other_c1:
                        bonus -= 6.5

            if card.abilities.deals_willpower_damage_in_challenges:
                high_wp_allies = sum(1 for e in p.characters_in_play if e.effective_willpower >= 3 and e.effective_strength <= 1)
                bonus += high_wp_allies * 3.5

            # Targeted removal / bounce when opponent threatens lethal
            if opp_threatens_lethal and (ops & {EffectOpType.BANISH_CHAR, EffectOpType.BANISH_LOW_STR_CHAR, EffectOpType.BOUNCE_OPP_CHAR, EffectOpType.DEAL_DAMAGE, EffectOpType.INK_OPP_CHAR, EffectOpType.EXERT_OPP_CHAR}):
                bonus += 24.0

            # Strongly prefer deploying Characters over hard-casting Songs with ink in the early/mid game!
            if card.is_song:
                has_playable_char = any(
                    c.card_type == "Character" and p.effective_play_cost(c) <= p.total_payable_ink for c in p.hand
                )
                if has_playable_char and not opp_threatens_lethal:
                    bonus -= 10.0
                elif p.inkwell_total <= 3:
                    bonus -= 4.0
            elif EffectOpType.LOOT_SELF in ops and any(c.card_type == "Character" and c.cost >= 6 for c in p.hand):
                bonus += 5.0

            if card.card_type == "Character":
                bonus += 3.0  # Board presence bonus
                # Play around known board wipes in opponent's revealed hand if we already have >= 4 characters on board!
                if opp_has_revealed_sweeper and len(p.characters_in_play) >= 4 and len(card.abilities.clauses) == 0 and p.lore + my_ready_lore < 18:
                    bonus -= 5.5
                if card.cost <= 2 and (card.abilities.singer_cost or 0) >= 3:
                    bonus += 3.5
                if card.abilities.has_underdog and state.turn_number == 2:
                    bonus += 5.0
                if EffectOpType.DISCOUNT_NEXT_CHAR in ops:
                    my_eff_cost = p.effective_play_cost(card, is_second_player_first_turn=(state.turn_number == 2 and state.active_player != state.starting_player))
                    rem_ink = p.total_payable_ink - my_eff_cost
                    can_chain_now = any(
                        c.card_type == "Character" and i != action.hand_index and max(0, p.effective_play_cost(c) - 1) <= rem_ink
                        for i, c in enumerate(p.hand)
                    )
                    bonus += 9.5 if can_chain_now else 3.0
                if "3 or more exerted characters" in card_lower and len(p.characters_in_play) >= 2:
                    bonus += 5.5
                if any(cl.op == EffectOpType.BOUNCE_OPP_CHAR and cl.cost_ink > 0 for cl in card.abilities.clauses):
                    if opp.characters_in_play and p.total_payable_ink >= card.cost + 2:
                        bonus += 4.5
                if action.enter_exerted:
                    if card.abilities.keywords & KeywordFlag.BODYGUARD:
                        if len(p.characters_in_play) >= 1:
                            bonus += 2.5
                    elif "whenever one of your other characters quests" in card_lower:
                        ready_questers_cnt = sum(1 for e in p.characters_in_play if e.can_quest())
                        if len(opp.characters_in_play) > 0 and ready_questers_cnt > 0:
                            bonus += 14.0  # Enter exerted BEFORE other characters quest to debuff opp -2 STR!
                        else:
                            bonus -= 3.0
            return 14.0 + card.cost * 2.2 + card.lore * 2.5 + bonus

        # 6. Challenge vs Quest (Characters & Locations per CR Rule 4.6.8)
        if action.action_type == ActionType.CHALLENGE:
            attacker = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            defender = next((e for e in opp.board if e.instance_id == action.target_instance_id), None)
            if attacker is None or defender is None:
                return -5.0

            # Never challenge if we already have lethal lore on board right now!
            if p.lore + my_ready_lore >= 20:
                return -25.0

            p_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in p.characters_in_play)
            opp_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in opp.characters_in_play)
            base_atk = attacker.effective_willpower if p_uses_wp else attacker.effective_strength
            atk_pwr = (
                base_atk
                + attacker.card.abilities.challenger_bonus
                + attacker.temp_challenger_mod
            )
            def_resist = defender.effective_resist(opp)
            dmg_to_def = max(0, atk_pwr - def_resist)
            if dmg_to_def <= 0:
                return -50.0
            base_def = (
                0
                if defender.card.card_type == "Location"
                else (defender.effective_willpower if opp_uses_wp else defender.effective_strength)
            )
            atk_resist = attacker.effective_resist(p)
            dmg_to_atk = (
                0
                if defender.card.card_type == "Location"
                else max(0, base_def - atk_resist)
            )

            def_rem_wp = defender.effective_willpower - defender.damage
            kills_def = dmg_to_def >= def_rem_wp
            survives = (attacker.damage + dmg_to_atk) < attacker.effective_willpower

            # Check if other ready allies on board can finish off `defender` this turn (Cooperative Team-Kill!)
            other_allies_dmg = 0
            if not kills_def:
                def_evasive = defender.has_keyword(KeywordFlag.EVASIVE)
                for ally in p.characters_in_play:
                    if ally.instance_id != attacker.instance_id and ally.can_challenge():
                        if def_evasive and not (ally.has_keyword(KeywordFlag.EVASIVE) or ally.has_keyword(KeywordFlag.ALERT)):
                            continue
                        ally_base = ally.effective_willpower if p_uses_wp else ally.effective_strength
                        ally_pwr = ally_base + ally.card.abilities.challenger_bonus + ally.temp_challenger_mod
                        other_allies_dmg += max(0, ally_pwr - def_resist)
            can_team_kill = (not kills_def) and ((dmg_to_def + other_allies_dmg) >= def_rem_wp)

            opp_board_lore = sum(e.effective_lore(opp, p) for e in opp.characters_in_play)
            def_lore = defender.effective_lore(opp, p) if defender.card.card_type == "Character" else defender.card.lore
            atk_lore = attacker.effective_lore(p, opp)

            # Emergency Anti-Lethal bonus when opponent threatens lethal next turn!
            anti_lethal_bonus = 0.0
            if opp_threatens_lethal and def_lore > 0:
                if kills_def:
                    anti_lethal_bonus = 45.0 + def_lore * 8.0
                elif can_team_kill:
                    anti_lethal_bonus = 38.0 + def_lore * 6.0

            # Revealed Hand Shift Denial: if opponent's revealed hand contains a Shift card that can shift onto `defender`, prioritize killing `defender`!
            shift_denial_bonus = 0.0
            if (kills_def or can_team_kill) and defender.card.card_type == "Character":
                if any(oc.abilities.shift_cost is not None and oc.can_shift_onto_single(defender.card) for oc in opp_revealed_cards):
                    shift_denial_bonus = 14.0

            # The Prince - Searching for His Love: draws a card when banishing a character in a challenge!
            prince_draw_bonus = (
                6.5
                if kills_def
                and defender.card.card_type == "Character"
                and "banishes another character in a challenge, draw a card" in attacker.card.full_text.lower()
                else 0.0
            )

            score = 5.0 + anti_lethal_bonus + shift_denial_bonus + prince_draw_bonus
            if kills_def and survives:
                score += 22.0 + def_lore * 5.0 + defender.card.cost * 1.5
            elif kills_def and not survives:
                if (
                    def_lore > atk_lore
                    or opp.lore >= 13
                    or opp_board_lore >= 4
                    or defender.card.cost > attacker.card.cost
                    or len(defender.card.abilities.clauses) > 0
                    or shift_denial_bonus > 0
                ):
                    score += 15.5 + def_lore * 4.5 - atk_lore * 1.8
                else:
                    score -= 3.0
            elif can_team_kill:
                # Two smaller characters ganging up to kill a high-value quester (e.g. Elinor 1/4, Cindy 1/4, Lionheart 2/4)
                if def_lore >= 2 or opp.lore >= 11 or opp_board_lore >= 4 or defender.card.cost >= 3 or shift_denial_bonus > 0:
                    score += 16.5 + def_lore * 4.5 - atk_lore * 1.5
                else:
                    score -= 2.0
            else:
                # Cannot kill defender this turn even with all allies: only chip Locations or 2+ lore threats when surviving!
                if defender.card.card_type == "Location" and defender.card.lore >= 1:
                    score += 11.0 + def_lore * 2.5
                elif survives and def_lore >= 2 and (opp.lore >= 14 or opp_board_lore >= 6):
                    score += 10.5 + def_lore * 2.0
                else:
                    score -= 6.0

            # Elinor - Renowned Diplomat synergy: if we have Elinor in play and 2 exerted characters, attacking with a 3rd exerts it and triggers Elinor at end of turn!
            has_elinor = any("3 or more exerted characters" in e.card.full_text.lower() for e in p.characters_in_play)
            exerted_cnt = sum(1 for e in p.characters_in_play if e.exerted)
            if has_elinor and exerted_cnt == 2:
                score += 6.5

            return score

        if action.action_type == ActionType.QUEST:
            quester = next((e for e in p.characters_in_play if e.instance_id == action.source_instance_id), None)
            if quester is None:
                return 0.0
            q_lore = quester.effective_lore(p, opp)
            if p.lore + q_lore >= 20:
                return 1000.0  # Immediate victory!
            # If all our ready questers together reach 20 lore this turn, quest immediately!
            if p.lore + my_ready_lore >= 20:
                return 500.0 + q_lore * 10.0

            # Aurora - Holding Court (Set 9): quest BEFORE playing a Princess or Queen character to apply -1 ink discount this turn!
            if any(
                cl.trigger == TriggerType.ON_QUEST and cl.op == EffectOpType.DISCOUNT_NEXT_PRINCESS_QUEEN
                for cl in quester.card.abilities.clauses
            ):
                if any(p.is_princess_or_queen(c) for c in p.hand):
                    return 38.5
            elif any(
                cl.trigger == TriggerType.ON_QUEST and cl.op == EffectOpType.DISCOUNT_NEXT_CHAR
                for cl in quester.card.abilities.clauses
            ):
                if any(c.card_type == "Character" for c in p.hand):
                    return 38.5

            # Anti-Lethal check: if opponent has lethal on board next turn and `quester` can challenge an exerted opposing quester, DO NOT quest before challenging!
            if opp_threatens_lethal and quester.can_challenge() and not p.cant_challenge_this_turn:
                p_uses_wp = any(e.card.abilities.deals_willpower_damage_in_challenges for e in p.characters_in_play)
                q_atk = quester.effective_willpower if p_uses_wp else quester.effective_strength
                exerted_opp_questers = [
                    e for e in opp.characters_in_play
                    if e.exerted
                    and e.effective_lore(opp, p) > 0
                    and (not e.has_keyword(KeywordFlag.EVASIVE) or quester.has_keyword(KeywordFlag.EVASIVE) or quester.has_keyword(KeywordFlag.ALERT))
                ]
                if exerted_opp_questers and q_atk + quester.card.abilities.challenger_bonus > 0:
                    return -18.0

            bonus = 0.0
            if opp_has_revealed_sweeper:
                bonus += 3.5  # Cash in lore before opponent's revealed sweeper!
            if any(cl.op == EffectOpType.REPLAY_SONG_FROM_DISCARD for cl in quester.card.abilities.clauses):
                if any(c.is_song and c.cost <= 4 for c in p.discard):
                    bonus += 18.0
                else:
                    bonus += 4.0
            if any(
                cl.trigger == TriggerType.ON_QUEST and cl.op in (EffectOpType.DRAW, EffectOpType.REVEAL_TOP_CHAR_BOTH)
                for cl in quester.card.abilities.clauses
            ):
                bonus += 6.0

            # Elinor - Renowned Diplomat synergy: if we have Elinor in play and 2 exerted characters, exerting a 3rd draws a card at end of turn!
            has_elinor = any("3 or more exerted characters" in e.card.full_text.lower() for e in p.characters_in_play)
            exerted_cnt = sum(1 for e in p.characters_in_play if e.exerted)
            if has_elinor and exerted_cnt == 2:
                bonus += 6.5

            # Pocahontas Fog: if opponent cannot challenge, all ready characters can quest with 100% safety
            if opp.cant_challenge_this_turn:
                bonus += 14.0

            # Dale trap-trades: if Dale is on board, 4 WP characters hit for 4 in counter-trades
            has_dale = any(e.card.abilities.deals_willpower_damage_in_challenges for e in p.characters_in_play)
            if has_dale and quester.effective_willpower >= 4:
                bonus += 4.5

            # Syndrome - Out for Revenge: questing returns robot from discard and plays robot <= 8 for free!
            if "syndrome" in quester.card.full_name.lower() and any(cl.trigger == TriggerType.ON_QUEST for cl in quester.card.abilities.clauses):
                bonus += 16.0

            # Mickey Mouse - Best in Town: gains +1 lore at end of turn only if exerted!
            if "if this character is exerted" in quester.card.full_text.lower():
                bonus += 5.0

            return 10.0 + q_lore * 4.5 + bonus

        if action.action_type == ActionType.MOVE_TO_LOCATION:
            loc = next((e for e in p.locations_in_play if e.instance_id == action.target_instance_id), None)
            if loc is None:
                return -5.0
            return 4.0 if p.total_payable_ink > loc.card.move_cost + 2 else -2.0

        if action.action_type == ActionType.USE_ABILITY:
            ent = next((e for e in p.board if e.instance_id == action.source_instance_id), None)
            if ent is None:
                return -5.0
            for cl in ent.card.abilities.clauses:
                if cl.op == EffectOpType.DISCOUNT_NEXT_CHAR:
                    char_costs = [p.effective_play_cost(c) for c in p.hand if c.card_type == "Character"]
                    needs_discount = (
                        any(0 < cost - cl.amount <= p.total_payable_ink for cost in char_costs)
                        and sum(char_costs) > p.total_payable_ink
                    )
                    if not needs_discount:
                        return -20.0
                    score = 34.5
                    # Tactical guideline on Pluto / Fragile ramp:
                    # If exerting Pluto exposes him to a free kill from an opposing 1-drop/attacker,
                    # and we don't have a Bodyguard (David) in play or entering to protect him, avoid exerting!
                    if cl.requires_exert and ent.effective_willpower <= 2:
                        has_bodyguard_shield = any(
                            bool(b.card.abilities.keywords & KeywordFlag.BODYGUARD) and b.exerted
                            for b in p.characters_in_play
                        ) or any(
                            bool(c.abilities.keywords & KeywordFlag.BODYGUARD)
                            for c in p.hand if p.effective_play_cost(c) <= p.total_payable_ink
                        )
                        opp_threatens_kill = any(
                            (e.effective_strength >= ent.effective_willpower - ent.damage)
                            for e in opp.characters_in_play
                        )
                        if opp_threatens_kill and not has_bodyguard_shield:
                            score -= 18.0
                    return score
                if cl.op == EffectOpType.GAIN_LORE:
                    return 32.0
                if cl.op == EffectOpType.RAMP_INK:
                    if p.inkwell_total < 7 and len(p.hand) >= 3:
                        return 19.0
                    return -15.0
                if cl.op == EffectOpType.DEBUFF_OPP_STR:
                    return 16.0 if len(opp.characters_in_play) > 0 else -10.0
                if cl.op == EffectOpType.RETRO_EVOLVE:
                    # Retro Evolution Device: banish a character to play a bigger character for free!
                    max_c = max((e.card.cost for e in p.characters_in_play), default=0) + 2
                    has_target = any(c.card_type == "Character" and c.cost <= max_c and c.cost >= 3 for c in p.hand)
                    return 28.5 if has_target else -20.0
                if cl.op == EffectOpType.PUT_UNDER_FROM_DISCARD:
                    # The Black Cauldron: put character from discard under cauldron
                    return 20.0 if any(c.card_type == "Character" for c in p.discard) else -20.0
                if cl.op == EffectOpType.MOVE_DAMAGE:
                    # Luisa Madrigal: move damage
                    return 18.0 if any(e.damage > 0 for e in p.characters_in_play if e != ent) else -10.0
                if cl.op == EffectOpType.GIVE_RUSH_EVASIVE:
                    # Demona: give gargoyle rush/evasive
                    return 22.0 if len(p.hand) >= 2 and any("Gargoyle" in e.card.subtypes for e in p.characters_in_play) else -15.0
            return 11.0

        if action.action_type == ActionType.PASS_TURN:
            if p.inkwell_total == 0 and any(c.inkable for c in p.hand):
                return -100.0  # Mandatory Lorcana rule: never pass turn 1 with 0 ink when holding inkable cards!
            return 0.0

        return 0.0
