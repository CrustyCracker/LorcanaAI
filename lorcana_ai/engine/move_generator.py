"""
Legal Move Generator for Disney Lorcana (Season 14 Core Constructed: Sets 9-14).
Compliant with Official Disney Lorcana Comprehensive Rules v2.3.0 (Effective October 16, 2026):
- Mulligan (2.2.4), Inkwell (4.3), Card Play (4.4), Bodyguard enter-exerted option (8.3.2)
- All 6 Official Shift Variants (8.10.8): Standard, Classification, Universal, Duo, Combo, Temporary, Potato
- Singing Songs & Sing Together X (8.11 & 8.12)
- Challenging Characters & Locations (4.6.8, Bodyguard 8.3.3, Evasive 8.5, Alert 8.15, Rush 8.9, Reckless 8.7.3)
- Moving Characters to Locations (4.7)
- Adventurous (8.16.2 cannot challenge, 8.16.3 cannot declare end of turn if ready and can quest)
- Boost X (8.17) & Activated Abilities (6.2)
- Ink Drop Counters (1.13.1.2)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from lorcana_ai.data.effect_dsl import EffectOpType, KeywordFlag, TriggerType
from lorcana_ai.engine.game_state import BoardEntity, GameState


class ActionType(str, Enum):
    MULLIGAN = "MULLIGAN"
    INK_CARD = "INK_CARD"
    PLAY_CARD = "PLAY_CARD"
    SHIFT_CARD = "SHIFT_CARD"
    SING_SONG = "SING_SONG"
    QUEST = "QUEST"
    CHALLENGE = "CHALLENGE"
    MOVE_TO_LOCATION = "MOVE_TO_LOCATION"
    USE_ABILITY = "USE_ABILITY"
    PASS_TURN = "PASS_TURN"


@dataclass(frozen=True)
class GameAction:
    """Represents a single discrete decision in Lorcana."""
    action_type: ActionType
    hand_index: int = -1
    source_instance_id: int = -1
    target_instance_id: int = -1
    secondary_target_id: int = -1
    enter_exerted: bool = False
    mulligan_keep_mask: tuple[bool, ...] = ()
    description: str = ""

    def action_slot_id(self, max_space: int = 256) -> int:
        """Maps any GameAction deterministically into [0, max_space - 1] for Policy Network logits."""
        if self.action_type == ActionType.PASS_TURN:
            return 0
        if self.action_type == ActionType.MULLIGAN:
            mask_bits = sum((1 << i) for i, k in enumerate(self.mulligan_keep_mask) if k)
            return 1 + (mask_bits % 15)
        if self.action_type == ActionType.INK_CARD:
            return 16 + (max(0, self.hand_index) % 16)
        if self.action_type == ActionType.PLAY_CARD:
            offset = 16 if self.enter_exerted else 0
            return 32 + ((max(0, self.hand_index) + offset) % 32)
        if self.action_type == ActionType.SHIFT_CARD:
            return 64 + ((max(0, self.hand_index) * 3 + max(0, self.target_instance_id)) % 24)
        if self.action_type == ActionType.SING_SONG:
            return 88 + ((max(0, self.hand_index) * 5 + max(0, self.source_instance_id)) % 32)
        if self.action_type == ActionType.QUEST:
            return 120 + (max(0, self.source_instance_id) % 32)
        if self.action_type == ActionType.CHALLENGE:
            return 152 + ((max(0, self.source_instance_id) * 7 + max(0, self.target_instance_id)) % 64)
        if self.action_type == ActionType.MOVE_TO_LOCATION:
            return 216 + ((max(0, self.source_instance_id) + max(0, self.target_instance_id)) % 16)
        if self.action_type == ActionType.USE_ABILITY:
            return 232 + (max(0, self.source_instance_id) % 24)
        return 0


class MoveGenerator:
    """Generates all legal actions for the active player in a given GameState."""

    @classmethod
    def generate_legal_actions(cls, state: GameState) -> list[GameAction]:
        if state.phase == "GAME_OVER":
            return []

        p = state.players[state.active_player]
        opp = state.players[1 - state.active_player]

        # 1. Mulligan phase (CR Rule 2.2.4)
        if state.phase.startswith("MULLIGAN"):
            return cls._generate_mulligan_actions(state)

        actions: list[GameAction] = []
        payable_ink = p.total_payable_ink  # CR Rule 1.13.1.2: inkwell_available + ink_drops

        # 2. Ink a card (once per turn, CR Rule 4.3)
        if not p.inked_this_turn:
            seen_ink_ids: set[int] = set()
            for idx, card in enumerate(p.hand):
                if card.inkable and card.card_id not in seen_ink_ids:
                    seen_ink_ids.add(card.card_id)
                    actions.append(
                        GameAction(
                            action_type=ActionType.INK_CARD,
                            hand_index=idx,
                            description=f"Ink {card.full_name}",
                        )
                    )

        # 3. Play cards from hand / Shift (all 6 variants, CR Rule 8.10.8) / Sing Songs (CR Rule 8.11 & 8.12)
        ready_dry_chars = [e for e in p.characters_in_play if not e.exerted and not e.drying]
        total_chorus_power = sum((e.card.abilities.singer_cost or e.card.cost) for e in ready_dry_chars)
        is_p2_first_turn = (state.turn_number == 2 and state.active_player != state.starting_player)
        has_char_in_discard = any(c.card_type == "Character" for c in p.discard)

        seen_play_ids: set[int] = set()
        for idx, card in enumerate(p.hand):
            if card.card_id in seen_play_ids:
                continue
            seen_play_ids.add(card.card_id)

            # Toulouse - Rough and Tumble (Set 14): "opponents can't play actions until the start of your next turn"
            if card.card_type == "Action" and p.cant_play_actions_this_turn:
                continue

            is_pure_reanimate = (
                card.card_type == "Action"
                and any(cl.op == EffectOpType.REANIMATE_CHAR for cl in card.abilities.clauses)
            )
            if is_pure_reanimate and not has_char_in_discard:
                continue

            # Standard Ink Play (CR Rule 4.4 + Underdog / Akood et Emuti / Aurora Princess-Queen discounts)
            eff_cost = p.effective_play_cost(card, is_second_player_first_turn=is_p2_first_turn, opp=opp)
            if eff_cost <= payable_ink:
                if card.card_type == "Character" and (
                    bool(card.abilities.keywords & KeywordFlag.BODYGUARD)
                    or "this character may enter play exerted" in card.full_text.lower()
                ):
                    # CR Rule 8.3.2 / Mickey Mouse - Expedition Leader: may enter play exerted instead of ready
                    actions.append(
                        GameAction(
                            action_type=ActionType.PLAY_CARD,
                            hand_index=idx,
                            enter_exerted=True,
                            description=f"Play {card.full_name} exerted ({eff_cost} ink)",
                        )
                    )
                actions.append(
                    GameAction(
                        action_type=ActionType.PLAY_CARD,
                        hand_index=idx,
                        enter_exerted=False,
                        description=f"Play {card.full_name} ({eff_cost} ink)",
                    )
                )

            # Shift Play — All 6 Official Shift Variants (CR Rule 8.10.8)
            shift_cost = card.abilities.shift_cost
            if shift_cost is not None:
                eff_shift_cost = p.effective_shift_cost(card)
            else:
                eff_shift_cost = 999
            if shift_cost is not None and eff_shift_cost <= payable_ink:
                # 3a. Single-target Shift (Standard, Classification, Universal, Combo-1, Temporary, Potato)
                for ent in p.board:
                    if card.can_shift_onto_single(ent.card) and ent.card.full_name != card.full_name:
                        shift_label = (
                            "Temp-Shift"
                            if card.abilities.is_temporary_shift
                            else (
                                "Potato-Shift"
                                if card.abilities.is_potato_shift
                                else (
                                    f"{card.abilities.shift_classification}-Shift"
                                    if card.abilities.shift_classification
                                    else ("Combo-Shift" if card.abilities.is_combo_shift else "Shift")
                                )
                            )
                        )
                        actions.append(
                            GameAction(
                                action_type=ActionType.SHIFT_CARD,
                                hand_index=idx,
                                target_instance_id=ent.instance_id,
                                description=f"{shift_label} {card.full_name} onto {ent.card.full_name} ({eff_shift_cost} ink)",
                            )
                        )

                # 3b. Two-target Shift (Duo Shift 8.10.8.3 & Combo Shift onto 2 characters 8.10.8.4)
                if card.abilities.is_duo_shift or card.abilities.is_combo_shift:
                    chars = p.characters_in_play
                    for i in range(len(chars)):
                        for j in range(i + 1, len(chars)):
                            c1, c2 = chars[i], chars[j]
                            if card.can_shift_onto_pair(c1.card, c2.card):
                                pair_label = "Duo-Shift" if card.abilities.is_duo_shift else "Combo-Shift (2)"
                                actions.append(
                                    GameAction(
                                        action_type=ActionType.SHIFT_CARD,
                                        hand_index=idx,
                                        target_instance_id=c1.instance_id,
                                        secondary_target_id=c2.instance_id,
                                        description=f"{pair_label} {card.full_name} onto {c1.card.base_name} & {c2.card.base_name} ({eff_shift_cost} ink)",
                                    )
                                )

            # Sing Song Action (Single Singer 8.12 or Sing Together Chorus 8.11)
            if card.is_song:
                st_cost = card.abilities.sing_together_cost
                if st_cost is not None and total_chorus_power >= st_cost and len(ready_dry_chars) >= 1:
                    actions.append(
                        GameAction(
                            action_type=ActionType.SING_SONG,
                            hand_index=idx,
                            source_instance_id=-99,  # Sing Together Chorus marker
                            description=f"Sing Together {card.full_name} (cost {st_cost})",
                        )
                    )
                for ent in ready_dry_chars:
                    if ent.can_sing(card.cost):
                        actions.append(
                            GameAction(
                                action_type=ActionType.SING_SONG,
                                hand_index=idx,
                                source_instance_id=ent.instance_id,
                                description=f"{ent.card.full_name} sings {card.full_name}",
                            )
                        )

        # 4. Challenge exerted opposing characters AND opposing locations (CR Rule 4.6, 4.6.8, 8.3.3, 8.7.3)
        opp_has_exerted_diablo = any(
            e.exerted and "your villain characters can't be challenged" in e.card.full_text.lower()
            for e in opp.characters_in_play
        )
        exerted_opp_chars = [
            e for e in opp.characters_in_play
            if e.exerted and not (opp_has_exerted_diablo and "Villain" in e.card.subtypes)
        ]
        bodyguards = [e for e in exerted_opp_chars if e.has_keyword(KeywordFlag.BODYGUARD)]
        opp_locations = opp.locations_in_play
        has_lead_detective = any(
            "your detective characters gain alert and resist +2" in e.card.full_text.lower()
            for e in p.characters_in_play
        )

        has_forced_reckless_challenge = False
        if not p.cant_challenge_this_turn:
            for my_char in p.characters_in_play:
                if not my_char.can_challenge():
                    continue
                my_has_evasive = (
                    my_char.has_keyword(KeywordFlag.EVASIVE)
                    or my_char.has_keyword(KeywordFlag.ALERT)
                    or (has_lead_detective and "Detective" in my_char.card.subtypes)
                )
                valid_targets: list[BoardEntity] = []
                # CR Rule 8.3.3: Bodyguard applies when choosing an opposing character to challenge
                candidate_pool = bodyguards if bodyguards else exerted_opp_chars
                for target in candidate_pool:
                    if target.has_keyword(KeywordFlag.EVASIVE) and not my_has_evasive:
                        continue
                    valid_targets.append(target)
                if not valid_targets and bodyguards:
                    for target in exerted_opp_chars:
                        if target.has_keyword(KeywordFlag.EVASIVE) and not my_has_evasive:
                            continue
                        valid_targets.append(target)

                # CR Rule 4.6.8: Opposing locations are always valid challenge targets
                valid_targets.extend(opp_locations)

                # CR Rule 8.7.3: Reckless prevents declaring end of turn if ready and can challenge an opposing character or location
                if valid_targets and my_char.has_keyword(KeywordFlag.RECKLESS):
                    has_forced_reckless_challenge = True

                for target in valid_targets:
                    actions.append(
                        GameAction(
                            action_type=ActionType.CHALLENGE,
                            source_instance_id=my_char.instance_id,
                            target_instance_id=target.instance_id,
                            description=f"{my_char.card.full_name} challenges {target.card.full_name}",
                        )
                    )

        # 5. Quest with ready dry characters (CR Rule 4.5 & Adventurous 8.16.3)
        has_forced_adventurous_quest = False
        has_4wp_char = any(e.effective_willpower >= 4 for e in p.characters_in_play)
        for my_char in p.characters_in_play:
            if my_char.can_quest():
                if "unless you have a character with 4" in my_char.card.full_text.lower() and not has_4wp_char:
                    continue
                if my_char.has_keyword(KeywordFlag.ADVENTUROUS):
                    # CR Rule 8.16.3: You can't declare the end of your turn if this character is ready and can quest
                    has_forced_adventurous_quest = True
                actions.append(
                    GameAction(
                        action_type=ActionType.QUEST,
                        source_instance_id=my_char.instance_id,
                        description=f"Quest with {my_char.card.full_name} (+{my_char.effective_lore(p, opp)} lore)",
                    )
                )

        # 6. Move a character to a Location (CR Rule 4.7)
        for loc in p.locations_in_play:
            m_cost = loc.card.move_cost
            if m_cost <= payable_ink:
                for my_char in p.characters_in_play:
                    if my_char.location_instance_id != loc.instance_id:
                        actions.append(
                            GameAction(
                                action_type=ActionType.MOVE_TO_LOCATION,
                                source_instance_id=my_char.instance_id,
                                target_instance_id=loc.instance_id,
                                description=f"Move {my_char.card.full_name} to {loc.card.full_name} ({m_cost} ink)",
                            )
                        )

        # 7. Activated abilities on Items/Characters + Boost X mechanic (CR Rule 6.2 & 8.17)
        for ent in p.board:
            b_cost = ent.card.abilities.boost_cost
            if (
                b_cost is not None
                and not ent.boosted_this_turn
                and b_cost <= payable_ink
                and len(p.deck) > 0
            ):
                actions.append(
                    GameAction(
                        action_type=ActionType.USE_ABILITY,
                        source_instance_id=ent.instance_id,
                        target_instance_id=-88,  # Boost marker
                        description=f"Boost {ent.card.full_name} ({b_cost} ink)",
                    )
                )

            for cl in ent.card.abilities.clauses:
                if cl.trigger == TriggerType.ACTIVATED:
                    if cl.requires_exert and (ent.exerted or (ent.card.card_type == "Character" and ent.drying)):
                        continue
                    if cl.cost_ink > payable_ink:
                        continue
                    actions.append(
                        GameAction(
                            action_type=ActionType.USE_ABILITY,
                            source_instance_id=ent.instance_id,
                            description=f"Activate {ent.card.full_name} ({cl.op.value})",
                        )
                    )

        # 8. Pass Turn (CR Rule 4.8 — blocked by Reckless 8.7.3 or Adventurous 8.16.3 if forced action exists)
        if (not has_forced_reckless_challenge and not has_forced_adventurous_quest) or len(actions) == 0:
            actions.append(
                GameAction(
                    action_type=ActionType.PASS_TURN,
                    description="Pass Turn",
                )
            )

        return actions

    @classmethod
    def _generate_mulligan_actions(cls, state: GameState) -> list[GameAction]:
        p = state.players[state.active_player]
        hand = p.hand
        n = len(hand)
        if n == 0:
            return [GameAction(ActionType.MULLIGAN, mulligan_keep_mask=(), description="Mulligan: Empty")]

        masks: dict[tuple[bool, ...], str] = {}

        # Strategy 1: Strict 1-2 Drop Characters Only (deduped, max 1 uninkable) — guarantees Turn 1 & Turn 2 plays!
        seen_early_ids: set[int] = set()
        unink_early = 0
        mask_strict_early: list[bool] = []
        for c in hand:
            if (
                c.card_type == "Character"
                and c.cost <= 2
                and c.card_id not in seen_early_ids
                and (c.inkable or unink_early < 1)
            ):
                seen_early_ids.add(c.card_id)
                if not c.inkable:
                    unink_early += 1
                mask_strict_early.append(True)
            else:
                mask_strict_early.append(False)
        masks[tuple(mask_strict_early)] = "Mulligan: Dig for 1-2 Drop Characters"

        # Strategy 2: Fast Early Game (all 1-2 drop characters + at most 1 cheap removal/3-drop if we already have a 1-2 drop)
        has_any_1_2_char = any(c.card_type == "Character" and c.cost <= 2 for c in hand)
        mask_early = tuple(
            (c.card_type == "Character" and c.cost <= 2)
            or (has_any_1_2_char and c.cost <= 3 and c.inkable)
            for c in hand
        )
        masks[mask_early] = "Mulligan: Fast Early Game (Keep 1-2 drops & curve)"

        # Strategy 3: Balanced Curve Keep (cost <= 3, max 2 uninkables, no duplicate 3-drops)
        unink_cnt = 0
        seen_3_ids: set[int] = set()
        mask_curve: list[bool] = []
        for c in hand:
            if c.cost <= 3 and (c.inkable or unink_cnt < 2) and not (c.cost == 3 and c.card_id in seen_3_ids):
                if not c.inkable:
                    unink_cnt += 1
                if c.cost == 3:
                    seen_3_ids.add(c.card_id)
                mask_curve.append(True)
            else:
                mask_curve.append(False)
        masks[tuple(mask_curve)] = "Mulligan: Balanced Curve (Keep <=3 cost)"

        # Strategy 4: Hard Hunt for Turn-1 Play (keep 1-drops + at most 1 best 2-drop character)
        kept_2_drop = False
        mask_t1: list[bool] = []
        for c in hand:
            if c.card_type == "Character" and c.cost == 1:
                mask_t1.append(True)
            elif c.card_type == "Character" and c.cost == 2 and not kept_2_drop:
                kept_2_drop = True
                mask_t1.append(True)
            else:
                mask_t1.append(False)
        masks[tuple(mask_t1)] = "Mulligan: Hard Hunt for Turn-1 & Turn-2 Curve"

        # Strategy 5: Full 7 Reset (throw back all 7 when hand has no early curve)
        masks[tuple(False for _ in hand)] = "Mulligan: Full 7 Reset (Dig for 1-2 drops)"

        # Strategy 6: Keep All 7
        masks[tuple(True for _ in hand)] = "Mulligan: Keep All 7"

        return [
            GameAction(
                action_type=ActionType.MULLIGAN,
                mulligan_keep_mask=mask,
                description=desc,
            )
            for mask, desc in masks.items()
        ]
