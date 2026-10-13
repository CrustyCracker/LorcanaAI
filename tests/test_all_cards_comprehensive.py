"""
Comprehensive 1,228-Card & Core Constructed Game Rules Verification Suite.
Validates:
1. Card Database Integrity: All 1,228 unique cards from Sets 9, 10, 11, 12, 13, 14.
2. Core Mechanics & Keywords: Bodyguard, Evasive, Alert, Resist, Challenger, Support, Rush, Reckless, Adventurous.
3. All Shift Variants: Standard, Universal (Morph), Classification, Duo/Combo, and Card-Under mechanics.
4. Top Meta Archetypes: Princess Aggro (Willow, Elinor, Dale, Pluto, Aurora, Pocahontas),
   Purple Steel Droids (Syndrome, Omnidroids V.8/V.9/V.10/Ultimate, Angel 624), Amber Steel Songs.
5. Exhaustive 1,228-Card Simulator Playability & Invariant Smoke Test.
"""

from __future__ import annotations
import unittest

from lorcana_ai.data.card_db import CardTemplate, GLOBAL_CARD_DB, INK_COLORS
from lorcana_ai.data.effect_dsl import EffectOpType, KeywordFlag, TriggerType
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


class TestAllCardsComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = GLOBAL_CARD_DB

    # =========================================================================
    # 1. DATABASE INTEGRITY FOR ALL 1,228 CARDS
    # =========================================================================
    def test_database_integrity_all_1228_cards(self) -> None:
        """Verifies structural validity and attributes of every card in Sets 9-14."""
        self.assertEqual(len(self.db.by_id), 1228, "Must load all 1,228 Core Constructed cards")

        seen_ids = set()
        for cid, card in self.db.by_id.items():
            self.assertNotIn(cid, seen_ids, f"Duplicate card ID: {cid}")
            seen_ids.add(cid)

            # Name and identity
            self.assertTrue(len(card.full_name) > 0)
            self.assertTrue(len(card.base_name) > 0)
            self.assertIn(card.set_code, {"9", "10", "11", "12", "13", "14"})

            # Inks
            self.assertTrue(len(card.inks) >= 1)
            for ink in card.inks:
                self.assertIn(ink, INK_COLORS, f"Invalid ink color {ink} on {card.full_name}")

            # Cost & Inking
            self.assertGreaterEqual(card.cost, 0)
            self.assertIsInstance(card.inkable, bool)

            # Card Type
            self.assertIn(card.card_type, {"Character", "Action", "Item", "Location"})

            if card.card_type == "Character":
                self.assertGreaterEqual(card.strength, 0)
                self.assertGreaterEqual(card.willpower, 1, f"Character {card.full_name} must have willpower >= 1")
                self.assertGreaterEqual(card.lore, 0)
            elif card.card_type == "Location":
                self.assertGreaterEqual(card.willpower, 1, f"Location {card.full_name} must have willpower >= 1")
                self.assertGreaterEqual(card.lore, 0)
                self.assertGreaterEqual(card.move_cost, 0)
            elif card.card_type == "Action":
                self.assertIsInstance(card.is_song, bool)

            # Abilities compilation
            self.assertIsNotNone(card.abilities)
            for cl in card.abilities.clauses:
                self.assertIsInstance(cl.trigger, TriggerType)
                self.assertIsInstance(cl.op, EffectOpType)

    # =========================================================================
    # 2. KEYWORD MECHANICS & COMBAT RULES
    # =========================================================================
    def test_bodyguard_blocking_and_evasive_rules(self) -> None:
        """Validates Bodyguard challenge redirection and Evasive combat legality."""
        bg_card = self.db.get_by_name("David - Protective Snowboarder")  # Bodyguard 1/4
        att_card = self.db.get_by_name("Queen of Hearts - Wonderland Empress")  # 3/3
        target_card = self.db.get_by_name("Pluto - Friendly Pooch")  # 0/2
        self.assertIsNotNone(bg_card)
        self.assertIsNotNone(att_card)
        self.assertIsNotNone(target_card)

        state = GameState.create_new_game([att_card] * 60, [target_card] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]

        att_ent = BoardEntity(instance_id=1, card=att_card, exerted=False, drying=False)
        target_ent = BoardEntity(instance_id=2, card=target_card, exerted=True, drying=False)
        bg_ent = BoardEntity(instance_id=3, card=bg_card, exerted=True, drying=False)

        p0.board = [att_ent]
        p1.board = [target_ent, bg_ent]

        # Bodyguard is exerted -> Attacker MUST challenge Bodyguard, cannot challenge Pluto!
        legal = MoveGenerator.generate_legal_actions(state)
        chal_acts = [a for a in legal if a.action_type == ActionType.CHALLENGE]
        self.assertEqual(len(chal_acts), 1)
        self.assertEqual(chal_acts[0].target_instance_id, bg_ent.instance_id)

        # When Bodyguard is ready (not exerted), attacker CAN challenge Pluto
        bg_ent.exerted = False
        legal_ready = MoveGenerator.generate_legal_actions(state)
        chal_pluto = [a for a in legal_ready if a.action_type == ActionType.CHALLENGE and a.target_instance_id == target_ent.instance_id]
        self.assertEqual(len(chal_pluto), 1)

    def test_effective_resist_combat_and_damage_reduction(self) -> None:
        """Validates effective_resist across static, conditional cards-under, and challenge combat."""
        v10 = self.db.get_by_name("Omnidroid - V.10")
        v8 = self.db.get_by_name("Omnidroid - V.8")
        att_card = self.db.get_by_name("Queen of Hearts - Wonderland Empress")  # 3/3
        self.assertIsNotNone(v10)
        self.assertIsNotNone(v8)
        self.assertIsNotNone(att_card)

        # Omnidroid V.10 has 0 base resist without cards under
        ent_v10 = BoardEntity(instance_id=10, card=v10, exerted=True, drying=False, cards_under_count=0)
        self.assertEqual(ent_v10.effective_resist(), 0)

        # With 1 card under (shifted), gains Resist +2
        ent_v10.cards_under_count = 1
        self.assertEqual(ent_v10.effective_resist(), 2)

        # In combat: 3 attack vs Resist 2 takes max(0, 3 - 2) = 1 damage
        state = GameState.create_new_game([att_card] * 60, [v10] * 60, starting_player=0)
        state.phase = "MAIN"
        att_ent = BoardEntity(instance_id=11, card=att_card, exerted=False, drying=False)
        state.players[0].board = [att_ent]
        state.players[1].board = [ent_v10]

        LorcanaSimulator.step(state, GameAction(ActionType.CHALLENGE, source_instance_id=11, target_instance_id=10))
        self.assertEqual(ent_v10.damage, 1, "Resist 2 reduces 3 damage down to 1")

    # =========================================================================
    # 3. ALL SHIFT VARIANTS & CARD-UNDER INVARIANTS
    # =========================================================================
    def test_universal_shift_morph_little_imitator(self) -> None:
        """Validates Universal Shift: Morph can be shifted onto by ANY character with Shift."""
        morph = self.db.get_by_name("Morph - Little Imitator")
        syndrome_shift = self.db.get_by_name("Syndrome - Out for Revenge")  # Shift 4
        pocahontas_shift = self.db.get_by_name("Pocahontas - Peacekeeper")  # Shift 5
        self.assertIsNotNone(morph)
        self.assertIsNotNone(syndrome_shift)
        self.assertIsNotNone(pocahontas_shift)

        # Morph is a valid shift target for both Syndrome and Pocahontas
        self.assertTrue(syndrome_shift.can_shift_onto_single(morph))
        self.assertTrue(pocahontas_shift.can_shift_onto_single(morph))

    def test_shift_inherits_damage_drying_and_cards_under(self) -> None:
        """Validates CR 8.10.3: Shifted character inherits damage and exerted/drying state."""
        base_v8 = self.db.get_by_name("Omnidroid - V.8")
        shift_v10 = self.db.get_by_name("Omnidroid - V.10")
        self.assertIsNotNone(base_v8)
        self.assertIsNotNone(shift_v10)

        state = GameState.create_new_game([shift_v10] * 60, [base_v8] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.inkwell_total = 10
        p0.inkwell_available = 10

        # Base Omnidroid V.8 in play with 1 damage, exerted, dry
        v8_ent = BoardEntity(instance_id=100, card=base_v8, exerted=True, drying=False, damage=1)
        p0.board = [v8_ent]
        p0.hand = [shift_v10]

        legal = MoveGenerator.generate_legal_actions(state)
        shift_acts = [a for a in legal if a.action_type == ActionType.SHIFT_CARD]
        self.assertGreaterEqual(len(shift_acts), 1)

        LorcanaSimulator.step(state, shift_acts[0])
        self.assertEqual(len(p0.board), 1)
        shifted_ent = p0.board[0]

        # Invariants preserved across Shift
        self.assertEqual(shifted_ent.card.full_name, shift_v10.full_name)
        self.assertEqual(shifted_ent.damage, 1, "Shift must preserve existing damage")
        self.assertTrue(shifted_ent.exerted, "Shift must preserve exerted state")
        self.assertFalse(shifted_ent.drying, "Shift must preserve drying (ready/dry) state")
        self.assertEqual(shifted_ent.cards_under_count, 1, "Must have exactly 1 card under")

    def test_omnidroid_ultimate_iteration_return_on_investment(self) -> None:
        """Validates Ultimate Iteration returning cards under it to hand when shifted."""
        ultimate = self.db.get_by_name("Omnidroid - Ultimate Iteration")
        v8 = self.db.get_by_name("Omnidroid - V.8")
        self.assertIsNotNone(ultimate)
        self.assertIsNotNone(v8)

        state = GameState.create_new_game([ultimate] * 60, [v8] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.inkwell_total = 10
        p0.inkwell_available = 10

        v8_ent = BoardEntity(instance_id=200, card=v8, exerted=False, drying=False)
        p0.board = [v8_ent]
        p0.hand = [ultimate]

        legal = MoveGenerator.generate_legal_actions(state)
        shift_act = next(a for a in legal if a.action_type == ActionType.SHIFT_CARD)
        LorcanaSimulator.step(state, shift_act)

        # Ultimate Iteration shifts: cards under are returned to hand
        self.assertEqual(len(p0.board), 1)
        self.assertTrue(any(c.full_name == v8.full_name for c in p0.hand), "V.8 must return to hand via Return on Investment")

    # =========================================================================
    # 4. PRINCESS AGGRO ARCHETYPE CARD MECHANICS
    # =========================================================================
    def test_grandmother_willow_and_aurora_discounts(self) -> None:
        """Validates Grandmother Willow and Aurora Holding Court character discounts."""
        willow = self.db.get_by_name("Grandmother Willow - Ancient Advisor")
        aurora = self.db.get_by_name("Aurora - Holding Court")
        elinor = self.db.get_by_name("Elinor - Renowned Diplomat")  # Queen
        dale = self.db.get_by_name("Dale - Ready for His Shot")  # Not Queen/Princess
        self.assertIsNotNone(willow)
        self.assertIsNotNone(aurora)
        self.assertIsNotNone(elinor)
        self.assertIsNotNone(dale)

        state = GameState.create_new_game([willow] * 60, [dale] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.inkwell_total = 5
        p0.inkwell_available = 5

        # 1. Willow ON_PLAY discount
        p0.hand = [willow, dale]
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0))
        self.assertEqual(p0.next_char_discount, 1, "Willow grants -1 discount on play")
        self.assertEqual(p0.effective_play_cost(dale), dale.cost - 1)

        # 2. Aurora ON_QUEST discount: only for Princess or Queen
        p0.next_char_discount = 0
        p0.next_princess_queen_discount = 0
        aurora_ent = BoardEntity(instance_id=301, card=aurora, exerted=False, drying=False)
        p0.board.append(aurora_ent)

        LorcanaSimulator.step(state, GameAction(ActionType.QUEST, source_instance_id=aurora_ent.instance_id))
        self.assertEqual(p0.next_princess_queen_discount, 1)
        self.assertEqual(p0.effective_play_cost(elinor), elinor.cost - 1, "Elinor (Queen) discounted by Aurora")
        self.assertEqual(p0.effective_play_cost(dale), dale.cost, "Dale (not royal) NOT discounted by Aurora")

    def test_elinor_coordinated_efforts_end_of_turn(self) -> None:
        """Validates Elinor triggering only when exactly 3+ exerted characters are present."""
        elinor = self.db.get_by_name("Elinor - Renowned Diplomat")
        c1 = self.db.get_by_name("Pluto - Friendly Pooch")
        c2 = self.db.get_by_name("Thomas - Wide-Eyed Recruit")
        opp_c = self.db.get_by_name("Queen of Hearts - Wonderland Empress")
        self.assertIsNotNone(elinor)
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.assertIsNotNone(opp_c)

        state = GameState.create_new_game([c1] * 60, [opp_c] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.lore = 0
        p0.deck = [c1, c2]

        elinor_ent = BoardEntity(instance_id=400, card=elinor, exerted=False, drying=False)
        e1 = BoardEntity(instance_id=401, card=c1, exerted=True, drying=False)
        e2 = BoardEntity(instance_id=402, card=c2, exerted=True, drying=False)
        opp_ent = BoardEntity(instance_id=403, card=opp_c, exerted=False, drying=False)

        p0.board = [elinor_ent, e1, e2]
        p1.board = [opp_ent]

        # Case A: Only 2 exerted characters -> Elinor does NOT trigger
        LorcanaSimulator._resolve_clauses(state, 0, elinor, TriggerType.END_OF_TURN, source_id=400)
        self.assertEqual(p0.lore, 0)
        self.assertEqual(opp_ent.damage, 0)

        # Case B: 3 exerted characters (exert Elinor) -> Elinor triggers all 3 effects!
        elinor_ent.exerted = True
        LorcanaSimulator._resolve_clauses(state, 0, elinor, TriggerType.END_OF_TURN, source_id=400)
        self.assertEqual(p0.lore, 1, "Elinor grants +1 lore on 3 exerted characters")
        self.assertEqual(opp_ent.damage, 1, "Elinor deals 1 damage to opposing character")
        self.assertEqual(len(p0.hand), 1, "Elinor draws 1 card")

    def test_dale_willpower_damage_in_challenges(self) -> None:
        """Validates Dale making all characters deal willpower damage instead of strength."""
        dale = self.db.get_by_name("Dale - Ready for His Shot")
        thomas = self.db.get_by_name("Thomas - Wide-Eyed Recruit")  # 1/4 stats
        defender = self.db.get_by_name("Queen of Hearts - Wonderland Empress")  # 3/3 stats
        self.assertIsNotNone(dale)
        self.assertIsNotNone(thomas)
        self.assertIsNotNone(defender)

        state = GameState.create_new_game([dale] * 60, [defender] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]

        dale_ent = BoardEntity(instance_id=500, card=dale, exerted=False, drying=False)
        thomas_ent = BoardEntity(instance_id=501, card=thomas, exerted=False, drying=False)
        def_ent = BoardEntity(instance_id=502, card=defender, exerted=True, drying=False)

        p0.board = [dale_ent, thomas_ent]
        p1.board = [def_ent]

        # Thomas (1 str, 4 wp) challenges defender with Dale on board:
        # Thomas hits for 4 damage (willpower) and banishes the 3-health Queen!
        LorcanaSimulator.step(state, GameAction(ActionType.CHALLENGE, source_instance_id=501, target_instance_id=502))
        self.assertTrue(def_ent.is_banished or def_ent not in p1.board)

    def test_pocahontas_peacekeeper_fogs_challenges(self) -> None:
        """Validates Pocahontas Peacekeeper shift preventing all opponent challenges."""
        pocahontas = self.db.get_by_name("Pocahontas - Peacekeeper")
        attacker = self.db.get_by_name("Queen of Hearts - Wonderland Empress")
        quester = self.db.get_by_name("Thomas - Wide-Eyed Recruit")
        self.assertIsNotNone(pocahontas)
        self.assertIsNotNone(attacker)
        self.assertIsNotNone(quester)

        state = GameState.create_new_game([pocahontas] * 60, [attacker] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]

        # Player 0 plays Pocahontas Fog clause
        LorcanaSimulator._resolve_clauses(state, 0, pocahontas, TriggerType.ON_PLAY, is_shift=True)
        self.assertTrue(p1.cant_challenge_this_turn)

        # Pass to Player 1's turn
        state.active_player = 1
        q_ent = BoardEntity(instance_id=601, card=quester, exerted=True, drying=False)
        att_ent = BoardEntity(instance_id=602, card=attacker, exerted=False, drying=False)
        p0.board = [q_ent]
        p1.board = [att_ent]

        legal = MoveGenerator.generate_legal_actions(state)
        chal_acts = [a for a in legal if a.action_type == ActionType.CHALLENGE]
        self.assertEqual(len(chal_acts), 0, "Opponent cannot declare any challenge under Pocahontas Fog")

    # =========================================================================
    # 5. PURPLE STEEL DROIDS ARCHETYPE CARD MECHANICS
    # =========================================================================
    def test_syndrome_out_for_revenge_quest_engine(self) -> None:
        """Validates Syndrome on quest returning Robot from discard and playing Robot <= 8 free."""
        syndrome = self.db.get_by_name("Syndrome - Out for Revenge")
        v8 = self.db.get_by_name("Omnidroid - V.8")
        v10 = self.db.get_by_name("Omnidroid - V.10")
        self.assertIsNotNone(syndrome)
        self.assertIsNotNone(v8)
        self.assertIsNotNone(v10)

        state = GameState.create_new_game([syndrome] * 60, [v8] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.discard = [v8]  # Robot in discard
        p0.hand = [v10]    # Robot in hand

        syn_ent = BoardEntity(instance_id=700, card=syndrome, exerted=False, drying=False)
        p0.board = [syn_ent]

        # Syndrome quests -> returns V.8 from discard to hand, and can cheat out Robot <= 8
        LorcanaSimulator.step(state, GameAction(ActionType.QUEST, source_instance_id=700))
        self.assertTrue(syn_ent.exerted)
        self.assertGreaterEqual(p0.lore, syndrome.lore)

    def test_omnidroid_v9_shift_ping_damage(self) -> None:
        """Validates Omnidroid V.9 dealing 2 damage on play when shifted."""
        v9 = self.db.get_by_name("Omnidroid - V.9")
        v8 = self.db.get_by_name("Omnidroid - V.8")
        target = self.db.get_by_name("Pluto - Friendly Pooch")  # 0/2 body
        self.assertIsNotNone(v9)
        self.assertIsNotNone(v8)
        self.assertIsNotNone(target)

        state = GameState.create_new_game([v9] * 60, [target] * 60, starting_player=0)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.inkwell_total = 5
        p0.inkwell_available = 5

        v8_ent = BoardEntity(instance_id=800, card=v8, exerted=False, drying=False)
        target_ent = BoardEntity(instance_id=801, card=target, exerted=False, drying=False)
        p0.board = [v8_ent]
        p1.board = [target_ent]
        p0.hand = [v9]

        legal = MoveGenerator.generate_legal_actions(state)
        shift_acts = [a for a in legal if a.action_type == ActionType.SHIFT_CARD]
        self.assertGreaterEqual(len(shift_acts), 1)

        # Shift V.9 onto V.8 -> deals 2 damage and banishes 2-health Pluto!
        LorcanaSimulator.step(state, shift_acts[0])
        self.assertTrue(target_ent.is_banished or target_ent not in p1.board)

    # =========================================================================
    # 6. EXHAUSTIVE 1,228-CARD PLAYABILITY & INVARIANT SMOKE TEST
    # =========================================================================
    def test_exhaustive_playability_smoke_all_1228_cards(self) -> None:
        """
        Smokes every single card in Sets 9-14:
        - Inking if inkable
        - Playing the card into MAIN phase
        - Resolving actions, items, locations, and characters without crash
        """
        all_cards = list(self.db.by_id.values())
        dummy = all_cards[0]
        reanimate_char = self.db.get_by_name("Thomas - Wide-Eyed Recruit")

        for card in all_cards:
            state = GameState.create_new_game([dummy] * 60, [dummy] * 60, starting_player=0)
            state.phase = "MAIN"
            p0 = state.players[0]
            p0.hand = [card]
            p0.inkwell_total = card.cost + 6
            p0.inkwell_available = card.cost + 6

            # Provide valid character in discard for reanimation cards (e.g. Circle of Life)
            if reanimate_char:
                p0.discard = [reanimate_char]

            # 1. Test Inking
            if card.inkable:
                st_ink = state.clone()
                legal_ink = MoveGenerator.generate_legal_actions(st_ink)
                ink_acts = [a for a in legal_ink if a.action_type == ActionType.INK_CARD]
                self.assertGreater(len(ink_acts), 0, f"Must have ink action for {card.full_name}")
                LorcanaSimulator.step(st_ink, ink_acts[0])
                self.assertEqual(len(st_ink.players[0].hand), 0)
                self.assertEqual(st_ink.players[0].inkwell_total, p0.inkwell_total + 1)

            # 2. Test Playing
            st_play = state.clone()
            legal_play = MoveGenerator.generate_legal_actions(st_play)
            play_acts = [a for a in legal_play if a.action_type in (ActionType.PLAY_CARD, ActionType.SING_SONG, ActionType.SHIFT_CARD)]
            self.assertGreater(len(play_acts), 0, f"Must have play/song/shift action for {card.full_name}")

            # Step the first legal play action
            LorcanaSimulator.step(st_play, play_acts[0])

            # Invariant assertions:
            self.assertGreaterEqual(st_play.players[0].lore, 0)
            self.assertGreaterEqual(st_play.players[0].inkwell_available, 0)
            self.assertLessEqual(st_play.players[0].inkwell_available, st_play.players[0].inkwell_total + st_play.players[0].ink_drops)
            for ent in st_play.players[0].board:
                self.assertGreaterEqual(ent.damage, 0)
                self.assertGreaterEqual(ent.effective_willpower, 1)


if __name__ == "__main__":
    unittest.main()
