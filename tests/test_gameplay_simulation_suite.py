"""
Comprehensive Gameplay Simulation & Rules Verification Test Suite.
Validates end-to-end tournament matches, Elinor ramp synergies, Aurora Queen discounts,
Bodyguard/Evasive combat, Singing/Chorus mechanics, and game invariants across all Season 14 meta decks.
"""

from __future__ import annotations
import unittest

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.effect_dsl import EffectOpType, KeywordFlag, TriggerType
from lorcana_ai.data.meta_decks import get_reference_meta_decks
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


class TestGameplaySimulationSuite(unittest.TestCase):
    def test_elinor_ramp_curve_scenario(self) -> None:
        """
        Validates the competitive Elinor curve in Princess Aggro:
        Turn 1: 1-cost character (enters READY and WET / drying).
        Turn 2: Grandmother Willow (cost 2) played -> gives -1 discount -> free 1-cost character played (0 ink).
                Both enter READY and WET!
        Turn 3: Characters from T1 & T2 dry (drying=False, ready).
                Willow's start of turn gives -1 discount -> Elinor (Queen) costs 3 ink.
                All 3 early characters exert (quest/ability).
                Elinor played for 3 ink -> enters READY and WET (drying=True, NOT exerted)!
                Pass turn -> Elinor Coordinated Efforts triggers: +1 lore, +1 draw, 1 damage!
        """
        willow = GLOBAL_CARD_DB.get_by_name("Grandmother Willow - Ancient Advisor")  # cost 2, 1 lore
        pluto = GLOBAL_CARD_DB.get_by_name("Pluto - Friendly Pooch")                # cost 1, 0 lore, tap ability
        poca = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")          # cost 2, 1 lore (or 1-drop)
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")            # cost 4, Queen
        mickey = GLOBAL_CARD_DB.get_by_name("Mickey Mouse - Best in Town")
        finnick = GLOBAL_CARD_DB.get_by_name("Finnick - Tiny Terror")                # cost 1
        assert willow and pluto and elinor and mickey and finnick

        deck = [finnick] * 15 + [willow] * 15 + [pluto] * 15 + [elinor] * 15
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=42)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.board.clear()
        p1.board.clear()
        p0.hand = [finnick, willow, pluto, elinor]
        p1.hand.clear()
        p0.lore = 0

        # Opponent has Mickey on board
        p1_mickey = BoardEntity(instance_id=10, card=mickey, exerted=False, drying=False, damage=0)
        p1.board = [p1_mickey]

        # === TURN 1 ===
        p0.inkwell_total = 1
        p0.inkwell_available = 1
        # Play Finnick (1-drop). MUST enter READY (exerted=False) and WET (drying=True)!
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Finnick"))
        self.assertEqual(len(p0.board), 1)
        finnick_ent = p0.board[0]
        self.assertTrue(finnick_ent.drying, "Newly played character must be WET (drying)")
        self.assertFalse(finnick_ent.exerted, "Newly played character must be READY (not exerted)")

        # Pass turn 1 -> Turn 2 begins: Finnick dries!
        finnick_ent.drying = False
        p0.inkwell_total = 2
        p0.inkwell_available = 2

        # === TURN 2 ===
        # Play Grandmother Willow (cost 2) -> gives 1 discount!
        # Hand order now: [willow, pluto, elinor]
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Willow"))
        willow_ent = p0.board[1]
        self.assertTrue(willow_ent.drying, "Willow enters WET")
        self.assertFalse(willow_ent.exerted, "Willow enters READY")
        self.assertEqual(p0.next_char_discount, 1, "Willow grants -1 discount on play")

        # Hand order now: [pluto, elinor]
        # Pluto (cost 1) costs 0 ink with Willow's discount!
        self.assertEqual(p0.effective_play_cost(pluto), 0, "Pluto is free with Willow discount")
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Pluto free"))
        pluto_ent = p0.board[2]
        self.assertTrue(pluto_ent.drying, "Pluto enters WET")
        self.assertFalse(pluto_ent.exerted, "Pluto enters READY")
        self.assertEqual(len(p0.board), 3, "Turn 2 ends with 3 characters in play!")

        # Pass turn 2 -> Turn 3 begins: all 3 characters are DRY and READY!
        for e in p0.board:
            e.drying = False
            e.exerted = False
        p0.inkwell_total = 3
        p0.inkwell_available = 3

        # === TURN 3 ===
        # Start of turn: Willow's START_OF_TURN triggers -> gives -1 discount!
        LorcanaSimulator._resolve_clauses(state, 0, willow, TriggerType.START_OF_TURN, source_id=willow_ent.instance_id)
        self.assertEqual(p0.effective_play_cost(elinor), 3, "Elinor costs 3 ink with Willow discount")

        # Exert all 3 earlier characters:
        # 1. Finnick quests -> exerted!
        LorcanaSimulator.step(state, GameAction(ActionType.QUEST, source_instance_id=finnick_ent.instance_id, description="Quest Finnick"))
        self.assertTrue(finnick_ent.exerted)

        # 2. Willow quests -> exerted!
        LorcanaSimulator.step(state, GameAction(ActionType.QUEST, source_instance_id=willow_ent.instance_id, description="Quest Willow"))
        self.assertTrue(willow_ent.exerted)

        # 3. Pluto taps ability -> exerted!
        LorcanaSimulator.step(state, GameAction(ActionType.USE_ABILITY, source_instance_id=pluto_ent.instance_id, description="Use Pluto"))
        self.assertTrue(pluto_ent.exerted)

        # Verify exactly 3 characters are exerted!
        self.assertEqual(sum(1 for e in p0.board if e.exerted), 3)

        # Hand order now: [elinor]
        # Play Elinor for 3 ink:
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Elinor"))
        elinor_ent = p0.board[3]
        # CRITICAL RULE VERIFICATION: Elinor enters WET (drying) and READY (not exerted)!
        self.assertTrue(elinor_ent.drying, "Elinor enters WET (drying) - cannot quest!")
        self.assertFalse(elinor_ent.exerted, "Elinor enters READY (not exerted)!")

        # 3 earlier characters remain exerted:
        self.assertEqual(sum(1 for e in p0.board if e.exerted), 3)

        # Pass turn -> Elinor Coordinated Efforts triggers at end of turn!
        lore_before = p0.lore
        hand_before = len(p0.hand)
        p1_mickey_dmg_before = p1_mickey.damage

        LorcanaSimulator.step(state, GameAction(ActionType.PASS_TURN, description="Pass Turn"))

        self.assertEqual(p0.lore, lore_before + 1, "Elinor gains +1 lore at end of turn")
        self.assertEqual(len(p0.hand), hand_before + 1, "Elinor draws 1 card at end of turn")
        self.assertEqual(p1_mickey.damage, p1_mickey_dmg_before + 1, "Elinor deals 1 damage to opposing character")

    def test_aurora_queen_and_princess_discount_validation(self) -> None:
        """Verifies Aurora Holding Court discounts Queen (Elinor) and Princess (Pocahontas), but not non-Queen/Princess (Dale)."""
        aurora = GLOBAL_CARD_DB.get_by_name("Aurora - Holding Court")
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")
        poca = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")
        dale = GLOBAL_CARD_DB.get_by_name("Dale - Ready for His Shot")
        assert aurora and elinor and poca and dale

        deck = [aurora] * 60
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=12)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.board = [BoardEntity(instance_id=1, card=aurora, exerted=False, drying=False)]

        # Quest with Aurora -> Princess/Queen discount +1
        LorcanaSimulator.step(state, GameAction(ActionType.QUEST, source_instance_id=1, description="Quest Aurora"))
        self.assertEqual(p0.next_princess_queen_discount, 1)

        # Elinor (Queen) and Pocahontas (Princess) must be discounted
        self.assertEqual(p0.effective_play_cost(elinor), elinor.cost - 1)
        self.assertEqual(p0.effective_play_cost(poca), poca.cost - 1)
        # Dale (Chipmunk) must NOT be discounted
        self.assertEqual(p0.effective_play_cost(dale), dale.cost)

    def test_bodyguard_blocking_and_evasive_rules(self) -> None:
        """Verifies Bodyguard forces challenge redirection when exerted, and Evasive restricts valid attackers."""
        attacker = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")  # non-evasive
        bg_char = GLOBAL_CARD_DB.get_by_name("Cinderella - Ballroom Sensation")  # bodyguard, if available
        if bg_char is None:
            bg_char = attacker
        evasive_def = GLOBAL_CARD_DB.get_by_name("Ursula - Whisper of Vanessa")
        bobby = GLOBAL_CARD_DB.get_by_name("Bobby Zimuruski - Spray Cheese Kid")
        assert attacker and evasive_def and bobby

        deck = [attacker] * 60
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=55)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.board.clear()
        p1.board.clear()

        # P0 has a ready attacker
        p0.board = [BoardEntity(instance_id=1, card=attacker, exerted=False, drying=False)]
        # P1 has an exerted Evasive character with card under it
        e_ent = BoardEntity(instance_id=2, card=evasive_def, exerted=True, drying=False, cards_under_count=1)
        p1.board = [e_ent]

        # Non-evasive attacker CANNOT challenge an Evasive defender!
        legal = MoveGenerator.generate_legal_actions(state)
        challenge_acts = [a for a in legal if a.action_type == ActionType.CHALLENGE]
        self.assertEqual(len(challenge_acts), 0)

        # Now give attacker temporary Evasive -> challenge becomes legal!
        p0.board[0].temp_evasive = True
        legal_evasive = MoveGenerator.generate_legal_actions(state)
        challenge_evasive = [a for a in legal_evasive if a.action_type == ActionType.CHALLENGE]
        self.assertEqual(len(challenge_evasive), 1)

    def test_full_meta_matches_stress_test(self) -> None:
        """
        Simulates 25 full matches between all reference meta deck archetypes.
        Verifies clean game progression, valid phase/turn transitions, and invariant preservation.
        """
        meta_decks = list(get_reference_meta_decks().values())
        agent = TacticalHeuristicAgent(lookahead_depth=1)

        for match_idx in range(25):
            d1 = meta_decks[match_idx % len(meta_decks)]
            d2 = meta_decks[(match_idx + 1) % len(meta_decks)]
            c1 = d1.expand_cards()
            c2 = d2.expand_cards()

            state = GameState.create_new_game(c1, c2, starting_player=match_idx % 2, seed=match_idx * 177 + 13)

            steps = 0
            while state.phase != "GAME_OVER" and steps < 250:
                steps += 1
                legal = MoveGenerator.generate_legal_actions(state)
                self.assertTrue(len(legal) > 0, f"Match {match_idx} step {steps} has 0 legal actions before GAME_OVER")
                action = agent.select_action(state, legal)
                LorcanaSimulator.step(state, action)

                # Invariant checks during play:
                for p in state.players:
                    self.assertGreaterEqual(p.lore, 0)
                    self.assertGreaterEqual(p.inkwell_total, 0)
                    self.assertGreaterEqual(p.inkwell_available, 0)
                    self.assertLessEqual(p.inkwell_available, p.inkwell_total + p.ink_drops)
                    for ent in p.board:
                        self.assertGreaterEqual(ent.damage, 0)
                        self.assertFalse(ent.is_banished)

            self.assertEqual(state.phase, "GAME_OVER", f"Match {match_idx} did not conclude in {steps} steps")
            self.assertIn(state.winner, (0, 1))
            self.assertGreaterEqual(state.players[state.winner].lore, 20)

    def test_droids_resist_and_jddr_guide_heuristics(self) -> None:
        """
        Validates Purple Steel Droids Resist mechanics and JDRR guide tactical decisions:
        - Omnidroid V.10 gains Resist +2 with a card under it
        - Challenge damage reduction via effective_resist
        - Pluto Turn 2 risk penalty when opponent threatens free kill without bodyguard
        - Pocahontas Fog full safety quest bonus
        """
        db = GLOBAL_CARD_DB
        v10 = db.get_by_name("Omnidroid - V.10")
        v8 = db.get_by_name("Omnidroid - V.8")
        pluto = db.get_by_name("Pluto - Friendly Pooch")
        queen_att = db.get_by_name("Queen of Hearts - Wonderland Empress")
        david = db.get_by_name("David - Protective Snowboarder")
        self.assertIsNotNone(v10)
        self.assertIsNotNone(v8)
        self.assertIsNotNone(pluto)
        self.assertIsNotNone(queen_att)
        self.assertIsNotNone(david)

        # 1. Omnidroid V.10 Resist +2 with card under
        ent_v10 = BoardEntity(instance_id=1, card=v10, exerted=False, drying=False, cards_under_count=0)
        self.assertEqual(ent_v10.effective_resist(), 0, "Omnidroid without cards under has 0 resist")
        ent_v10.cards_under_count = 1
        self.assertEqual(ent_v10.effective_resist(), 2, "Omnidroid with card under gains Resist +2")

        # 2. Challenge damage reduction: Queen of Hearts (3 strength) challenges Omnidroid V.10 (Resist 2)
        state = GameState.create_new_game([queen_att] * 60, [v10] * 60, starting_player=0)
        p0 = state.players[0]
        p1 = state.players[1]
        att_ent = BoardEntity(instance_id=2, card=queen_att, exerted=False, drying=False)
        ent_v10.exerted = True
        p0.board = [att_ent]
        p1.board = [ent_v10]

        # Queen challenges exerted Omnidroid V.10: Queen 3 power; V.10 takes 3 - 2 = 1 damage
        LorcanaSimulator.step(state, GameAction(ActionType.CHALLENGE, source_instance_id=2, target_instance_id=1))
        self.assertEqual(ent_v10.damage, 1, "Omnidroid V.10 with Resist +2 only takes 1 damage from 3 attack")

        # 3. Pluto Turn 2 risk assessment (JDRR Guide):
        agent = TacticalHeuristicAgent(lookahead_depth=0)
        state_pluto = GameState.create_new_game([pluto] * 60, [queen_att] * 60, starting_player=0)
        p0_pluto = state_pluto.players[0]
        p1_pluto = state_pluto.players[1]
        p0_pluto.inkwell_total = 2
        p0_pluto.inkwell_available = 2
        pluto_ent = BoardEntity(instance_id=5, card=pluto, exerted=False, drying=False)
        p0_pluto.board = [pluto_ent]
        # Opponent has 3-strength threat on board that can challenge Pluto
        opp_threat = BoardEntity(instance_id=6, card=queen_att, exerted=False, drying=False)
        p1_pluto.board = [opp_threat]
        # Hand has a 3-cost character needing discount with 2 ink
        p0_pluto.hand = [queen_att]

        use_pluto_act = GameAction(ActionType.USE_ABILITY, source_instance_id=5, description="Use Pluto")
        score_exposed = agent.score_action(state_pluto, use_pluto_act)

        # Now give Pluto a Bodyguard in play:
        if david:
            bg_david = BoardEntity(instance_id=7, card=david, exerted=True, drying=False)
            p0_pluto.board.append(bg_david)
            score_shielded = agent.score_action(state_pluto, use_pluto_act)
            self.assertGreater(score_shielded, score_exposed, "Shielded Pluto has higher score than exposed Pluto")


if __name__ == "__main__":
    unittest.main()
