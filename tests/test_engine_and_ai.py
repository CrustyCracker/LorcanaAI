"""
Unit tests verifying Lorcana Simulator rules, EffectDSLCompiler accuracy,
TacticalHeuristicAgent lookahead & anti-blunder logic, and Game Replay generation.
"""

from __future__ import annotations
import unittest

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.effect_dsl import EffectOpType, TriggerType
from lorcana_ai.data.meta_decks import get_reference_meta_decks
from lorcana_ai.engine.game_replay import simulate_and_format_replay
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.move_generator import ActionType
from lorcana_ai.engine.simulator import LorcanaSimulator


class TestEngineAndAI(unittest.TestCase):
    def test_dsl_compiler_fixes(self) -> None:
        # 1. Nick Wilde - Toy Drive Officer: Support reminder text in parens must not turn END_OF_TURN draw into ON_QUEST
        nick = GLOBAL_CARD_DB.get_by_name("Nick Wilde - Toy Drive Officer")
        self.assertIsNotNone(nick)
        assert nick is not None
        triggers = [cl.trigger for cl in nick.abilities.clauses if cl.op == EffectOpType.DRAW]
        self.assertIn(TriggerType.END_OF_TURN, triggers)
        self.assertNotIn(TriggerType.ON_QUEST, triggers)

        # 2. Judy Hopps - Always Vigilant: BANISH_CHAR must have threshold == -5 (5 strength or more)
        judy_av = GLOBAL_CARD_DB.get_by_name("Judy Hopps - Always Vigilant")
        self.assertIsNotNone(judy_av)
        assert judy_av is not None
        banish_clauses = [cl for cl in judy_av.abilities.clauses if cl.op == EffectOpType.BANISH_CHAR]
        self.assertEqual(len(banish_clauses), 1)
        self.assertEqual(banish_clauses[0].threshold, -5)

        # 3. Lionheart - Incumbent Mayor & Pocahontas - Finding the Way: BUFF_ALLY_LORE instead of unconditional GAIN_LORE
        lionheart = GLOBAL_CARD_DB.get_by_name("Lionheart - Incumbent Mayor")
        self.assertIsNotNone(lionheart)
        assert lionheart is not None
        self.assertTrue(any(cl.op == EffectOpType.BUFF_ALLY_LORE and cl.amount == 2 for cl in lionheart.abilities.clauses))

        poca_ftw = GLOBAL_CARD_DB.get_by_name("Pocahontas - Finding the Way")
        self.assertIsNotNone(poca_ftw)
        assert poca_ftw is not None
        self.assertTrue(any(cl.op == EffectOpType.BUFF_ALLY_LORE and cl.amount == 1 for cl in poca_ftw.abilities.clauses))

    def test_judy_hopps_always_vigilant_requires_another_char_and_5_str(self) -> None:
        judy_av = GLOBAL_CARD_DB.get_by_name("Judy Hopps - Always Vigilant")
        poca_2 = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")
        self.assertIsNotNone(judy_av)
        self.assertIsNotNone(poca_2)
        assert judy_av is not None and poca_2 is not None

        deck = [judy_av] * 30 + [poca_2] * 30
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=42)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.board.clear()
        p1.board.clear()

        # Put a 2-strength Pocahontas on P1's board
        p1.board.append(BoardEntity(instance_id=10, card=poca_2, exerted=False, drying=False))
        # Even if P0 already played another character this turn, 2-strength Pocahontas must NOT be banished
        p0.chars_played_this_turn = 2
        p0.board.append(BoardEntity(instance_id=11, card=judy_av, exerted=False, drying=True))
        LorcanaSimulator._resolve_clauses(state, 0, judy_av, TriggerType.ON_PLAY, source_id=11)
        self.assertEqual(len(p1.board), 1)

    def test_ai_never_inks_last_card_and_prioritizes_lethal(self) -> None:
        aurora = GLOBAL_CARD_DB.get_by_name("Aurora - Holding Court")
        assert aurora is not None
        deck = [aurora] * 60
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=7)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.hand = [aurora]
        p0.inkwell_total = 3
        p0.inkwell_available = 3

        agent = TacticalHeuristicAgent(lookahead_depth=1)
        chosen = agent.select_action(state)
        self.assertNotEqual(chosen.action_type, ActionType.INK_CARD)
        self.assertEqual(chosen.action_type, ActionType.PLAY_CARD)

    def test_ai_anti_lethal_challenges_instead_of_questing(self) -> None:
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")
        mickey = GLOBAL_CARD_DB.get_by_name("Mickey Mouse - Best in Town")
        assert elinor is not None and mickey is not None
        deck = [elinor] * 30 + [mickey] * 30
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=99)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.hand.clear()
        p1.hand.clear()
        p0.lore = 10
        p1.lore = 19  # Opponent is 1 lore away from winning next turn!

        # P0 has a ready 3/4 Elinor; P1 has an exerted 0/4 Mickey with 1 damage
        p0.board = [BoardEntity(instance_id=1, card=elinor, exerted=False, drying=False)]
        p1.board = [BoardEntity(instance_id=2, card=mickey, exerted=True, drying=False, damage=1)]

        agent = TacticalHeuristicAgent(lookahead_depth=1)
        chosen = agent.select_action(state)
        self.assertEqual(chosen.action_type, ActionType.CHALLENGE)

    def test_aurora_holding_court_discounts_only_princess_or_queen(self) -> None:
        aurora = GLOBAL_CARD_DB.get_by_name("Aurora - Holding Court")
        bobby = GLOBAL_CARD_DB.get_by_name("Bobby Zimuruski - Spray Cheese Kid")
        dale = GLOBAL_CARD_DB.get_by_name("Dale - Ready for His Shot")
        poca = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")
        assert all(c is not None for c in (aurora, bobby, dale, poca, elinor))
        assert aurora and bobby and dale and poca and elinor

        # 1. Verify DSL compilation uses DISCOUNT_NEXT_PRINCESS_QUEEN
        q_clauses = [cl for cl in aurora.abilities.clauses if cl.trigger == TriggerType.ON_QUEST]
        self.assertEqual(len(q_clauses), 1)
        self.assertEqual(q_clauses[0].op, EffectOpType.DISCOUNT_NEXT_PRINCESS_QUEEN)

        # 2. Verify Simulator & PlayerState discount behavior
        deck = [aurora] * 60
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=11)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.board = [BoardEntity(instance_id=1, card=aurora, exerted=False, drying=False)]
        p0.hand = [bobby, poca, elinor]
        p0.inkwell_total = 4
        p0.inkwell_available = 4

        # Quest with Aurora -> +1 next_princess_queen_discount
        from lorcana_ai.engine.move_generator import GameAction
        LorcanaSimulator.step(state, GameAction(ActionType.QUEST, source_instance_id=1, description="Quest Aurora"))
        self.assertEqual(p0.next_princess_queen_discount, 1)
        self.assertEqual(p0.next_char_discount, 0)

        # Dale (Chipmunk, printed cost 4) and Bobby (printed cost 1) must NOT get Aurora's discount!
        self.assertEqual(p0.effective_play_cost(dale), 4)
        self.assertEqual(p0.effective_play_cost(bobby), 1)
        # Pocahontas (Princess, cost 2) and Elinor (Queen, cost 4) MUST get Aurora's -1 discount!
        self.assertEqual(p0.effective_play_cost(poca), 1)
        self.assertEqual(p0.effective_play_cost(elinor), 3)

        # Playing Bobby (non-Princess/non-Queen, cost 1) costs 1 ink and does NOT consume Aurora's Princess/Queen discount!
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Bobby"))
        self.assertEqual(p0.inkwell_available, 3)
        self.assertEqual(p0.next_princess_queen_discount, 1)

        # Now playing Pocahontas (Princess, cost 2 -> 1) consumes Aurora's discount and leaves 2 ink available!
        p0.hand = [poca, elinor]
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Pocahontas"))
        self.assertEqual(p0.inkwell_available, 2)
        self.assertEqual(p0.next_princess_queen_discount, 0)
        # Elinor now costs full 4 ink again because the discount was consumed by Pocahontas
        self.assertEqual(p0.effective_play_cost(elinor), 4)

    def test_ursula_deceiver_reveals_hand_and_mowgli_non_char_filter(self) -> None:
        ursula = GLOBAL_CARD_DB.get_by_name("Ursula - Deceiver")
        mowgli = GLOBAL_CARD_DB.get_by_name("Mowgli - Man Cub")
        col = GLOBAL_CARD_DB.get_by_name("Circle of Life")
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")
        poca_pk = GLOBAL_CARD_DB.get_by_name("Pocahontas - Peacekeeper")
        assert ursula and mowgli and col and elinor and poca_pk

        deck = [ursula] * 30 + [elinor] * 30
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=22)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.inkwell_total = 4
        p0.inkwell_available = 4
        p0.hand = [ursula, mowgli]
        p1.hand = [col, elinor, poca_pk]
        p1.revealed_hand_ids.clear()

        from lorcana_ai.engine.move_generator import GameAction
        # 1. Play Ursula - Deceiver -> reveals all 3 cards in P1's hand and discards Circle of Life (Song), leaving Elinor & Peacekeeper revealed!
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Ursula"))
        self.assertEqual(len(p1.hand), 2)
        self.assertIn(col, p1.discard)
        self.assertEqual(set(p1.revealed_hand_ids), {elinor.card_id, poca_pk.card_id})

        # 2. Play Mowgli - Man Cub ("discards a non-character card") against P1's remaining hand of 2 Characters -> must discard 0 cards!
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Mowgli"))
        self.assertEqual(len(p1.hand), 2)

    def test_core_constructed_sets_9_to_14_mechanics(self) -> None:
        vixey = GLOBAL_CARD_DB.get_by_name("Vixey - Forest Friend")
        tod = GLOBAL_CARD_DB.get_by_name("Tod - Knows All the Tricks")
        olaf = GLOBAL_CARD_DB.get_by_name("Olaf - Snowman of Action")
        toulouse = GLOBAL_CARD_DB.get_by_name("Toulouse - Rough and Tumble")
        strike = GLOBAL_CARD_DB.get_by_name("Strike a Good Match")
        assert vixey and tod and olaf and toulouse and strike

        deck = [vixey] * 30 + [toulouse] * 30
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=33)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]

        # Vixey (Set 12, printed cost 3) costs 1 less when Tod is in play (3 -> 2)
        self.assertEqual(p0.effective_play_cost(vixey, opp=p1), 3)
        p0.board = [BoardEntity(instance_id=50, card=tod, exerted=False, drying=False, damage=1)]
        self.assertEqual(p0.effective_play_cost(vixey, opp=p1), 2)

        # Olaf - Snowman of Action (Set 11) costs 1 less for each Action card in your discard
        p0.discard = [strike]
        self.assertEqual(p0.effective_play_cost(olaf, opp=p1), olaf.cost - 1)

        # Toulouse (Set 14) prevents opponent from playing Actions on their next turn
        p0.inkwell_total = 5
        p0.inkwell_available = 5
        p0.hand = [toulouse]
        p1.hand = [strike]
        p1.inkwell_total = 4
        p1.inkwell_available = 4
        from lorcana_ai.engine.move_generator import GameAction, MoveGenerator
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Toulouse"))
        LorcanaSimulator.step(state, GameAction(ActionType.PASS_TURN, description="Pass"))
        # Now it is P1's turn and P1 has `cant_play_actions_this_turn == True`
        self.assertTrue(p1.cant_play_actions_this_turn)
        p1_legal = MoveGenerator.generate_legal_actions(state)
        self.assertFalse(any("Play Strike a Good Match" in a.description for a in p1_legal))

    def test_replay_generator_produces_valid_transcript_with_revealed_hand(self) -> None:
        decks = get_reference_meta_decks()
        d1 = decks["Amber/Emerald Princess Aggro"]
        d2 = next(v for k, v in decks.items() if "Circle of Life" in k or "Detectives" in k)
        txt = simulate_and_format_replay(d1, d2, starting_player=0, seed=105, lookahead_depth=1)
        self.assertIn("PEŁNY ZAPIS ROZGRYWKI", txt)
        self.assertIn("FAZA MULLIGANU", txt)
        self.assertIn("KONIEC GRY", txt)
        self.assertIn("TURA GRY  1 (Runda 1 — Tura 1 gracza", txt)


if __name__ == "__main__":
    unittest.main()
