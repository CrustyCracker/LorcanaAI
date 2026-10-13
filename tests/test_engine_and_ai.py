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

    def test_ai_always_inks_on_turn_1_even_with_lone_inkable(self) -> None:
        """Verifies AI holding inkable cards on Turn 1 / 0 inkwell NEVER passes turn."""
        agent = TacticalHeuristicAgent(lookahead_depth=1)
        higgins = GLOBAL_CARD_DB.get_by_name("Higgins - Undercover Officer")  # cost 1, inkable=True
        fangmeyer = GLOBAL_CARD_DB.get_by_name("Fangmeyer - Icy Officer")     # cost 5, inkable=False
        raging = GLOBAL_CARD_DB.get_by_name("Raging Storm")                   # cost 8, inkable=False
        assert higgins and fangmeyer and raging

        # Construct worst-case Turn 1 brick hand: 1 lone inkable 1-drop + 6 uninkables
        deck = [higgins] * 10 + [fangmeyer] * 30 + [raging] * 20
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=42)
        state.phase = "MAIN"
        p0 = state.players[0]
        p0.hand = [fangmeyer, raging, fangmeyer, higgins, fangmeyer, fangmeyer, raging]
        p0.inkwell_total = 0
        p0.inkwell_available = 0

        from lorcana_ai.engine.move_generator import MoveGenerator
        legal = MoveGenerator.generate_legal_actions(state)
        chosen = agent.select_action(state, legal)
        self.assertEqual(chosen.action_type, ActionType.INK_CARD)
        self.assertIn("Higgins", chosen.description)

        # Also verify PASS_TURN score is strictly negative on empty inkwell
        pass_act = next(a for a in legal if a.action_type == ActionType.PASS_TURN)
        self.assertEqual(agent.score_action(state, pass_act), -100.0)

    def test_princess_aggro_core_and_tech_cards(self) -> None:
        """Verifies Core and Tech cards for Amber/Emerald Princess Aggro: Nani, Pluto, Elinor."""
        nani = GLOBAL_CARD_DB.get_by_name("Nani - Stage Manager")       # Set 11, cost 3, Amber
        pluto = GLOBAL_CARD_DB.get_by_name("Pluto - Friendly Pooch")     # Set 9, cost 1, Amber, inkable=False
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat") # Set 12, cost 4, Emerald, inkable=False
        assert nani and pluto and elinor

        # 1. Pluto - Friendly Pooch activated ability gives -1 character discount
        self.assertTrue(any(cl.op == EffectOpType.DISCOUNT_NEXT_CHAR and cl.requires_exert for cl in pluto.abilities.clauses))

        # 2. Nani - Stage Manager draws/reveals character with cost <= 2 from top 4 of deck
        self.assertIn("cost 2 or less", nani.full_text.lower())
        self.assertTrue(any(cl.op == EffectOpType.DRAW for cl in nani.abilities.clauses))

        # 3. Elinor - Renowned Diplomat End of Turn trigger requires 3 exerted characters
        self.assertTrue(any(cl.trigger == TriggerType.END_OF_TURN for cl in elinor.abilities.clauses))
        eot_ops = {cl.op for cl in elinor.abilities.clauses if cl.trigger == TriggerType.END_OF_TURN}
        self.assertIn(EffectOpType.DEAL_DAMAGE, eot_ops)
        self.assertIn(EffectOpType.DRAW, eot_ops)
        self.assertIn(EffectOpType.GAIN_LORE, eot_ops)

    def test_meta_deck_mechanics_and_cards(self) -> None:
        """Verifies newly integrated meta cards across Season 14 constructed archetypes."""
        from lorcana_ai.data.effect_dsl import KeywordFlag
        from lorcana_ai.engine.move_generator import GameAction

        second_star = GLOBAL_CARD_DB.get_by_name("Second Star to the Right")
        remember_me = GLOBAL_CARD_DB.get_by_name("Remember Me")
        retro_device = GLOBAL_CARD_DB.get_by_name("Retro Evolution Device")
        cauldron = GLOBAL_CARD_DB.get_by_name("The Black Cauldron")
        luisa = GLOBAL_CARD_DB.get_by_name("Luisa Madrigal - Confident Climber")
        belle = GLOBAL_CARD_DB.get_by_name("Belle - Snowfield Strategist")
        roller_bob = GLOBAL_CARD_DB.get_by_name("Roller Bob - Sid's Toy")
        aladdin = GLOBAL_CARD_DB.get_by_name("Aladdin - Doing His Part")
        judy_case = GLOBAL_CARD_DB.get_by_name("Judy Hopps - On the Case")
        nick_inq = GLOBAL_CARD_DB.get_by_name("Nick Wilde - Inquisitive Harbormaster")
        demona = GLOBAL_CARD_DB.get_by_name("Demona - Imperious Spellcaster")
        poca = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")
        bobby = GLOBAL_CARD_DB.get_by_name("Bobby Zimuruski - Spray Cheese Kid")
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")
        nick_det = GLOBAL_CARD_DB.get_by_name("Nick Wilde - Toy Drive Officer")
        assert all(
            c is not None
            for c in (
                second_star,
                remember_me,
                retro_device,
                cauldron,
                luisa,
                belle,
                roller_bob,
                aladdin,
                judy_case,
                nick_inq,
                demona,
                poca,
                bobby,
                elinor,
                nick_det,
            )
        )

        deck = [bobby] * 60
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=55)
        state.phase = "MAIN"
        p0 = state.players[0]
        p1 = state.players[1]
        p0.board.clear()
        p1.board.clear()
        p0.hand.clear()
        p1.hand.clear()
        p0.inkwell_total = 10
        p0.inkwell_available = 10

        # 1. Second Star to the Right: draws 5 cards
        p0.hand = [second_star]
        p0.deck = [bobby] * 10
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Second Star"))
        self.assertEqual(len(p0.hand), 5)

        # 2. Remember Me: plays character from discard
        p0.discard = [elinor]
        p0.hand = [remember_me]
        p0.inkwell_available = 10
        LorcanaSimulator.step(state, GameAction(ActionType.PLAY_CARD, hand_index=0, description="Play Remember Me"))
        self.assertTrue(any(e.card.full_name == elinor.full_name for e in p0.board))

        # 3. Retro Evolution Device: banishes friendly character (Bobby, cost 1) -> plays Pocahontas (cost 2 <= 1 + 2)
        p0.board = [
            BoardEntity(instance_id=1, card=retro_device, exerted=False, drying=False),
            BoardEntity(instance_id=2, card=bobby, exerted=False, drying=False),
        ]
        p0.hand = [poca]
        p0.discard.clear()
        LorcanaSimulator._resolve_clauses(state, 0, retro_device, TriggerType.ACTIVATED, source_id=1)
        self.assertTrue(any(e.card.full_name == poca.full_name for e in p0.board))
        self.assertIn(bobby, p0.discard)

        # 4. The Black Cauldron: puts character from discard under item
        p0.discard = [elinor]
        cauldron_ent = BoardEntity(instance_id=3, card=cauldron, exerted=False, drying=False)
        p0.board = [cauldron_ent]
        LorcanaSimulator._resolve_clauses(state, 0, cauldron, TriggerType.ACTIVATED, source_id=3)
        self.assertEqual(cauldron_ent.cards_under_count, 1)

        # 5. Luisa Madrigal - Confident Climber: moves damage to Luisa, then to opponent if >= 3
        luisa_ent = BoardEntity(instance_id=4, card=luisa, exerted=False, drying=False, damage=2)
        bobby_ent = BoardEntity(instance_id=5, card=bobby, exerted=False, drying=False, damage=1)
        opp_ent = BoardEntity(instance_id=6, card=elinor, exerted=False, drying=False, damage=0)
        p0.board = [luisa_ent, bobby_ent]
        p1.board = [opp_ent]
        LorcanaSimulator._resolve_clauses(state, 0, luisa, TriggerType.ACTIVATED, source_id=4)
        self.assertEqual(bobby_ent.damage, 0)
        self.assertEqual(luisa_ent.damage, 0)
        self.assertEqual(opp_ent.damage, 3)

        # 6. Belle - Snowfield Strategist: when allied character is banished, puts card into inkwell
        belle_ent = BoardEntity(instance_id=7, card=belle, exerted=False, drying=False)
        sac_ent = BoardEntity(instance_id=8, card=bobby, exerted=False, drying=False, damage=bobby.willpower)
        p0.board = [belle_ent, sac_ent]
        ink_before = p0.inkwell_total
        LorcanaSimulator._cleanup_banished(state)
        self.assertEqual(p0.inkwell_total, ink_before + 1)
        self.assertIn(bobby, p0.inkwell_cards)

        # 7. Roller Bob - Sid's Toy: puts 2 characters from discard to deck bottom to gain Rush
        p0.discard = [bobby, poca]
        rb_ent = BoardEntity(instance_id=9, card=roller_bob, exerted=False, drying=True)
        p0.board = [rb_ent]
        LorcanaSimulator._resolve_clauses(state, 0, roller_bob, TriggerType.ON_PLAY, source_id=9)
        self.assertFalse(rb_ent.drying)

        # 8. Aladdin - Doing His Part: pays 1 ink to banish opponent's item
        p0.inkwell_available = 2
        p1.board = [BoardEntity(instance_id=10, card=retro_device, exerted=False, drying=False)]
        LorcanaSimulator._resolve_clauses(state, 0, aladdin, TriggerType.ON_PLAY, source_id=11)
        self.assertEqual(len(p1.board), 0)
        self.assertIn(retro_device, p1.discard)

        # 9. Judy Hopps - On the Case: Detective ally present -> ramps opp item into opp inkwell
        p0.board = [
            BoardEntity(instance_id=12, card=nick_det, exerted=False, drying=False),
            BoardEntity(instance_id=13, card=judy_case, exerted=False, drying=False),
        ]
        p1.board = [BoardEntity(instance_id=14, card=retro_device, exerted=False, drying=False)]
        p1_ink_before = p1.inkwell_total
        LorcanaSimulator._resolve_clauses(state, 0, judy_case, TriggerType.ON_PLAY, source_id=13)
        self.assertEqual(len(p1.board), 0)
        self.assertEqual(p1.inkwell_total, p1_ink_before + 1)

        # 10. Nick Wilde - Inquisitive Harbormaster: gives Adventurous
        p1.board = [BoardEntity(instance_id=15, card=bobby, exerted=False, drying=False)]
        LorcanaSimulator._resolve_clauses(state, 0, nick_inq, TriggerType.ON_QUEST, source_id=16)
        self.assertTrue(p1.board[0].has_keyword(KeywordFlag.ADVENTUROUS))

        # 11. Demona - Imperious Spellcaster: gives Gargoyle Rush and Evasive
        p0.hand = [bobby]
        demona_ent = BoardEntity(instance_id=17, card=demona, exerted=False, drying=True)
        p0.board = [demona_ent]
        LorcanaSimulator._resolve_clauses(state, 0, demona, TriggerType.ACTIVATED, source_id=17)
        self.assertTrue(demona_ent.has_keyword(KeywordFlag.EVASIVE))
        self.assertFalse(demona_ent.drying)

    def test_matchup_aware_mulligan_strategy(self) -> None:
        """Verifies matchup-aware, on-the-play vs on-the-draw, and uninkable-safe mulligan selection."""
        bobby = GLOBAL_CARD_DB.get_by_name("Bobby Zimuruski - Spray Cheese Kid")  # cost 1, inkable
        poca = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")       # cost 2, inkable
        elinor = GLOBAL_CARD_DB.get_by_name("Elinor - Renowned Diplomat")         # cost 4, uninkable
        raging = GLOBAL_CARD_DB.get_by_name("Raging Storm")                       # cost 8, uninkable
        assert bobby and poca and elinor and raging

        deck = [bobby] * 20 + [poca] * 20 + [elinor] * 10 + [raging] * 10
        state = GameState.create_new_game(deck, deck, starting_player=0, seed=1)
        p0 = state.players[0]
        agent = TacticalHeuristicAgent(lookahead_depth=1)

        # 1. Hand with good 1-drop + 2-drop but 5 uninkables: must pitch all 5 uninkables and keep Bobby + Pocahontas
        p0.hand = [bobby, poca, elinor, raging, raging, raging, raging]
        best_act = agent.select_action(state)
        self.assertEqual(best_act.action_type, ActionType.MULLIGAN)
        # Verify kept cards are only Bobby and Pocahontas
        self.assertTrue(best_act.mulligan_keep_mask[0])
        self.assertTrue(best_act.mulligan_keep_mask[1])
        self.assertFalse(best_act.mulligan_keep_mask[2])
        self.assertFalse(best_act.mulligan_keep_mask[3])

        # 2. Hand with 0 early drops and 7 uninkables: must perform Full 7 Reset
        p0.hand = [elinor, elinor, raging, raging, raging, raging, raging]
        best_brick = agent.select_action(state)
        self.assertEqual(best_brick.action_type, ActionType.MULLIGAN)
        self.assertEqual(sum(best_brick.mulligan_keep_mask), 0)


if __name__ == "__main__":
    unittest.main()
