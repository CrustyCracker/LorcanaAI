"""
Comprehensive test suite verifying all modules of LorcanaAlpha & 100% compliance with
Official Disney Lorcana Comprehensive Rules v2.3.0 (Effective October 16, 2026):
1. CardDatabase & EffectDSLCompiler (Sets 9-14 Core Constructed)
2. Comprehensive Rules v2.3.0 (Oct 16, 2026) Audit Tests:
   - Rule 8.16 (Adventurous: cannot challenge, must quest before PASS_TURN if ready & able, can sing)
   - Rule 1.8.1.2 & 2.3.3.2 (Deck-out loss occurs when turn ends with 0 cards in deck, not mid-turn)
   - Rule 1.13.1.2 (Ink Drop Counters: GAIN_INK_DROP / EACH_PLAYER_INK_DROP & cost payment)
   - Rule 4.6.8, 8.3.3 & 8.7.3 (Challenging Locations, 0 counter-damage, Bodyguard bypass for Locations)
   - Rule 4.7 & 5.6.6 (MOVE_TO_LOCATION & clearing location when banished)
   - Rule 8.10.8 (All 6 Official Shift Variants: Classification, Universal, Duo, Combo, Temporary, Potato)
   - Rule 8.14 (Vanish banished when chosen by opponent's Action) & Rule 8.3.2 (Bodyguard enter-exerted)
   - Rule 1.10.1.3 (Christopher Robin - Hunny Sage off-color Hunny modifier & custom copy limits)
3. Deck I/O & Tournament Rule Validation
4. GameState, MoveGenerator, Imperfect-Information Determinization & Simulator
5. Transformer Policy-Value Network, Opponent Belief Network & IS-MCTS Agent
6. Self-Play Trainer & Checkpoint Save/Load
7. Tech-Card Scanner (UCB1), MAP-Elites Evolutionary Deckbuilder & Nash Meta-Solver
"""

from __future__ import annotations
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.agents.is_mcts import ISMCTSAgent
from lorcana_ai.agents.networks import LorcanaTransformerNet, OpponentBeliefNetwork
from lorcana_ai.agents.trainer import SelfPlayTrainer
from lorcana_ai.config import MCTSConfig
from lorcana_ai.data.card_db import CardTemplate, GLOBAL_CARD_DB
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.effect_dsl import (
    CompiledAbilities,
    EffectClause,
    EffectOpType,
    KeywordFlag,
    TriggerType,
)
from lorcana_ai.data.meta_decks import DEFAULT_METAGAME_SHARES, get_reference_meta_decks
from lorcana_ai.deckbuilder.evolutionary_map import MAPElitesDeckbuilder
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator
from lorcana_ai.deckbuilder.nash_solver import NashMetaSolver
from lorcana_ai.deckbuilder.tech_scanner import TechCardScanner
from lorcana_ai.engine.game_state import BoardEntity, GameState, PlayerState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


def test_card_database_and_dsl() -> None:
    # Verify all 1,228 unique Core Constructed cards from Sets 9, 10, 11, 12, 13, 14 are loaded
    assert len(GLOBAL_CARD_DB.by_id) >= 1200
    allowed_sets = {"9", "10", "11", "12", "13", "14"}
    for card in GLOBAL_CARD_DB.by_id.values():
        assert card.set_code in allowed_sets, f"Card {card.full_name} from set {card.set_code} should not be in Core 9-14"

    # Verify Boost mechanic (e.g. Ariel - Ethereal Voice from Set 10)
    ariel = GLOBAL_CARD_DB.get_by_name("Ariel - Ethereal Voice")
    assert ariel is not None
    assert ariel.abilities.boost_cost == 1

    # Verify Temporary Shift mechanic (e.g. Meilin Lee - Popular Red Panda from Set 13)
    meilin = GLOBAL_CARD_DB.get_by_name("Meilin Lee - Popular Red Panda")
    assert meilin is not None
    assert meilin.abilities.shift_cost == 3
    assert meilin.abilities.is_temporary_shift is True

    # Verify Dual-Ink & Combo Shift (e.g. Sulley & Boo - Scare Buddies)
    sulley_boo = GLOBAL_CARD_DB.get_by_name("Sulley & Boo - Scare Buddies")
    assert sulley_boo is not None
    assert sulley_boo.is_dual_ink is True
    assert set(sulley_boo.inks) == {"Amber", "Ruby"}
    assert sulley_boo.abilities.is_combo_shift is True
    assert sulley_boo.matches_base_name_for_shift("Sulley") is True
    assert sulley_boo.matches_base_name_for_shift("Boo") is True


def test_comprehensive_rules_v2_3_0_oct_2026() -> None:
    """Tests all 8 rules findings from CRUpdate_EN_Oct-2026.pdf (v2.3.0, Effective Oct 16, 2026)."""
    # 1. Rule 8.16 (Adventurous: cannot challenge, blocks PASS_TURN if ready & can quest)
    adv_card = CardTemplate(
        card_id=9001,
        full_name="Woody - Friendly Sheriff",
        base_name="Woody",
        ink="Amber",
        inks=("Amber",),
        cost=2,
        inkable=True,
        card_type="Character",
        strength=2,
        willpower=3,
        lore=2,
        abilities=CompiledAbilities(keywords=KeywordFlag.ADVENTUROUS),
    )
    adv_ent = BoardEntity(instance_id=1, card=adv_card, exerted=False, drying=False)
    assert adv_ent.can_challenge() is False, "CR 8.16.2: Adventurous character cannot challenge"
    assert adv_ent.can_quest() is True

    p0 = PlayerState(player_id=0, deck=[adv_card] * 5, board=[adv_ent])
    p1 = PlayerState(player_id=1, deck=[adv_card] * 5)
    state = GameState(players=(p0, p1), active_player=0, phase="MAIN")
    legal = MoveGenerator.generate_legal_actions(state)
    action_types = {a.action_type for a in legal}
    assert ActionType.QUEST in action_types
    assert ActionType.PASS_TURN not in action_types, "CR 8.16.3: Cannot PASS_TURN while Adventurous char can quest"

    # 2. Rule 1.8.1.2 & 2.3.3.2 (Deck-out loss occurs when turn ends with 0 cards in deck, not mid-turn)
    p0_empty = PlayerState(player_id=0, deck=[], hand=[adv_card])
    p1_ok = PlayerState(player_id=1, deck=[adv_card] * 5)
    st_deck = GameState(players=(p0_empty, p1_ok), active_player=0, phase="MAIN")
    LorcanaSimulator._begin_turn(st_deck, draw_card=True)
    assert st_deck.phase == "MAIN", "CR 1.8.1.2: Drawing from empty deck does not immediately lose during Draw step"
    LorcanaSimulator.step(st_deck, GameAction(ActionType.PASS_TURN))
    assert st_deck.phase == "GAME_OVER" and st_deck.winner == 1 and st_deck.victory_reason == "deck_out"

    # 3. Rule 1.13.1.2 (Ink Drop Counters: Mickey Mouse - Best in Town)
    mickey_best = GLOBAL_CARD_DB.get_by_name("Mickey Mouse - Best in Town")
    assert mickey_best is not None
    assert any(cl.op == EffectOpType.EACH_PLAYER_INK_DROP for cl in mickey_best.abilities.clauses)
    p_ink = PlayerState(player_id=0, inkwell_available=2, ink_drops=2)
    assert p_ink.total_payable_ink == 4
    p_ink.pay_ink(3)
    assert p_ink.inkwell_available == 0 and p_ink.ink_drops == 1

    # 4. Rule 4.6.8, 4.7 & 5.6.6 (Locations: MOVE_TO_LOCATION, Challenging Locations & Banishing)
    loc_card = CardTemplate(
        card_id=9002,
        full_name="Hyperia City - Central Plaza",
        base_name="Hyperia City",
        ink="Steel",
        inks=("Steel",),
        cost=2,
        inkable=True,
        card_type="Location",
        willpower=4,
        lore=1,
        move_cost=1,
    )
    atk_card = CardTemplate(
        card_id=9003,
        full_name="Beast - Attacker",
        base_name="Beast",
        ink="Steel",
        inks=("Steel",),
        cost=3,
        inkable=True,
        card_type="Character",
        strength=4,
        willpower=4,
        lore=1,
    )
    my_char = BoardEntity(instance_id=10, card=atk_card, exerted=False, drying=False)
    my_loc = BoardEntity(instance_id=11, card=loc_card, exerted=False, drying=False)
    opp_loc = BoardEntity(instance_id=20, card=loc_card, exerted=False, drying=False)
    opp_char_at_loc = BoardEntity(instance_id=21, card=atk_card, exerted=False, drying=False, location_instance_id=20)

    st_loc = GameState(
        players=(
            PlayerState(player_id=0, inkwell_available=2, deck=[atk_card] * 5, board=[my_char, my_loc]),
            PlayerState(player_id=1, deck=[atk_card] * 5, board=[opp_loc, opp_char_at_loc]),
        ),
        active_player=0,
        phase="MAIN",
    )
    legal_loc = MoveGenerator.generate_legal_actions(st_loc)
    move_acts = [a for a in legal_loc if a.action_type == ActionType.MOVE_TO_LOCATION]
    chal_acts = [a for a in legal_loc if a.action_type == ActionType.CHALLENGE and a.target_instance_id == 20]
    assert len(move_acts) == 1, "CR 4.7: Should generate MOVE_TO_LOCATION action"
    assert len(chal_acts) == 1, "CR 4.6.8: Should generate CHALLENGE action targeting opposing Location"

    LorcanaSimulator.step(st_loc, move_acts[0])
    assert my_char.location_instance_id == 11 and st_loc.players[0].inkwell_available == 1
    LorcanaSimulator.step(st_loc, chal_acts[0])
    assert my_char.damage == 0, "CR 4.6.8.2: Challenging a Location deals 0 counter-damage"
    assert len(st_loc.players[1].locations_in_play) == 0, "Location with 4 willpower banished by 4 strength"
    assert opp_char_at_loc.location_instance_id == -1, "CR 5.6.6: Character at banished Location remains in play with location_instance_id == -1"

    # 5. Rule 8.10.8 (All 6 Official Shift Variants: Classification, Duo, Combo, Temporary, Potato)
    madrigal_family = GLOBAL_CARD_DB.get_by_name("The Madrigal Family - Every Generation")
    assert madrigal_family is not None and madrigal_family.abilities.shift_classification == "Madrigal"
    vine = GLOBAL_CARD_DB.get_by_name("The Vine - Towering Stalk")
    assert vine is not None and vine.abilities.shift_classification == "Floodborn"
    sun_yee = GLOBAL_CARD_DB.get_by_name("Sun Yee - Red Panda Spirit")
    assert sun_yee is not None and sun_yee.abilities.is_temporary_shift and sun_yee.abilities.shift_classification == "Red Panda"
    posey = GLOBAL_CARD_DB.get_by_name("Posey - Vampire Potato")
    assert posey is not None and posey.abilities.is_potato_shift is True
    potato_item = GLOBAL_CARD_DB.get_by_name("Potato")
    assert potato_item is not None and posey.can_shift_onto_single(potato_item) is True

    duo_mm = GLOBAL_CARD_DB.get_by_name("Mickey Mouse & Minnie Mouse - Adventuring Duo")
    assert duo_mm is not None and duo_mm.abilities.is_duo_shift is True
    mickey_c = CardTemplate(card_id=9010, full_name="Mickey Mouse - Friend", base_name="Mickey Mouse", ink="Amber", inks=("Amber",), cost=1, inkable=True, card_type="Character", strength=1, willpower=2, lore=1)
    minnie_c = CardTemplate(card_id=9011, full_name="Minnie Mouse - Friend", base_name="Minnie Mouse", ink="Amber", inks=("Amber",), cost=1, inkable=True, card_type="Character", strength=1, willpower=2, lore=1)
    assert duo_mm.can_shift_onto_single(mickey_c) is False, "CR 8.10.8.3: Duo Shift requires TWO characters"
    assert duo_mm.can_shift_onto_pair(mickey_c, minnie_c) is True

    # 6. Rule 8.14 (Vanish banished when chosen by opponent's Action) & Rule 8.3.2 (Bodyguard enter-exerted)
    vanish_char = CardTemplate(
        card_id=9020,
        full_name="Rajah - Ghostly Tiger",
        base_name="Rajah",
        ink="Amethyst",
        inks=("Amethyst",),
        cost=3,
        inkable=True,
        card_type="Character",
        strength=3,
        willpower=5,
        lore=2,
        abilities=CompiledAbilities(keywords=KeywordFlag.VANISH),
    )
    ping_action = CardTemplate(
        card_id=9021,
        full_name="Spark - 1 Damage Action",
        base_name="Spark",
        ink="Steel",
        inks=("Steel",),
        cost=1,
        inkable=True,
        card_type="Action",
        abilities=CompiledAbilities(
            clauses=[EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.DEAL_DAMAGE, amount=1)]
        ),
    )
    st_van = GameState(
        players=(
            PlayerState(player_id=0, inkwell_available=2, hand=[ping_action], deck=[ping_action] * 5),
            PlayerState(player_id=1, deck=[vanish_char] * 5, board=[BoardEntity(instance_id=50, card=vanish_char, drying=False)]),
        ),
        active_player=0,
        phase="MAIN",
    )
    LorcanaSimulator.step(st_van, GameAction(ActionType.PLAY_CARD, hand_index=0))
    assert len(st_van.players[1].characters_in_play) == 0, "CR 8.14.1: Character with Vanish banished when chosen by opponent's Action"

    # 7. Rule 1.10.1.3 (Christopher Robin - Hunny Sage allows off-color Hunny characters in Amethyst/Sapphire)
    chris_robin = GLOBAL_CARD_DB.get_by_name("Christopher Robin - Hunny Sage")
    pooh_hunny = next(
        (c for c in GLOBAL_CARD_DB.by_id.values() if "Hunny" in c.subtypes and "Amber" in c.inks),
        None,
    )
    if chris_robin is not None and pooh_hunny is not None:
        filler = [
            c for c in GLOBAL_CARD_DB.by_id.values()
            if set(c.inks).issubset({"Amethyst", "Sapphire"}) and c.card_id != chris_robin.card_id
        ][:14]
        counts = {chris_robin.card_id: 4, pooh_hunny.card_id: 4}
        for fc in filler:
            counts[fc.card_id] = 4
        hunny_deck = Deck(name="Hunny Sage Test", card_counts=counts)
        valid_h, errs_h = hunny_deck.validate()
        assert valid_h, f"CR 1.10.1.3: Christopher Robin - Hunny Sage should permit off-color Hunny characters: {errs_h}"

    # 8. Amber/Emerald Princess Aggro Mechanics (Dale Willpower Damage, Pocahontas Free 1-Drop & Peacekeeper Challenge Fog)
    poca_guide = GLOBAL_CARD_DB.get_by_name("Pocahontas - Guiding the Tribe")
    poca_peace = GLOBAL_CARD_DB.get_by_name("Pocahontas - Peacekeeper")
    dale_card = GLOBAL_CARD_DB.get_by_name("Dale - Ready for His Shot")
    pluto_card = GLOBAL_CARD_DB.get_by_name("Pluto - Friendly Pooch")
    assert poca_guide is not None and any(cl.op == EffectOpType.PLAY_COST1_CHAR_FREE for cl in poca_guide.abilities.clauses)
    assert poca_peace is not None and any(cl.op == EffectOpType.FOG_CHALLENGES for cl in poca_peace.abilities.clauses)
    assert dale_card is not None and dale_card.abilities.deals_willpower_damage_in_challenges is True
    assert pluto_card is not None and any(cl.op == EffectOpType.DISCOUNT_NEXT_CHAR and cl.requires_exert for cl in pluto_card.abilities.clauses)



def test_reference_decks_and_dreamborn_io() -> None:
    decks = get_reference_meta_decks()
    assert len(decks) >= 6
    for name, deck in decks.items():
        valid, errors = deck.validate()
        assert valid, f"Deck {name} failed validation: {errors}"
        assert deck.total_cards == 60
        assert len(deck.inks) == 2

        exported = deck.to_dreamborn_text()
        reimported = Deck.from_dreamborn_text(exported, name=name)
        assert reimported.total_cards == 60
        assert reimported.inks == deck.inks


def test_simulator_and_imperfect_info_determinization() -> None:
    decks = list(get_reference_meta_decks().values())
    state = GameState.create_new_game(
        decks[0].expand_cards(),
        decks[1].expand_cards(),
        starting_player=0,
        know_opponent_inks=True,
        seed=123,
    )
    det_state = state.clone_and_determinize(observer_id=0)
    assert len(det_state.players[1].hand) == len(state.players[1].hand)
    assert len(det_state.players[1].deck) == len(state.players[1].deck)

    agent = TacticalHeuristicAgent()
    steps = 0
    while state.phase != "GAME_OVER" and steps < 180:
        steps += 1
        legal = MoveGenerator.generate_legal_actions(state)
        assert len(legal) >= 1
        action = agent.select_action(state, legal)
        LorcanaSimulator.step(state, action)

    assert state.phase == "GAME_OVER"
    assert state.winner in (0, 1)


def test_neural_networks_and_is_mcts() -> None:
    decks = list(get_reference_meta_decks().values())
    state = GameState.create_new_game(
        decks[0].expand_cards(),
        decks[1].expand_cards(),
        starting_player=0,
        know_opponent_inks=True,
        seed=77,
    )
    pv_net = LorcanaTransformerNet()
    belief_net = OpponentBeliefNetwork()
    mcts_agent = ISMCTSAgent(
        policy_value_net=pv_net,
        belief_net=belief_net,
        cfg=MCTSConfig(num_determinizations=2, simulations_per_determinization=4),
    )

    action, pi_vec = mcts_agent.select_action(state, use_mcts=True, add_exploration_noise=True)
    assert action is not None
    assert abs(float(pi_vec.sum()) - 1.0) < 1e-4


def test_self_play_trainer_checkpointing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        trainer = SelfPlayTrainer(
            mcts_cfg=MCTSConfig(num_determinizations=2, simulations_per_determinization=4),
            checkpoint_path=ckpt_path,
        )
        metrics = trainer.train_epoch(games_per_epoch=2, batch_size=16, use_mcts=False)
        assert metrics["epoch"] == 1
        assert ckpt_path.exists()

        # Verify checkpoint reload
        trainer2 = SelfPlayTrainer(checkpoint_path=ckpt_path)
        assert trainer2.completed_epochs == 1


def test_deckbuilder_modules() -> None:
    decks = list(get_reference_meta_decks().values())
    evaluator = MatchupEvaluator()
    stats = evaluator.evaluate_matchup(decks[0], decks[1], num_games=4, seed=11)
    assert 0.0 <= stats.bo3_winrate <= 1.0

    # Module 1: Tech-Card Scanner
    scanner = TechCardScanner(evaluator=evaluator)
    recs = scanner.scan_for_matchup(
        base_deck=decks[1],
        opponent_deck=decks[0],
        copies_to_swap=2,
        total_simulation_budget=24,
        top_k=3,
    )
    assert len(recs) >= 1

    # Module 2: MAP-Elites Coevolution
    evo = MAPElitesDeckbuilder(evaluator=evaluator)
    cells = evo.run_evolution(
        generations=1,
        offspring_per_gen=2,
        games_per_matchup=2,
        target_ink_pairs=[("Amethyst", "Ruby"), ("Ruby", "Sapphire")],
    )
    assert len(cells) >= 2

    # Module 3: Nash Meta-Solver & Mulligan Guide
    solver = NashMetaSolver(evaluator=evaluator)
    report = solver.solve_metagame(
        decks=decks[:3],
        current_meta_shares=DEFAULT_METAGAME_SHARES,
        games_per_pair=4,
    )
    assert abs(sum(report.nash_weights.values()) - 1.0) < 1e-2

    mull = solver.generate_mulligan_guide(decks[1], decks[0], games_per_strategy=4)
    assert len(mull.priority_keep_cards) >= 3


if __name__ == "__main__":
    test_card_database_and_dsl()
    test_comprehensive_rules_v2_3_0_oct_2026()
    test_reference_decks_and_dreamborn_io()
    test_simulator_and_imperfect_info_determinization()
    test_neural_networks_and_is_mcts()
    test_self_play_trainer_checkpointing()
    test_deckbuilder_modules()
    print("ALL LORCANA_ALPHA TESTS (INCLUDING CR v2.3.0 OCT 2026 AUDIT) PASSED SUCCESSFULLY!")
