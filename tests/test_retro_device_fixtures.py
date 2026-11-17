"""Critical-loop fixtures for Emerald/Ruby Retro Device (Bob recycle, Sid↔Bob, Leviathan)."""

from __future__ import annotations

from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.effect_dsl import EffectOpType, TriggerType
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.simulator import LorcanaSimulator


def _card(name: str):
    c = GLOBAL_CARD_DB.get_by_name(name)
    assert c is not None, f"Missing card: {name}"
    return c


def _blank_state(filler):
    state = GameState.create_new_game([filler] * 60, [filler] * 60, starting_player=0, seed=42)
    state.phase = "MAIN"
    state.active_player = 0
    p0, p1 = state.players
    p0.board.clear()
    p1.board.clear()
    p0.hand.clear()
    p1.hand.clear()
    p0.discard.clear()
    p1.discard.clear()
    p0.inkwell_total = 12
    p0.inkwell_available = 12
    return state, p0, p1


def test_retro_key_cards_compile():
    device = _card("Retro Evolution Device")
    assert any(c.op == EffectOpType.RETRO_EVOLVE for c in device.abilities.clauses)

    for name in ("Improvise", "Inner Strength"):
        c = _card(name)
        ops = {cl.op for cl in c.abilities.clauses}
        assert EffectOpType.BUFF_ALLY_STR in ops
        assert EffectOpType.DRAW in ops

    strange = _card("Strange Things")
    assert any(cl.op == EffectOpType.CANT_QUEST_CHARS for cl in strange.abilities.clauses)

    rivera = _card("Rivera Family Photo")
    assert any(cl.op == EffectOpType.RIVERA_CHOOSE for cl in rivera.abilities.clauses)
    assert not any(cl.op == EffectOpType.GAIN_LORE for cl in rivera.abilities.clauses)

    torn = _card("The Torn Corner")
    assert any(
        cl.trigger == TriggerType.ACTIVATED and cl.op == EffectOpType.DRAW for cl in torn.abilities.clauses
    )

    sid = _card("Sid Phillips - Toy Surgeon")
    assert any(cl.op == EffectOpType.SAC_FORCE_OPP_BANISH for cl in sid.abilities.clauses)
    assert not any(cl.op == EffectOpType.GAIN_LORE for cl in sid.abilities.clauses)

    levi = _card("The Leviathan - Guardian of Atlantis")
    assert any(cl.op == EffectOpType.BANISH_OPP_TOTAL_STR for cl in levi.abilities.clauses)

    bob = _card("Roller Bob - Sid's Toy")
    assert any(cl.op == EffectOpType.GIVE_SELF_RUSH for cl in bob.abilities.clauses)

    olaf = _card("Olaf - Snowman of Action")
    assert any(cl.op == EffectOpType.LOSE_OPP_LORE for cl in olaf.abilities.clauses)

    phillip = _card("Prince Phillip - Vanquisher of Foes")
    assert any(cl.op == EffectOpType.BANISH_ALL_DAMAGED_OPP for cl in phillip.abilities.clauses)

    max_k = _card("Max Goof - Karaoke Star")
    assert max_k.abilities.has_five_songs_lore_bonus
    assert any(cl.op == EffectOpType.LOOT_SELF for cl in max_k.abilities.clauses)

    rabbit = _card("White Rabbit - Late Again")
    assert rabbit.abilities.has_underdog

    tod = _card("Tod - Clever Fox")
    assert any(cl.op == EffectOpType.LOOT_SELF for cl in tod.abilities.clauses)

    malicious = _card("Malicious, Mean, and Scary")
    assert any(cl.op == EffectOpType.DEAL_DAMAGE_ALL_OPP for cl in malicious.abilities.clauses)


def test_roller_bob_gy_to_bottom_rush():
    bob = _card("Roller Bob - Sid's Toy")
    bobby = _card("Bobby Zimuruski - Spray Cheese Kid")
    poca = _card("Pocahontas - Guiding the Tribe")
    state, p0, _ = _blank_state(bobby)
    p0.discard = [bobby, poca]
    rb = BoardEntity(instance_id=1, card=bob, drying=True)
    p0.board = [rb]
    LorcanaSimulator._resolve_clauses(state, 0, bob, TriggerType.ON_PLAY, source_id=1)
    assert rb.drying is False
    assert len(p0.discard) == 0
    assert bobby in p0.deck and poca in p0.deck


def test_sid_bob_toy_lore_and_sac():
    sid = _card("Sid Phillips - Toy Surgeon")
    bob = _card("Roller Bob - Sid's Toy")
    bobby = _card("Bobby Zimuruski - Spray Cheese Kid")
    elinor = _card("Elinor - Renowned Diplomat")
    state, p0, p1 = _blank_state(bobby)
    sid_ent = BoardEntity(1, sid, drying=False)
    bob_ent = BoardEntity(2, bob, drying=False)
    opp_ent = BoardEntity(3, elinor, drying=False)
    p0.board = [sid_ent, bob_ent]
    p1.board = [opp_ent]
    lore_before = p0.lore
    LorcanaSimulator._resolve_clauses(state, 0, sid, TriggerType.ON_PLAY, source_id=1)
    # Sac Bob (Toy) → Sid DOUBLE PRIZES +2 lore; opp loses a character
    assert bob in p0.discard
    assert p0.lore == lore_before + 2
    assert len(p1.board) == 0
    assert elinor in p1.discard


def test_leviathan_same_turn_gy_wipe():
    levi = _card("The Leviathan - Guardian of Atlantis")
    bobby = _card("Bobby Zimuruski - Spray Cheese Kid")
    elinor = _card("Elinor - Renowned Diplomat")
    state, p0, p1 = _blank_state(bobby)
    p0.cards_discarded_this_turn = 2
    p0.board = [BoardEntity(1, levi, drying=True)]
    # Two mid-strength bodies totaling <= 10
    p1.board = [
        BoardEntity(2, bobby, drying=False),
        BoardEntity(3, elinor, drying=False),
    ]
    LorcanaSimulator._resolve_clauses(state, 0, levi, TriggerType.ON_PLAY, source_id=1)
    assert len(p1.board) == 0

    # Gate: without 2 discards this turn, wipe does not fire
    state2, p0b, p1b = _blank_state(bobby)
    p0b.cards_discarded_this_turn = 0
    p0b.board = [BoardEntity(1, levi, drying=True)]
    p1b.board = [BoardEntity(2, elinor, drying=False)]
    LorcanaSimulator._resolve_clauses(state2, 0, levi, TriggerType.ON_PLAY, source_id=1)
    assert len(p1b.board) == 1


def test_rivera_mill_recycles_torn_corner():
    rivera = _card("Rivera Family Photo")
    torn = _card("The Torn Corner")
    bobby = _card("Bobby Zimuruski - Spray Cheese Kid")
    state, p0, _ = _blank_state(bobby)
    p0.board = [BoardEntity(1, rivera, drying=False, exerted=False)]
    # Deck pop() is from the end — place Torn on top
    p0.deck = [bobby] * 10 + [bobby, torn]
    p0.discard.clear()
    LorcanaSimulator._resolve_clauses(state, 0, rivera, TriggerType.ACTIVATED, source_id=1)
    # Mill top 2; Torn free-plays back via MEND THE PHOTO
    assert any(e.card.full_name == torn.full_name for e in p0.board)
    assert torn not in p0.discard


def test_improvise_buffs_strength_and_strange_locks_quest():
    improvise = _card("Improvise")
    strange = _card("Strange Things")
    bobby = _card("Bobby Zimuruski - Spray Cheese Kid")
    elinor = _card("Elinor - Renowned Diplomat")
    state, p0, p1 = _blank_state(bobby)
    ally = BoardEntity(1, bobby, drying=False, exerted=False)
    p0.board = [ally]
    LorcanaSimulator._execute_single_clause(
        state,
        0,
        next(cl for cl in improvise.abilities.clauses if cl.op == EffectOpType.BUFF_ALLY_STR),
        source_id=-1,
        is_action_source=True,
        card_text=improvise.full_text.lower(),
    )
    assert ally.temp_strength_mod >= 1

    opp = BoardEntity(2, elinor, drying=False, exerted=False)
    p1.board = [opp]
    LorcanaSimulator._execute_single_clause(
        state,
        0,
        next(cl for cl in strange.abilities.clauses if cl.op == EffectOpType.CANT_QUEST_CHARS),
        source_id=-1,
        is_action_source=True,
        card_text=strange.full_text.lower(),
    )
    assert opp.temp_cant_quest is True
    assert opp.can_quest() is False
