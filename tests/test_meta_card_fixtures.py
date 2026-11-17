"""Fixture tests for critical meta cards (Sets 9–14) — not smoke play."""

from __future__ import annotations

from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.data.effect_dsl import AuraFilter, ChoiceKind, EffectOpType, KeywordFlag, TriggerType
from lorcana_ai.engine.game_state import BoardEntity, GameState, PlayerState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator


def _card(name: str):
    c = GLOBAL_CARD_DB.get_by_name(name)
    assert c is not None, f"Missing card: {name}"
    return c


def test_amber_champion_willpower_aura_and_singer():
    """Mickey Mouse - Amber Champion: +2 willpower to other Amber; Singer 8 with 2+ other Amber."""
    ac = _card("Mickey Mouse - Amber Champion")
    assert ac.abilities.singer_cost == ac.cost  # not native Singer 8
    assert any(a.willpower == 2 and a.filter_kind == AuraFilter.INK and a.filter_value == "Amber" for a in ac.abilities.aura_statics)
    assert any(g.grant_singer == 8 and g.condition == "other_amber_ge_2" for g in ac.abilities.conditional_grants)

    amber_others = [
        c for c in GLOBAL_CARD_DB.by_id.values()
        if "Amber" in c.inks and c.card_type == "Character" and c.full_name != ac.full_name
    ][:2]
    assert len(amber_others) == 2

    p = PlayerState(0)
    champ = BoardEntity(1, ac, drying=False, exerted=False)
    a1 = BoardEntity(2, amber_others[0], drying=False)
    a2 = BoardEntity(3, amber_others[1], drying=False)
    p.board = [champ, a1, a2]

    assert a1.effective_willpower_with(p) == a1.card.willpower + 2
    assert a2.effective_willpower_with(p) == a2.card.willpower + 2
    # Aura excludes self
    assert champ.effective_willpower_with(p) == champ.card.willpower
    assert champ.effective_singer_cost(p) == 8
    assert champ.can_sing(8, owner=p)


def test_gadget_support_lore_aura_compiled():
    gadget = _card("Gadget Hackwrench - Resourceful Mechanic")
    assert any(a.filter_kind.value == "HAS_SUPPORT" and a.lore >= 1 for a in gadget.abilities.aura_statics)


def test_support_opens_choice_node_interactive():
    support_card = next(
        c for c in GLOBAL_CARD_DB.by_id.values()
        if c.card_type == "Character" and (c.abilities.keywords & KeywordFlag.SUPPORT) and c.lore > 0
    )
    ally = next(
        c for c in GLOBAL_CARD_DB.by_id.values()
        if c.card_type == "Character" and c.full_name != support_card.full_name and c.lore > 0
    )
    state = GameState.create_new_game([support_card] * 30 + [ally] * 30, [ally] * 60, starting_player=0, seed=7)
    # Force MAIN with two ready dry characters
    state.phase = "MAIN"
    state.active_player = 0
    p = state.players[0]
    p.hand.clear()
    p.board = [
        BoardEntity(state.next_instance_id, support_card, drying=False, exerted=False),
        BoardEntity(state.next_instance_id + 1, ally, drying=False, exerted=False),
    ]
    state.next_instance_id += 2
    quester = p.board[0]
    LorcanaSimulator.interactive_choices = True
    try:
        LorcanaSimulator.step(
            state,
            GameAction(
                action_type=ActionType.QUEST,
                source_instance_id=quester.instance_id,
                description="Quest Support",
            ),
        )
        assert state.phase == "CHOICE"
        assert state.pending_choice is not None
        assert state.pending_choice.choice_kind == ChoiceKind.SUPPORT_TARGET.value
        legal = MoveGenerator.generate_legal_actions(state)
        assert any(a.action_type == ActionType.RESOLVE_CHOICE for a in legal)
        assert any(a.decline_choice for a in legal)
    finally:
        LorcanaSimulator.interactive_choices = False


def test_beast_gracious_prince_princess_aura():
    beast = _card("Beast - Gracious Prince")
    assert any(
        a.filter_kind == AuraFilter.CLASSIFICATION and a.filter_value.lower() == "princess" and (a.strength or a.willpower)
        for a in beast.abilities.aura_statics
    )


def test_aurora_dreaming_guardian_ward_aura():
    aurora = _card("Aurora - Dreaming Guardian")
    assert any(a.grant_keywords & KeywordFlag.WARD for a in aurora.abilities.aura_statics)


def test_lost_in_the_woods_debuff_clause():
    card = _card("Lost in the Woods")
    assert any(c.op == EffectOpType.DEBUFF_OPP_STR and c.threshold >= 99 for c in card.abilities.clauses)


def test_choice_annotation_on_banish():
    # Any banish-chosen card should expose TARGET_OPP_CHAR
    card = next(
        c for c in GLOBAL_CARD_DB.by_id.values()
        if any(cl.op == EffectOpType.BANISH_CHAR for cl in c.abilities.clauses)
    )
    ban = next(cl for cl in card.abilities.clauses if cl.op == EffectOpType.BANISH_CHAR)
    assert ban.choice_kind == ChoiceKind.TARGET_OPP_CHAR
