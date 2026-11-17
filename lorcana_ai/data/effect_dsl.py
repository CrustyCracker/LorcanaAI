"""
Modular Effect DSL (Domain Specific Language) & Text Parser for Disney Lorcana (Sets 9-14 Core Constructed).
Compliant with Official Disney Lorcana Comprehensive Rules v2.3.0 (Effective October 16, 2026):
- Standard Keywords (Section 8): Rush, Ward, Evasive, Bodyguard, Support, Vanish, Reckless, Alert, Adventurous (8.16)
- All 6 Official Shift Variants (Rule 8.10.8):
  - Standard Shift X (8.10)
  - [Classification] Shift X (8.10.8.1: Madrigal Shift, Floodborn Shift, Red Panda Shift, etc.)
  - Universal Shift X (8.10.8.2)
  - Duo Shift X (8.10.8.3)
  - Combo Shift X (8.10.8.4)
  - Temporary Shift X (8.10.8.5)
  - Potato Shift X (8.10.8.6)
- Parametric Keywords: Singer X, Sing Together X, Boost X, Challenger +X, Resist +X
- Counters (Rule 1.13.1.2): Ink Drop Counters (GAIN_INK_DROP / EACH_PLAYER_INK_DROP)
- Full Circle of Life / Season 14 Meta Synergy Ops:
  - REANIMATE_CHAR (Circle of Life)
  - LOOT_SELF (Strike a Good Match, Tod - Clever Fox, Max Goof - Karaoke Star, Bobby Zimuruski)
  - BANISH_ALL_DAMAGED_OPP (Prince Phillip - Vanquisher of Foes)
  - DEAL_DAMAGE_ALL_DAMAGED_OPP (To Wither a Flower)
  - DISCOUNT_NEXT_CHAR (Akood et Emuti, Aurora - Holding Court)
  - DRAW_PER_INK_TYPE (With a Few Good Friends)
  - REPLAY_SONG_FROM_DISCARD (Max Goof - Chart Topper)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntFlag
import re


class KeywordFlag(IntFlag):
    NONE = 0
    RUSH = 1 << 0
    WARD = 1 << 1
    EVASIVE = 1 << 2
    BODYGUARD = 1 << 3
    SUPPORT = 1 << 4
    VANISH = 1 << 5
    RECKLESS = 1 << 6
    ALERT = 1 << 7
    ADVENTUROUS = 1 << 8


class TriggerType(str, Enum):
    ON_PLAY = "ON_PLAY"
    ON_QUEST = "ON_QUEST"
    ON_CHALLENGE = "ON_CHALLENGE"
    ON_BANISH = "ON_BANISH"
    ON_LEAVE_PLAY = "ON_LEAVE_PLAY"
    START_OF_TURN = "START_OF_TURN"
    END_OF_TURN = "END_OF_TURN"
    ACTIVATED = "ACTIVATED"
    PASSIVE = "PASSIVE"


class EffectOpType(str, Enum):
    DRAW = "DRAW"
    LOOT_SELF = "LOOT_SELF"
    DRAW_PER_INK_TYPE = "DRAW_PER_INK_TYPE"
    DEAL_DAMAGE = "DEAL_DAMAGE"
    DEAL_DAMAGE_ALL_OPP = "DEAL_DAMAGE_ALL_OPP"
    DEAL_DAMAGE_ALL_DAMAGED_OPP = "DEAL_DAMAGE_ALL_DAMAGED_OPP"
    BANISH_CHAR = "BANISH_CHAR"
    BANISH_ALL_CHARS = "BANISH_ALL_CHARS"
    BANISH_ALL_DAMAGED_OPP = "BANISH_ALL_DAMAGED_OPP"
    BANISH_LOW_STR_CHAR = "BANISH_LOW_STR_CHAR"
    REANIMATE_CHAR = "REANIMATE_CHAR"
    REPLAY_SONG_FROM_DISCARD = "REPLAY_SONG_FROM_DISCARD"
    DISCOUNT_NEXT_CHAR = "DISCOUNT_NEXT_CHAR"
    RAMP_INK = "RAMP_INK"
    GAIN_INK_DROP = "GAIN_INK_DROP"
    EACH_PLAYER_INK_DROP = "EACH_PLAYER_INK_DROP"
    GAIN_LORE = "GAIN_LORE"
    LOSE_OPP_LORE = "LOSE_OPP_LORE"
    BOUNCE_OWN_CHAR = "BOUNCE_OWN_CHAR"
    BOUNCE_OPP_CHAR = "BOUNCE_OPP_CHAR"
    DISCARD_OPP = "DISCARD_OPP"
    BUFF_ALLY_STR = "BUFF_ALLY_STR"
    BUFF_ALLY_LORE = "BUFF_ALLY_LORE"
    BUFF_ALLY_WILLPOWER = "BUFF_ALLY_WILLPOWER"
    DEBUFF_OPP_STR = "DEBUFF_OPP_STR"
    HEAL_DRAW = "HEAL_DRAW"
    HEAL = "HEAL"
    READY_CHAR = "READY_CHAR"
    READY_ALL_CHARS = "READY_ALL_CHARS"
    BOOST_UNDER = "BOOST_UNDER"
    PLAY_COST1_CHAR_FREE = "PLAY_COST1_CHAR_FREE"
    FOG_CHALLENGES = "FOG_CHALLENGES"
    REVEAL_TOP_CHAR_BOTH = "REVEAL_TOP_CHAR_BOTH"
    DISCOUNT_NEXT_PRINCESS_QUEEN = "DISCOUNT_NEXT_PRINCESS_QUEEN"
    EXERT_OPP_CHAR = "EXERT_OPP_CHAR"
    INK_OPP_CHAR = "INK_OPP_CHAR"
    RETRO_EVOLVE = "RETRO_EVOLVE"
    PUT_UNDER_FROM_DISCARD = "PUT_UNDER_FROM_DISCARD"
    GIVE_SELF_RUSH = "GIVE_SELF_RUSH"
    MOVE_DAMAGE = "MOVE_DAMAGE"
    INK_ITEM = "INK_ITEM"
    BANISH_ITEM = "BANISH_ITEM"
    GIVE_ADVENTUROUS = "GIVE_ADVENTUROUS"
    GIVE_RUSH_EVASIVE = "GIVE_RUSH_EVASIVE"
    SUPPORT_BUFF = "SUPPORT_BUFF"
    LOOK_AT_TOP = "LOOK_AT_TOP"
    GAIN_EVASIVE_SELF = "GAIN_EVASIVE_SELF"
    PUT_CARD_UNDER_SELF = "PUT_CARD_UNDER_SELF"
    NOOP_MARKED = "NOOP_MARKED"


class ChoiceKind(str, Enum):
    """Player decision required when resolving a clause (bots resolve via policy)."""
    NONE = "NONE"
    OPTIONAL_YES_NO = "OPTIONAL_YES_NO"  # "you may"
    TARGET_ALLY_CHAR = "TARGET_ALLY_CHAR"
    TARGET_OPP_CHAR = "TARGET_OPP_CHAR"
    TARGET_ANY_CHAR = "TARGET_ANY_CHAR"
    TARGET_ALLY_ITEM = "TARGET_ALLY_ITEM"
    TARGET_ANY_ITEM = "TARGET_ANY_ITEM"
    SUPPORT_TARGET = "SUPPORT_TARGET"
    DISCARD_OWN = "DISCARD_OWN"
    REANIMATE_DISCARD = "REANIMATE_DISCARD"


class StatMod(str, Enum):
    STRENGTH = "STRENGTH"
    WILLPOWER = "WILLPOWER"
    LORE = "LORE"
    RESIST = "RESIST"
    CHALLENGER = "CHALLENGER"


class AuraFilter(str, Enum):
    ALL_OTHER = "ALL_OTHER"
    ALL = "ALL"
    CLASSIFICATION = "CLASSIFICATION"
    INK = "INK"
    HAS_SUPPORT = "HAS_SUPPORT"
    SELF = "SELF"


@dataclass(frozen=True)
class AuraStatic:
    """Compiled continuous/static modifier — preferred over full_text.lower() hacks."""
    strength: int = 0
    willpower: int = 0
    lore: int = 0
    resist: int = 0
    challenger: int = 0
    grant_keywords: KeywordFlag = KeywordFlag.NONE
    grant_singer: int | None = None
    filter_kind: AuraFilter = AuraFilter.ALL_OTHER
    filter_value: str = ""  # subtype or ink name
    while_exerted: bool = False
    while_your_turn: bool = False
    while_condition: str = ""  # opaque tag for rare conditions
    this_turn_only: bool = False  # quest-triggered temp auras use clauses instead
    exclude_self: bool = True


@dataclass(frozen=True)
class ConditionalGrant:
    """Conditional self-grant compiled from text (e.g. Amber Champion Singer 8)."""
    grant_singer: int | None = None
    grant_keywords: KeywordFlag = KeywordFlag.NONE
    resist: int = 0
    strength: int = 0
    willpower: int = 0
    lore: int = 0
    condition: str = ""  # e.g. other_amber_ge_2, hand_empty, lore_behind, card_under


@dataclass(frozen=True)
class EffectClause:
    """Single compiled effect instruction executed by the game simulator."""
    trigger: TriggerType
    op: EffectOpType
    amount: int = 1
    threshold: int = 0
    cost_ink: int = 0
    requires_exert: bool = False
    requires_card_under: bool = False
    optional: bool = False
    choice_kind: ChoiceKind = ChoiceKind.NONE
    target_filter: str = ""  # subtype / ink gate for chosen targets
    classification: str = ""
    unsupported_reason: str = ""


@dataclass
class CompiledAbilities:
    """All compiled static keywords and triggered/activated DSL clauses for a card."""
    keywords: KeywordFlag = KeywordFlag.NONE
    shift_cost: int | None = None
    is_temporary_shift: bool = False
    is_combo_shift: bool = False
    is_duo_shift: bool = False
    is_universal_shift: bool = False
    is_potato_shift: bool = False
    shift_classification: str | None = None
    singer_cost: int | None = None
    sing_together_cost: int | None = None
    boost_cost: int | None = None
    challenger_bonus: int = 0
    resist_bonus: int = 0
    has_underdog: bool = False
    has_five_songs_lore_bonus: bool = False
    deals_willpower_damage_in_challenges: bool = False
    clauses: list[EffectClause] = field(default_factory=list)
    aura_statics: list[AuraStatic] = field(default_factory=list)
    conditional_grants: list[ConditionalGrant] = field(default_factory=list)
    unsupported_notes: list[str] = field(default_factory=list)

    def to_rust_bitflags(self) -> tuple[int, int]:
        """Packs abilities into fast integer bitflags and primary effect value for lorcana_rs."""
        flags = int(self.keywords)
        if self.singer_cost is not None or self.sing_together_cost is not None:
            flags |= 16
        effect_val = 0
        for cl in self.clauses:
            if cl.op in (
                EffectOpType.DEAL_DAMAGE,
                EffectOpType.DEAL_DAMAGE_ALL_OPP,
                EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP,
                EffectOpType.BANISH_CHAR,
                EffectOpType.BANISH_ALL_CHARS,
                EffectOpType.BANISH_ALL_DAMAGED_OPP,
                EffectOpType.BANISH_LOW_STR_CHAR,
            ):
                flags |= 32
                effect_val = max(effect_val, cl.amount if cl.amount > 0 else 4)
            elif cl.op in (
                EffectOpType.DRAW,
                EffectOpType.LOOT_SELF,
                EffectOpType.DRAW_PER_INK_TYPE,
                EffectOpType.HEAL_DRAW,
                EffectOpType.REANIMATE_CHAR,
            ):
                flags |= 64
                effect_val = max(effect_val, cl.amount)
        return flags, effect_val


class EffectDSLCompiler:
    """Compiles raw LorcanaJSON card data and rules text into CompiledAbilities."""

    _RE_TEMP_CLASS_SHIFT = re.compile(
        r"Temporary\s+(?:(Red\s+Panda|Madrigal|Floodborn|Hero|Villain|Whisper)\s+)?Shift\s+(\d+)",
        re.IGNORECASE,
    )
    _RE_DUO_SHIFT = re.compile(r"\bDuo\s+Shift\s+(\d+)", re.IGNORECASE)
    _RE_COMBO_SHIFT = re.compile(r"\bCombo\s+Shift\s+(\d+)", re.IGNORECASE)
    _RE_UNIVERSAL_SHIFT = re.compile(r"\bUniversal\s+Shift\s+(\d+)", re.IGNORECASE)
    _RE_POTATO_SHIFT = re.compile(r"\bPotato\s+Shift\s+(\d+)", re.IGNORECASE)
    _RE_CLASSIFICATION_SHIFT = re.compile(
        r"\b(Madrigal|Floodborn|Red\s+Panda|Puppy|Pirate|Hero|Villain|Princess|Prince|Knight|Alien|Robot|Whisper|Inventor|Detective|Dragon|Fairy|Hyena|Broom|Musketeer|Racer|Hunny)\s+Shift\s+(\d+)",
        re.IGNORECASE,
    )
    _RE_SHIFT = re.compile(r"\bShift\s+(\d+)", re.IGNORECASE)
    _RE_SING_TOGETHER = re.compile(r"Sing\s+Together\s+(\d+)", re.IGNORECASE)
    _RE_SINGER = re.compile(r"Singer\s+(\d+)", re.IGNORECASE)
    _RE_BOOST = re.compile(r"Boost\s+(\d+)", re.IGNORECASE)
    _RE_CHALLENGER = re.compile(r"Challenger\s*\+(\d+)", re.IGNORECASE)
    _RE_RESIST = re.compile(r"Resist\s*\+(\d+)", re.IGNORECASE)
    _RE_DRAW = re.compile(r"draws?\s+(\d+|a|one|two|three|four|five|six|seven)\s+cards?", re.IGNORECASE)
    _RE_DAMAGE = re.compile(r"deal\s+(\d+)\s+damage", re.IGNORECASE)
    _RE_PUT_DMG_COUNTER = re.compile(r"put\s+(\d+|a)\s+damage\s+counters?\s+on\s+each\s+opposing", re.IGNORECASE)
    _RE_DISCOUNT_CHAR = re.compile(r"you\s+pay\s+(\d+)\s*⬡?\s*less\s+for\s+the\s+next\s+[\w\s]*character", re.IGNORECASE)
    _RE_GAIN_LORE = re.compile(r"gain\s+(\d+)\s+lore", re.IGNORECASE)
    _RE_LOSE_LORE = re.compile(r"loses?\s+(\d+)\s+lore", re.IGNORECASE)
    _RE_STR_THRESH = re.compile(r"(\d+)\s*(?:\{S\}|¤|Strength)\s+or\s+less", re.IGNORECASE)
    _RE_STR_MIN_THRESH = re.compile(r"(\d+)\s*(?:\{S\}|¤|Strength)\s+or\s+more", re.IGNORECASE)

    _WORD_TO_INT = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

    @classmethod
    def compile_from_text(
        cls,
        full_text: str,
        card_type: str = "Character",
        cost: int = 1,
        keyword_list: list[str] | None = None,
        explicit_clauses: list[EffectClause] | None = None,
    ) -> CompiledAbilities:
        abilities = CompiledAbilities()
        text = " ".join((full_text or "").split())
        lower = text.lower()
        kw_set = {k.lower() for k in (keyword_list or [])}

        # 1. Parse Native Keywords (CR Section 8) — avoid false positives from "lose Rush" / "give chosen character ..."
        def _has_native_kw(kw_name: str) -> bool:
            if kw_name.lower() in kw_set:
                return True
            if keyword_list is None:
                return bool(re.search(rf"\b{kw_name}\b", text, re.IGNORECASE))
            return bool(
                re.search(
                    rf"(?<!gains\s)(?<!gain\s)(?<!lose\s)(?<!character\s)\b{kw_name}\s*\(",
                    text,
                    re.IGNORECASE,
                )
            )

        if _has_native_kw("Rush"):
            abilities.keywords |= KeywordFlag.RUSH
        if _has_native_kw("Ward"):
            abilities.keywords |= KeywordFlag.WARD
        if _has_native_kw("Evasive"):
            abilities.keywords |= KeywordFlag.EVASIVE
        if _has_native_kw("Bodyguard"):
            abilities.keywords |= KeywordFlag.BODYGUARD
        if _has_native_kw("Support") or "this character gains support" in lower:
            abilities.keywords |= KeywordFlag.SUPPORT
        if _has_native_kw("Vanish"):
            abilities.keywords |= KeywordFlag.VANISH
        if _has_native_kw("Reckless"):
            abilities.keywords |= KeywordFlag.RECKLESS
        if _has_native_kw("Alert"):
            abilities.keywords |= KeywordFlag.ALERT
        if _has_native_kw("Adventurous"):
            abilities.keywords |= KeywordFlag.ADVENTUROUS

        # Underdog, 5+ Songs Lore bonus, and Dale's Willpower damage in challenges
        if "if this is your first turn and you're not the first player, you pay 1" in lower:
            abilities.has_underdog = True
        if "5 or more song cards in your discard, this character gets +3" in lower:
            abilities.has_five_songs_lore_bonus = True
        if "during challenges, your characters deal damage with their" in lower:
            abilities.deals_willpower_damage_in_challenges = True

        # 2. All 6 Official Shift Variants (CR Rule 8.10.8)
        m_tshift = cls._RE_TEMP_CLASS_SHIFT.search(text)
        m_duo = cls._RE_DUO_SHIFT.search(text)
        m_combo = cls._RE_COMBO_SHIFT.search(text)
        m_univ = cls._RE_UNIVERSAL_SHIFT.search(text)
        m_potato = cls._RE_POTATO_SHIFT.search(text)
        m_class = cls._RE_CLASSIFICATION_SHIFT.search(text)
        m_shift = cls._RE_SHIFT.search(text)

        if m_tshift:
            abilities.shift_cost = int(m_tshift.group(2))
            abilities.is_temporary_shift = True
            if m_tshift.group(1):
                abilities.shift_classification = " ".join(m_tshift.group(1).split())
        elif m_duo:
            abilities.shift_cost = int(m_duo.group(1))
            abilities.is_duo_shift = True
        elif m_combo:
            abilities.shift_cost = int(m_combo.group(1))
            abilities.is_combo_shift = True
        elif m_univ:
            abilities.shift_cost = int(m_univ.group(1))
            abilities.is_universal_shift = True
        elif m_potato:
            abilities.shift_cost = int(m_potato.group(1))
            abilities.is_potato_shift = True
        elif m_class:
            abilities.shift_cost = int(m_class.group(2))
            abilities.shift_classification = " ".join(m_class.group(1).split())
        elif m_shift:
            abilities.shift_cost = int(m_shift.group(1))

        # Sing Together & Singer
        m_st = cls._RE_SING_TOGETHER.search(text)
        if m_st:
            abilities.sing_together_cost = int(m_st.group(1))

        # Native Singer X (with reminder paren), NOT conditional "gains Singer X"
        m_native_singer = re.search(r"(?<!gains\s)\bSinger\s+(\d+)\s*\(", text, re.IGNORECASE)
        m_singer_plain = cls._RE_SINGER.search(text)
        if m_native_singer:
            abilities.singer_cost = int(m_native_singer.group(1))
        elif m_singer_plain and "gains singer" not in lower:
            abilities.singer_cost = int(m_singer_plain.group(1))
        elif card_type == "Character":
            abilities.singer_cost = cost

        # Boost X
        m_boost = cls._RE_BOOST.search(text)
        if m_boost:
            abilities.boost_cost = int(m_boost.group(1))

        m_chal = cls._RE_CHALLENGER.search(text)
        if m_chal:
            if "give chosen character challenger" in lower:
                abilities.clauses.append(
                    EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.BUFF_ALLY_STR, amount=int(m_chal.group(1)))
                )
            else:
                abilities.challenger_bonus = int(m_chal.group(1))

        m_res = cls._RE_RESIST.search(text)
        if (
            m_res
            and "give chosen character" not in lower
            and "your detective characters gain" not in lower
            and "your other characters gain" not in lower
            and "card under this character" not in lower
            and "cards in your hand" not in lower
        ):
            abilities.resist_bonus = int(m_res.group(1))

        if explicit_clauses:
            abilities.clauses.extend(explicit_clauses)
            return abilities

        # 3. Automatic DSL extraction from LorcanaJSON card rules text (strip reminder text in parentheses first!)
        no_rem = " ".join(re.sub(r"\([^)]*\)", " ", text).split())
        lower = no_rem.lower()
        needs_card_under = ("card under" in lower) or ("cards under" in lower)

        default_trig = (
            TriggerType.ON_PLAY
            if card_type in ("Action", "Character", "Item")
            else TriggerType.START_OF_TURN
        )
        default_exert = False
        default_cost_ink = 0
        if "⟳" in no_rem and "when you play" not in lower:
            default_trig = TriggerType.ACTIVATED
            default_exert = True
            m_act_ink = re.search(r"⟳\s*,\s*(\d+)\s*⬡", no_rem)
            if m_act_ink:
                default_cost_ink = int(m_act_ink.group(1))
        elif ("whenever this character quests" in lower or "whenever she quests" in lower or "whenever he quests" in lower) and "when you play" not in lower:
            default_trig = TriggerType.ON_QUEST
        elif (
            "when this character is banished" in lower
            or "leaves play" in lower
            or "whenever this character is challenged" in lower
            or "if this character would be banished" in lower
        ) and "when you play" not in lower:
            default_trig = TriggerType.ON_LEAVE_PLAY
        elif "at the end of your turn" in lower and "when you play" not in lower:
            default_trig = TriggerType.END_OF_TURN
        elif "at the start of your turn" in lower:
            default_trig = TriggerType.START_OF_TURN

        m_thresh = cls._RE_STR_THRESH.search(no_rem)
        str_thresh = int(m_thresh.group(1)) if m_thresh else 0
        m_min_thresh = cls._RE_STR_MIN_THRESH.search(no_rem)
        min_str_thresh = int(m_min_thresh.group(1)) if m_min_thresh else 0

        # A. Reanimation, Song Replay & Free Cost-1 Character Deployment (Circle of Life, Max Goof, Pocahontas)
        if "play a character from your discard for free" in lower and "that player" not in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.REANIMATE_CHAR, amount=1)
            )
        if "play a song card with cost 4 or less from your discard for free" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_QUEST, op=EffectOpType.REPLAY_SONG_FROM_DISCARD, amount=1, threshold=4)
            )
        if "play a character with cost 1 for free" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.PLAY_COST1_CHAR_FREE, amount=1)
            )
        if "you may play characters from your discard" in lower and "that player" not in lower:
            # Remember Me (Set 14 Song)
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.REANIMATE_CHAR, amount=1)
            )
        if "if any of the cards that were under them are character cards" in lower and "play those characters from your discard for free" in lower:
            # Sulley & Boo - Scare Buddies (Set 14 Combo Shift)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_LEAVE_PLAY, op=EffectOpType.REANIMATE_CHAR, amount=1, requires_card_under=True)
            )
        if "play a character with cost up to 2 more than the banished character for free" in lower:
            # Retro Evolution Device (Set 14 Item)
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ACTIVATED,
                    op=EffectOpType.RETRO_EVOLVE,
                    amount=2,
                    cost_ink=1,
                    requires_exert=True,
                )
            )
        if "put a character card from your discard under this item" in lower:
            # The Black Cauldron (Set 14 Item)
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ACTIVATED,
                    op=EffectOpType.PUT_UNDER_FROM_DISCARD,
                    amount=1,
                    cost_ink=1,
                    requires_exert=True,
                )
            )
        if "put 2 character cards from your discard on the bottom of your deck to give this character rush" in lower:
            # Roller Bob - Sid's Toy (Set 12)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.GIVE_SELF_RUSH, amount=1)
            )
        if "move up to 1 damage from chosen character of yours to this character" in lower:
            # Luisa Madrigal - Confident Climber (Set 13)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ACTIVATED, op=EffectOpType.MOVE_DAMAGE, amount=1, cost_ink=1)
            )
        if "put chosen item into its player's inkwell" in lower:
            # Judy Hopps - On the Case (Set 13)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.INK_ITEM, amount=1)
            )
        if "banish chosen item" in lower:
            # Aladdin - Doing His Part (Set 12)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.BANISH_ITEM, amount=1, cost_ink=1)
            )
        if "give chosen character adventurous" in lower:
            # Nick Wilde - Inquisitive Harbormaster (Set 12)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_QUEST, op=EffectOpType.GIVE_ADVENTUROUS, amount=1)
            )
        if "chosen gargoyle character gains rush" in lower:
            # Demona - Imperious Spellcaster (Set 10)
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ACTIVATED, op=EffectOpType.GIVE_RUSH_EVASIVE, amount=1)
            )
        if "characters can't challenge until the start of your next turn" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.FOG_CHALLENGES, amount=1)
            )
        if "each player reveals the top card of their deck" in lower and "character card" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.REVEAL_TOP_CHAR_BOTH, amount=1)
            )
            if "whenever he quests" in lower or "whenever this character quests" in lower:
                abilities.clauses.append(
                    EffectClause(trigger=TriggerType.ON_QUEST, op=EffectOpType.REVEAL_TOP_CHAR_BOTH, amount=1)
                )

        # B. Board Wipes & Targeted Removal (Prince Phillip - Vanquisher of Foes, Under the Sea, Judy Hopps - Always Vigilant)
        if "banish all opposing damaged characters" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.BANISH_ALL_DAMAGED_OPP, amount=1)
            )
        elif "banish all characters" in lower or "put all opposing characters" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ON_PLAY,
                    op=EffectOpType.BANISH_ALL_CHARS,
                    amount=1,
                    threshold=str_thresh,
                )
            )
        elif (
            ("banish chosen character" in lower and "of yours" not in lower)
            or "banish chosen opposing character" in lower
            or "shuffle chosen opposing character into their player's deck" in lower
            or ("put chosen opposing character" in lower and "deck" in lower)
        ):
            op_t = EffectOpType.BANISH_LOW_STR_CHAR if str_thresh > 0 else EffectOpType.BANISH_CHAR
            eff_thresh = str_thresh if str_thresh > 0 else (-min_str_thresh if min_str_thresh > 0 else 0)
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=op_t,
                    amount=1,
                    threshold=eff_thresh,
                    cost_ink=default_cost_ink,
                    requires_exert=default_exert,
                    requires_card_under=needs_card_under,
                )
            )

        # C. Damage & Damage Counters (Malicious, Mean, and Scary; To Wither a Flower; Don Karnage)
        m_put_dmg = cls._RE_PUT_DMG_COUNTER.search(no_rem)
        m_dmg = cls._RE_DAMAGE.search(no_rem)
        if m_put_dmg:
            raw_d = m_put_dmg.group(1).lower()
            dmg_val = 1 if raw_d == "a" else int(raw_d)
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.DEAL_DAMAGE_ALL_OPP, amount=dmg_val)
            )
        elif m_dmg:
            dmg = int(m_dmg.group(1))
            if "damage to each opposing damaged character" in lower:
                abilities.clauses.append(
                    EffectClause(
                        trigger=default_trig,
                        op=EffectOpType.DEAL_DAMAGE_ALL_DAMAGED_OPP,
                        amount=dmg,
                        requires_card_under=needs_card_under,
                    )
                )
            elif "each opposing character" in lower:
                abilities.clauses.append(
                    EffectClause(
                        trigger=default_trig,
                        op=EffectOpType.DEAL_DAMAGE_ALL_OPP,
                        amount=dmg,
                        requires_card_under=needs_card_under,
                    )
                )
            else:
                abilities.clauses.append(
                    EffectClause(
                        trigger=default_trig,
                        op=EffectOpType.DEAL_DAMAGE,
                        amount=dmg,
                        requires_card_under=needs_card_under,
                    )
                )
        elif "equal to the number of characters" in lower and "damage" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.DEAL_DAMAGE, amount=4)
            )

        # D. Next-Character Ink Discount (Akood et Emuti, Aurora - Holding Court, Grandmother Willow, Pluto)
        m_disc_char = cls._RE_DISCOUNT_CHAR.search(no_rem)
        if m_disc_char:
            disc_amt = int(m_disc_char.group(1))
            disc_op = (
                EffectOpType.DISCOUNT_NEXT_PRINCESS_QUEEN
                if ("princess or queen character" in lower)
                else EffectOpType.DISCOUNT_NEXT_CHAR
            )
            if "⟳" in no_rem:
                abilities.clauses.append(
                    EffectClause(
                        trigger=TriggerType.ACTIVATED,
                        op=disc_op,
                        amount=disc_amt,
                        requires_exert=True,
                    )
                )
            else:
                abilities.clauses.append(
                    EffectClause(
                        trigger=default_trig,
                        op=disc_op,
                        amount=disc_amt,
                    )
                )
                if "once during your turn, you pay 1" in lower:
                    abilities.clauses.append(
                        EffectClause(
                            trigger=TriggerType.START_OF_TURN,
                            op=disc_op,
                            amount=disc_amt,
                        )
                    )

        # E. Self-Looting vs Pure Draw vs Ink-Type Draw vs Song/Character Recursion & Healing
        is_self_loot = (
            ("then choose and discard" in lower)
            or ("choose and discard a song card. if you do, draw" in lower)
        )
        m_draw = cls._RE_DRAW.search(no_rem)
        draw_trig = (
            TriggerType.ON_QUEST
            if ("whenever this character quests, draw" in lower)
            else default_trig
        )
        if "remove all damage from chosen character of yours" in lower and "draw a card for each" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.HEAL_DRAW, amount=99)
            )
        elif "remove up to 2 damage from chosen character" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.HEAL_DRAW, amount=0)
            )
        elif "draws a card for each different ink type" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.DRAW_PER_INK_TYPE, amount=2)
            )
        elif is_self_loot and m_draw:
            raw_num = m_draw.group(1).lower()
            cnt = cls._WORD_TO_INT.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.LOOT_SELF, amount=cnt)
            )
            if "at the end of your turn, if you played 2 or more characters this turn, draw a card" in lower:
                abilities.clauses.append(
                    EffectClause(trigger=TriggerType.END_OF_TURN, op=EffectOpType.DRAW, amount=1)
                )
        elif m_draw:
            raw_num = m_draw.group(1).lower()
            cnt = cls._WORD_TO_INT.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
            abilities.clauses.append(
                EffectClause(
                    trigger=draw_trig,
                    op=EffectOpType.DRAW,
                    amount=cnt,
                    cost_ink=default_cost_ink,
                    requires_exert=default_exert,
                    requires_card_under=needs_card_under,
                )
            )
        elif (
            ("look at the top" in lower and "put" in lower and "hand" in lower)
            or ("return" in lower and "from your discard to your hand" in lower)
            or ("draw that number of cards plus 1" in lower)
            or ("draws until they have 3" in lower)
        ):
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.DRAW,
                    amount=1,
                    cost_ink=default_cost_ink,
                    requires_exert=default_exert,
                )
            )
            if "when you play this character and whenever she quests" in lower:
                abilities.clauses.append(
                    EffectClause(trigger=TriggerType.ON_QUEST, op=EffectOpType.DRAW, amount=1)
                )
        if "put the top 3 cards of your deck into your discard" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.BUFF_ALLY_STR, amount=3)
            )

        # F. Ink Drop Counters (Rule 1.13.1.2) vs Opponent Inkwell Removal vs Standard Inkwell Ramp
        if "each player gets 1 ink drop" in lower or "each get 1 ink drop" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.EACH_PLAYER_INK_DROP, amount=1)
            )
        elif "ink drop" in lower and ("gets 1 ink drop" in lower or "get 1 ink drop" in lower or "gain 1 ink drop" in lower):
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.GAIN_INK_DROP, amount=1)
            )
        elif "put chosen opposing character" in lower and "into their player's inkwell" in lower:
            max_c = 2 if "cost 2 or less" in lower else 99
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.INK_OPP_CHAR, amount=1, threshold=max_c)
            )
        elif "into your inkwell" in lower and "whenever one of your characters is banished" not in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.RAMP_INK, amount=1)
            )

        # F2. Exert Opposing Character(s) (Isis Vanderchill, Peter Pan's Shadow, Demona - Scourge of the Wyvern Clan)
        if "exert all opposing characters" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.EXERT_OPP_CHAR, amount=99)
            )
        elif "exert chosen opposing character" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.EXERT_OPP_CHAR, amount=1)
            )

        # G. Lore Gain / Loss (including Lionheart +1 Lore to 2 chars / Pocahontas Finding the Way +1 Lore to 1 char)
        m_lore = cls._RE_GAIN_LORE.search(no_rem)
        if m_lore:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.GAIN_LORE,
                    amount=int(m_lore.group(1)),
                    cost_ink=default_cost_ink,
                    requires_exert=default_exert,
                    requires_card_under=needs_card_under,
                )
            )
        elif "up to 2 chosen characters get +1 ◊ this turn" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.BUFF_ALLY_LORE, amount=2)
            )
        elif "chosen character gets +1 ◊ this turn" in lower and "pay" not in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_PLAY, op=EffectOpType.BUFF_ALLY_LORE, amount=1)
            )

        m_lose_lore = cls._RE_LOSE_LORE.search(no_rem)
        if m_lose_lore:
            lose_trig = (
                TriggerType.END_OF_TURN
                if "at the end of your turn" in lower
                else default_trig
            )
            abilities.clauses.append(
                EffectClause(
                    trigger=lose_trig,
                    op=EffectOpType.LOSE_OPP_LORE,
                    amount=int(m_lose_lore.group(1)),
                    requires_card_under=needs_card_under,
                )
            )

        # H. Bounce (Nani - Stage Manager self-bounce vs Milo Thatch, Finnick, Rapunzel, Mother Knows Best opp-bounce)
        if "return another chosen character of yours" in lower or "return chosen character of yours" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.BOUNCE_OWN_CHAR, amount=1)
            )
        elif "return chosen character" in lower or "return chosen opposing character" in lower:
            pay_2 = 2 if "you may pay 2" in lower else 0
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.BOUNCE_OPP_CHAR,
                    amount=1,
                    threshold=str_thresh,
                    cost_ink=pay_2,
                )
            )
            if "if 2 or more cards were put into your discard this turn, draw a card" in lower:
                abilities.clauses.append(
                    EffectClause(trigger=TriggerType.END_OF_TURN, op=EffectOpType.DRAW, amount=1)
                )

        # I. Opponent Discard (only when targeting an opponent!)
        if (
            "opponent reveals their hand and discards" in lower
            or "opponent chooses and discards" in lower
            or "each opponent chooses and discards" in lower
            or ("chooses and discards" in lower and not is_self_loot)
        ):
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.DISCARD_OPP,
                    amount=1,
                    optional="you may" in lower and "opponent" not in lower.split("you may")[0][-20:],
                    choice_kind=ChoiceKind.NONE,
                )
            )

        # J. Ready / Heal / Support / Look / Under-card / Evasive-self (wave families)
        cls._compile_ready_heal_support(abilities, no_rem, lower, default_trig, default_exert, default_cost_ink)
        cls._compile_aura_statics(abilities, text, no_rem, lower)
        cls._compile_conditional_grants(abilities, text, no_rem, lower)
        cls._compile_wave_families(abilities, no_rem, lower, default_trig, default_exert, default_cost_ink)
        cls._annotate_choice_nodes(abilities, lower)

        return abilities

    @classmethod
    def _compile_wave_families(
        cls,
        abilities: CompiledAbilities,
        no_rem: str,
        lower: str,
        default_trig: TriggerType,
        default_exert: bool,
        default_cost_ink: int,
    ) -> None:
        """Additional Sets 9–14 mechanic families toward near-zero empty non-reminder text."""

        # Self heal at start of turn (Pluto - Determined Defender)
        m_self_heal = re.search(r"at the start of your turn, remove up to (\d+) damage from this character", lower)
        if m_self_heal:
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.START_OF_TURN,
                    op=EffectOpType.HEAL,
                    amount=int(m_self_heal.group(1)),
                    choice_kind=ChoiceKind.NONE,
                    target_filter="self",
                )
            )

        # Activated heal chosen (Cinderella - Gentle and Kind)
        m_act_heal = re.search(r"⟳\s*[—\-]\s*remove up to (\d+) damage from chosen", lower)
        if m_act_heal and not any(c.op == EffectOpType.HEAL and c.trigger == TriggerType.ACTIVATED for c in abilities.clauses):
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ACTIVATED,
                    op=EffectOpType.HEAL,
                    amount=int(m_act_heal.group(1)),
                    requires_exert=True,
                    optional=False,
                    choice_kind=ChoiceKind.TARGET_ALLY_CHAR,
                    classification="Princess" if "princess" in lower else "",
                )
            )

        # Banish chosen exerted with strength threshold (Minnie Sweetheart)
        m_banish_ex = re.search(
            r"banish chosen exerted character with (\d+)\s*(?:¤|\{S\}|strength)\s+or more",
            lower,
        )
        if m_banish_ex:
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ON_QUEST if "whenever this character quests" in lower else default_trig,
                    op=EffectOpType.BANISH_CHAR,
                    amount=1,
                    threshold=-int(m_banish_ex.group(1)),  # negative => min strength (see BANISH_CHAR)
                    optional="you may" in lower,
                    choice_kind=ChoiceKind.TARGET_OPP_CHAR,
                    target_filter="exerted",
                )
            )

        # Ready your other exerted [Class] characters (Moana)
        if "ready your other exerted" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ON_QUEST if "whenever this character quests" in lower else default_trig,
                    op=EffectOpType.READY_ALL_CHARS,
                    amount=1,
                    optional="you may" in lower,
                    classification="Princess" if "princess" in lower else "",
                )
            )

        # Board-wide opposing strength debuff (Lost in the Woods)
        m_all_debuff = re.search(
            r"all opposing characters get -(\d+)\s*(?:¤|\{S\}|strength)",
            lower,
        )
        if m_all_debuff:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.DEBUFF_OPP_STR,
                    amount=int(m_all_debuff.group(1)),
                    threshold=99,  # apply to all
                )
            )

        # Move damage counters (Belle / Mama Odie)
        m_move = re.search(r"move up to (\d+) damage counters? from chosen character to chosen opposing", lower)
        if m_move and not any(c.op == EffectOpType.MOVE_DAMAGE for c in abilities.clauses):
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.MOVE_DAMAGE,
                    amount=int(m_move.group(1)),
                    optional="you may" in lower,
                    choice_kind=ChoiceKind.TARGET_ANY_CHAR,
                )
            )

        # Exert up to N chosen characters (Elsa Spirit of Winter)
        m_exert_n = re.search(r"exert up to (\d+) chosen characters?", lower)
        if m_exert_n:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.EXERT_OPP_CHAR,
                    amount=int(m_exert_n.group(1)),
                    choice_kind=ChoiceKind.TARGET_ANY_CHAR,
                )
            )

        # Chosen character gains Evasive (Dumbo)
        if "chosen character gains evasive" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.GAIN_EVASIVE_SELF,  # applied to chosen ally via choice path
                    amount=1,
                    choice_kind=ChoiceKind.TARGET_ALLY_CHAR,
                )
            )

        # Named-character Support grant as aura (Minnie → Mickey Mouse)
        m_named = re.search(r"your characters named ([^.]+?) gain support", lower)
        if m_named:
            abilities.aura_statics.append(
                AuraStatic(
                    grant_keywords=KeywordFlag.SUPPORT,
                    filter_kind=AuraFilter.CLASSIFICATION,  # reuse filter_value as name match tag
                    filter_value="name:" + m_named.group(1).strip().title(),
                    exclude_self=False,
                )
            )

        # Self +N strength per card in hand (Jafar Keeper)
        m_hand_str = re.search(r"this character gets \+(\d+)\s*(?:¤|\{S\}|strength) for each card in your hand", lower)
        if m_hand_str:
            abilities.conditional_grants.append(
                ConditionalGrant(strength=int(m_hand_str.group(1)), condition="per_card_in_hand")
            )

        # Your other characters with Rush gain Evasive (Peter Pan's Shadow)
        if "your other characters with rush gain evasive" in lower:
            abilities.aura_statics.append(
                AuraStatic(
                    grant_keywords=KeywordFlag.EVASIVE,
                    filter_kind=AuraFilter.CLASSIFICATION,
                    filter_value="keyword:RUSH",
                    exclude_self=True,
                )
            )

        # Reveal top / put character in hand (Pongo) — approximate as LOOK_AT_TOP
        if "reveal the top card of your deck" in lower and "character card" in lower:
            if not any(c.op == EffectOpType.LOOK_AT_TOP for c in abilities.clauses):
                abilities.clauses.append(
                    EffectClause(
                        trigger=TriggerType.ACTIVATED if "once during your turn" in lower else default_trig,
                        op=EffectOpType.LOOK_AT_TOP,
                        amount=1,
                        cost_ink=2 if "pay 2" in lower else default_cost_ink,
                        optional="you may" in lower,
                        choice_kind=ChoiceKind.OPTIONAL_YES_NO,
                    )
                )

        # Can't ready at start of next turn — mark unsupported residual with reason if only that
        if "can't ready at the start of their next turn" in lower and not abilities.clauses:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.EXERT_OPP_CHAR,
                    amount=1,
                    choice_kind=ChoiceKind.TARGET_OPP_CHAR,
                    unsupported_reason="persist-can't-ready approximated as exert",
                )
            )
            abilities.unsupported_notes.append("can't-ready-next-turn approximated as exert")

        # While this character has no damage → self statics
        m_no_dmg = re.search(
            r"while this character has no damage[^.]*gets?\s+([^.]{3,80})",
            lower,
        )
        if m_no_dmg or "while this character has no damage" in lower:
            body = m_no_dmg.group(1) if m_no_dmg else lower
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(body)
            if "resist +" in body or "gains resist" in body:
                m_r = re.search(r"resist\s*\+(\d+)", body)
                if m_r:
                    res_a = max(res_a, int(m_r.group(1)))
            if str_a or will_a or lore_a or res_a:
                abilities.aura_statics.append(
                    AuraStatic(
                        strength=str_a,
                        willpower=will_a,
                        lore=lore_a,
                        resist=res_a,
                        challenger=chal_a,
                        filter_kind=AuraFilter.SELF,
                        while_condition="no_damage",
                        exclude_self=False,
                    )
                )

        # While this character has damage
        if "while this character has damage" in lower:
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(
                lower.split("while this character has damage", 1)[1][:80]
            )
            if str_a or will_a or lore_a:
                abilities.aura_statics.append(
                    AuraStatic(
                        strength=str_a,
                        willpower=will_a,
                        lore=lore_a,
                        filter_kind=AuraFilter.SELF,
                        while_condition="has_damage",
                        exclude_self=False,
                    )
                )

        # Location "while here" auras
        if "while here" in lower:
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(lower)
            grant_kw = KeywordFlag.NONE
            if re.search(r"gain evasive", lower):
                grant_kw |= KeywordFlag.EVASIVE
            if str_a or will_a or lore_a or res_a or chal_a or grant_kw:
                abilities.aura_statics.append(
                    AuraStatic(
                        strength=str_a,
                        willpower=will_a,
                        lore=lore_a,
                        resist=res_a,
                        challenger=chal_a,
                        grant_keywords=grant_kw,
                        filter_kind=AuraFilter.ALL,
                        while_condition="at_this_location",
                        exclude_self=False,
                    )
                )

        # Enters play with N damage
        m_enter_dmg = re.search(r"enters play with (\d+) damage", lower)
        if m_enter_dmg:
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ON_PLAY,
                    op=EffectOpType.HEAL,  # reused: amount negative via threshold hack
                    amount=0,
                    threshold=int(m_enter_dmg.group(1)),
                    target_filter="enter_damage",
                    unsupported_reason="",
                )
            )

        # Named character resist/bodyguard grants
        m_named_resist = re.search(
            r"your characters named ([^.]+?) gain resist\s*\+(\d+)",
            lower,
        )
        if m_named_resist:
            abilities.aura_statics.append(
                AuraStatic(
                    resist=int(m_named_resist.group(2)),
                    filter_kind=AuraFilter.CLASSIFICATION,
                    filter_value="name:" + m_named_resist.group(1).strip().title(),
                    exclude_self=False,
                )
            )
        m_named_bg = re.search(r"your characters named ([^.]+?) gain bodyguard", lower)
        if m_named_bg:
            abilities.aura_statics.append(
                AuraStatic(
                    grant_keywords=KeywordFlag.BODYGUARD,
                    filter_kind=AuraFilter.CLASSIFICATION,
                    filter_value="name:" + m_named_bg.group(1).strip().title(),
                    exclude_self=False,
                )
            )

        # Self +N lore/str per location / per opp hand
        m_per_loc = re.search(
            r"this character gets \+(\d+)\s*(?:◊|\{L\}|lore) for each location",
            lower,
        )
        if m_per_loc:
            abilities.conditional_grants.append(
                ConditionalGrant(lore=int(m_per_loc.group(1)), condition="per_location")
            )
        m_per_opp_hand = re.search(
            r"this character gets \+(\d+)\s*(?:¤|\{S\}|strength) for each card in opponents'? hands?",
            lower,
        )
        if m_per_opp_hand:
            abilities.conditional_grants.append(
                ConditionalGrant(strength=int(m_per_opp_hand.group(1)), condition="per_opp_hand")
            )

        # During your turn self +str (Pacha)
        m_turn_str = re.search(
            r"during your turn, this character gets \+(\d+)\s*(?:¤|\{S\}|strength)",
            lower,
        )
        if m_turn_str:
            abilities.aura_statics.append(
                AuraStatic(
                    strength=int(m_turn_str.group(1)),
                    filter_kind=AuraFilter.SELF,
                    while_your_turn=True,
                    exclude_self=False,
                )
            )

    @classmethod
    def _parse_stat_amounts(cls, fragment: str) -> tuple[int, int, int, int, int]:
        """Return (str, will, lore, resist, challenger) deltas from a text fragment."""
        strength = willpower = lore = resist = challenger = 0
        for m in re.finditer(r"\+(\d+)\s*(?:¤|\{S\}|strength)", fragment, re.I):
            strength += int(m.group(1))
        for m in re.finditer(r"\+(\d+)\s*(?:⛉|\{W\}|willpower)", fragment, re.I):
            willpower += int(m.group(1))
        for m in re.finditer(r"\+(\d+)\s*(?:◊|\{L\}|lore)", fragment, re.I):
            lore += int(m.group(1))
        # Generic "+1 ¤ and +1 ⛉" already covered; also bare "+1 ◊ this turn"
        for m in re.finditer(r"resist\s*\+(\d+)", fragment, re.I):
            resist += int(m.group(1))
        for m in re.finditer(r"challenger\s*\+(\d+)", fragment, re.I):
            challenger += int(m.group(1))
        return strength, willpower, lore, resist, challenger

    @classmethod
    def _compile_ready_heal_support(
        cls,
        abilities: CompiledAbilities,
        no_rem: str,
        lower: str,
        default_trig: TriggerType,
        default_exert: bool,
        default_cost_ink: int,
    ) -> None:
        if "ready all your characters" in lower or "ready all characters" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.READY_ALL_CHARS,
                    amount=1,
                    cost_ink=default_cost_ink,
                    requires_exert=default_exert,
                )
            )
        elif "ready chosen character" in lower or "ready another chosen character" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.READY_CHAR,
                    amount=1,
                    cost_ink=default_cost_ink,
                    requires_exert=default_exert,
                    optional="you may" in lower,
                    choice_kind=ChoiceKind.TARGET_ALLY_CHAR,
                )
            )

        m_heal = re.search(r"remove up to (\d+) damage from chosen character", lower)
        m_heal_all = re.search(r"remove all damage from chosen character", lower)
        if m_heal_all and "draw" not in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.HEAL,
                    amount=99,
                    optional="you may" in lower,
                    choice_kind=ChoiceKind.TARGET_ALLY_CHAR,
                )
            )
        elif m_heal and not any(c.op == EffectOpType.HEAL_DRAW for c in abilities.clauses):
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.HEAL,
                    amount=int(m_heal.group(1)),
                    optional="you may" in lower,
                    choice_kind=ChoiceKind.TARGET_ANY_CHAR,
                )
            )

        # Support (CR 8.13): resolved at quest-time as a choice node in the simulator
        # (not a DSL clause — avoids double-fire with ON_QUEST). Marked via keyword flag.

        if "look at the top" in lower and not any(c.op == EffectOpType.LOOK_AT_TOP for c in abilities.clauses):
            m_n = re.search(r"look at the top (\d+|card)", lower)
            n = 1
            if m_n and m_n.group(1).isdigit():
                n = int(m_n.group(1))
            # Prefer LOOK_AT_TOP over generic DRAW if we only had the look/put heuristic
            abilities.clauses = [
                c for c in abilities.clauses
                if not (c.op == EffectOpType.DRAW and c.amount == 1 and "look at the top" in lower)
            ]
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.LOOK_AT_TOP,
                    amount=n,
                    optional="you may" in lower,
                    choice_kind=ChoiceKind.OPTIONAL_YES_NO if "you may" in lower else ChoiceKind.NONE,
                )
            )

        if "put the top card of your deck facedown under" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=TriggerType.ON_QUEST if "whenever this character quests" in lower else default_trig,
                    op=EffectOpType.PUT_CARD_UNDER_SELF,
                    amount=1,
                )
            )

        if "during your turn, this character gains evasive" in lower:
            abilities.aura_statics.append(
                AuraStatic(
                    grant_keywords=KeywordFlag.EVASIVE,
                    filter_kind=AuraFilter.SELF,
                    while_your_turn=True,
                    exclude_self=False,
                )
            )

    @classmethod
    def _compile_aura_statics(
        cls,
        abilities: CompiledAbilities,
        text: str,
        no_rem: str,
        lower: str,
    ) -> None:
        # Continuous auras: "Your [other] [Class|Ink] characters get/gain ..."
        # Skip quest-triggered "get +X this turn" — those are temp buffs via clauses.
        aura_pat = re.compile(
            r"your\s+(other\s+)?(?:(\w+(?:\s+\w+)?)\s+)?"
            r"characters?\s+"
            r"(get|gain|gains)\s+([^.]{3,120})",
            re.I,
        )
        for m in aura_pat.finditer(no_rem):
            frag = m.group(0)
            frag_l = frag.lower()
            if "this turn" in frag_l and ("whenever" in lower or "when you play" in lower):
                # Temporary triggered buff — compile as clause when possible
                str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(frag)
                trig = TriggerType.ON_QUEST if "whenever this character quests" in lower else TriggerType.ON_PLAY
                if lore_a and not any(c.op == EffectOpType.BUFF_ALLY_LORE and c.trigger == trig for c in abilities.clauses):
                    abilities.clauses.append(
                        EffectClause(
                            trigger=trig,
                            op=EffectOpType.BUFF_ALLY_LORE,
                            amount=lore_a,
                            classification=(m.group(2) or "").strip(),
                            choice_kind=ChoiceKind.NONE,
                        )
                    )
                if str_a and not any(c.op == EffectOpType.BUFF_ALLY_STR and c.trigger == trig for c in abilities.clauses):
                    abilities.clauses.append(
                        EffectClause(
                            trigger=trig,
                            op=EffectOpType.BUFF_ALLY_STR,
                            amount=str_a,
                            classification=(m.group(2) or "").strip(),
                        )
                    )
                continue

            other = bool(m.group(1))
            class_or_ink = (m.group(2) or "").strip()
            verb = m.group(3).lower()
            rest = m.group(4)
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(rest)
            grant_kw = KeywordFlag.NONE
            if re.search(r"\bward\b", rest, re.I):
                grant_kw |= KeywordFlag.WARD
            if re.search(r"\bsupport\b", rest, re.I):
                grant_kw |= KeywordFlag.SUPPORT
            if re.search(r"\bevasive\b", rest, re.I):
                grant_kw |= KeywordFlag.EVASIVE
            if re.search(r"\balert\b", rest, re.I):
                grant_kw |= KeywordFlag.ALERT
            if re.search(r"\brush\b", rest, re.I):
                grant_kw |= KeywordFlag.RUSH

            filter_kind = AuraFilter.ALL_OTHER if other else AuraFilter.ALL
            filter_value = ""
            ink_names = {"Amber", "Amethyst", "Emerald", "Ruby", "Sapphire", "Steel"}
            if class_or_ink:
                titled = " ".join(w.capitalize() for w in class_or_ink.split())
                if titled in ink_names:
                    filter_kind = AuraFilter.INK
                    filter_value = titled
                elif class_or_ink.lower() == "other":
                    filter_kind = AuraFilter.ALL_OTHER
                else:
                    filter_kind = AuraFilter.CLASSIFICATION
                    filter_value = titled

            # Genie-style: aura only while exerted when the while-clause scopes this fragment
            while_exerted = False
            if "while this character is exerted" in lower:
                m_ex = re.search(
                    r"while this character is exerted[^.]*your other characters (?:get|gain)",
                    lower,
                    re.I,
                )
                while_exerted = bool(m_ex) and (
                    "your other characters get" in frag_l or "your other characters gain" in frag_l
                )

            if str_a or will_a or lore_a or res_a or chal_a or grant_kw:
                abilities.aura_statics.append(
                    AuraStatic(
                        strength=str_a,
                        willpower=will_a,
                        lore=lore_a,
                        resist=res_a,
                        challenger=chal_a,
                        grant_keywords=grant_kw,
                        filter_kind=filter_kind,
                        filter_value=filter_value,
                        while_exerted=while_exerted and "while this character is exerted" in lower,
                        while_your_turn="during your turn" in lower and "your" in frag_l,
                        exclude_self=other or filter_kind != AuraFilter.ALL,
                    )
                )

        # "Your characters with Support get +1 ◊"
        if "your characters with support get" in lower:
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(
                lower.split("your characters with support get", 1)[1][:40]
            )
            abilities.aura_statics.append(
                AuraStatic(
                    strength=str_a,
                    willpower=will_a,
                    lore=lore_a or 1,
                    resist=res_a,
                    challenger=chal_a,
                    filter_kind=AuraFilter.HAS_SUPPORT,
                    exclude_self=False,
                )
            )

        # Self statics: "this character gets +X while ..."
        self_pat = re.compile(
            r"while ([^.]{5,100}), this (?:character|location) gets?\s+([^.]{3,80})",
            re.I,
        )
        for m in self_pat.finditer(no_rem):
            cond = m.group(1).lower()
            rest = m.group(2)
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(rest)
            cond_tag = ""
            if "card under" in cond:
                cond_tag = "card_under"
            elif "no cards in your hand" in cond:
                cond_tag = "hand_empty"
            elif "opponent has more lore" in cond:
                cond_tag = "lore_behind"
            elif "villain character in play" in cond:
                cond_tag = "villain_in_play"
            elif "10 or more cards in your discard" in cond:
                cond_tag = "discard_ge_10"
            elif "exerted" in cond:
                cond_tag = "self_exerted"
            else:
                cond_tag = "misc:" + cond[:40]
            if str_a or will_a or lore_a or res_a:
                abilities.aura_statics.append(
                    AuraStatic(
                        strength=str_a,
                        willpower=will_a,
                        lore=lore_a,
                        resist=res_a,
                        challenger=chal_a,
                        filter_kind=AuraFilter.SELF,
                        while_condition=cond_tag,
                        exclude_self=False,
                    )
                )
            if "resist +" in rest.lower():
                m_r = re.search(r"resist\s*\+(\d+)", rest, re.I)
                if m_r:
                    abilities.conditional_grants.append(
                        ConditionalGrant(resist=int(m_r.group(1)), condition=cond_tag)
                    )

        # Debuff auras on opposing damaged while self exerted (Karnage)
        if "each opposing damaged character gets -1" in lower:
            abilities.aura_statics.append(
                AuraStatic(
                    lore=-1,
                    filter_kind=AuraFilter.ALL,
                    while_exerted=True,
                    while_condition="opp_damaged_debuff",
                    exclude_self=True,
                )
            )

    @classmethod
    def _compile_conditional_grants(
        cls,
        abilities: CompiledAbilities,
        text: str,
        no_rem: str,
        lower: str,
    ) -> None:
        # Mickey Mouse - Amber Champion: Singer 8 while 2+ other Amber
        if "gains singer" in lower and "other amber" in lower:
            m = re.search(r"gains singer (\d+)", lower)
            if m:
                abilities.conditional_grants.append(
                    ConditionalGrant(grant_singer=int(m.group(1)), condition="other_amber_ge_2")
                )
        # Hand-empty resist
        if "while you have no cards in your hand" in lower and "resist" in lower:
            m = re.search(r"resist\s*\+(\d+)", lower)
            if m:
                abilities.conditional_grants.append(
                    ConditionalGrant(resist=int(m.group(1)), condition="hand_empty")
                )
        # Card-under keyword grants
        if "while there's a card under this character" in lower:
            kw = KeywordFlag.NONE
            if "gains evasive" in lower:
                kw |= KeywordFlag.EVASIVE
            str_a, will_a, lore_a, res_a, chal_a = (0, 0, 0, 0, 0)
            # Parse "+N" after the while clause
            m_body = re.search(
                r"while there's a card under this character[^.]*\.\s*([^.]*)",
                no_rem,
                re.I,
            )
            body = m_body.group(1) if m_body else no_rem
            str_a, will_a, lore_a, res_a, chal_a = cls._parse_stat_amounts(body)
            if "resist +" in lower:
                m_r = re.search(r"resist\s*\+(\d+)", lower)
                if m_r:
                    res_a = max(res_a, int(m_r.group(1)))
            abilities.conditional_grants.append(
                ConditionalGrant(
                    grant_keywords=kw,
                    strength=str_a,
                    willpower=will_a,
                    lore=lore_a,
                    resist=res_a,
                    condition="card_under",
                )
            )

    @classmethod
    def _annotate_choice_nodes(cls, abilities: CompiledAbilities, lower: str) -> None:
        """Stamp choice_kind / optional on clauses that need player decisions."""
        annotated: list[EffectClause] = []
        for c in abilities.clauses:
            if c.choice_kind != ChoiceKind.NONE or c.optional:
                annotated.append(c)
                continue
            opt = c.optional
            kind = c.choice_kind
            # Infer from op
            if c.op in (
                EffectOpType.BANISH_CHAR,
                EffectOpType.BANISH_LOW_STR_CHAR,
                EffectOpType.DEAL_DAMAGE,
                EffectOpType.EXERT_OPP_CHAR,
                EffectOpType.BOUNCE_OPP_CHAR,
                EffectOpType.DEBUFF_OPP_STR,
                EffectOpType.INK_OPP_CHAR,
            ):
                kind = ChoiceKind.TARGET_OPP_CHAR
            elif c.op in (
                EffectOpType.BOUNCE_OWN_CHAR,
                EffectOpType.READY_CHAR,
                EffectOpType.BUFF_ALLY_STR,
                EffectOpType.BUFF_ALLY_LORE,
                EffectOpType.BUFF_ALLY_WILLPOWER,
                EffectOpType.MOVE_DAMAGE,
                EffectOpType.GIVE_ADVENTUROUS,
                EffectOpType.GIVE_RUSH_EVASIVE,
            ):
                kind = ChoiceKind.TARGET_ALLY_CHAR
            elif c.op in (EffectOpType.HEAL, EffectOpType.HEAL_DRAW):
                kind = ChoiceKind.TARGET_ANY_CHAR
            elif c.op in (EffectOpType.BANISH_ITEM, EffectOpType.INK_ITEM):
                kind = ChoiceKind.TARGET_ANY_ITEM
            elif c.op == EffectOpType.REANIMATE_CHAR:
                kind = ChoiceKind.REANIMATE_DISCARD
            elif c.op == EffectOpType.SUPPORT_BUFF:
                kind = ChoiceKind.SUPPORT_TARGET
                opt = True
            if "you may" in lower and kind != ChoiceKind.NONE:
                opt = True
            if kind != c.choice_kind or opt != c.optional:
                annotated.append(
                    EffectClause(
                        trigger=c.trigger,
                        op=c.op,
                        amount=c.amount,
                        threshold=c.threshold,
                        cost_ink=c.cost_ink,
                        requires_exert=c.requires_exert,
                        requires_card_under=c.requires_card_under,
                        optional=opt,
                        choice_kind=kind,
                        target_filter=c.target_filter,
                        classification=c.classification,
                        unsupported_reason=c.unsupported_reason,
                    )
                )
            else:
                annotated.append(c)
        abilities.clauses = annotated
