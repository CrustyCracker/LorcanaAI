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
    DEBUFF_OPP_STR = "DEBUFF_OPP_STR"
    HEAL_DRAW = "HEAL_DRAW"
    READY_CHAR = "READY_CHAR"
    BOOST_UNDER = "BOOST_UNDER"


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
    clauses: list[EffectClause] = field(default_factory=list)

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
    _RE_DRAW = re.compile(r"draw\s+(\d+|a|one|two|three|four)\s+cards?", re.IGNORECASE)
    _RE_DAMAGE = re.compile(r"deal\s+(\d+)\s+damage", re.IGNORECASE)
    _RE_PUT_DMG_COUNTER = re.compile(r"put\s+(\d+|a)\s+damage\s+counters?\s+on\s+each\s+opposing", re.IGNORECASE)
    _RE_DISCOUNT_CHAR = re.compile(r"you\s+pay\s+(\d+)\s*⬡?\s*less\s+for\s+the\s+next\s+[\w\s]*character", re.IGNORECASE)
    _RE_GAIN_LORE = re.compile(r"gain\s+(\d+)\s+lore", re.IGNORECASE)
    _RE_LOSE_LORE = re.compile(r"loses?\s+(\d+)\s+lore", re.IGNORECASE)
    _RE_STR_THRESH = re.compile(r"(\d+)\s*(?:\{S\}|¤|Strength)\s+or\s+less", re.IGNORECASE)

    _WORD_TO_INT = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4}

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

        # 1. Parse Keywords (CR Section 8)
        if "rush" in kw_set or re.search(r"\bRush\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.RUSH
        if "ward" in kw_set or re.search(r"\bWard\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.WARD
        if "evasive" in kw_set or re.search(r"\bEvasive\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.EVASIVE
        if "bodyguard" in kw_set or re.search(r"\bBodyguard\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.BODYGUARD
        if "support" in kw_set or re.search(r"\bSupport\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.SUPPORT
        if "vanish" in kw_set or re.search(r"\bVanish\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.VANISH
        if "reckless" in kw_set or re.search(r"\bReckless\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.RECKLESS
        if "alert" in kw_set or re.search(r"\bAlert\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.ALERT
        if "adventurous" in kw_set or re.search(r"\bAdventurous\b", text, re.IGNORECASE):
            abilities.keywords |= KeywordFlag.ADVENTUROUS

        # Underdog & 5+ Songs in discard Lore bonus (Max Goof - Karaoke Star)
        if "if this is your first turn and you're not the first player, you pay 1" in lower:
            abilities.has_underdog = True
        if "5 or more song cards in your discard, this character gets +3" in lower:
            abilities.has_five_songs_lore_bonus = True

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

        m_singer = cls._RE_SINGER.search(text)
        if m_singer:
            abilities.singer_cost = int(m_singer.group(1))
        elif card_type == "Character":
            abilities.singer_cost = cost

        # Boost X
        m_boost = cls._RE_BOOST.search(text)
        if m_boost:
            abilities.boost_cost = int(m_boost.group(1))

        m_chal = cls._RE_CHALLENGER.search(text)
        if m_chal:
            abilities.challenger_bonus = int(m_chal.group(1))

        m_res = cls._RE_RESIST.search(text)
        if m_res:
            abilities.resist_bonus = int(m_res.group(1))

        if explicit_clauses:
            abilities.clauses.extend(explicit_clauses)
            return abilities

        # 3. Automatic DSL extraction from LorcanaJSON card rules text
        needs_card_under = ("card under" in lower) or ("cards under" in lower)

        default_trig = (
            TriggerType.ON_PLAY
            if card_type in ("Action", "Character", "Item")
            else TriggerType.START_OF_TURN
        )
        if "whenever this character quests" in lower and "when you play" not in lower:
            default_trig = TriggerType.ON_QUEST
        elif "when this character is banished" in lower or "leaves play" in lower:
            default_trig = TriggerType.ON_LEAVE_PLAY
        elif "at the end of your turn" in lower and "when you play" not in lower:
            default_trig = TriggerType.END_OF_TURN
        elif "at the start of your turn" in lower:
            default_trig = TriggerType.START_OF_TURN

        m_thresh = cls._RE_STR_THRESH.search(text)
        str_thresh = int(m_thresh.group(1)) if m_thresh else 0

        # A. Reanimation & Song Replay (Circle of Life, Max Goof - Chart Topper)
        if "play a character from your discard for free" in lower and "that player" not in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.REANIMATE_CHAR, amount=1)
            )
        if "play a song card with cost 4 or less from your discard for free" in lower:
            abilities.clauses.append(
                EffectClause(trigger=TriggerType.ON_QUEST, op=EffectOpType.REPLAY_SONG_FROM_DISCARD, amount=1, threshold=4)
            )

        # B. Board Wipes & Targeted Removal (Prince Phillip - Vanquisher of Foes, Under the Sea, etc.)
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
            "banish chosen character" in lower
            or "banish chosen opposing character" in lower
            or "shuffle chosen opposing character into their player's deck" in lower
        ):
            op_t = EffectOpType.BANISH_LOW_STR_CHAR if str_thresh > 0 else EffectOpType.BANISH_CHAR
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=op_t,
                    amount=1,
                    threshold=str_thresh,
                    requires_card_under=needs_card_under,
                )
            )

        # C. Damage & Damage Counters (Malicious, Mean, and Scary; To Wither a Flower; Don Karnage)
        m_put_dmg = cls._RE_PUT_DMG_COUNTER.search(text)
        m_dmg = cls._RE_DAMAGE.search(text)
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

        # D. Next-Character Ink Discount (Akood et Emuti, Aurora - Holding Court, etc.)
        m_disc_char = cls._RE_DISCOUNT_CHAR.search(text)
        if m_disc_char:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.DISCOUNT_NEXT_CHAR,
                    amount=int(m_disc_char.group(1)),
                )
            )

        # E. Self-Looting vs Pure Draw vs Ink-Type Draw vs Song Recursion
        is_self_loot = (
            ("then choose and discard" in lower)
            or ("choose and discard a song card. if you do, draw" in lower)
        )
        m_draw = cls._RE_DRAW.search(text)
        if "draws a card for each different ink type" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.DRAW_PER_INK_TYPE, amount=2)
            )
        elif is_self_loot and m_draw:
            raw_num = m_draw.group(1).lower()
            cnt = cls._WORD_TO_INT.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.LOOT_SELF, amount=cnt)
            )
        elif m_draw:
            raw_num = m_draw.group(1).lower()
            cnt = cls._WORD_TO_INT.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.DRAW,
                    amount=cnt,
                    requires_card_under=needs_card_under,
                )
            )
        elif (
            ("look at the top" in lower and "put" in lower and "hand" in lower)
            or ("return a song card" in lower and "from your discard to your hand" in lower)
        ):
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.DRAW, amount=1)
            )

        # F. Ink Drop Counters (Rule 1.13.1.2) vs Standard Inkwell Ramp
        if "each player gets 1 ink drop" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.EACH_PLAYER_INK_DROP, amount=1)
            )
        elif "ink drop" in lower and ("gets 1 ink drop" in lower or "get 1 ink drop" in lower or "gain 1 ink drop" in lower):
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.GAIN_INK_DROP, amount=1)
            )
        elif "into your inkwell" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.RAMP_INK, amount=1)
            )

        # G. Lore Gain / Loss
        m_lore = cls._RE_GAIN_LORE.search(text)
        if m_lore:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.GAIN_LORE,
                    amount=int(m_lore.group(1)),
                    requires_card_under=needs_card_under,
                )
            )

        m_lose_lore = cls._RE_LOSE_LORE.search(text)
        if m_lose_lore:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.LOSE_OPP_LORE,
                    amount=int(m_lose_lore.group(1)),
                    requires_card_under=needs_card_under,
                )
            )

        # H. Bounce (Milo Thatch, Mother Knows Best, etc.)
        if "return chosen character" in lower or "return chosen opposing character" in lower:
            abilities.clauses.append(
                EffectClause(
                    trigger=default_trig,
                    op=EffectOpType.BOUNCE_OPP_CHAR,
                    amount=1,
                    threshold=str_thresh,
                )
            )
            if "if 2 or more cards were put into your discard this turn, draw a card" in lower:
                abilities.clauses.append(
                    EffectClause(trigger=TriggerType.END_OF_TURN, op=EffectOpType.DRAW, amount=1)
                )
        elif "return another chosen character of yours" in lower:
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.BOUNCE_OWN_CHAR, amount=1)
            )

        # I. Opponent Discard (only when targeting an opponent!)
        if (
            "opponent reveals their hand and discards" in lower
            or "opponent chooses and discards" in lower
            or "each opponent chooses and discards" in lower
            or ("chooses and discards" in lower and not is_self_loot)
        ):
            abilities.clauses.append(
                EffectClause(trigger=default_trig, op=EffectOpType.DISCARD_OPP, amount=1)
            )

        return abilities
