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
    DEBUFF_OPP_STR = "DEBUFF_OPP_STR"
    HEAL_DRAW = "HEAL_DRAW"
    READY_CHAR = "READY_CHAR"
    BOOST_UNDER = "BOOST_UNDER"
    PLAY_COST1_CHAR_FREE = "PLAY_COST1_CHAR_FREE"
    FOG_CHALLENGES = "FOG_CHALLENGES"
    REVEAL_TOP_CHAR_BOTH = "REVEAL_TOP_CHAR_BOTH"
    DISCOUNT_NEXT_PRINCESS_QUEEN = "DISCOUNT_NEXT_PRINCESS_QUEEN"
    EXERT_OPP_CHAR = "EXERT_OPP_CHAR"
    INK_OPP_CHAR = "INK_OPP_CHAR"


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
    deals_willpower_damage_in_challenges: bool = False
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
    _RE_STR_MIN_THRESH = re.compile(r"(\d+)\s*(?:\{S\}|¤|Strength)\s+or\s+more", re.IGNORECASE)

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
                EffectClause(trigger=default_trig, op=EffectOpType.DISCARD_OPP, amount=1)
            )

        return abilities
