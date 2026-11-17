"""
Full GameState representation for Disney Lorcana (Season 14 Core Constructed: Sets 9-14),
compliant with Comprehensive Rules v2.3.0 (Effective October 16, 2026):
- Rule 8.16 (Adventurous: cannot challenge, must quest before end of turn if ready and able)
- Rule 1.13.1.2 (Ink Drop Counters: removable counters that pay ink costs)
- Rule 4.7 & 5.6 (Locations & Moving Characters to Locations)
- Rule 8.10.8 (All 6 Shift Variants: Classification, Universal, Duo, Combo, Temporary, Potato)
- Imperfect Information Determinization for IS-MCTS
"""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Sequence

from lorcana_ai.data.card_db import CardTemplate
from lorcana_ai.data.effect_dsl import AuraFilter, AuraStatic, ConditionalGrant, KeywordFlag


@dataclass
class BoardEntity:
    """An active Character, Item, or Location on the board."""
    instance_id: int
    card: CardTemplate
    exerted: bool = False
    drying: bool = True
    damage: int = 0
    temp_strength_mod: int = 0
    temp_challenger_mod: int = 0
    temp_lore_mod: int = 0
    temp_willpower_mod: int = 0
    cards_under_count: int = 0
    boosted_this_turn: bool = False
    temp_evasive: bool = False
    temp_adventurous: bool = False
    temp_cant_quest: bool = False
    shifted_under_card: CardTemplate | None = None
    shifted_under_cards: list[CardTemplate] = field(default_factory=list)
    location_instance_id: int = -1

    def _condition_met(
        self,
        condition: str,
        owner: PlayerState | None,
        opp: PlayerState | None,
        source: BoardEntity | None = None,
    ) -> bool:
        src = source or self
        if not condition:
            return True
        if condition == "card_under":
            return src.cards_under_count > 0
        if condition == "hand_empty":
            return owner is not None and len(owner.hand) == 0
        if condition == "lore_behind":
            return owner is not None and opp is not None and opp.lore > owner.lore
        if condition == "villain_in_play":
            return owner is not None and any("Villain" in e.card.subtypes for e in owner.characters_in_play)
        if condition == "discard_ge_10":
            return owner is not None and len(owner.discard) >= 10
        if condition == "self_exerted":
            return src.exerted
        if condition == "other_amber_ge_2":
            if owner is None:
                return False
            return (
                sum(
                    1
                    for e in owner.characters_in_play
                    if e.instance_id != src.instance_id and "Amber" in e.card.inks
                )
                >= 2
            )
        if condition == "opp_damaged_debuff":
            return True
        if condition == "per_card_in_hand":
            return owner is not None and len(owner.hand) > 0
        if condition == "no_damage":
            return self.damage == 0
        if condition == "has_damage":
            return self.damage > 0
        if condition == "per_location":
            return owner is not None and len(owner.locations_in_play) > 0
        if condition == "per_opp_hand":
            return opp is not None and len(opp.hand) > 0
        if condition == "at_this_location":
            # Applied only when character.location_instance_id matches aura source
            return True
        if condition.startswith("misc:"):
            return False
        return False

    def _aura_targets_self(self, aura: AuraStatic, source: BoardEntity, owner: PlayerState) -> bool:
        if aura.filter_kind == AuraFilter.SELF:
            return source.instance_id == self.instance_id
        if aura.exclude_self and source.instance_id == self.instance_id:
            return False
        if aura.filter_kind == AuraFilter.HAS_SUPPORT:
            return self.has_keyword(KeywordFlag.SUPPORT, owner=owner, check_auras=False)
        if aura.filter_kind == AuraFilter.INK:
            return aura.filter_value in self.card.inks
        if aura.filter_kind == AuraFilter.CLASSIFICATION:
            fv = aura.filter_value
            if fv.startswith("name:"):
                return self.card.base_name.lower() == fv[5:].lower()
            if fv.startswith("keyword:"):
                try:
                    flag = KeywordFlag[fv.split(":", 1)[1].upper()]
                except KeyError:
                    return False
                return self.has_keyword(flag, owner=owner, check_auras=False)
            return any(s.lower() == fv.lower() for s in self.card.subtypes)
        if aura.filter_kind in (AuraFilter.ALL, AuraFilter.ALL_OTHER):
            return True
        return False

    def _iter_relevant_auras(
        self,
        owner: PlayerState | None,
        opp: PlayerState | None,
        *,
        for_opp_debuff: bool = False,
    ):
        if owner is None:
            return
        # Own auras
        for src in owner.board:
            for aura in src.card.abilities.aura_statics:
                if aura.while_condition == "opp_damaged_debuff":
                    continue
                if aura.while_exerted and not src.exerted:
                    continue
                if aura.while_your_turn and owner is None:
                    continue
                if aura.while_condition == "at_this_location":
                    if src.card.card_type != "Location" or self.location_instance_id != src.instance_id:
                        continue
                elif aura.while_condition and not self._condition_met(aura.while_condition, owner, opp, source=src):
                    continue
                if self._aura_targets_self(aura, src, owner):
                    yield aura
            for grant in src.card.abilities.conditional_grants:
                if src.instance_id != self.instance_id:
                    continue
                if self._condition_met(grant.condition, owner, opp, source=src):
                    yield grant
        # Opposing debuff auras (Karnage-style)
        if for_opp_debuff and opp is not None and self.damage > 0:
            for src in opp.characters_in_play:
                for aura in src.card.abilities.aura_statics:
                    if aura.while_condition != "opp_damaged_debuff":
                        continue
                    if aura.while_exerted and not src.exerted:
                        continue
                    yield aura

    def _stat_from_auras(
        self,
        owner: PlayerState | None,
        opp: PlayerState | None,
        attr: str,
    ) -> int:
        total = 0
        for item in self._iter_relevant_auras(owner, opp, for_opp_debuff=(attr == "lore")):
            if isinstance(item, AuraStatic):
                total += getattr(item, attr, 0)
            elif isinstance(item, ConditionalGrant):
                amt = getattr(item, attr, 0)
                if item.condition == "per_card_in_hand" and owner is not None:
                    total += amt * len(owner.hand)
                elif item.condition == "per_location" and owner is not None:
                    total += amt * len(owner.locations_in_play)
                elif item.condition == "per_opp_hand" and opp is not None:
                    total += amt * len(opp.hand)
                else:
                    total += amt
        return total

    @property
    def effective_strength(self) -> int:
        # Without owner context, only self conditional grants with card_under
        val = self.card.strength + self.temp_strength_mod
        for grant in self.card.abilities.conditional_grants:
            if grant.condition == "card_under" and self.cards_under_count > 0:
                val += grant.strength
        return max(0, val)

    def effective_strength_with(self, owner: PlayerState | None = None, opp: PlayerState | None = None) -> int:
        val = self.card.strength + self.temp_strength_mod
        val += self._stat_from_auras(owner, opp, "strength")
        if owner is None:
            for grant in self.card.abilities.conditional_grants:
                if grant.condition == "card_under" and self.cards_under_count > 0:
                    val += grant.strength
        return max(0, val)

    @property
    def effective_willpower(self) -> int:
        val = self.card.willpower + self.temp_willpower_mod
        for grant in self.card.abilities.conditional_grants:
            if grant.condition == "card_under" and self.cards_under_count > 0:
                val += grant.willpower
        return max(1, val)

    def effective_willpower_with(self, owner: PlayerState | None = None, opp: PlayerState | None = None) -> int:
        val = self.card.willpower + self.temp_willpower_mod
        val += self._stat_from_auras(owner, opp, "willpower")
        if owner is None:
            for grant in self.card.abilities.conditional_grants:
                if grant.condition == "card_under" and self.cards_under_count > 0:
                    val += grant.willpower
        return max(1, val)

    @property
    def is_banished(self) -> bool:
        return self.damage >= self.effective_willpower

    def effective_resist(self, owner: PlayerState | None = None, opp: PlayerState | None = None) -> int:
        base = self.card.abilities.resist_bonus
        base += self._stat_from_auras(owner, opp, "resist")
        if owner is None:
            for grant in self.card.abilities.conditional_grants:
                if grant.condition == "card_under" and self.cards_under_count > 0:
                    base += grant.resist
        # Detective Lead lateral-thinking aura via compiled auras (Alert+Resist on Detectives)
        if owner is not None and "Detective" in self.card.subtypes:
            for src in owner.characters_in_play:
                for aura in src.card.abilities.aura_statics:
                    if aura.filter_kind == AuraFilter.CLASSIFICATION and aura.filter_value.lower() == "detective":
                        if aura.while_your_turn:
                            base += aura.resist
        return max(0, base)

    def has_keyword(
        self,
        kw: KeywordFlag | str,
        owner: PlayerState | None = None,
        opp: PlayerState | None = None,
        check_auras: bool = True,
    ) -> bool:
        if isinstance(kw, str):
            try:
                flag = KeywordFlag[kw.upper()]
            except KeyError:
                return False
        else:
            flag = kw

        if flag == KeywordFlag.EVASIVE and self.temp_evasive:
            return True
        if flag == KeywordFlag.ADVENTUROUS and self.temp_adventurous:
            return True
        if bool(self.card.abilities.keywords & flag):
            return True
        for grant in self.card.abilities.conditional_grants:
            if grant.grant_keywords & flag:
                if self._condition_met(grant.condition, owner, opp):
                    return True
        if check_auras and owner is not None:
            for item in self._iter_relevant_auras(owner, opp):
                if isinstance(item, AuraStatic) and (item.grant_keywords & flag):
                    return True
        return False

    def effective_lore(self, owner: PlayerState | None = None, opp: PlayerState | None = None) -> int:
        base = self.card.lore + self.temp_lore_mod
        base += self._stat_from_auras(owner, opp, "lore")
        if owner is not None and self.card.abilities.has_five_songs_lore_bonus:
            songs_in_discard = sum(1 for c in owner.discard if c.is_song)
            if songs_in_discard >= 5:
                base += 3
        # Villain-in-play strength bump is separate; lore from compiled auras covers Diablo etc.
        if owner is None:
            for grant in self.card.abilities.conditional_grants:
                if grant.condition == "card_under" and self.cards_under_count > 0:
                    base += grant.lore
        return max(0, base)

    def can_quest(self) -> bool:
        if self.card.card_type != "Character":
            return False
        if self.exerted or self.drying:
            return False
        if self.temp_cant_quest:
            return False
        # CR Rule 8.7.2: Reckless character can't quest
        if self.has_keyword(KeywordFlag.RECKLESS):
            return False
        return (self.card.lore + self.temp_lore_mod) > 0

    def can_challenge(self) -> bool:
        if self.card.card_type != "Character":
            return False
        if self.exerted:
            return False
        # CR Rule 8.16.2: Adventurous ("This character can't challenge.") or static "This character can't challenge."
        if self.has_keyword(KeywordFlag.ADVENTUROUS):
            return False
        if "this character can't challenge" in self.card.full_text.lower() and "unless" not in self.card.full_text.lower():
            return False
        if self.drying and not self.has_keyword(KeywordFlag.RUSH):
            return False
        return True

    def effective_singer_cost(self, owner: PlayerState | None = None) -> int:
        """Printed Singer X, plus conditional Singer grants (e.g. Mickey Mouse - Amber Champion)."""
        singer_val = self.card.abilities.singer_cost or self.card.cost
        for grant in self.card.abilities.conditional_grants:
            if grant.grant_singer is not None and self._condition_met(grant.condition, owner, None):
                singer_val = max(singer_val, grant.grant_singer)
        return singer_val

    def can_sing(self, song_cost: int, owner: PlayerState | None = None) -> bool:
        if self.card.card_type != "Character" or self.exerted or self.drying:
            return False
        return self.effective_singer_cost(owner) >= song_cost

    def clone(self) -> BoardEntity:
        return BoardEntity(
            instance_id=self.instance_id,
            card=self.card,
            exerted=self.exerted,
            drying=self.drying,
            damage=self.damage,
            temp_strength_mod=self.temp_strength_mod,
            temp_challenger_mod=self.temp_challenger_mod,
            temp_lore_mod=self.temp_lore_mod,
            temp_willpower_mod=self.temp_willpower_mod,
            cards_under_count=self.cards_under_count,
            boosted_this_turn=self.boosted_this_turn,
            temp_evasive=self.temp_evasive,
            temp_adventurous=self.temp_adventurous,
            temp_cant_quest=self.temp_cant_quest,
            shifted_under_card=self.shifted_under_card,
            shifted_under_cards=list(self.shifted_under_cards),
            location_instance_id=self.location_instance_id,
        )


@dataclass
class PlayerState:
    """State of one player in a Lorcana match."""
    player_id: int
    lore: int = 0
    inkwell_total: int = 0
    inkwell_available: int = 0
    ink_drops: int = 0  # CR Rule 1.13.1.2: Ink Drop Counters
    next_char_discount: int = 0  # Discount for next character played this turn (e.g. Grandmother Willow, Pluto, Akood et Emuti)
    next_princess_queen_discount: int = 0  # Discount for next Princess or Queen character played this turn (Aurora - Holding Court)
    inked_this_turn: bool = False
    challenged_this_turn: bool = False
    cant_challenge_this_turn: bool = False
    cant_play_actions_this_turn: bool = False
    chars_played_this_turn: int = 0
    princess_played_this_turn: bool = False
    songs_played_this_turn: int = 0
    cards_discarded_this_turn: int = 0
    hand: list[CardTemplate] = field(default_factory=list)
    deck: list[CardTemplate] = field(default_factory=list)
    discard: list[CardTemplate] = field(default_factory=list)
    inkwell_cards: list[CardTemplate] = field(default_factory=list)
    board: list[BoardEntity] = field(default_factory=list)
    revealed_hand_ids: list[int] = field(default_factory=list)

    @property
    def total_payable_ink(self) -> int:
        """Total ink cost payable using ready inkwell cards + Ink Drop counters (CR Rule 1.13.1.2)."""
        return self.inkwell_available + self.ink_drops

    @staticmethod
    def is_princess_or_queen(card: CardTemplate) -> bool:
        return "Princess" in card.subtypes or "Queen" in card.subtypes

    def effective_play_cost(
        self,
        card: CardTemplate,
        is_second_player_first_turn: bool = False,
        opp: PlayerState | None = None,
    ) -> int:
        """
        Computes effective ink cost of playing `card` after Underdog, general next-character discount,
        Princess/Queen discount (Aurora - Holding Court), and Sets 9-14 conditional cost reductions.
        """
        cost = card.cost
        if card.card_type == "Character":
            cost -= self.next_char_discount
            if self.is_princess_or_queen(card):
                cost -= self.next_princess_queen_discount
            if card.abilities.has_underdog and is_second_player_first_turn:
                cost -= 1
            txt_lower = card.full_text.lower()
            if "for each action card in your discard, you pay 1" in txt_lower:
                cost -= sum(1 for c in self.discard if c.card_type == "Action")
            if "for each toy character card in your discard, you pay 1" in txt_lower:
                cost -= sum(1 for c in self.discard if "Toy" in c.subtypes)
            if "if you have a character named tod in play, you pay 1" in txt_lower:
                if any(e.card.base_name == "Tod" for e in self.characters_in_play):
                    cost -= 1
            if "if you have a character named meilin lee in play, you pay 1" in txt_lower:
                if any(e.card.base_name == "Meilin Lee" for e in self.characters_in_play):
                    cost -= 1
            if "if you played a song this turn, you pay 1" in txt_lower and self.songs_played_this_turn > 0:
                cost -= 1
            if (
                ("for each card in opponents' inkwells" in txt_lower or "for each card in an opponent's inkwell" in txt_lower)
                and opp is not None
            ):
                cost -= opp.inkwell_total
            if "for each damaged character in play, you pay 1" in txt_lower:
                dmg_cnt = sum(1 for e in self.characters_in_play if e.damage > 0)
                if opp is not None:
                    dmg_cnt += sum(1 for e in opp.characters_in_play if e.damage > 0)
                cost -= dmg_cnt
        return max(0, cost)

    def effective_shift_cost(self, card: CardTemplate) -> int:
        """Computes effective ink cost of Shifting `card` after applicable character/Princess/Queen discounts."""
        base_shift = card.abilities.shift_cost if card.abilities.shift_cost is not None else card.cost
        if card.card_type == "Character":
            base_shift -= self.next_char_discount
            if self.is_princess_or_queen(card):
                base_shift -= self.next_princess_queen_discount
        return max(0, base_shift)

    def consume_character_discounts(self, card: CardTemplate) -> None:
        """Consumes active next-character discounts when a Character `card` is played or Shifted."""
        if card.card_type == "Character":
            self.next_char_discount = 0
            if self.is_princess_or_queen(card):
                self.next_princess_queen_discount = 0

    def pay_ink(self, cost: int) -> None:
        """
        Spends ready inkwell cards first, then removes Ink Drop counters for any remaining cost
        (CR Rule 1.13.1.2).
        """
        if cost <= 0:
            return
        from_inkwell = min(self.inkwell_available, cost)
        self.inkwell_available -= from_inkwell
        rem = cost - from_inkwell
        if rem > 0:
            self.ink_drops = max(0, self.ink_drops - rem)

    @property
    def characters_in_play(self) -> list[BoardEntity]:
        return [e for e in self.board if e.card.card_type == "Character"]

    @property
    def items_in_play(self) -> list[BoardEntity]:
        return [e for e in self.board if e.card.card_type == "Item"]

    @property
    def locations_in_play(self) -> list[BoardEntity]:
        return [e for e in self.board if e.card.card_type == "Location"]

    def clone(self) -> PlayerState:
        return PlayerState(
            player_id=self.player_id,
            lore=self.lore,
            inkwell_total=self.inkwell_total,
            inkwell_available=self.inkwell_available,
            ink_drops=self.ink_drops,
            next_char_discount=self.next_char_discount,
            next_princess_queen_discount=self.next_princess_queen_discount,
            inked_this_turn=self.inked_this_turn,
            challenged_this_turn=self.challenged_this_turn,
            cant_challenge_this_turn=self.cant_challenge_this_turn,
            cant_play_actions_this_turn=self.cant_play_actions_this_turn,
            chars_played_this_turn=self.chars_played_this_turn,
            princess_played_this_turn=self.princess_played_this_turn,
            songs_played_this_turn=self.songs_played_this_turn,
            cards_discarded_this_turn=self.cards_discarded_this_turn,
            hand=list(self.hand),
            deck=list(self.deck),
            discard=list(self.discard),
            inkwell_cards=list(self.inkwell_cards),
            board=[e.clone() for e in self.board],
            revealed_hand_ids=list(self.revealed_hand_ids),
        )


@dataclass
class PendingChoice:
    """In-flight player choice from an effect clause (you may / chosen target / Support)."""
    controller_id: int
    choice_kind: str
    source_instance_id: int = -1
    clause_op: str = ""
    amount: int = 0
    optional: bool = False
    candidate_ids: tuple[int, ...] = ()  # board instance ids or hand/discard indices encoded
    prompt: str = ""


@dataclass
class GameState:
    """
    Complete state of a 1v1 Disney Lorcana game.
    Supports both Game 1 (blind mulligan) and Game 2/3 (known opponent archetype).
    """
    players: tuple[PlayerState, PlayerState]
    active_player: int = 0
    starting_player: int = 0
    turn_number: int = 1
    phase: str = "MULLIGAN_P0"  # "MULLIGAN_P0", "MULLIGAN_P1", "MAIN", "GAME_OVER", "CHOICE"
    winner: int | None = None
    victory_reason: str = ""
    next_instance_id: int = 1
    known_opponent_inks: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    pending_choice: PendingChoice | None = None
    phase_before_choice: str = "MAIN"

    @classmethod
    def create_new_game(
        cls,
        deck0_cards: Sequence[CardTemplate],
        deck1_cards: Sequence[CardTemplate],
        starting_player: int = 0,
        know_opponent_inks: bool = False,
        seed: int | None = None,
    ) -> GameState:
        rng = random.Random(seed)
        d0 = list(deck0_cards)
        d1 = list(deck1_cards)
        rng.shuffle(d0)
        rng.shuffle(d1)

        h0 = [d0.pop() for _ in range(min(7, len(d0)))]
        h1 = [d1.pop() for _ in range(min(7, len(d1)))]

        p0 = PlayerState(player_id=0, hand=h0, deck=d0)
        p1 = PlayerState(player_id=1, hand=h1, deck=d1)

        inks0 = tuple(sorted({ink for c in deck0_cards for ink in c.inks})) if know_opponent_inks else ()
        inks1 = tuple(sorted({ink for c in deck1_cards for ink in c.inks})) if know_opponent_inks else ()

        first_phase = f"MULLIGAN_P{starting_player}"
        return cls(
            players=(p0, p1),
            active_player=starting_player,
            starting_player=starting_player,
            turn_number=1,
            phase=first_phase,
            known_opponent_inks=(inks1, inks0),
        )

    def clone(self) -> GameState:
        pc = None
        if self.pending_choice is not None:
            pc = PendingChoice(
                controller_id=self.pending_choice.controller_id,
                choice_kind=self.pending_choice.choice_kind,
                source_instance_id=self.pending_choice.source_instance_id,
                clause_op=self.pending_choice.clause_op,
                amount=self.pending_choice.amount,
                optional=self.pending_choice.optional,
                candidate_ids=self.pending_choice.candidate_ids,
                prompt=self.pending_choice.prompt,
            )
        return GameState(
            players=(self.players[0].clone(), self.players[1].clone()),
            active_player=self.active_player,
            starting_player=self.starting_player,
            turn_number=self.turn_number,
            phase=self.phase,
            winner=self.winner,
            victory_reason=self.victory_reason,
            next_instance_id=self.next_instance_id,
            known_opponent_inks=self.known_opponent_inks,
            pending_choice=pc,
            phase_before_choice=self.phase_before_choice,
        )

    def clone_and_determinize(
        self,
        observer_id: int,
        belief_card_weights: dict[int, float] | None = None,
        rng: random.Random | None = None,
    ) -> GameState:
        rng = rng or random.Random()
        det = self.clone()
        opp_id = 1 - observer_id
        opp = det.players[opp_id]

        hidden_pool = list(opp.hand) + list(opp.deck)
        if not hidden_pool:
            return det

        locked_hand: list[CardTemplate] = []
        remaining_pool = list(hidden_pool)
        for rev_id in opp.revealed_hand_ids:
            for idx, c in enumerate(remaining_pool):
                if c.card_id == rev_id:
                    locked_hand.append(remaining_pool.pop(idx))
                    break

        needed_in_hand = max(0, len(opp.hand) - len(locked_hand))
        if belief_card_weights and needed_in_hand > 0 and len(remaining_pool) > needed_in_hand:
            weights = [max(0.05, belief_card_weights.get(c.card_id, 1.0)) for c in remaining_pool]
            sampled_hand: list[CardTemplate] = []
            for _ in range(needed_in_hand):
                chosen_idx = rng.choices(range(len(remaining_pool)), weights=weights, k=1)[0]
                sampled_hand.append(remaining_pool.pop(chosen_idx))
                weights.pop(chosen_idx)
            opp.hand = locked_hand + sampled_hand
            rng.shuffle(remaining_pool)
            opp.deck = remaining_pool
        else:
            rng.shuffle(remaining_pool)
            opp.hand = locked_hand + remaining_pool[:needed_in_hand]
            opp.deck = remaining_pool[needed_in_hand:]

        rng.shuffle(det.players[observer_id].deck)
        return det
