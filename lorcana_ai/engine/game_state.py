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
from lorcana_ai.data.effect_dsl import KeywordFlag


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
    cards_under_count: int = 0
    boosted_this_turn: bool = False
    shifted_under_card: CardTemplate | None = None
    shifted_under_cards: list[CardTemplate] = field(default_factory=list)
    location_instance_id: int = -1

    @property
    def effective_strength(self) -> int:
        return max(0, self.card.strength + self.temp_strength_mod)

    @property
    def effective_willpower(self) -> int:
        return max(1, self.card.willpower)

    @property
    def is_banished(self) -> bool:
        return self.damage >= self.effective_willpower

    def has_keyword(self, kw: KeywordFlag) -> bool:
        if bool(self.card.abilities.keywords & kw):
            return True
        # Ursula - Whisper of Vanessa (Set 10): "While there's a card under this character, she gets +1 ◊ and gains Evasive."
        if (
            kw == KeywordFlag.EVASIVE
            and self.cards_under_count > 0
            and "while there's a card under this character" in self.card.full_text.lower()
            and "gains evasive" in self.card.full_text.lower()
        ):
            return True
        return False

    def effective_lore(self, owner: PlayerState | None = None, opp: PlayerState | None = None) -> int:
        base = self.card.lore + self.temp_lore_mod
        txt_lower = self.card.full_text.lower()
        if owner is not None:
            if self.card.abilities.has_five_songs_lore_bonus:
                songs_in_discard = sum(1 for c in owner.discard if c.is_song)
                if songs_in_discard >= 5:
                    base += 3
            # Ursula - Whisper of Vanessa (Set 10): +1 ◊ while card under
            if self.cards_under_count > 0 and "while there's a card under this character, she gets +1" in txt_lower:
                base += 1
            # Diablo - Stone Servant (Set 12): +2 ¤ and +1 ◊ while you have a Villain character in play
            if "while you have a villain character in play" in txt_lower:
                if any("Villain" in e.card.subtypes for e in owner.characters_in_play):
                    base += 1
            # Land of the Dead - Marigold Bridge (Set 14): +2 ◊ while 10+ cards in discard
            if "while you have 10 or more cards in your discard, this location gets +2" in txt_lower:
                if len(owner.discard) >= 10:
                    base += 2
            # Gadget Hackwrench - Resourceful Mechanic (Set 12): "Your characters with Support get +1 ◊."
            if self.has_keyword(KeywordFlag.SUPPORT):
                gadget_cnt = sum(
                    1 for e in owner.characters_in_play if "your characters with support get +1" in e.card.full_text.lower()
                )
                base += gadget_cnt
        if opp is not None:
            # The Queen - Devious Disguise (Set 12): "While an opponent has more lore than you, this character gets +2 ◊."
            if owner is not None and "while an opponent has more lore than you, this character gets +2" in txt_lower:
                if opp.lore > owner.lore:
                    base += 2
            # Don Karnage - Debonair Pirate (Set 14): "While this character is exerted, each opposing damaged character gets -1 ◊."
            if self.damage > 0:
                karnage_debuff = sum(
                    1
                    for e in opp.characters_in_play
                    if e.exerted and "each opposing damaged character gets -1" in e.card.full_text.lower()
                )
                base -= karnage_debuff
        return max(0, base)

    def can_quest(self) -> bool:
        if self.card.card_type != "Character":
            return False
        if self.exerted or self.drying:
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

    def can_sing(self, song_cost: int) -> bool:
        if self.card.card_type != "Character" or self.exerted or self.drying:
            return False
        singer_val = self.card.abilities.singer_cost or self.card.cost
        return singer_val >= song_cost

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
            cards_under_count=self.cards_under_count,
            boosted_this_turn=self.boosted_this_turn,
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
class GameState:
    """
    Complete state of a 1v1 Disney Lorcana game.
    Supports both Game 1 (blind mulligan) and Game 2/3 (known opponent archetype).
    """
    players: tuple[PlayerState, PlayerState]
    active_player: int = 0
    starting_player: int = 0
    turn_number: int = 1
    phase: str = "MULLIGAN_P0"  # "MULLIGAN_P0", "MULLIGAN_P1", "MAIN", "GAME_OVER"
    winner: int | None = None
    victory_reason: str = ""
    next_instance_id: int = 1
    known_opponent_inks: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())

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
