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
        return bool(self.card.abilities.keywords & kw)

    def effective_lore(self, owner: PlayerState | None = None) -> int:
        base = self.card.lore
        if owner is not None and self.card.abilities.has_five_songs_lore_bonus:
            songs_in_discard = sum(1 for c in owner.discard if c.is_song)
            if songs_in_discard >= 5:
                base += 3
        return max(0, base)

    def can_quest(self) -> bool:
        if self.card.card_type != "Character":
            return False
        if self.exerted or self.drying:
            return False
        # CR Rule 8.7.2: Reckless character can't quest
        if self.has_keyword(KeywordFlag.RECKLESS):
            return False
        return self.card.lore > 0

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
    next_char_discount: int = 0  # Discount for next character played this turn (e.g. Akood et Emuti)
    inked_this_turn: bool = False
    challenged_this_turn: bool = False
    cant_challenge_this_turn: bool = False
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

    def effective_play_cost(self, card: CardTemplate, is_second_player_first_turn: bool = False) -> int:
        """Computes effective ink cost of playing `card` after Underdog and next-character discounts."""
        cost = card.cost
        if card.card_type == "Character":
            cost -= self.next_char_discount
            if card.abilities.has_underdog and is_second_player_first_turn:
                cost -= 1
        return max(0, cost)

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
            inked_this_turn=self.inked_this_turn,
            challenged_this_turn=self.challenged_this_turn,
            cant_challenge_this_turn=self.cant_challenge_this_turn,
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
