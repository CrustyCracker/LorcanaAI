"""
Neural State & Action Mask Vectorizer for LorcanaAlpha.
Encodes imperfect-information GameState into compact tensors for the Transformer Policy-Value
and Opponent Belief Networks on NVIDIA A100 GPUs.
"""

from __future__ import annotations
import numpy as np
import torch

from lorcana_ai.config import EngineConfig, NetworkConfig
from lorcana_ai.data.card_db import INK_COLORS
from lorcana_ai.data.effect_dsl import KeywordFlag
from lorcana_ai.engine.game_state import BoardEntity, GameState
from lorcana_ai.engine.move_generator import GameAction

# 15 canonical 2-ink combinations in Lorcana
INK_PAIRS: list[tuple[str, str]] = [
    (INK_COLORS[i], INK_COLORS[j])
    for i in range(len(INK_COLORS))
    for j in range(i + 1, len(INK_COLORS))
]


class StateVectorizer:
    """
    Converts a GameState (from observer_id's perspective) into numerical feature arrays:
    - global_features: (24,) float32 vector (lore, ink, hand/deck/discard sizes, turn, play/draw, known opp inks)
    - hand_features: (MAX_HAND, 14) float32 tensor of observer's hand cards
    - my_board_features: (MAX_BOARD, 16) float32 tensor of observer's board entities
    - opp_board_features: (MAX_BOARD, 16) float32 tensor of opponent's board entities
    - opp_public_ink_counts: (6,) float32 vector of opponent's observed inks in play/discard/inkwell
    """

    GLOBAL_DIM = 24
    CARD_FEAT_DIM = 16
    MAX_HAND = 12
    MAX_BOARD = 10

    @classmethod
    def encode_state(cls, state: GameState, observer_id: int | None = None) -> dict[str, np.ndarray]:
        obs = state.active_player if observer_id is None else observer_id
        opp = 1 - obs
        me_p = state.players[obs]
        opp_p = state.players[opp]

        # 1. Global Scalar Features (24,)
        g = np.zeros(cls.GLOBAL_DIM, dtype=np.float32)
        g[0] = me_p.lore / 20.0
        g[1] = opp_p.lore / 20.0
        g[2] = (me_p.lore - opp_p.lore) / 20.0
        g[3] = me_p.inkwell_total / 10.0
        g[4] = me_p.inkwell_available / 10.0
        g[5] = opp_p.inkwell_total / 10.0
        g[6] = 1.0 if me_p.inked_this_turn else 0.0
        g[7] = len(me_p.hand) / 10.0
        g[8] = len(opp_p.hand) / 10.0
        g[9] = len(me_p.deck) / 60.0
        g[10] = len(opp_p.deck) / 60.0
        g[11] = state.turn_number / 30.0
        g[12] = 1.0 if state.starting_player == obs else 0.0  # On the Play flag
        g[13] = 1.0 if state.phase.startswith("MULLIGAN") else 0.0

        # Observed opponent ink histogram (from opponent board, discard, and known Bo3 inks)
        opp_ink_hist = np.zeros(len(INK_COLORS), dtype=np.float32)
        for ent in opp_p.board:
            for ink_c in ent.card.inks:
                if ink_c in INK_COLORS:
                    opp_ink_hist[INK_COLORS.index(ink_c)] += 1.0
        for c in opp_p.discard:
            for ink_c in c.inks:
                if ink_c in INK_COLORS:
                    opp_ink_hist[INK_COLORS.index(ink_c)] += 1.0
        for ink_name in state.known_opponent_inks[obs]:
            if ink_name in INK_COLORS:
                opp_ink_hist[INK_COLORS.index(ink_name)] += 3.0

        g[14:20] = np.clip(opp_ink_hist / 10.0, 0.0, 1.0)
        g[20] = sum(e.card.lore for e in me_p.characters_in_play) / 10.0
        g[21] = sum(e.card.lore for e in opp_p.characters_in_play) / 10.0
        g[22] = sum(e.effective_strength for e in me_p.characters_in_play) / 20.0
        g[23] = sum(e.effective_strength for e in opp_p.characters_in_play) / 20.0

        # 2. Hand Card Features (MAX_HAND, 16)
        hand_arr = np.zeros((cls.MAX_HAND, cls.CARD_FEAT_DIM), dtype=np.float32)
        for idx, card in enumerate(me_p.hand[: cls.MAX_HAND]):
            hand_arr[idx] = cls._encode_card_template(card)

        # 3. My Board Features (MAX_BOARD, 16)
        my_board_arr = np.zeros((cls.MAX_BOARD, cls.CARD_FEAT_DIM), dtype=np.float32)
        for idx, ent in enumerate(me_p.board[: cls.MAX_BOARD]):
            my_board_arr[idx] = cls._encode_board_entity(ent)

        # 4. Opponent Board Features (MAX_BOARD, 16)
        opp_board_arr = np.zeros((cls.MAX_BOARD, cls.CARD_FEAT_DIM), dtype=np.float32)
        for idx, ent in enumerate(opp_p.board[: cls.MAX_BOARD]):
            opp_board_arr[idx] = cls._encode_board_entity(ent)

        return {
            "global": g,
            "hand": hand_arr,
            "my_board": my_board_arr,
            "opp_board": opp_board_arr,
            "opp_ink_hist": opp_ink_hist,
        }

    @classmethod
    def _encode_card_template(cls, card) -> np.ndarray:
        v = np.zeros(cls.CARD_FEAT_DIM, dtype=np.float32)
        v[0] = 1.0  # Present flag
        v[1] = card.cost / 10.0
        v[2] = 1.0 if card.inkable else 0.0
        v[3] = 1.0 if card.card_type == "Character" else 0.0
        v[4] = 1.0 if card.is_song else 0.0
        v[5] = card.strength / 10.0
        v[6] = card.willpower / 10.0
        v[7] = card.lore / 4.0
        kw = card.abilities.keywords
        v[8] = 1.0 if (kw & KeywordFlag.RUSH) else 0.0
        v[9] = 1.0 if (kw & KeywordFlag.EVASIVE) else 0.0
        v[10] = 1.0 if (kw & KeywordFlag.WARD) else 0.0
        v[11] = 1.0 if (kw & KeywordFlag.BODYGUARD) else 0.0
        v[12] = (card.abilities.shift_cost or 0) / 10.0
        v[13] = (card.abilities.singer_cost or 0) / 10.0
        flags, eff_val = card.abilities.to_rust_bitflags()
        v[14] = 1.0 if (flags & 32) else 0.0  # Removal / Damage
        v[15] = eff_val / 5.0
        return v

    @classmethod
    def _encode_board_entity(cls, ent: BoardEntity) -> np.ndarray:
        v = cls._encode_card_template(ent.card)
        v[5] = ent.effective_strength / 10.0
        v[6] = max(0, ent.effective_willpower - ent.damage) / 10.0
        v[12] = 1.0 if ent.exerted else 0.0
        v[13] = 1.0 if ent.drying else 0.0
        return v

    @classmethod
    def build_action_mask(
        cls,
        legal_actions: list[GameAction],
        action_space_size: int = 256,
    ) -> tuple[np.ndarray, dict[int, GameAction]]:
        mask = np.zeros(action_space_size, dtype=bool)
        slot_to_action: dict[int, GameAction] = {}
        for act in legal_actions:
            slot = act.action_slot_id(action_space_size)
            # Resolve rare hash collisions by probing next free slot
            for offset in range(action_space_size):
                cand = (slot + offset) % action_space_size
                if not mask[cand]:
                    mask[cand] = True
                    slot_to_action[cand] = act
                    break
        return mask, slot_to_action
