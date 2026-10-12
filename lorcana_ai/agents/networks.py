"""
PyTorch Neural Architectures for LorcanaAlpha (Optimized for NVIDIA A100 GPUs on PLGrid Athena):
1. OpponentBeliefNetwork: Predicts opponent's 2-Ink archetype and probability of holding key cards.
2. LorcanaTransformerNet: Multi-Head Attention Policy-Value network over Hand & Board tokens.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lorcana_ai.config import NetworkConfig
from lorcana_ai.data.card_db import GLOBAL_CARD_DB
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import GameAction, MoveGenerator
from lorcana_ai.engine.vectorizer import StateVectorizer


def np_to_tensor(
    arr: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Safely converts numpy array to torch.Tensor even when PyTorch C-ABI was built for NumPy 1.x."""
    try:
        t = torch.from_numpy(arr)
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t
    except RuntimeError:
        if dtype is None:
            dtype = torch.bool if arr.dtype == np.bool_ else torch.float32
        return torch.tensor(arr.tolist(), dtype=dtype, device=device)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """Safely converts torch.Tensor to numpy array even when PyTorch C-ABI was built for NumPy 1.x."""
    t_cpu = t.detach().cpu()
    try:
        return t_cpu.numpy()
    except RuntimeError:
        return np.array(t_cpu.tolist(), dtype=np.float32)


class OpponentBeliefNetwork(nn.Module):
    """
    Takes public game observations (global features, opponent board, opponent ink histogram)
    and predicts:
    - archetype_logits: (B, 15) distribution over the 15 two-ink combinations
    - threat_probs: (B, num_cards) probability that opponent currently holds specific cards in hand
    """

    def __init__(self, cfg: NetworkConfig | None = None, num_known_cards: int = 120) -> None:
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        self.num_known_cards = num_known_cards
        in_dim = StateVectorizer.GLOBAL_DIM + StateVectorizer.MAX_BOARD * StateVectorizer.CARD_FEAT_DIM + 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.archetype_head = nn.Linear(256, self.cfg.num_archetype_classes)
        self.card_hold_head = nn.Linear(256, num_known_cards)

    def forward(
        self,
        global_feat: torch.Tensor,
        opp_board: torch.Tensor,
        opp_ink_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = global_feat.shape[0]
        flat_opp = opp_board.reshape(b, -1)
        x = torch.cat([global_feat, flat_opp, opp_ink_hist], dim=-1)
        h = self.net(x)
        arch_logits = self.archetype_head(h)
        card_probs = torch.sigmoid(self.card_hold_head(h))
        return arch_logits, card_probs

    @torch.no_grad()
    def predict_card_weights(self, state: GameState, observer_id: int, device: torch.device) -> dict[int, float]:
        """Returns dictionary mapping card_id -> relative likelihood of being held in opponent's hand."""
        enc = StateVectorizer.encode_state(state, observer_id=observer_id)
        g = np_to_tensor(enc["global"], device=device, dtype=torch.float32).unsqueeze(0)
        ob = np_to_tensor(enc["opp_board"], device=device, dtype=torch.float32).unsqueeze(0)
        ink_h = np_to_tensor(enc["opp_ink_hist"], device=device, dtype=torch.float32).unsqueeze(0)
        _, probs = self.forward(g, ob, ink_h)
        probs_np = tensor_to_np(probs[0])

        sorted_cids = sorted(GLOBAL_CARD_DB.by_id.keys())
        weights: dict[int, float] = {}
        for idx, cid in enumerate(sorted_cids):
            p_val = float(probs_np[idx % len(probs_np)])
            card = GLOBAL_CARD_DB.by_id[cid]
            # Boost weight for high-cost sweepers/bombs if opponent has held card for several turns
            if card.cost >= 5:
                p_val *= 1.25
            weights[cid] = max(0.1, p_val)
        return weights


class LorcanaTransformerNet(nn.Module):
    """
    Transformer Policy-Value Network for Disney Lorcana.
    Processes a sequence of tokens:
      [CLS_GLOBAL (1), HAND_TOKENS (12), MY_BOARD_TOKENS (10), OPP_BOARD_TOKENS (10)] = 33 tokens
    Outputs:
      - policy_logits: (B, 256)
      - state_value: (B, 1) in [-1, +1]
    """

    def __init__(self, cfg: NetworkConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        d_model = self.cfg.hidden_dim

        self.global_proj = nn.Sequential(
            nn.Linear(StateVectorizer.GLOBAL_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.card_proj = nn.Sequential(
            nn.Linear(StateVectorizer.CARD_FEAT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Zone embeddings: 0=Global CLS, 1=My Hand, 2=My Board, 3=Opponent Board
        self.zone_embedding = nn.Embedding(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.cfg.num_heads,
            dim_feedforward=d_model * 2,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.num_transformer_layers,
            enable_nested_tensor=False,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.cfg.action_space_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        global_feat: torch.Tensor,
        hand_feat: torch.Tensor,
        my_board_feat: torch.Tensor,
        opp_board_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = global_feat.shape[0]
        device = global_feat.device

        t_global = self.global_proj(global_feat).unsqueeze(1)  # (B, 1, D)
        t_hand = self.card_proj(hand_feat)                     # (B, 12, D)
        t_my_b = self.card_proj(my_board_feat)                 # (B, 10, D)
        t_opp_b = self.card_proj(opp_board_feat)               # (B, 10, D)

        z_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.ones(StateVectorizer.MAX_HAND, dtype=torch.long, device=device),
                torch.full((StateVectorizer.MAX_BOARD,), 2, dtype=torch.long, device=device),
                torch.full((StateVectorizer.MAX_BOARD,), 3, dtype=torch.long, device=device),
            ]
        ).unsqueeze(0).expand(b, -1)

        tokens = torch.cat([t_global, t_hand, t_my_b, t_opp_b], dim=1) + self.zone_embedding(z_ids)
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0, :]  # (B, D)

        logits = self.policy_head(cls_out)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        value = self.value_head(cls_out)
        return logits, value

    @torch.no_grad()
    def evaluate_state(
        self,
        state: GameState,
        legal_actions: list[GameAction] | None = None,
        device: torch.device | None = None,
    ) -> tuple[dict[GameAction, float], float]:
        """
        Evaluates a single GameState and returns (action_prior_dict, value_estimate).
        """
        device = device or next(self.parameters()).device
        actions = legal_actions if legal_actions is not None else MoveGenerator.generate_legal_actions(state)
        if not actions:
            return {}, 0.0

        enc = StateVectorizer.encode_state(state, observer_id=state.active_player)
        mask_np, slot_to_act = StateVectorizer.build_action_mask(actions, self.cfg.action_space_size)

        g = np_to_tensor(enc["global"], device=device, dtype=torch.float32).unsqueeze(0)
        h = np_to_tensor(enc["hand"], device=device, dtype=torch.float32).unsqueeze(0)
        mb = np_to_tensor(enc["my_board"], device=device, dtype=torch.float32).unsqueeze(0)
        ob = np_to_tensor(enc["opp_board"], device=device, dtype=torch.float32).unsqueeze(0)
        mask_t = np_to_tensor(mask_np, device=device, dtype=torch.bool).unsqueeze(0)

        use_amp = device.type == "cuda" and self.cfg.use_amp_bf16
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, val = self.forward(g, h, mb, ob, action_mask=mask_t)
            probs = tensor_to_np(F.softmax(logits.float(), dim=-1)[0])
            value_scalar = float(val.float()[0, 0].cpu().item())

        priors: dict[GameAction, float] = {}
        for slot, act in slot_to_act.items():
            priors[act] = float(probs[slot])
        return priors, value_scalar
