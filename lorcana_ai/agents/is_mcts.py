"""
Information Set Monte Carlo Tree Search (IS-MCTS) for Disney Lorcana.
Handles imperfect information (opponent's hidden hand & deck order) by sampling
belief-weighted determinizations and running PUCT search guided by LorcanaTransformerNet.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
import numpy as np
import torch

from lorcana_ai.agents.heuristic_agent import TacticalHeuristicAgent
from lorcana_ai.agents.networks import LorcanaTransformerNet, OpponentBeliefNetwork
from lorcana_ai.config import MCTSConfig
from lorcana_ai.engine.game_state import GameState
from lorcana_ai.engine.move_generator import ActionType, GameAction, MoveGenerator
from lorcana_ai.engine.simulator import LorcanaSimulator
from lorcana_ai.engine.vectorizer import StateVectorizer


@dataclass
class MCTSNode:
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[GameAction, MCTSNode] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class ISMCTSAgent:
    """
    Dual-mode AI Agent:
    - Fast Policy Mode (use_mcts=False): Uses Transformer Policy-Value + Heuristic fusion for rapid screening.
    - Deep IS-MCTS Mode (use_mcts=True): Runs multi-determinization Information Set MCTS with Belief Network.
    """

    def __init__(
        self,
        policy_value_net: LorcanaTransformerNet | None = None,
        belief_net: OpponentBeliefNetwork | None = None,
        cfg: MCTSConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg or MCTSConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = (policy_value_net or LorcanaTransformerNet()).to(self.device)
        self.belief_net = (belief_net or OpponentBeliefNetwork()).to(self.device)
        self.heuristic = TacticalHeuristicAgent()

    def select_action(
        self,
        state: GameState,
        use_mcts: bool = True,
        add_exploration_noise: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[GameAction, np.ndarray]:
        """
        Returns (chosen_action, action_prob_target_vector_256).
        """
        rng = rng or random.Random()
        legal_actions = MoveGenerator.generate_legal_actions(state)
        action_space = self.net.cfg.action_space_size
        pi_vec = np.zeros(action_space, dtype=np.float32)

        if not legal_actions:
            return GameAction(ActionType.PASS_TURN, description="Pass Turn"), pi_vec

        if len(legal_actions) == 1:
            slot = legal_actions[0].action_slot_id(action_space)
            pi_vec[slot] = 1.0
            return legal_actions[0], pi_vec

        if not use_mcts:
            priors = self._compute_blended_priors(state, legal_actions)
            best_act = max(legal_actions, key=lambda a: priors.get(a, 0.0))
            for act, p_val in priors.items():
                pi_vec[act.action_slot_id(action_space)] = p_val
            pi_sum = pi_vec.sum()
            if pi_sum > 0:
                pi_vec /= pi_sum
            return best_act, pi_vec

        # Information Set MCTS across K belief-weighted determinizations
        root_player = state.active_player
        card_weights = None
        if self.cfg.use_belief_weighting:
            card_weights = self.belief_net.predict_card_weights(state, observer_id=root_player, device=self.device)

        aggregated_visits: dict[GameAction, int] = {a: 0 for a in legal_actions}

        for det_idx in range(self.cfg.num_determinizations):
            det_rng = random.Random(rng.randint(1, 10**9) + det_idx * 7919)
            det_state = state.clone_and_determinize(
                observer_id=root_player,
                belief_card_weights=card_weights,
                rng=det_rng,
            )
            root = MCTSNode(prior=1.0)
            self._expand_node(root, det_state, root_player, add_noise=(add_exploration_noise and det_idx == 0))

            for _ in range(self.cfg.simulations_per_determinization):
                sim_state = det_state.clone()
                self._simulate_single_path(root, sim_state, root_player, det_rng)

            for act, child in root.children.items():
                if act in aggregated_visits:
                    aggregated_visits[act] += child.visit_count

        total_v = sum(aggregated_visits.values())
        if total_v == 0:
            best_act = self.heuristic.select_action(state, legal_actions)
            pi_vec[best_act.action_slot_id(action_space)] = 1.0
            return best_act, pi_vec

        best_act = max(legal_actions, key=lambda a: aggregated_visits.get(a, 0))
        _, slot_to_act = StateVectorizer.build_action_mask(legal_actions, action_space)
        act_to_slot = {act: sl for sl, act in slot_to_act.items()}
        for act, v_cnt in aggregated_visits.items():
            if act in act_to_slot:
                pi_vec[act_to_slot[act]] = v_cnt / total_v

        return best_act, pi_vec

    def _simulate_single_path(
        self,
        root: MCTSNode,
        sim_state: GameState,
        root_player: int,
        rng: random.Random,
    ) -> float:
        node = root
        search_path: list[tuple[MCTSNode, int]] = [(node, sim_state.active_player)]
        depth = 0

        while node.children and sim_state.phase != "GAME_OVER" and depth < 16:
            depth += 1
            best_act, next_node = self._select_puct_child(node, sim_state.active_player == root_player)
            LorcanaSimulator.step(sim_state, best_act, rng=rng)
            node = next_node
            search_path.append((node, sim_state.active_player))

        # Evaluate leaf
        if sim_state.phase == "GAME_OVER":
            leaf_val = 1.0 if sim_state.winner == root_player else -1.0
        else:
            leaf_val = self._expand_node(node, sim_state, root_player, add_noise=False)

        # Backpropagate value
        for path_node, _ in search_path:
            path_node.visit_count += 1
            path_node.value_sum += leaf_val

        return leaf_val

    def _select_puct_child(self, node: MCTSNode, is_root_player_turn: bool) -> tuple[GameAction, MCTSNode]:
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score = -1e9
        best_pair: tuple[GameAction, MCTSNode] | None = None

        for act, child in node.children.items():
            q = child.q_value if is_root_player_turn else -child.q_value
            u = self.cfg.c_puct * child.prior * (sqrt_total / (1 + child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_pair = (act, child)

        assert best_pair is not None
        return best_pair

    def _expand_node(
        self,
        node: MCTSNode,
        state: GameState,
        root_player: int,
        add_noise: bool = False,
    ) -> float:
        legal_actions = MoveGenerator.generate_legal_actions(state)
        if not legal_actions:
            return 0.0

        net_priors, net_val = self.net.evaluate_state(state, legal_actions=legal_actions, device=self.device)
        # Convert value to root_player perspective
        if state.active_player != root_player:
            net_val = -net_val

        # Blend with domain heuristic scores for sharp tactical precision
        h_scores = np.array([self.heuristic.score_action(state, a) for a in legal_actions], dtype=np.float32)
        h_scores = h_scores - np.max(h_scores)
        h_probs = np.exp(h_scores / 8.0)
        h_probs /= np.sum(h_probs) + 1e-8

        priors_arr = np.zeros(len(legal_actions), dtype=np.float32)
        for i, act in enumerate(legal_actions):
            priors_arr[i] = 0.65 * net_priors.get(act, 1.0 / len(legal_actions)) + 0.35 * h_probs[i]
        priors_arr /= np.sum(priors_arr) + 1e-8

        if add_noise and len(legal_actions) > 1:
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(legal_actions))
            eps = self.cfg.dirichlet_epsilon
            priors_arr = (1.0 - eps) * priors_arr + eps * noise

        for i, act in enumerate(legal_actions):
            node.children[act] = MCTSNode(prior=float(priors_arr[i]))

        # Blend neural value with board-state heuristic evaluation
        p_me = state.players[root_player]
        p_opp = state.players[1 - root_player]
        lore_diff = (p_me.lore - p_opp.lore) / 20.0
        board_diff = (
            sum(e.card.lore * 2 + e.effective_strength for e in p_me.characters_in_play)
            - sum(e.card.lore * 2 + e.effective_strength for e in p_opp.characters_in_play)
        ) / 25.0
        heur_val = float(np.tanh(lore_diff * 1.5 + board_diff * 0.8))

        return 0.7 * net_val + 0.3 * heur_val

    def _compute_blended_priors(self, state: GameState, legal_actions: list[GameAction]) -> dict[GameAction, float]:
        net_priors, _ = self.net.evaluate_state(state, legal_actions=legal_actions, device=self.device)
        h_scores = np.array([self.heuristic.score_action(state, a) for a in legal_actions], dtype=np.float32)
        h_scores = h_scores - np.max(h_scores)
        h_probs = np.exp(h_scores / 6.0)
        h_probs /= np.sum(h_probs) + 1e-8

        out: dict[GameAction, float] = {}
        for i, act in enumerate(legal_actions):
            out[act] = 0.6 * net_priors.get(act, 1.0 / len(legal_actions)) + 0.4 * float(h_probs[i])
        return out
