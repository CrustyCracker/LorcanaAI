"""
Central configuration and hyperparameters for LorcanaAlpha.
Automatically adapts to PLGrid Athena (A100 GPUs + 128 CPU cores) or local workstation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path
import multiprocessing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class EngineConfig:
    """Rules and simulation limits for Disney Lorcana Core Constructed (Season 14: Sets 9-14)."""
    core_constructed_sets: tuple[str, ...] = ("9", "10", "11", "12", "13", "14")
    target_lore: int = 20
    starting_hand_size: int = 7
    min_deck_size: int = 60
    max_copies_per_card: int = 4
    max_ink_colors: int = 2
    max_turns_per_game: int = 65
    max_actions_per_turn: int = 40
    max_board_slots: int = 12
    max_hand_slots: int = 15


@dataclass
class NetworkConfig:
    """Transformer Policy-Value & Opponent Belief Network parameters."""
    card_embedding_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_transformer_layers: int = 4
    dropout: float = 0.1
    action_space_size: int = 256
    num_archetype_classes: int = 15  # 15 two-ink combinations in Lorcana
    use_amp_bf16: bool = True        # Native A100 TensorCore BFloat16 acceleration


@dataclass
class MCTSConfig:
    """Information Set MCTS (IS-MCTS) parameters."""
    num_determinizations: int = 8
    simulations_per_determinization: int = 32
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    use_belief_weighting: bool = True


@dataclass
class DeckbuilderConfig:
    """Parameters for Tech-Card Scanner, MAP-Elites Coevolution, and Nash Solver."""
    max_uninkable_count: int = 18
    recommended_uninkable_range: tuple[int, int] = (8, 16)
    ucb_exploration_c: float = 1.414
    tech_scan_games_per_round: int = 40
    map_elites_generations: int = 30
    map_elites_games_per_eval: int = 24
    mutation_rate: float = 0.25
    nash_iterations: int = 500


@dataclass
class ClusterConfig:
    """PLGrid Athena & local runtime hardware detection."""
    cpu_workers: int = field(
        default_factory=lambda: int(
            os.environ.get("SLURM_CPUS_PER_TASK", max(1, multiprocessing.cpu_count()))
        )
    )
    slurm_job_id: str | None = field(
        default_factory=lambda: os.environ.get("SLURM_JOB_ID")
    )
    lorcana_json_url: str = "https://lorcanajson.org/files/current/en/allCards.json"
