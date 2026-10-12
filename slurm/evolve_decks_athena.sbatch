#!/bin/bash
#SBATCH --job-name=lorcana_deck_evo
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglorcana-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/evolve_decks_%j.out
#SBATCH --error=logs/evolve_decks_%j.err

# ==============================================================================
# Skrypt SLURM: Masowe odkrywanie nowych talii (MAP-Elites w 15 parach kolorów),
# wyliczanie równowagi Nasha Bo3 oraz skanowanie Tech-Kart na PLGrid Athena.
# ==============================================================================

mkdir -p logs checkpoints results

module purge || true
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1 || true

VENV_DIR="${SCRATCH:-$HOME}/lorcana_alpha_venv"
source "${VENV_DIR}/bin/activate"

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

echo "[1/3] Uruchamianie ewolucji talii MAP-Elites na ${RAYON_NUM_THREADS} wątkach EPYC..."
python lorcana_cli.py evolve --generations 40 --offspring 16 --games 32

echo "[2/3] Wyliczanie pełnej macierzy metagame Bo3 i równowagi Nasha..."
python lorcana_cli.py nash-meta --games 64

echo "[3/3] Skanowanie najlepszych Tech-Kart na główne matchupy turniejowe..."
python lorcana_cli.py scan-tech \
    --my-deck "Amethyst/Ruby Bounce Control" \
    --opp-deck "Ruby/Sapphire Ramp Control" \
    --copies 3 \
    --budget 600 \
    --top-k 8

echo "[GOTOWE] Wyniki zapisane w katalogu results/."
