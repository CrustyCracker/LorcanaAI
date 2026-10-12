#!/bin/bash
#SBATCH --job-name=lorcana_deck_evo
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

module purge 2>/dev/null || true
module load GCCcore/12.3.0 2>/dev/null || true
module load Python/3.11.3 2>/dev/null || module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || module load Python 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv_athena/bin/activate ]; then
    source venv_athena/bin/activate
elif [ -f "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate" ]; then
    source "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate"
fi

PY_REAL=$(readlink -f "$(which python3)")
PY_LIB_DIR="$(dirname "$(dirname "$PY_REAL")")/lib"
export LD_LIBRARY_PATH="${PY_LIB_DIR}:${LD_LIBRARY_PATH}"
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONIOENCODING=utf-8

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
