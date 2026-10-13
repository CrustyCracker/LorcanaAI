#!/bin/bash
#SBATCH --job-name=lorcana_green_steel_opt
#SBATCH --output=logs/green_steel_tech_%j.out
#SBATCH --error=logs/green_steel_tech_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -e
mkdir -p logs results checkpoints decks

module purge 2>/dev/null || true
module load GCCcore/12.3.0 2>/dev/null || true
module load Python/3.11.3 2>/dev/null || module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || module load Python 2>/dev/null || true
module load SciPy-bundle/2023.07 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv_athena/bin/activate ]; then
    source venv_athena/bin/activate
elif [ -f "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate" ]; then
    source "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate"
elif [ -f "${SCRATCH:-$HOME}/lorcana_venv/bin/activate" ]; then
    source "${SCRATCH:-$HOME}/lorcana_venv/bin/activate"
fi

# Automatically resolve and export the directory containing libpython3.11.so.1.0
PY_REAL=$(readlink -f "$(which python3)")
PY_LIB_DIR="$(dirname "$(dirname "$PY_REAL")")/lib"
export LD_LIBRARY_PATH="${PY_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo "[Athena] Running Multi-Core Green Steel Tech Optimizer"
echo "Grant: plgvideocnn0-gpu-a100 | Archetype: Emerald/Steel Green Steel"
echo "Node: $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK:-32} | Date: $(date)"
echo "=================================================================="

# 1. Optymalizacja 20 slotów flex/tech wokół 40-kartowego rdzenia Green Steel
#    vs Top 10 talii Duels.ink (1200 gier na matchup, 48 pakietów flex, 32 wątki CPU)
python3 scripts/optimize_green_steel.py \
    --top-meta 10 \
    --games 1200 \
    --packages 48 \
    --tech-candidates 40 \
    --lookahead-depth 1 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --seed 2137

# 2. Generowanie szczegółowych zapisów rozgrywek (Play-by-Play Replays) vs kluczowe talie metagry
python3 lorcana_cli.py watch-game \
    --deck-a "Green Steel" \
    --deck-b "Amber/Emerald Princess Aggro" \
    --games 4 \
    --lookahead-depth 2 \
    --seed 2137

python3 lorcana_cli.py watch-game \
    --deck-a "Green Steel" \
    --deck-b "Amber/Emerald Circle of Life" \
    --games 4 \
    --lookahead-depth 2 \
    --seed 2137

echo "=================================================================="
echo "[OK] Zakończono optymalizację Green Steel na Athenie."
echo "Wyniki zapisano w decks/, results/ oraz logs/."
echo "=================================================================="
