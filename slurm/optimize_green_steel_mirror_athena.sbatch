#!/bin/bash
#SBATCH --job-name=lorcana_gs_mirror_opt
#SBATCH --output=logs/green_steel_mirror_%j.out
#SBATCH --error=logs/green_steel_mirror_%j.err
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
echo "[Athena] Running Multi-Core Green Steel MIRROR Optimizer"
echo "Grant: plgvideocnn0-gpu-a100 | Matchup: Green Steel Mirror"
echo "Node: $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK:-32} | Date: $(date)"
echo "=================================================================="

# 1. Głęboka symulacja mirrora: 300 gier na tech w Fazie 1, 800 gier na pakiet w Fazie 2 na 32 rdzeniach
python3 scripts/optimize_green_steel_mirror.py \
    --games-tech 300 \
    --games-packages 800 \
    --lookahead-depth 2 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --seed 2137

# 2. Generowanie szczegółowych zapisów rozgrywek (Play-by-Play Replays) vs Baseline Mirror
python3 lorcana_cli.py watch-game \
    --deck-a "decks/emerald_steel_green_steel_mirror_optimized.txt" \
    --deck-b "Green Steel" \
    --games 4 \
    --lookahead-depth 2 \
    --seed 2137

echo "=================================================================="
echo "[OK] Zakończono optymalizację Green Steel Mirror na Athenie."
echo "Wyniki zapisano w decks/, results/ oraz logs/."
echo "=================================================================="
