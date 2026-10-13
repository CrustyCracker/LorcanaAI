#!/bin/bash
#SBATCH --job-name=lorcana_inkdecks_gs
#SBATCH --output=logs/inkdecks_gs_mirror_%j.out
#SBATCH --error=logs/inkdecks_gs_mirror_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -e
mkdir -p logs results checkpoints decks data_cache

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

PY_REAL=$(readlink -f "$(which python3)")
PY_LIB_DIR="$(dirname "$(dirname "$PY_REAL")")/lib"
export LD_LIBRARY_PATH="${PY_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo "[Athena] Running INKDECKS.COM Real Tournament Techs Mirror Optimizer"
echo "Source: https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats"
echo "Grant: plgvideocnn0-gpu-a100 | CPUs: ${SLURM_CPUS_PER_TASK:-32} | Date: $(date)"
echo "=================================================================="

# 1. Głęboka symulacja realnych techów z Inkdecks w mirrorze Green/Steel (depth=2, seed=2137)
python3 scripts/test_inkdecks_green_steel_techs.py \
    --games-tech 250 \
    --games-packages 600 \
    --lookahead-depth 2 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --seed 2137

# 2. Replay meczu turniejowego
python3 lorcana_cli.py watch-game \
    --deck-a "decks/emerald_steel_inkdecks_mirror_optimized.txt" \
    --deck-b "Green Steel" \
    --games 4 \
    --lookahead-depth 2 \
    --seed 2137

echo "=================================================================="
echo "[OK] Zakończono symulację Inkdecks Green Steel Mirror na Athenie."
echo "Wyniki zapisano w decks/, results/ oraz logs/."
echo "=================================================================="
