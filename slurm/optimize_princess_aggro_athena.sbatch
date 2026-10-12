#!/bin/bash
#SBATCH --job-name=lorcana_princess_opt
#SBATCH --output=logs/princess_tech_%j.out
#SBATCH --error=logs/princess_tech_%j.err
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
echo "[Athena] Running Massive Multi-Core Optimizer (~235,000+ games)"
echo "Grant: plgvideocnn0-gpu-a100 | Archetype: Amber/Emerald Princess Aggro"
echo "Node: $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK:-32} | Date: $(date)"
echo "=================================================================="

# 1. Optimize the 16 Flex Slots around the 44-card Locked Core against Top 10 Duels.ink Meta Decks
python3 lorcana_cli.py optimize-core \
    --archetype "Amber/Emerald Princess Aggro" \
    --core-file decks/amber_emerald_princess_aggro_core44.txt \
    --baseline-file decks/amber_emerald_princess_aggro_baseline.txt \
    --top-meta 10 \
    --games 500 \
    --packages 32 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --seed 2026

# 2. Run Targeted UCB1 Tech Scans vs the Top Opponents in Duels.ink
python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Princess Aggro" \
    --opp-deck "Amber/Sapphire Detectives" \
    --copies 3 \
    --budget 600 \
    --top-k 5

python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Princess Aggro" \
    --opp-deck "Emerald/Steel Midrange" \
    --copies 3 \
    --budget 600 \
    --top-k 5

python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Princess Aggro" \
    --opp-deck "Amber/Emerald Circle of Life" \
    --copies 3 \
    --budget 600 \
    --top-k 5

echo "[Athena] Done! Optimized decklist saved to: decks/amber_emerald_princess_aggro_optimized.txt"
echo "[Athena] Detailed JSON report saved to:    results/princess_aggro_optimized_report.json"
echo "[Athena] Logs saved in:                    logs/"
