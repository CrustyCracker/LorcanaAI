#!/bin/bash
#SBATCH --job-name=lorcana_col_tech
#SBATCH --output=logs/col_tech_%j.out
#SBATCH --error=logs/col_tech_%j.err
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
echo "[Athena] Running Deep Lookahead Multi-Core Optimizer (~750,000+ games)"
echo "Grant: plgvideocnn0-gpu-a100 | Archetype: Amber/Emerald Circle of Life"
echo "Node: $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK:-32} | Date: $(date)"
echo "=================================================================="

# 1. Optimize the 16 Flex Slots around the 44-card Locked Core against Top 10 Duels.ink Meta Decks
python3 lorcana_cli.py optimize-core \
    --archetype "Amber/Emerald Circle of Life" \
    --top-meta 10 \
    --games 1200 \
    --packages 50 \
    --lookahead-depth 1 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --seed 2137

# 2. Record Detailed Turn-by-Turn & Action-by-Action Replays into logs/
python3 lorcana_cli.py watch-game \
    --deck-a "Amber/Emerald Circle of Life" \
    --deck-b "Amber/Sapphire Detectives" \
    --games 4 \
    --lookahead-depth 2 \
    --output logs/replay_col_vs_detectives.log

python3 lorcana_cli.py watch-game \
    --deck-a "Amber/Emerald Circle of Life" \
    --deck-b "Emerald/Steel Midrange" \
    --games 4 \
    --lookahead-depth 2 \
    --output logs/replay_col_vs_emerald_steel.log

# 3. Run High-Budget Targeted UCB1 Tech Scans vs the Top Hardest Opponents in Duels.ink
python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Circle of Life" \
    --opp-deck "Emerald/Steel Midrange" \
    --copies 3 \
    --budget 1500 \
    --lookahead-depth 1 \
    --top-k 6

python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Circle of Life" \
    --opp-deck "Amber/Emerald Princess Aggro" \
    --copies 3 \
    --budget 1500 \
    --lookahead-depth 1 \
    --top-k 6

echo "[Athena] Done! Optimized decklist saved to: decks/amber_emerald_circle_of_life_optimized.txt"
echo "[Athena] Detailed JSON report saved to:    results/circle_of_life_optimized_report.json"
echo "[Athena] Play-by-Play game replays in:     logs/circle_of_life_sample_games.log & logs/replay_col_*.log"
