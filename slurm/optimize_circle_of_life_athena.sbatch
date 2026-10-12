#!/bin/bash
#SBATCH --job-name=lorcana_col_tech
#SBATCH --output=logs/col_tech_%j.out
#SBATCH --error=logs/col_tech_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e
mkdir -p logs results checkpoints decks

module purge
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1 Rust/1.75.0 2>/dev/null || true

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv_athena/bin/activate ]; then
    source venv_athena/bin/activate
fi
export RAYON_NUM_THREADS=16
export OMP_NUM_THREADS=8
export PYTHONIOENCODING=utf-8

echo "=================================================================="
echo "[Athena] Running High-Sample Duels.ink Meta Tech & Flex-Slot Optimizer"
echo "Grant: plgvideocnn0-gpu-a100 | Archetype: Amber/Emerald Circle of Life"
echo "Node: $(hostname) | Threads: $RAYON_NUM_THREADS"
echo "=================================================================="

# 1. Optimize the 16 Flex Slots around the 44-card Locked Core against Top 10 Duels.ink Meta Decks
python3 lorcana_cli.py optimize-core \
    --archetype "Amber/Emerald Circle of Life" \
    --top-meta 10 \
    --games 200 \
    --packages 50 \
    --seed 2026

# 2. Run Targeted UCB1 Tech Scans vs the Top Hardest Opponents in Duels.ink
python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Circle of Life" \
    --opp-deck "Emerald/Steel Midrange" \
    --copies 3 \
    --budget 600 \
    --top-k 6

python3 lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Circle of Life" \
    --opp-deck "Amber/Emerald Princess Aggro" \
    --copies 3 \
    --budget 600 \
    --top-k 6

echo "[Athena] Done! Optimized decklist saved to decks/amber_emerald_circle_of_life_optimized.txt"
echo "[Athena] Detailed JSON report saved to results/circle_of_life_optimized_report.json"
