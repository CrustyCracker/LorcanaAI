#!/bin/bash
#SBATCH --job-name=lorcana_col_tech
#SBATCH --output=logs/col_tech_%j.out
#SBATCH --error=logs/col_tech_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglorcana-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -e
mkdir -p logs results checkpoints decks

module purge
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1 Rust/1.75.0 2>/dev/null || true

source venv_athena/bin/activate
export RAYON_NUM_THREADS=64
export OMP_NUM_THREADS=8

echo "=================================================================="
echo "[Athena] Running High-Sample Duels.ink Meta Tech & Flex-Slot Optimizer"
echo "Archetype: Amber/Emerald Circle of Life (44 Locked Core + 16 Flex Slots)"
echo "Node: $(hostname) | Threads: $RAYON_NUM_THREADS"
echo "=================================================================="

# 1. Optimize the 16 Flex Slots around the 44-card Locked Core against Top 10 Duels.ink Meta Decks
python lorcana_cli.py optimize-core \
    --archetype "Amber/Emerald Circle of Life" \
    --top-meta 10 \
    --games 120 \
    --packages 32 \
    --seed 2026

# 2. Run Targeted UCB1 Tech Scans vs the Top 3 Hardest Opponents in Duels.ink
python lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Circle of Life" \
    --opp-deck "Emerald/Steel Midrange" \
    --copies 3 \
    --budget 600 \
    --top-k 6

python lorcana_cli.py scan-tech \
    --my-deck "Amber/Emerald Circle of Life" \
    --opp-deck "Amber/Sapphire Detectives" \
    --copies 3 \
    --budget 600 \
    --top-k 6

echo "[Athena] Done! Optimized decklist saved to decks/amber_emerald_circle_of_life_optimized.txt"
echo "[Athena] Detailed JSON report saved to results/circle_of_life_optimized_report.json"
