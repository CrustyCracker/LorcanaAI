#!/bin/bash
#SBATCH --job-name=lorcana_retro_opt
#SBATCH --output=logs/retro_device_%j.out
#SBATCH --error=logs/retro_device_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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

WEEK_PERIOD="${RETRO_META_WEEK:-week:2026-11-15}"
WORKERS="${SLURM_CPUS_PER_TASK:-32}"
DECK="decks/user_emerald_ruby_retro_device.txt"
CORE="decks/user_emerald_ruby_retro_device_core_loops.txt"
ARCHETYPE="Emerald/Ruby Action Retro"

echo "=================================================================="
echo "[Athena] Emerald/Ruby Retro Device — DSL sim optimize package"
echo "Grant: plgvideocnn0-gpu-a100 | Archetype: ${ARCHETYPE}"
echo "Meta period: ${WEEK_PERIOD} | use_full_dsl_sim=True (CLI default)"
echo "Node: $(hostname) | CPUs: ${WORKERS} | Date: $(date)"
echo "=================================================================="

# 0. Sync this week's Duels.ink Core BO1 meta (requires dated week:, not bare --period week)
python3 lorcana_cli.py sync-meta --period "${WEEK_PERIOD}" --top-n 15

# 1. CoreValue leave-one-out ablation on the user 60 vs top meta (Python DSL only)
python3 lorcana_cli.py core-ablate \
    --deck "${DECK}" \
    --core-deck "${CORE}" \
    --vs-meta cache \
    --top-meta 10 \
    --games 48 \
    --lookahead-depth 1 \
    --workers "${WORKERS}" \
    --seed 2137

# 2. Optimize flex around locked loop core (Device/Photo/Torn/Bob/Sid/Olaf/Leviathan/Phillip/Max/Malicious)
python3 lorcana_cli.py optimize-core \
    --archetype "${ARCHETYPE}" \
    --core-file "${CORE}" \
    --baseline-file "${DECK}" \
    --top-meta 10 \
    --games 48 \
    --packages 16 \
    --tech-candidates 40 \
    --lookahead-depth 1 \
    --workers "${WORKERS}" \
    --seed 2137

# 3. High-budget UCB1 tech scans vs week pressure matchups
python3 lorcana_cli.py scan-tech \
    --my-deck "${DECK}" \
    --opp-deck "Emerald/Steel Midrange" \
    --copies 3 \
    --budget 1500 \
    --lookahead-depth 1 \
    --top-k 6

python3 lorcana_cli.py scan-tech \
    --my-deck "${DECK}" \
    --opp-deck "Amber/Emerald Circle of Life" \
    --copies 3 \
    --budget 1500 \
    --lookahead-depth 1 \
    --top-k 6

python3 lorcana_cli.py scan-tech \
    --my-deck "${DECK}" \
    --opp-deck "Amber/Sapphire Detectives" \
    --copies 3 \
    --budget 1500 \
    --lookahead-depth 1 \
    --top-k 6

echo "[Athena] Done."
echo "[Athena] Ablation JSON:     results/core_ablation_report.json"
echo "[Athena] Optimize JSON:     results/retro_device_optimized_report.json"
echo "[Athena] Optimized deck:    decks/user_emerald_ruby_retro_device_optimized.txt"
echo "[Athena] scan-tech logs:    logs/retro_device_*.out"
