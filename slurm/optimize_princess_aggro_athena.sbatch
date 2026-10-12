#!/bin/bash
#SBATCH --job-name=lorcana_princess_opt
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "================================================================"
echo "[Athena SLURM] Massive Simulation: Amber/Emerald Princess Aggro"
echo "Node: $(hostname) | Job ID: ${SLURM_JOB_ID:-local} | Date: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo "================================================================"

module purge
module load GCCcore/12.3.0 || true
module load Python/3.11.3 || true
module load CUDA/12.1.1 || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export SCRATCH_WORK="${SCRATCH:-$PWD}/lorcana_runs"
mkdir -p "${SCRATCH_WORK}/logs" "${SCRATCH_WORK}/results" "${SCRATCH_WORK}/checkpoints"

VENV_DIR="${SCRATCH:-$PWD}/lorcana_venv"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
elif [ -f "$HOME/lorcana_venv/bin/activate" ]; then
    source "$HOME/lorcana_venv/bin/activate"
fi

# Ensure libpython3.11.so.1.0 is on LD_LIBRARY_PATH on compute nodes
if [ -n "${EBROOTPYTHON:-}" ] && [ -d "${EBROOTPYTHON}/lib" ]; then
    export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH:-}"
fi
for py_lib_dir in /net/pr2/projects/plgrid/plggllms/software/*/Python/3.11*/lib /net/software/local/*/Python/3.11*/lib /cvmfs/*/software/*/Python/3.11*/lib; do
    if [ -d "$py_lib_dir" ]; then
        export LD_LIBRARY_PATH="${py_lib_dir}:${LD_LIBRARY_PATH:-}"
    fi
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[Step 1/2] Running Massive Multi-Core Optimizer (~235,000+ games) for Amber/Emerald Princess Aggro..."
python3 lorcana_cli.py optimize-core \
    --archetype "Amber/Emerald Princess Aggro" \
    --core-file decks/amber_emerald_princess_aggro_core44.txt \
    --baseline-file decks/amber_emerald_princess_aggro_baseline.txt \
    --top-meta 10 \
    --games 500 \
    --packages 32 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --seed 2026

echo "[Step 2/2] Running UCB1 Matchup Tech-Scanner vs Top Meta Opponents..."
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

cp -f results/princess_aggro_optimized_report.json "${SCRATCH_WORK}/results/" 2>/dev/null || true
echo "[DONE] Optimized deck saved to: decks/amber_emerald_princess_aggro_optimized.txt"
echo "[DONE] JSON report saved to:    results/princess_aggro_optimized_report.json"
