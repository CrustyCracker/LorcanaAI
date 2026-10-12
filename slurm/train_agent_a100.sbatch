#!/bin/bash
#SBATCH --job-name=lorcana_rl_a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgvideocnn0-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_a100_%j.out
#SBATCH --error=logs/train_a100_%j.err
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Skrypt SLURM: Trening Self-Play (IS-MCTS + Transformer Policy-Value + Belief Net)
# na karcie NVIDIA A100 na klastrze PLGrid Athena.
# Automatycznie wznawia trening z ostatniego checkpointu (checkpoints/latest_agent.pt).
# ==============================================================================

mkdir -p logs checkpoints results

module purge || true
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1 || true

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv_athena/bin/activate ]; then
    source venv_athena/bin/activate
elif [ -f "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate" ]; then
    source "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTHONIOENCODING=utf-8

# Obsługa sygnału przed końcem czasu zadania w SLURM (auto-requeue)
trap 'echo "[SLURM] Otrzymano sygnał USR1 - zapisano checkpoint, ponowne kolejkowanie zadania..."; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

echo "[START] Węzeł: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1) | Rdzenie CPU: ${RAYON_NUM_THREADS}"

python lorcana_cli.py train \
    --epochs 250 \
    --games-per-epoch 32 \
    --batch-size 128 \
    --determinizations 8 \
    --simulations 32 \
    --use-mcts &

wait $!
echo "[KONIEC] Zadanie treningowe zakończone pomyślnie."
