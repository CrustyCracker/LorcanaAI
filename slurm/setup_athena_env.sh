#!/usr/bin/env bash
# ==============================================================================
# Inicjalizacja środowiska LorcanaAlpha na klastrze PLGrid Athena (ACK Cyfronet)
# Uruchom ten skrypt JEDEN RAZ na węźle dostępowym (login node, gdzie działa internet):
#   bash slurm/setup_athena_env.sh
# ==============================================================================
set -euo pipefail

echo "[1/5] Ładowanie modułów na PLGrid Athena..."
module purge || true
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1 Rust/1.75.0 || true

VENV_DIR="${SCRATCH:-$HOME}/lorcana_alpha_venv"
echo "[2/5] Tworzenie środowiska wirtualnego Python w: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[3/5] Instalacja PyTorch (CUDA 12.1 dla A100) oraz zależności..."
pip install --upgrade pip wheel maturin
pip install -r requirements.txt

echo "[4/5] Kompilacja natywnego silnika Rust (lorcana_rs) z optymalizacją AVX2/EPYC..."
RUSTFLAGS="-C target-cpu=native" maturin develop --release

echo "[5/5] Pobieranie bazy kart LorcanaJSON do pamięci podręcznej (dla zadań offline na węzłach GPU)..."
python lorcana_cli.py sync-cards

echo "=============================================================================="
echo "[GOTOWE] Środowisko LorcanaAlpha na PLGrid Athena jest skonfigurowane!"
echo "Możesz teraz zlecić zadania do kolejki SLURM:"
echo "  sbatch slurm/train_agent_a100.sbatch"
echo "  sbatch slurm/evolve_decks_athena.sbatch"
echo "=============================================================================="
