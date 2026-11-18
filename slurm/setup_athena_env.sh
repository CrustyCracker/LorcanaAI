#!/usr/bin/env bash
# ==============================================================================
# Inicjalizacja środowiska LorcanaAlpha na klastrze PLGrid Athena (ACK Cyfronet)
# Uruchom ten skrypt JEDEN RAZ na węźle dostępowym (login node, gdzie działa internet):
#   bash slurm/setup_athena_env.sh
#
# Rust/maturin is optional: if native build fails, Python DSL-only still works
# for optimize-core / scan-tech / core-ablate (use_full_dsl_sim=True).
# ==============================================================================
set -euo pipefail

echo "[1/5] Ładowanie modułów na PLGrid Athena..."
module purge || true
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1 Rust/1.75.0 || true

VENV_DIR="${SCRATCH:-$HOME}/lorcana_alpha_venv"
echo "[2/5] Tworzenie środowiska wirtualnego Python w: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[3/5] Instalacja PyTorch (CUDA 12.1 dla A100) oraz zależności..."
pip install --upgrade pip wheel maturin
pip install -r requirements.txt

echo "[4/5] Kompilacja natywnego silnika Rust (lorcana_rs) — soft-fail (DSL-only OK)..."
if command -v rustc >/dev/null 2>&1; then
  echo "[4/5] rustc: $(rustc --version 2>/dev/null || echo missing)"
else
  echo "[4/5] WARN: rustc not on PATH — skipping maturin develop"
fi
set +e
RUSTFLAGS="-C target-cpu=native" maturin develop --release
MATURIN_RC=$?
set -e
if [ "${MATURIN_RC}" -ne 0 ]; then
  echo "[4/5] WARN: maturin develop failed (exit ${MATURIN_RC}). Continuing with Python DSL-only."
  echo "[4/5] WARN: optimize/scan/core-ablate still work via use_full_dsl_sim=True."
else
  echo "[4/5] OK: lorcana_rs built successfully."
fi

echo "[5/5] Pobieranie bazy kart LorcanaJSON do pamięci podręcznej (dla zadań offline na węzłach GPU)..."
python lorcana_cli.py sync-cards

echo "=============================================================================="
echo "[GOTOWE] Środowisko LorcanaAlpha na PLGrid Athena jest skonfigurowane!"
echo "Venv: ${VENV_DIR}"
if [ "${MATURIN_RC}" -ne 0 ]; then
  echo "Mode: Python DSL-only (Rust engine unavailable)"
else
  echo "Mode: Python DSL + native lorcana_rs"
fi
echo "Możesz teraz zlecić zadania do kolejki SLURM:"
echo "  sbatch slurm/train_agent_a100.sbatch"
echo "  sbatch slurm/evolve_decks_athena.sbatch"
echo "  sbatch slurm/optimize_retro_device_athena.sbatch"
echo "=============================================================================="
