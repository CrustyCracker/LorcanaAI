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
#SBATCH --mem=48G
#SBATCH --time=10:00:00

set -euo pipefail
mkdir -p logs results checkpoints decks

module purge 2>/dev/null || true
module load GCCcore/12.3.0 2>/dev/null || true
module load Python/3.11.3 2>/dev/null || module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || module load Python 2>/dev/null || true
module load SciPy-bundle/2023.07 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(pwd -P)"

if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
elif [ -f venv_athena/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv_athena/bin/activate
elif [ -f "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRATCH:-$HOME}/lorcana_alpha_venv/bin/activate"
elif [ -f "${SCRATCH:-$HOME}/lorcana_venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRATCH:-$HOME}/lorcana_venv/bin/activate"
else
    echo "[FATAL] No venv found under ./venv, ./venv_athena, or \${SCRATCH}/lorcana_*_venv" >&2
    echo "[FATAL] Run: bash slurm/setup_athena_env.sh on a login node first" >&2
    exit 1
fi

# Automatically resolve and export the directory containing libpython3.11.so.1.0
PY_REAL=$(readlink -f "$(which python3)")
PY_LIB_DIR="$(dirname "$(dirname "$PY_REAL")")/lib"
export LD_LIBRARY_PATH="${PY_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Latest Duels.ink week = Monday of the current ISO week (override with RETRO_META_WEEK)
if [ -n "${RETRO_META_WEEK:-}" ]; then
    WEEK_PERIOD="${RETRO_META_WEEK}"
else
    WEEK_MONDAY="$(python3 - <<'PY'
from datetime import date, timedelta
d = date.today()
print((d - timedelta(days=d.weekday())).isoformat())
PY
)"
    WEEK_PERIOD="week:${WEEK_MONDAY}"
fi

WORKERS="${SLURM_CPUS_PER_TASK:-32}"
GAMES="${RETRO_GAMES:-112}"
PACKAGES="${RETRO_PACKAGES:-32}"
TECH_CANDIDATES="${RETRO_TECH_CANDIDATES:-80}"
TOP_META="${RETRO_TOP_META:-15}"
SCAN_BUDGET="${RETRO_SCAN_BUDGET:-3000}"
DECK="decks/user_emerald_ruby_retro_device.txt"
CORE="decks/user_emerald_ruby_retro_device_core_loops.txt"
ARCHETYPE="Emerald/Ruby Action Retro"

OUT_ROOT="${SCRATCH:-$HOME}/lorcana_retro_out"
OUT_DIR="${OUT_ROOT}/${SLURM_JOB_ID:-manual_$$}"
mkdir -p "${OUT_DIR}"

require_file() {
    local path="$1"
    local label="$2"
    if [ ! -s "${path}" ]; then
        echo "[FATAL] Missing or empty ${label}: ${path}" >&2
        echo "[FATAL] pwd=$(pwd -P) python=$(command -v python3)" >&2
        exit 1
    fi
    echo "[OK] ${label}: $(readlink -f "${path}" 2>/dev/null || realpath "${path}" 2>/dev/null || echo "${path}")"
}

stage_copy() {
    local src="$1"
    local dest_name="$2"
    require_file "${src}" "${dest_name}"
    cp -f "${src}" "${OUT_DIR}/${dest_name}"
    echo "[STAGED] ${OUT_DIR}/${dest_name}"
}

echo "=================================================================="
echo "[Athena] Emerald/Ruby Retro Device — STRONG DSL sim optimize package"
echo "Grant: plgvideocnn0-gpu-a100 | Archetype: ${ARCHETYPE}"
echo "Meta period: ${WEEK_PERIOD} | use_full_dsl_sim=True (CLI default)"
echo "Knobs: games=${GAMES} packages=${PACKAGES} tech-candidates=${TECH_CANDIDATES} top-meta=${TOP_META} scan-budget=${SCAN_BUDGET}"
echo "Node: $(hostname) | CPUs: ${WORKERS} | Date: $(date -Is)"
echo "pwd: ${REPO_ROOT}"
echo "python: $(command -v python3)"
python3 -V || true
echo "OUT_DIR (absolute): ${OUT_DIR}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "=================================================================="

require_file "${DECK}" "baseline deck"
require_file "${CORE}" "locked core loops"

# 0. Sync this week's Duels.ink Core BO1 meta (requires dated week:, not bare --period week)
echo "[STEP 0] sync-meta → ${WEEK_PERIOD} top-n=${TOP_META}"
echo "pwd=$(pwd -P) python=$(command -v python3) OUT_DIR=${OUT_DIR}"
python3 lorcana_cli.py sync-meta --period "${WEEK_PERIOD}" --top-n "${TOP_META}"

# 1. CoreValue leave-one-out ablation on the user 60 vs top meta (Python DSL only)
echo "[STEP 1] core-ablate games=${GAMES} top-meta=${TOP_META}"
echo "pwd=$(pwd -P) python=$(command -v python3) OUT_DIR=${OUT_DIR}"
python3 lorcana_cli.py core-ablate \
    --deck "${DECK}" \
    --core-deck "${CORE}" \
    --vs-meta cache \
    --top-meta "${TOP_META}" \
    --games "${GAMES}" \
    --lookahead-depth 1 \
    --workers "${WORKERS}" \
    --seed 2137
stage_copy "results/core_ablation_report.json" "core_ablation_report.json"

# 2. Optimize flex around locked loop core (Device/Photo/Torn/Bob/Sid/Olaf/Leviathan/Phillip/Max/Malicious)
echo "[STEP 2] optimize-core games=${GAMES} packages=${PACKAGES} tech-candidates=${TECH_CANDIDATES}"
echo "pwd=$(pwd -P) python=$(command -v python3) OUT_DIR=${OUT_DIR}"
python3 lorcana_cli.py optimize-core \
    --archetype "${ARCHETYPE}" \
    --core-file "${CORE}" \
    --baseline-file "${DECK}" \
    --top-meta "${TOP_META}" \
    --games "${GAMES}" \
    --packages "${PACKAGES}" \
    --tech-candidates "${TECH_CANDIDATES}" \
    --lookahead-depth 1 \
    --workers "${WORKERS}" \
    --seed 2137
stage_copy "results/retro_device_optimized_report.json" "retro_device_optimized_report.json"
stage_copy "decks/user_emerald_ruby_retro_device_optimized.txt" "user_emerald_ruby_retro_device_optimized.txt"

# 3. High-budget UCB1 tech scans vs week pressure matchups
run_scan() {
    local opp="$1"
    local tag="$2"
    echo "[STEP 3/${tag}] scan-tech vs ${opp} budget=${SCAN_BUDGET}"
    echo "pwd=$(pwd -P) python=$(command -v python3) OUT_DIR=${OUT_DIR}"
    python3 lorcana_cli.py scan-tech \
        --my-deck "${DECK}" \
        --opp-deck "${opp}" \
        --copies 3 \
        --budget "${SCAN_BUDGET}" \
        --lookahead-depth 1 \
        --top-k 8
    stage_copy "results/latest_tech_scan.json" "tech_scan_${tag}.json"
}

run_scan "Emerald/Steel Midrange" "emerald_steel"
run_scan "Amber/Emerald Circle of Life" "circle_of_life"
run_scan "Amber/Sapphire Detectives" "detectives"

# Persist SLURM log copies when available
if [ -n "${SLURM_JOB_ID:-}" ]; then
    if [ -f "logs/retro_device_${SLURM_JOB_ID}.out" ]; then
        cp -f "logs/retro_device_${SLURM_JOB_ID}.out" "${OUT_DIR}/" || true
    fi
    if [ -f "logs/retro_device_${SLURM_JOB_ID}.err" ]; then
        cp -f "logs/retro_device_${SLURM_JOB_ID}.err" "${OUT_DIR}/" || true
    fi
fi

echo "=================================================================="
echo "[Athena] Done — STRONG Retro Device package"
echo "[Athena] OUT_DIR (absolute): ${OUT_DIR}"
ls -la "${OUT_DIR}"
echo "[Athena] Required artifacts:"
for f in \
    core_ablation_report.json \
    retro_device_optimized_report.json \
    user_emerald_ruby_retro_device_optimized.txt \
    tech_scan_emerald_steel.json \
    tech_scan_circle_of_life.json \
    tech_scan_detectives.json
do
    require_file "${OUT_DIR}/${f}" "${f}"
done
echo "[Athena] Repo-local copies still under: ${REPO_ROOT}/results and ${REPO_ROOT}/decks"
echo "=================================================================="
