#!/bin/bash
#SBATCH --job-name=g1_intervention
#SBATCH --output=${PROJECT_ROOT}/pipeline/aml/logs/g1_intervention_%j.out
#SBATCH --error=${PROJECT_ROOT}/pipeline/aml/logs/g1_intervention_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1

############################################################
### G1: Prompt Intervention Eval (ICL-V)         ###
### CPU-only; connects to running vLLM server            ###
############################################################
#
# Pre-registered hypothesis (§6.5):
#   H1 (primary):   ΔF1 ≥ +5pp on HI-Small (V vs control)
#   H2 (mechanism): Δover-prediction-rate ≤ -10pp
#   H3 (guard):     Δunder-prediction-rate ≤ +5pp
#
# Tier strategy:
#   T1 Pilot:    GPT-OSS-120B × HI-Small × seed 42                  ~1.5h
#   T2 Confirm:  + Nemotron-3-Super-120B × HI+LI × seed 42          ~3h
#   T3 Stat:     T2 cells × 5 seeds                                  ~10h
#   T4 Robust:   remaining 5 models × HI-Small × seed 42 (optional)  ~7h
#
# Usage:
#   sbatch run_intervention_eval.sh GPT-OSS-120B <vllm_host> 18809 "HI-Small" "42"
#   sbatch run_intervention_eval.sh Nemotron-3-Super-120B <h> 18809 "HI-Small LI-Small" "42"
#   sbatch run_intervention_eval.sh GPT-OSS-120B <h> 18809 "HI-Small LI-Small" "42 123 456 789 1011"
#
# Args (positional):
#   $1 = MODEL_NAME (must match --served-model-name in vLLM)
#   $2 = VLLM_HOST  (node hosting vLLM, e.g. ${VLLM_NODE_4GPU})
#   $3 = VLLM_PORT  (default 18809)
#   $4 = DATASETS   (default "HI-Small")
#   $5 = SEEDS      (default "42")

set -euo pipefail

WORK="${AMLC_REPO:?set AMLC_REPO to this checkout}"
LOG_DIR=${PROJECT_ROOT}/pipeline/aml/logs
mkdir -p "$LOG_DIR"
cd "$WORK"

# Activate conda env. 'vllm' env has working vllm 0.19.0 + torch 2.10+cu128
# (matches driver 12.8). Override via CONDA_ENV var if needed.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-vllm}"

# ── Args ──────────────────────────────────────────────────────────────────────
MODEL_NAME="${1:-GPT-OSS-120B}"
VLLM_HOST="${2:-localhost}"
VLLM_PORT="${3:-18809}"
DATASETS="${4:-HI-Small}"
SEEDS="${5:-42}"
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# ── Threading ─────────────────────────────────────────────────────────────────
N_CORES=$(nproc); N_BLAS=$(( N_CORES / 2 ))
export OMP_NUM_THREADS=${N_BLAS}
export MKL_NUM_THREADS=${N_BLAS}
export OPENBLAS_NUM_THREADS=${N_BLAS}
export NUMEXPR_NUM_THREADS=${N_BLAS}

# ── PYTHONPATH (src/ must be importable) ──────────────────────────────────────
export PYTHONPATH="${WORK}/src:${PYTHONPATH:-}"

echo "=== G1 Prompt Intervention Eval ==="
echo "Repo       : ${WORK}"
echo "Node       : $(hostname)"
echo "Model      : ${MODEL_NAME}"
echo "vLLM URL   : ${VLLM_URL}"
echo "Datasets   : ${DATASETS}"
echo "Seeds      : ${SEEDS}"
echo "Methods    : ICL-V (pre-registered intervention)"
echo "Log dir    : ${LOG_DIR}"
echo "==================================="

# ── Wait for vLLM server ──────────────────────────────────────────────────────
echo "Checking vLLM server at ${VLLM_URL}..."
MAX_WAIT=300; WAITED=0
while ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server not reachable after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 10; WAITED=$((WAITED + 10))
done
echo "vLLM ready! Models:"
curl -s "${VLLM_URL}/models" | python -m json.tool 2>/dev/null || echo "(could not parse)"

# ── Run evaluation ────────────────────────────────────────────────────────────
echo ""
echo "==================================="
echo "  Running ICL-V"
echo "==================================="
python -u scripts/13_run_llm_eval.py \
    --mode llm \
    --vllm-url "${VLLM_URL}" \
    --model "${MODEL_NAME}" \
    --datasets ${DATASETS} \
    --seeds ${SEEDS} \
    --workers 14 \
    --promptings ICL-V

EXIT=$?
echo ""
echo "==================================="
echo "  Done (exit ${EXIT})"
echo "  Output: ${WORK}/outputs/${MODEL_NAME}/ICL-V/"
echo "==================================="
exit $EXIT
