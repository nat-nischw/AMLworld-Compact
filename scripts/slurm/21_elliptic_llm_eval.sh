#!/bin/bash
#SBATCH --job-name=elliptic_llm
#SBATCH --output=logs/elliptic_llm_%j.out
#SBATCH --error=logs/elliptic_llm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1

###############################################################################
# Phase B + C: LLM evaluation + Doubt Triage on the Elliptic Bitcoin Dataset.
#
# CPU-only client; assumes a vLLM server is already hosted on another node
# (e.g. via host_vllm_gpt_oss.sh).
#
# Pipeline (single submission):
#   B1. Serialize V2 coreset → k-hop subgraph text       (~30 s)
#   B2. Wait for vLLM server at $VLLM_URL                (≤5 min)
#   B3. LLM prediction on 3,249 Elliptic cases           (~10–60 min)
#   C.  Doubt Triage eval (ML / LLM / OR / AND / Thresh / Doubt Triage)  (~10 s)
#
# Usage:
#   sbatch elliptic_llm_eval.sh GPT-OSS-120B <vllm-host> 18809
#   sbatch elliptic_llm_eval.sh Qwen3.5-397B-A17B <vllm-host> 18809 42
#
# Args:
#   $1 = MODEL_NAME  (must match --served-model-name in vLLM)
#   $2 = VLLM_HOST   (cluster node hosting vLLM)
#   $3 = VLLM_PORT   (default: 18809)
#   $4 = SEED        (default: 42)
#   $5 = WORKERS     (default: 8)
###############################################################################

set -euo pipefail

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"
mkdir -p logs

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"

# ── Args ──────────────────────────────────────────────────────────────
MODEL_NAME="${1:-GPT-OSS-120B}"
VLLM_HOST="${2:-${VLLM_NODE_4GPU:-localhost}}"
VLLM_PORT="${3:-18809}"
SEED="${4:-42}"
WORKERS="${5:-8}"
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# ── Thread control ────────────────────────────────────────────────────
N_CORES=$(nproc)
N_BLAS=$(( N_CORES / 2 ))
export OMP_NUM_THREADS=${N_BLAS}
export MKL_NUM_THREADS=${N_BLAS}
export OPENBLAS_NUM_THREADS=${N_BLAS}
export NUMEXPR_NUM_THREADS=${N_BLAS}

echo "=== Elliptic LLM + Doubt Triage ==="
echo "  Node    : $(hostname)"
echo "  Model   : ${MODEL_NAME}"
echo "  vLLM URL: ${VLLM_URL}"
echo "  Seed    : ${SEED}"
echo "==========================="

# ── Sanity: V2 coreset must already exist ─────────────────────────────
if [[ ! -f outputs/elliptic/coreset/v2.npz ]]; then
    echo "ERROR: V2 coreset not found. Run elliptic_pipeline.sh first."
    exit 1
fi

# ── B1. Serialize V2 → JSONL (idempotent) ─────────────────────────────
echo ""
echo "=========================================="
echo "  B1: serialize V2 → cases.jsonl"
echo "=========================================="
if [[ ! -f outputs/elliptic/llm_data/cases.jsonl ]]; then
    python -u scripts/elliptic/serialize_coreset.py \
        --k-hop 2 --max-neighbors 30 --seed 42
else
    echo "  cases.jsonl already exists; skip"
    wc -l outputs/elliptic/llm_data/cases.jsonl
fi

# ── B2. Wait for vLLM server ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "  B2: wait for vLLM at ${VLLM_URL}"
echo "=========================================="
MAX_WAIT=300; WAITED=0
while ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM not reachable after ${MAX_WAIT}s"
        echo "Did you submit host_vllm_<model>.sh first?"
        exit 1
    fi
    echo "  waiting... (${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done
curl -s "${VLLM_URL}/models" | python -m json.tool 2>/dev/null || true

# ── B3. LLM prediction ────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  B3: LLM prediction on Elliptic V2"
echo "=========================================="
PRED_PATH="outputs/elliptic/llm/${MODEL_NAME}/seed_${SEED}.json"
if [[ -f "${PRED_PATH}" ]]; then
    echo "  predictions already exist at ${PRED_PATH}; skipping (delete to rerun)"
else
    python -u scripts/elliptic/run_llm_eval.py \
        --model "${MODEL_NAME}" \
        --vllm-url "${VLLM_URL}" \
        --seed "${SEED}" \
        --workers "${WORKERS}" \
        --temperature 0.6 --top-p 0.95 --max-tokens 2048
fi

# ── C. Doubt Triage eval ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  C: Doubt Triage eval"
echo "=========================================="
python -u scripts/elliptic/doubt_triage.py \
    --llm-preds "${PRED_PATH}" \
    --llm-name "${MODEL_NAME}"

echo ""
echo "=========================================="
echo "  DONE."
echo "    outputs/elliptic/llm/${MODEL_NAME}/seed_${SEED}.json"
echo "    outputs/elliptic/doubt_triage/results_${MODEL_NAME}.csv"
echo "=========================================="
