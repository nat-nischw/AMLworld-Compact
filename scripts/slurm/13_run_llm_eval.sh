#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --output=logs/llm_eval_%j.out
#SBATCH --error=logs/llm_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1

########################################
### LLM Eval (CPU-only)              ###
### Connects to running vLLM server  ###
########################################
#
# Usage:
#   sbatch run_llm_eval.sh GPT-OSS-120B ${VLLM_NODE_4GPU} 18809
#   sbatch run_llm_eval.sh Qwen3.5-397B-A17B ${VLLM_NODE_8GPU} 18809
#   sbatch run_llm_eval.sh Kimi-K2-Thinking ${VLLM_NODE_8GPU} 18809
#   sbatch run_llm_eval.sh Qwen3.5-27B ${VLLM_NODE_4GPU} 18809
#   sbatch run_llm_eval.sh Nemotron-3-Super-120B ${VLLM_NODE_8GPU} 18809
#   sbatch run_llm_eval.sh Nemotron-3-Nano-30B ${VLLM_NODE_4GPU} 18809
#   sbatch run_llm_eval.sh Qwen3.5-35B-A3B ${VLLM_NODE_4GPU} 18809
#   sbatch run_llm_eval.sh GLM-4.7-Flash ${VLLM_NODE_4GPU} 18810 "ICL-FS ICL-ZS" "HI-Small LI-Small"
#   sbatch run_llm_eval.sh Llama-4-Scout ${VLLM_NODE_4GPU} 18811 "ICL-FS ICL-ZS" "HI-Small LI-Small"
#   sbatch run_llm_eval.sh GPT-OSS-120B ${VLLM_NODE_4GPU} 18809 "ICL-FS"
#   sbatch run_llm_eval.sh Qwen3.5-397B-A17B ${VLLM_NODE_8GPU} 18809 "ICL-ZS"
#
#   # Re-run LI-Small only (after HI-Small already done):
#   sbatch run_llm_eval.sh Qwen3.5-397B-A17B ${VLLM_NODE_8GPU} 18809 "ICL-FS" "HI-Small LI-Small"
#   sbatch run_llm_eval.sh Qwen3.5-397B-A17B ${VLLM_NODE_8GPU} 18809 "ICL-ZS" "HI-Small LI-Small"
#   sbatch run_llm_eval.sh Nemotron-3-Super-120B ${VLLM_NODE_4GPU} 18809 "ICL-ZS" "HI-Small LI-Small"
#   sbatch run_llm_eval.sh GPT-OSS-20B ${VLLM_NODE_4GPU} 18809 "ICL-FS ICL-ZS" "HI-Small LI-Small"
#
# Args:
#   $1 = MODEL_NAME  (must match --served-model-name in vLLM)
#   $2 = VLLM_HOST   (node running vLLM server)
#   $3 = VLLM_PORT   (default: 18809)
#   $4 = PROMPTINGS  (optional: any of ICL-FS ICL-ZS ICL-V)
#   $5 = DATASETS    (optional: default "HI-Small LI-Small", e.g. "LI-Small" for re-run)

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"
export PATH="${CONDA_ROOT}/envs/${AMLC_ENV:-amlc-bench}/bin:$PATH"

# ── Args ──────────────────────────────────────────────────────────────────────
MODEL_NAME="${1:-GPT-OSS-120B}"
VLLM_HOST="${2:-${VLLM_NODE_8GPU}}"
VLLM_PORT="${3:-18809}"
METHODS="${4:-}"   # optional: any of ICL-FS ICL-ZS ICL-V
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# ── Thread control ────────────────────────────────────────────────────────────
N_CORES=$(nproc)
N_BLAS=$(( N_CORES / 2 ))
export OMP_NUM_THREADS=${N_BLAS}
export MKL_NUM_THREADS=${N_BLAS}
export OPENBLAS_NUM_THREADS=${N_BLAS}
export NUMEXPR_NUM_THREADS=${N_BLAS}

# ── Environment check ─────────────────────────────────────────────────────────
echo "=== LLM Evaluation ==="
echo "Node       : $(hostname)"
echo "Model      : ${MODEL_NAME}"
echo "vLLM URL   : ${VLLM_URL}"
echo "CPU cores  : ${N_CORES}"
echo "Python     : $(python --version)"
echo "======================"

# ── Wait for vLLM server to be ready ──────────────────────────────────────────
echo ""
echo "Checking vLLM server at ${VLLM_URL}..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server not reachable after ${MAX_WAIT}s"
        exit 1
    fi
    echo "  Waiting for vLLM server... (${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done
echo "vLLM server ready! Models:"
curl -s "${VLLM_URL}/models" | python -m json.tool 2>/dev/null || echo "(could not parse)"

# ── Run evaluation ─────────────────────────────────────────────────────────────
#
# Re-run mode: to resume after a crash (e.g. disk full), change --datasets
# to only the remaining dataset(s). Results merge via save_final_summary().
#
# FULL run (both datasets):
#   DATASETS="HI-Small LI-Small"
#
# Re-run LI-Small only (HI-Small already done):
#   DATASETS="LI-Small"
#
DATASETS="${5:-HI-Small LI-Small}"
SEEDS="${6:-42 123 456 789 1011}"
WORKERS="${7:-14}"

echo ""
echo "=================================================="
echo "  Running LLM Eval: ${MODEL_NAME}"
echo "  Datasets: ${DATASETS}"
echo "  Seeds: ${SEEDS}"
echo "=================================================="

# Build methods arg if specified
METHODS_ARG=""
if [ -n "${METHODS}" ]; then
    METHODS_ARG="--promptings ${METHODS}"
    echo "  Methods: ${METHODS}"
fi

python -u scripts/13_run_llm_eval.py \
    --mode llm \
    --vllm-url "${VLLM_URL}" \
    --model "${MODEL_NAME}" \
    --datasets ${DATASETS} \
    --seeds ${SEEDS} \
    --workers ${WORKERS:-8} \
    ${METHODS_ARG}

echo ""
echo "=================================================="
echo "  LLM Eval DONE: ${MODEL_NAME}"
echo "=================================================="
