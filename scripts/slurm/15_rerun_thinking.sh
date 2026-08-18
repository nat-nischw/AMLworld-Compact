#!/bin/bash
# Re-run LLM inference with thinking mode enabled
# This re-runs ICL-AML and ICL-AML-NoFS on both datasets for a given model
# to collect reasoning traces (thinking tokens).
#
# Output lands in the archive under the run directory's own name for the
# prompting, which amlc.archive maps from ICL-FS / ICL-ZS.
# with llm_raw_response.reasoning populated.
#
# Prerequisites:
#   1. Start vLLM server with --enable-reasoning (see host_vllm_*.sh scripts)
#   2. Ensure paper_format serialized data exists:
#      python scripts/11_serialize_coreset.py --datasets HI-Small LI-Small --formats paper_format
#   3. Ensure ICL examples exist:
#      python scripts/12_build_icl_examples.py --datasets HI-Small LI-Small --seed 42
#
# Usage:
#   bash scripts/thinking_rerun/run_thinking_rerun.sh <MODEL> <HOST> <PORT> [SEED]
#
# Examples:
#   bash scripts/thinking_rerun/run_thinking_rerun.sh Qwen3.5-35B-A3B localhost 18809
#   bash scripts/thinking_rerun/run_thinking_rerun.sh GPT-OSS-20B localhost 18809 42
#   bash scripts/thinking_rerun/run_thinking_rerun.sh Nemotron-3-Nano-30B localhost 18809

set -euo pipefail
cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

export HF_HOME="${HOME}/.cache/huggingface"

MODEL="${1:?Usage: $0 <MODEL> <HOST> <PORT> [SEED]}"
HOST="${2:?Usage: $0 <MODEL> <HOST> <PORT> [SEED]}"
PORT="${3:?Usage: $0 <MODEL> <HOST> <PORT> [SEED]}"
SEED="${4:-42}"  # default: seed 42 only (sufficient for trace analysis, std < 0.5pp)

VLLM_URL="http://${HOST}:${PORT}/v1"
DATASETS="HI-Small LI-Small"
METHODS="ICL-FS ICL-ZS"

echo "============================================"
echo "Thinking Re-run: ${MODEL}"
echo "vLLM URL: ${VLLM_URL}"
echo "Datasets: ${DATASETS}"
echo "Methods:  ${METHODS}"
echo "Seed:     ${SEED}"
echo "============================================"

# Check vLLM server is reachable
echo "Checking vLLM server..."
if ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not reachable at ${VLLM_URL}"
    echo "Start the server first: bash scripts/thinking_rerun/host_vllm_*.sh"
    exit 1
fi
echo "vLLM server OK: $(curl -s ${VLLM_URL}/models | python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo 'unknown')"

SEED_ARGS="--seeds ${SEED}"

for METHOD in ${METHODS}; do
    echo ""
    echo ">>> Running: ${MODEL} / ${METHOD} / ${DATASETS}"
    echo ""

    python -u scripts/13_run_llm_eval.py \
        --mode llm \
        --promptings "${METHOD}" \
        --model "${MODEL}" \
        --vllm-url "${VLLM_URL}" \
        --datasets ${DATASETS} \
        ${SEED_ARGS}

    echo ">>> Done: ${MODEL} / ${METHOD}"
done

echo ""
echo "============================================"
echo "All done! Check outputs/${MODEL}/ for results."
echo "Verify thinking traces:"
echo "  python scripts/slurm/check_traces.py --model ${MODEL}"
echo "============================================"
