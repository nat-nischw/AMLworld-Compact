#!/bin/bash
#SBATCH --job-name=vllm_gpt20b
#SBATCH --output=logs/vllm_gpt_oss_20b.out
#SBATCH --error=logs/vllm_gpt_oss_20b.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${VLLM_NODE_4GPU}

######################################
### vLLM: GPT-OSS-20B (1 GPU)     ###
### Dense 20B/1.3B active, ~10GB   ###
### Node: ${VLLM_NODE_4GPU}              ###
######################################

cd ${PROJECT_ROOT}/

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx (https://huggingface.co/settings/tokens)}"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUTF8=1
export PATH="${CONDA_ROOT:?set CONDA_ROOT}/bin:$PATH"
export FORCE_TORCHRUN=1
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME="${HOME}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME/hub" "$HUGGINGFACE_HUB_CACHE"
chmod -R 700 "$HF_HOME"

export VLLM_CACHE_ROOT="${HOME}/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_NO_USAGE_STATS=1

eval "$(conda shell.bash hook)"
conda activate "${AMLC_ENV:-amlc-bench}"

pip install --upgrade vllm transformers accelerate huggingface_hub 2>&1 | tail -5

echo "=== vLLM GPT-OSS-20B ==="
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
echo "========================="

python -m vllm.entrypoints.openai.api_server \
  --served-model-name GPT-OSS-20B \
  --model openai/gpt-oss-20b \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
  --dtype auto \
  --port 18809 \
  --max-model-len 131072 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
