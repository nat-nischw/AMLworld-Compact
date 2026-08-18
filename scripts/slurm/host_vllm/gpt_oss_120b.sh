#!/bin/bash
#SBATCH --job-name=vllm_gpt_oss
#SBATCH --output=logs/vllm_gpt_oss.out
#SBATCH --error=logs/vllm_gpt_oss.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=14-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${VLLM_NODE_4GPU}

######################################
### vLLM: GPT-OSS-120B (4 GPU)    ###
### MoE 117B/5.1B active, ~61GB   ###
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
# 'vllm' env has working vllm 0.19.0 + torch 2.10+cu128 (matches driver 12.8).
# Avoid 'amlc-vllm' which pulls vllm 0.20.0 cu13 (incompatible with our driver).
conda activate vllm

# ── DO NOT auto-upgrade vllm here ─────────────────────────────────────────────
# Previous version of this script ran `pip install --upgrade vllm` which pulled
# vllm 0.20.0 (built for CUDA 13, broken on our driver 12.8). vllm 0.19.0 in the
# 'vllm' env already supports GPT-OSS-120B.

echo "=== vLLM GPT-OSS-120B ==="
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
echo "========================="

python -m vllm.entrypoints.openai.api_server \
  --served-model-name GPT-OSS-120B \
  --model openai/gpt-oss-120b \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
  --dtype auto \
  --port 18809 \
  --max-model-len 131072 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
