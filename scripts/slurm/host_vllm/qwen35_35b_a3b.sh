#!/bin/bash
#SBATCH --job-name=vllm_qwen35_35b
#SBATCH --output=logs/vllm_qwen35_35b.out
#SBATCH --error=logs/vllm_qwen35_35b.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=14-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${VLLM_NODE_4GPU}
#SBATCH --gres-flags=enforce-binding

################################################
### vLLM: Qwen3.5-35B-A3B FP8 (4 GPU)      ###
### MoE 35B/3B active, Gated DeltaNet       ###
### ~35GB FP8, 262K context                  ###
### Thinking mode: temp=0.6, top_p=0.95,    ###
###   top_k=20 — model card recommended     ###
### Node: ${VLLM_NODE_4GPU}                        ###
################################################

cd ${PROJECT_ROOT}/

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx (https://huggingface.co/settings/tokens)}"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUTF8=1
export PATH="${CONDA_ROOT:?set CONDA_ROOT}/bin:$PATH"
export FORCE_TORCHRUN=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# ── CUDA paths for FlashInfer JIT linking (libcudart) ────────────────────────
export CUDA_HOME="/usr/local/cuda-12.8"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HOME}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME/hub" "$HUGGINGFACE_HUB_CACHE"
chmod -R 700 "$HF_HOME"

export VLLM_CACHE_ROOT="${HOME}/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_NO_USAGE_STATS=1

eval "$(conda shell.bash hook)"
conda activate "${AMLC_ENV:-amlc-bench}"

# ── Upgrade libs for new model support ────────────────────────────────────────
# NOTE: triton constexpr_function already patched in env
pip install --upgrade vllm transformers accelerate huggingface_hub 2>&1 | tail -5

echo "=== vLLM Qwen3.5-35B-A3B FP8 ==="
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "--- GPU Health Check ---"
nvidia-smi
echo "--- CUDA Test ---"
python -c "
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1
echo "--- vLLM version ---"
python -c "import vllm; print('vLLM:', vllm.__version__)"
echo "====================================="

# Qwen3.5-35B-A3B FP8: MoE with Gated DeltaNet (same family as 397B-A17B)
# Native 262K context, FP8 quantized (~35GB)
# TP=4 on 4x H100 80GB (~9GB/GPU, plenty of room for KV cache)
python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen3.5-35B-A3B \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
  --dtype auto \
  --port 18810 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
