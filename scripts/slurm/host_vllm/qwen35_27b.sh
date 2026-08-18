#!/bin/bash
#SBATCH --job-name=vllm_qwen35_27b
#SBATCH --output=logs/vllm_qwen35_27b.out
#SBATCH --error=logs/vllm_qwen35_27b.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=14-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${VLLM_NODE_4GPU}

################################################
### vLLM: Qwen3.5-27B-FP8 (4 GPU)           ###
### Dense 27B, Gated DeltaNet hybrid         ###
### Thinking model, ~27GB FP8                ###
### 256K native context                      ###
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

# CUDA paths for FlashInfer JIT kernel linking
export CUDA_HOME="/usr/local/cuda"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HOME}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME/hub" "$HUGGINGFACE_HUB_CACHE"
chmod -R 700 "$HF_HOME"

export VLLM_CACHE_ROOT="${HOME}/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_NO_USAGE_STATS=1

eval "$(conda shell.bash hook)"
conda activate "${AMLC_ENV:-amlc-bench}"

# ── Do NOT upgrade vLLM (nightly breaks CUDA 12.8 driver) ──
# vLLM 0.19.0 + torch 2.10.0+cu128 is the working combo
pip install importlib_metadata 2>&1 | tail -1

echo "=== vLLM Qwen3.5-27B ==="
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
echo "=========================="

# Clear stale torchinductor cache
rm -rf /tmp/torchinductor_${USER} 2>/dev/null || true

# Qwen3.5-27B: Gated DeltaNet + full attention hybrid (3:1 ratio)
# Native 256K context (max_position_embeddings=262144), rope_theta=10M
# --enforce-eager required: DeltaNet layers have CUDA graph / torch.compile
# bugs in vLLM (issues #35820, #35238, #36010). ~2x slower but stable.
# max-model-len capped at 131072 to keep VRAM reasonable on 4 GPU.
python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen3.5-27B \
  --model Qwen/Qwen3.5-27B-FP8 \
  --reasoning-parser qwen3 \
  --dtype auto \
  --port 18809 \
  --max-model-len 131072 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
