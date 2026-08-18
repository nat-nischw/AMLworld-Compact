#!/bin/bash
#SBATCH --job-name=vllm_nemotron3_nano
#SBATCH --output=logs/vllm_nemotron3_nano.out
#SBATCH --error=logs/vllm_nemotron3_nano.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=14-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${VLLM_NODE_4GPU}

###################################################################
### vLLM: Nemotron-3-Nano-30B-A3B FP8 (4 GPU)                  ###
###                                                             ###
### Architecture: Hybrid Mamba-2 + Transformer LatentMoE        ###
###   30B total, 3B active (same family as Super-120B)          ###
### VRAM: ~30GB FP8 → TP=4 on H100 80GB (~8GB/GPU)            ###
### Reasoning: --reasoning-parser super_v3                      ###
### Sampling: temp=1.0, top_p=0.95 (model card recommended)    ###
### Node: ${VLLM_NODE_4GPU}                                           ###
###                                                             ###
### Ref: same cookbook as Nemotron-3-Super                       ###
### Usage: sbatch host_vllm_nemotron3_nano.sh                   ###
###################################################################

cd ${PROJECT_ROOT}/

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx (https://huggingface.co/settings/tokens)}"
export PYTHONUTF8=1
export PATH="${CONDA_ROOT:?set CONDA_ROOT}/bin:$PATH"

# ── CUDA paths (required for FlashInfer kernel JIT) ──────────────────────────
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HOME}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME/hub" "$HUGGINGFACE_HUB_CACHE"
chmod -R 700 "$HF_HOME"

export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

export VLLM_CACHE_ROOT="${HOME}/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_NO_USAGE_STATS=1

eval "$(conda shell.bash hook)"
conda activate "${AMLC_ENV:-amlc-bench}"

# ── Environment info ─────────────────────────────────────────────────────────
echo ""
echo "=== vLLM Nemotron-3-Nano-30B-A3B FP8 TP=4 ==="
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "--- Version Check ---"
python -c "
import vllm; print('vLLM:', vllm.__version__)
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
try:
    import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)
except: print('mamba_ssm: NOT INSTALLED')
try:
    import causal_conv1d; print('causal_conv1d:', causal_conv1d.__version__)
except: print('causal_conv1d: NOT INSTALLED')
try:
    import flashinfer; print('flashinfer:', getattr(flashinfer, '__version__', 'installed'))
except: print('flashinfer: NOT INSTALLED')
" 2>&1
echo "================================================="

# Clear stale caches (FlashInfer JIT + torch inductor)
rm -rf /tmp/torchinductor_${USER} 2>/dev/null || true
rm -rf ${HOME}/.cache/flashinfer/0.6.4/90a/cached_ops/trtllm_comm 2>/dev/null || true

# ── Download custom reasoning parser plugin ──────────────────────────────────
# Same super_v3 parser as Nemotron-3-Super (same family)
wget -q -nc -O ${PROJECT_ROOT}/super_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/raw/main/super_v3_reasoning_parser.py \
  2>&1 || true

# ── Launch vLLM Server ────────────────────────────────────────────────────────
# Same architecture family as Super-120B but smaller (30B/3B active)
# FP8 variant: TP=4 on 4x H100 80GB (~8GB/GPU, plenty of room for KV cache)
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --swap-space 0 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-seqs 512 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
  --served-model-name Nemotron-3-Nano-30B \
  --host 0.0.0.0 \
  --port 18811 \
  --reasoning-parser-plugin ${PROJECT_ROOT}/super_v3_reasoning_parser.py \
  --reasoning-parser super_v3
