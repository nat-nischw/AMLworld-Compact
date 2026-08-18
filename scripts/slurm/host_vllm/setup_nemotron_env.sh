#!/bin/bash
#SBATCH --job-name=setup_nemo_env
#SBATCH --output=logs/setup_nemo_env_%j.out
#SBATCH --error=logs/setup_nemo_env_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${VLLM_NODE_4GPU}

##############################################################
### Create dedicated conda env for Nemotron-3-Super        ###
### Uses vLLM nightly wheel from model card                ###
###                                                        ###
### Usage: sbatch setup_nemotron_env.sh                    ###
##############################################################

set -e

export PATH="${CONDA_ROOT:?set CONDA_ROOT}/bin:$PATH"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

export CUDA_HOME="/usr/local/cuda-12.8"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)"

ENV_NAME="${AMLC_VLLM_NEMO_ENV:-amlc-vllm-nemo}"

echo "=== Setting up ${ENV_NAME} for Nemotron-3-Super ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Check if env already exists
if conda env list | grep -q "${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists, activating..."
    conda activate "${ENV_NAME}"
else
    echo "Creating new conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.12 -y
    conda activate "${ENV_NAME}"
fi

echo ""
echo "=== Installing vLLM nightly (model card recommended) ==="
# Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
pip install -U vllm --extra-index-url https://wheels.vllm.ai/097eb544e9a22810c9b7a59e586b61627b308362 2>&1 | tail -20

echo ""
echo "=== Installing additional dependencies ==="
pip install transformers accelerate huggingface_hub 2>&1 | tail -5

echo ""
echo "=== Installing Mamba-2 dependencies ==="
pip install mamba_ssm causal_conv1d 2>&1 | tail -5

echo ""
echo "=== Verifying installation ==="
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

# Check NemotronH support
try:
    from vllm.model_executor.models.registry import ModelRegistry
    assert 'NemotronHForCausalLM' in ModelRegistry.models
    print('NemotronHForCausalLM: REGISTERED')
except:
    print('NemotronHForCausalLM: NOT FOUND')

# Check FlashInfer version
try:
    import flashinfer
    print('FlashInfer:', flashinfer.__version__ if hasattr(flashinfer, '__version__') else 'installed (no version)')
except:
    print('FlashInfer: NOT INSTALLED')
" 2>&1

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run: sbatch host_vllm_nemotron3_super.sh"
