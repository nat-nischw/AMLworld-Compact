#!/bin/bash
#SBATCH --job-name=gcpal_v7      # Job name
#SBATCH --output=logs/gcpal_v7.out  # Standard output log
#SBATCH --error=logs/gcpal_v7.err   # Error log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --time=04:00:00          # 4 hours (inference only, no training)
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${BENCH_NODE}

######################
### Design v7      ###
######################
# Inference-only: load the GCPAL+GFP fine-tuned checkpoints, run on the temporal test split.
#
# v6: trained with random split → test probs NOT aligned with LightGBM/XGBoost
# v7: loads v6 models, infers on temporal test set → aligned for 3-model ensemble
#
# Source models : outputs/models/{dataset}/GCPAL_knn/finetuned_seed_*.pt
# Output probs  : outputs/test_probs/{dataset}/GCPAL_knn_temporal/seed_*.npy
# Metrics       : outputs/metrics/{dataset}/gcpal_knn_temporal_metrics.json

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"
export PATH="${CONDA_ROOT}/envs/${AMLC_ENV:-amlc-bench}/bin:$PATH"

echo "=== GCPAL v7 Environment Check ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "================================"

echo ""
echo "=================================================="
echo "  GCPAL v7 — Temporal Inference (HI-Small + LI-Small)"
echo "=================================================="

python -u scripts/05_infer_gcpal_temporal.py \
    --datasets HI-Small LI-Small \
    --seeds 42 123 456 789 1011 \
    --workers 16

echo ""
echo "=================================================="
echo "  Ensemble: LightGBM+GFP + XGBoost+GFP + GCPAL+GFP"
echo "=================================================="

python scripts/06_score_ensemble.py \
    --datasets HI-Small LI-Small \
    --members LightGBM+GFP XGBoost+GFP GCPAL+GFP ""
echo "=================================================="
echo "  GCPAL v7 DONE!"
echo "=================================================="
