#!/bin/bash
#SBATCH --job-name=gcpal_v6      # Job name
#SBATCH --output=logs/gcpal_v6.out  # Standard output log
#SBATCH --error=logs/gcpal_v6.err   # Error log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --time=1-00:00:00        # 1 day
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${BENCH_NODE}

######################
### Design v6      ###
######################
# KNN 3rd contrastive view (paper's full method)
#
# v5 best (no pretrain, no KNN):
#   HI-Small: Ens F1=0.6702, AUCPR=0.6140
#
# v6: enable pretrain + KNN 3-view contrastive
# KNN builds cosine-similarity graph from node features as 3rd view:
#   L = lam * L_NCE(G'1, G'2) + (1-lam) * L_NCE(G'2, G_KNN)
#
# Output folder: the archive's GCPAL_knn/, which amlc.archive maps from GCPAL+GFP

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"
export PATH="${CONDA_ROOT}/envs/${AMLC_ENV:-amlc-bench}/bin:$PATH"

echo "=== GCPAL v6 Environment Check ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "================================"

echo ""
echo "=================================================="
echo "  GCPAL v6 — KNN 3-view contrastive (paper method)"
echo "=================================================="

# ═══════════════════════════════════════════════════
#  Part 1: HI-Small — GIN + edge feats + pretrain + KNN
#  Same as v5 but WITH pretrain and KNN 3rd view
# ═══════════════════════════════════════════════════

echo ""
echo "--- Part 1: HI-Small, GIN + edge feats + pretrain + KNN, 2L/128, lg_k=5, w=5 ---"
python -u scripts/04_train_gcpal.py \
    --dataset HI-Small \
    --seeds 42 123 456 789 1011 \
    --hidden-dim 128 \
    --num-layers 2 \
    --pretrain-epochs 200 \
    --finetune-epochs 500 \
    --w-illicit 5.0 \
    --patience 80 \
    --lg-k 5 \
    --use-knn --knn-k 5 --lam 0.5 \
    --random-split \
    --workers 16 \
    --output-tag knn

# ═══════════════════════════════════════════════════
#  Part 2: LI-Small — same config
# ═══════════════════════════════════════════════════

echo ""
echo "--- Part 2: LI-Small, GIN + edge feats + pretrain + KNN, 2L/128, lg_k=5, w=5 ---"
python -u scripts/04_train_gcpal.py \
    --dataset LI-Small \
    --seeds 42 123 456 789 1011 \
    --hidden-dim 128 \
    --num-layers 2 \
    --pretrain-epochs 200 \
    --finetune-epochs 500 \
    --w-illicit 5.0 \
    --patience 80 \
    --lg-k 5 \
    --use-knn --knn-k 5 --lam 0.5 \
    --random-split \
    --workers 16 \
    --output-tag knn

echo ""
echo "=================================================="
echo "  GCPAL v6 DONE!"
echo "=================================================="
