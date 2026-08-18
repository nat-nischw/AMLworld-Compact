#!/bin/bash
#SBATCH --job-name=ml_aml     # Job name
#SBATCH --output=logs/ml_aml.out     # Standard output and error log
#SBATCH --error=logs/ml_aml.err      # Error log
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=32           # CPU cores
#SBATCH --gres=gpu:1                 # Request 1 GPU (H100 80GB)
#SBATCH --time=7-00:00:00            # 1 weeks for full experiment
#SBATCH --partition=batch            # Partition to submit to
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --nodelist=${BENCH_NODE}       # Required: conda env only on this node

######################
### Paper Design   ###
######################
# Reproducing AMLworld paper (2306.16424) ML baselines with temporal split.
#
# Pipeline:
#   Phase 0c: Hyperparameter tuning (random search + 3-fold CV on 80%)
#   Phase 1:  Train ML on train+val (80%), eval on full test (20%)
#   Phase 2:  LLM evaluation on sampled test (50:450 from same T2-T3)
#
# Temporal Split (60/20/20):
#   Train (0-60%):  HP tuning fitting
#   Val (60-80%):   HP tuning scoring + threshold optimization
#   Train+Val (0-80%): final model training (Phase 1)
#   Test (80-100%): evaluation (ML=full, LLM=sampled)
#
# Methods (9 total):
#   Non-LLM (3): LightGBM+GFP, XGBoost+GFP, PNA

######################
### Set environment ###
######################

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"
chmod -R 700 "$HF_HOME"

# Source conda
source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"
export PATH="${CONDA_ROOT}/envs/${AMLC_ENV:-amlc-bench}/bin:$PATH"

# Sanity-check
echo "=== Environment Check ==="
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
echo "========================="

######################
### Run Experiments ###
######################

# ============================================
# PHASE 0c: Hyperparameter Tuning
# ============================================
# Random search + 3-fold CV on train+val (80%)
# Saves: best_params, best_threshold (CV avg), actual_n_rounds (CV early stop avg)
# Results: data/tuned_params/{variant}/{model}.json

# echo ""
# echo "=========================================="
# echo "  PHASE 0c: Hyperparameter Tuning"
# echo "=========================================="

# # GBT tuning (wider ranges matching paper Table 10, 60 trials)
# python -u scripts/02_tune_hyperparams.py --dataset HI-Small --models lightgbm --n-candidates 60 --workers 16
# python -u scripts/02_tune_hyperparams.py --dataset HI-Small --models xgboost --n-candidates 60 --workers 16
# python -u scripts/02_tune_hyperparams.py --dataset LI-Small --models lightgbm --n-candidates 60 --workers 16
# python -u scripts/02_tune_hyperparams.py --dataset LI-Small --models xgboost --n-candidates 60 --workers 16

# echo "  Tuning complete."

# ============================================
# PHASE 1: Non-LLM Baselines — Full Test Set
# ============================================
# Train on train+val (80%) with actual_n_rounds from CV early stopping
# XGBoost: tuned threshold from Phase 0c (CV avg)
# LightGBM: val re-optimize (near oracle)
# PNA: transductive on 80% graph, loss on 60%, 500 epochs + eval_every=20 + early stopping
# Includes oracle F1 diagnostic per seed

echo ""
echo "=========================================="
echo "  PHASE 1: Non-LLM (tuned) — Full Test Set"
echo "=========================================="


python -u scripts/03_train_ml_baselines.py \
    --mode supervised \
    --members LightGBM+GFP XGBoost+GFP GCPAL+GFP \
    --datasets HI-Small LI-Small \
    --seeds 42 123 456 789 1011 \
    
    --workers 8

# ============================================
# PHASE 2: LLM Baselines — Sampled Test Set
# ============================================
# Same T2-T3 test region, downsampled to 500 cases (50:450)
# Uncomment below when vLLM server is ready

# VLLM_URL="http://localhost:8809/v1"
# MODEL="Qwen3-30B-A3B-Instruct-2507"
# MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
#
# echo ""
# echo "=========================================="
# echo "  PHASE 2: LLM on HI-Small and LI-Small"
# echo "=========================================="
#

echo ""
echo "=========================================="
echo "  DONE!"
echo "=========================================="
