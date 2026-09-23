#!/bin/bash
#SBATCH --job-name=downsample_v2
#SBATCH --output=logs/downsample_v2.out
#SBATCH --error=logs/downsample_v2.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # lightweight — no heavy multiprocessing
#SBATCH --time=02:00:00             # ~30 min expected
#SBATCH --partition=batch
#SBATCH --nodes=1

######################
### Design  (V2)   ###
######################
# Stratified Coreset + Importance-Weighted Downsampling
#
# Key differences from V1 (scripts/07b_build_naive_coreset.py):
#   1. Keeps ALL illicit edges  → exact recall (no sampling)
#   2. Smart-samples benign edges with hard-negative oversampling
#   3. Importance weighting  → provably unbiased P, R, F1
#   4. Tracks ΔPrecision, ΔRecall, ΔF1 separately
#
# This is CPU-light (no fork pool, no 1M-seed search).
# Each benign size is evaluated k=50 times for stability.
# Total work: ~12 sizes × 50 draws × 2 datasets = ~1,200 evaluations.
#
# Theory: Horvitz-Thompson estimator (1952)
#   w_illicit = 1.0  (all kept)
#   w_benign  = N_stratum / n_selected_from_stratum
#   Weighted P, R, F1 = unbiased estimates of full-set metrics
#
# Input  : outputs/test_probs/{dataset}/{method}/seed_*.npy
# Output : outputs/eval_subsets/{dataset}/subset_{dataset}_v2_best.npy
#          outputs/eval_subsets/{dataset}/weights_{dataset}_v2_best.npy
#          outputs/eval_subsets/{dataset}/downsample_v2_summary_{dataset}.json
#          outputs/eval_subsets/downsample_v2_summary_all.json

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"
export PATH="${CONDA_ROOT}/envs/${AMLC_ENV:-amlc-bench}/bin:$PATH"

# ── Thread control ──────────────────────────────────────────────────────────
N_CORES=$(nproc)
N_BLAS=$(( N_CORES / 2 ))
export OMP_NUM_THREADS=${N_BLAS}
export MKL_NUM_THREADS=${N_BLAS}
export OPENBLAS_NUM_THREADS=${N_BLAS}
export NUMEXPR_NUM_THREADS=${N_BLAS}

# ── Environment check ───────────────────────────────────────────────────────
echo "=== Downsample V2 — Environment Check ==="
echo "Node       : $(hostname)"
echo "CPU cores  : ${N_CORES}  (SLURM: ${SLURM_CPUS_PER_TASK})"
echo "Python     : $(python --version)"
python -c "
import numpy, sklearn
print('numpy   :', numpy.__version__)
print('sklearn :', sklearn.__version__)
"
echo "============================================"

echo ""
echo "=================================================="
echo "  Downsample V2: Stratified Coreset + IW"
echo "  (keep all illicit + smart-sample benign)"
echo "=================================================="

python -u scripts/07_build_ht_coreset.py \
    --datasets HI-Small LI-Small \
    --members LightGBM+GFP XGBoost+GFP GCPAL+GFP \
    --n-ben-sizes 12 \
    --hard-neg-ratio 0.30 \
    --metric-tol 0.05 \
    --k-repeats 50

echo ""
echo "=================================================="
echo "  Downsample V2 DONE!"
echo "=================================================="
