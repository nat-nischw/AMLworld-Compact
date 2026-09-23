#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --output=logs/downsample.out
#SBATCH --error=logs/downsample.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64         # all cores on node; Python uses 80%
#SBATCH --time=72:00:00            # ~20h expected (k=1M, fork pool)
#SBATCH --partition=batch
#SBATCH --nodes=1

######################
### Design         ###
######################
# CPU-only downsampling pipeline.  No GPU required.
#
# Search algorithm: exhaustive (3-phase binary search + dense ascending scan)
#   Phase 1: Geometric binary search (k=50/point) → [lo, hi] boundary
#   Phase 2: Dense ascending scan [lo*0.95, hi*1.05] (k=1,000,000/point)
#             → stops at first feasible n_illicit = global minimum
#   Phase 3: High-K confirmation (k=5,000,000) → best canonical subset
#
# Parallelism: multiprocessing.Pool with fork (no GIL, true parallelism)
#   - Datasets run SEQUENTIALLY (each gets full CPU for seed parallelism)
#   - Workers = 80% of detected CPU cores
#   - ~386 evals/sec at 76 workers → ~43 min/candidate at k=1M
#   - P(miss) < exp(-1e6 × P_feasible) ≈ 0 for any P_feasible > 1e-5
#   - OMP/MKL threads capped to avoid BLAS oversubscription
#
# Input  : outputs/test_probs/{dataset}/{method}/seed_*.npy
# Output : outputs/eval_subsets/{dataset}/subset_{dataset}_exhaustive_best.npy
#          outputs/eval_subsets/{dataset}/downsample_summary_{dataset}.json
#          outputs/eval_subsets/downsample_summary_all.json

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"
export PATH="${CONDA_ROOT}/envs/${AMLC_ENV:-amlc-bench}/bin:$PATH"

# ── Thread control ──────────────────────────────────────────────────────────
# With 2 ProcessPoolExecutor subprocesses each running numpy BLAS ops,
# cap BLAS threads per process so total = 2 × N_BLAS ≤ N_CORES.
N_CORES=$(nproc)
N_BLAS=$(( N_CORES / 2 ))          # 32 BLAS threads per subprocess
export OMP_NUM_THREADS=${N_BLAS}
export MKL_NUM_THREADS=${N_BLAS}
export OPENBLAS_NUM_THREADS=${N_BLAS}
export NUMEXPR_NUM_THREADS=${N_BLAS}

# ── Environment check ───────────────────────────────────────────────────────
echo "=== Downsample Eval — Environment Check ==="
echo "Node       : $(hostname)"
echo "CPU cores  : ${N_CORES}  (SLURM allocated: ${SLURM_CPUS_PER_TASK})"
echo "BLAS/OMP   : ${N_BLAS} threads/process  (2 processes × ${N_BLAS} = ${N_CORES})"
echo "Python     : $(python --version)"
python -c "
import os, numpy, sklearn, lightgbm
print('numpy   :', numpy.__version__)
print('sklearn :', sklearn.__version__)
print('lightgbm:', lightgbm.__version__)
print('Workers (80%) :', max(1, int((os.cpu_count() or 4) * 0.8)))
try:
    import optuna; print('optuna  :', optuna.__version__)
except ImportError:
    print('optuna  : NOT INSTALLED (not needed for exhaustive mode)')
"
echo "============================================"

echo ""
echo "=================================================="
echo "  Exhaustive Downsampling — HI-Small + LI-Small"
echo "  (sequential datasets, k=1,000,000/point, fork pool)"
echo "=================================================="

python -u scripts/07b_build_naive_coreset.py \
    --datasets HI-Small LI-Small \
    --members LightGBM+GFP XGBoost+GFP GCPAL+GFP \
    --n-sizes 8 \
    --search-mode exhaustive \
    --k-per-point 1000000 \
    --kl-dif-thresh 0.15 \
    --f1-tol 0.05

echo ""
echo "=================================================="
echo "  Downsample Eval DONE!"
echo "=================================================="
