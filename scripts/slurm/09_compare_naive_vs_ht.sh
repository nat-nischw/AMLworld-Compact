#!/bin/bash
#SBATCH --job-name=eval_v1v2
#SBATCH --output=logs/eval_v2_subset.out
#SBATCH --error=logs/eval_v2_subset.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00             # ~10 min expected (just numpy indexing)
#SBATCH --partition=batch
#SBATCH --nodes=1

##############################
### Eval V1 vs V2 Subsets  ###
##############################
# Compare saved model predictions on V1 (exhaustive) and V2 (coreset+IW) subsets.
#
# V1 subset : outputs/eval_subsets/{dataset}/subset_{dataset}_exhaustive_best.npy
# V2 subset : outputs/eval_subsets/{dataset}/subset_{dataset}_v2_best.npy
# V2 weights: outputs/eval_subsets/{dataset}/weights_{dataset}_v2_best.npy
#
# Loads pre-computed test probabilities (no model re-loading needed):
#   outputs/test_probs/{dataset}/{method}/seed_{seed}.npy
#
# Output:
#   outputs/eval_subsets/eval_v1_v2_subset_models.csv
#   outputs/eval_subsets/eval_v1_v2_subset_models.json
#   outputs/eval_subsets/plots/eval_v1_v2_*.png

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
echo "=== Eval V1 vs V2 Subset — Environment Check ==="
echo "Node       : $(hostname)"
echo "CPU cores  : ${N_CORES}  (SLURM: ${SLURM_CPUS_PER_TASK})"
echo "Python     : $(python --version)"
python -c "
import numpy, sklearn, matplotlib
print('numpy      :', numpy.__version__)
print('sklearn    :', sklearn.__version__)
print('matplotlib :', matplotlib.__version__)
"
echo "============================================"

echo ""
echo "=================================================="
echo "  Naive Coreset vs HT-Coreset: Per-Model Comparison"
echo "  (LightGBM+GFP, XGBoost+GFP, GCPAL+GFP)"
echo "=================================================="

python -u scripts/09_compare_naive_vs_ht.py \
    --datasets HI-Small LI-Small

echo ""
echo "=================================================="
echo "  Generating Plots"
echo "=================================================="

python -u scripts/09_compare_naive_vs_ht.py

echo ""
echo "=================================================="
echo "  Eval V1 vs V2 Subset DONE!"
echo "=================================================="
