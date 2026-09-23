#!/bin/bash
#SBATCH --job-name=baselines-v2
#SBATCH --output=logs/eval_baselines_v2.out
#SBATCH --error=logs/eval_baselines_v2.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00             # longer: B6/B7 need CSV loading + adjacency build
#SBATCH --partition=batch
#SBATCH --nodes=1

#################################
### Baseline Ablation Study v2###
### (B1-B7 + V2, 8 methods)  ###
#################################
# Compare 8 downsampling strategies at V2 best subset size:
#   B1: Random Uniform              — ablates everything
#   B2: Stratified Proportional     — ablates keep-all-illicit
#   B3: Keep-All-Illicit + Random   — ablates importance weighting
#   B4: V2 w/o Hard-Neg             — ablates hard-neg oversampling
#   B5: Fogliato et al. (Neyman)    — ECCV 2024 (k-means + Neyman alloc + HT)
#   B6: Leskovec Random Walk        — KDD 2006 (topology-preserving graph sampling)
#   B7: Gao et al. Stratified       — VLDB 2019 (KG eval, stratify by Payment Format)
#   V2: Coreset + IW (ours)         — reference
#
# Each baseline × k=50 draws → ensemble + per-model evaluation
# No GPU needed — loads pre-computed test probabilities (.npy).
# B6/B7 also load raw CSV for graph structure / edge attributes (first run caches).

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
echo "=== Baseline Ablation Study v2 — Environment Check ==="
echo "Node       : $(hostname)"
echo "CPU cores  : ${N_CORES}  (SLURM: ${SLURM_CPUS_PER_TASK})"
echo "Python     : $(python --version)"
python -c "
import numpy, sklearn, pandas
print('numpy      :', numpy.__version__)
print('sklearn    :', sklearn.__version__)
print('pandas     :', pandas.__version__)
"
echo "============================================"

echo ""
echo "=================================================="
echo "  Baseline Ablation Study v2 (B1-B7 + V2)"
echo "=================================================="

python -u scripts/08_run_ablation.py \
    --datasets HI-Small LI-Small \
    --k-repeats 50

echo ""
echo "=================================================="
echo "  Baseline Ablation Study v2 DONE!"
echo "=================================================="
