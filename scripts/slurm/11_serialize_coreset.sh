#!/bin/bash
#SBATCH --job-name=serialize_v2
#SBATCH --output=logs/serialize_v2_subset.out
#SBATCH --error=logs/serialize_v2_subset.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1

#################################
### V2 Subset → Graph Serial. ###
#################################
# Take V2 downsampled subset (best) → load AMLworld data →
# extract k-hop subgraphs → serialize to all 4 text formats
# (edge_list, structured, json, adjacency) → save JSONL datasets
# + GFP features + metadata for LLM evaluation.
#
# No GPU needed — graph loading, k-hop extraction, serialization.
#
# Input:
#   outputs/eval_subsets/{dataset}/subset_{dataset}_v2_best.npy
#   outputs/eval_subsets/{dataset}/weights_{dataset}_v2_best.npy
#   outputs/test_probs/{dataset}/test_labels.npy
#   data/{dataset}_Trans.csv + {dataset}_Patterns.txt
#   data/gfp_cache/{dataset}_gfp_b128.npy  (cached GFP features)
#
# Output:
#   outputs/llm_datasets/{dataset}/v2_subset/
#     cases_{format}.jsonl
#     features_gfp.npz
#     labels.npy, weights.npy
#     metadata.json, case_index.csv
#   outputs/llm_datasets/v2_subset_summary.json

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
echo "=== V2 Subset Graph Serialization — Environment Check ==="
echo "Node       : $(hostname)"
echo "CPU cores  : ${N_CORES}  (SLURM: ${SLURM_CPUS_PER_TASK})"
echo "Python     : $(python --version)"
python -c "
import numpy, networkx, pandas
print('numpy      :', numpy.__version__)
print('networkx   :', networkx.__version__)
print('pandas     :', pandas.__version__)
"
echo "============================================"

echo ""
echo "=================================================="
echo "  V2 Subset → Graph Serialization for LLM"
echo "=================================================="

python -u scripts/11_serialize_coreset.py \
    --datasets HI-Small LI-Small \
    --workers 8

echo ""
echo "=================================================="
echo "  V2 Subset Graph Serialization DONE!"
echo "=================================================="
