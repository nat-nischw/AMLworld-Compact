#!/bin/bash
#SBATCH --job-name=elliptic_v2
#SBATCH --output=logs/elliptic_v2.out
#SBATCH --error=logs/elliptic_v2.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --nodelist=${BENCH_NODE}

###############################################################################
# AMLworld-Compact — Real-data validation on the Elliptic Bitcoin Dataset
#
# Phases (CPU-only, ~1–2 hours total):
#   1. Download Elliptic CSVs (PyG mirror)
#   2. Prepare temporal split (t=1..34 train, 35..49 test)
#   3. Train LightGBM + XGBoost ensemble (5 seeds)
#   4. Build V2 stratified coreset + 7 ablation baselines (k=50 draws)
#   5. Evaluate ΔP/ΔR/ΔF1 vs full test set
#
# Optional phases (require vLLM hosting; see pipeline/slurm/host_vllm_*.sh):
#   6. Serialize the V2 subset to LLM-ready edge_list text
#   7. Run Doubt Triage eval (after LLM predictions are generated)
###############################################################################

set -euo pipefail

cd "${AMLC_REPO:?set AMLC_REPO to this checkout}"
mkdir -p logs

source "${CONDA_ROOT:?set CONDA_ROOT to your conda installation, see .env.example}/etc/profile.d/conda.sh"
conda activate "${AMLC_ENV:-amlc-bench}"

echo "=== Environment ==="
python -c "import lightgbm, xgboost, sklearn, numpy; print('lightgbm', lightgbm.__version__, '/ xgboost', xgboost.__version__, '/ numpy', numpy.__version__)"
echo "==================="

# ── Phase 1 — download ────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  PHASE 1: download Elliptic CSVs"
echo "=========================================="
python -u scripts/elliptic/download_elliptic.py --source pyg

# ── Phase 2 — prepare ────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  PHASE 2: temporal split (1-34 / 35-49)"
echo "=========================================="
python -u scripts/elliptic/prepare_elliptic.py

# ── Phase 3 — ML baselines ───────────────────────────────────────────
echo ""
echo "=========================================="
echo "  PHASE 3: LightGBM + XGBoost (5 seeds)"
echo "=========================================="
python -u scripts/elliptic/train_ml_baselines.py \
    --seeds 42 123 456 789 1011 --threshold-mode tuned

# ── Phase 4 — V2 coreset + 7 ablation baselines ─────────────────────
echo ""
echo "=========================================="
echo "  PHASE 4: V2 coreset + B1..B7 (50 draws)"
echo "=========================================="
python -u scripts/elliptic/build_ht_coreset.py \
    --seeds 42 123 456 789 1011 \
    --benign-multiplier 2.0 \
    --hard-neg-ratio 0.3 \
    --k-repeats 50

# ── Phase 5 — fidelity evaluation ───────────────────────────────────
echo ""
echo "=========================================="
echo "  PHASE 5: ΔP, ΔR, ΔF1 vs full test set"
echo "=========================================="
python -u scripts/elliptic/eval_ht_coreset.py --n-boot 2000

echo ""
echo "=========================================="
echo "  Phases 1-5 done. To run LLM/Doubt Triage on Elliptic:"
echo "    1. python scripts/elliptic/serialize_coreset.py"
echo "    2. host_vllm_<model>.sh + run_llm_eval.sh on the JSONL"
echo "    3. python scripts/elliptic/doubt_triage.py --llm-preds ..."
echo "=========================================="
