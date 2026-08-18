#!/bin/bash
#SBATCH --job-name=hybrid-eval
#SBATCH --output=${PROJECT_ROOT}/pipeline/aml/logs/hybrid_eval_%j.log
#SBATCH --error=${PROJECT_ROOT}/pipeline/aml/logs/hybrid_eval_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ──────────────────────────────────────────────────────────
# Step 1: Offline hybrid eval (grid search, no GPU needed)
# Uses existing ML+LLM predictions on V2 subset
# ──────────────────────────────────────────────────────────

set -euo pipefail

WORK="${AMLC_REPO:?set AMLC_REPO to this checkout}"
LOG_DIR=${PROJECT_ROOT}/pipeline/aml/logs
mkdir -p "$LOG_DIR"

cd "$WORK"

echo "$(date) — Starting hybrid offline evaluation"
echo "=========================================="

# Run all datasets × models × methods
python scripts/18_score_deferral_grid.py --all 2>&1 | tee "$LOG_DIR/hybrid_eval_full.log"

echo ""
echo "$(date) — Done. Results in: $WORK/outputs/hybrid/"
echo "Summary: $WORK/outputs/hybrid/hybrid_summary.csv"
