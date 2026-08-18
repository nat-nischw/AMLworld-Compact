#!/bin/bash
#SBATCH --job-name=dt-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dt_eval_%j.log

# ═══════════════════════════════════════════════════════════════════
# Doubt Triage — selective ML + LLM hybrid (offline, no GPU needed)
#
# Usage:
#   sbatch slurm/run_dt_eval.sh                     # all combinations
#   sbatch slurm/run_dt_eval.sh HI-Small             # single dataset
#   sbatch slurm/run_dt_eval.sh HI-Small GPT-OSS-120B  # dataset + model
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

# Activate conda env if CONDA_ENV is set
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

echo "=========================================="
echo "  Doubt Triage"
echo "  $(date)"
echo "=========================================="

ARGS=""
if [ "${1:-}" ]; then ARGS="$ARGS --dataset $1"; fi
if [ "${2:-}" ]; then ARGS="$ARGS --model $2"; fi
if [ "${3:-}" ]; then ARGS="$ARGS --method $3"; fi

python scripts/17_run_doubt_triage.py $ARGS

echo ""
echo "Done: $(date)"
