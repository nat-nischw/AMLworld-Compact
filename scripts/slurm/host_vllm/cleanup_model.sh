#!/bin/bash
#########################################
### Cleanup model weights from cache  ###
#########################################
#
# Usage:
#   bash cleanup_model.sh openai/gpt-oss-120b
#   bash cleanup_model.sh Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#   bash cleanup_model.sh meta-llama/Llama-4-Maverick-17B-128E-Instruct
#
# This deletes the HuggingFace cached model to free disk space
# before downloading the next model.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash cleanup_model.sh <model_id>"
    echo ""
    echo "Examples:"
    echo "  bash cleanup_model.sh openai/gpt-oss-120b"
    echo "  bash cleanup_model.sh Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    echo "  bash cleanup_model.sh meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    exit 1
fi

MODEL_ID="$1"
PERSONAL_CACHE="${HOME}/.cache/huggingface"
SHARED_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# HuggingFace stores models as: models--<org>--<model>  (/ replaced with --)
MODEL_DIR_NAME="models--${MODEL_ID//\//--}"

# Check all possible cache locations (personal + shared, hub/ + root)
CACHE_DIRS=(
    "${PERSONAL_CACHE}/hub/${MODEL_DIR_NAME}"
    "${PERSONAL_CACHE}/${MODEL_DIR_NAME}"
    "${SHARED_CACHE}/hub/${MODEL_DIR_NAME}"
    "${SHARED_CACHE}/${MODEL_DIR_NAME}"
)

echo "=== Model Cleanup ==="
echo "Model ID   : ${MODEL_ID}"

FOUND=0
for CACHE_DIR in "${CACHE_DIRS[@]}"; do
    if [ -d "$CACHE_DIR" ]; then
        FOUND=1
        SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
        echo "Cache dir  : ${CACHE_DIR}"
        echo "Size       : ${SIZE}"
        echo ""
        echo "Deleting..."
        rm -rf "$CACHE_DIR"
        echo "Done."
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "Cache directory not found in any location (already cleaned?)."
    echo "Searched:"
    for d in "${CACHE_DIRS[@]}"; do
        echo "  - $d"
    done
fi

echo ""
echo "Disk space:"
df -h /data
echo "===================="
