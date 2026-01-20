#!/usr/bin/env bash
# =============================================================================
# SpatialGT Reconstruction Evaluation Script
# =============================================================================
#
# This script runs reconstruction evaluation on a given dataset.
#
# Default settings:
#   - SpatialGT: 10 iteration steps
#   - SEDR: 1 iteration step
#   - KNN: 10 iteration steps
#   - Seed: 42
#   - Mask init: "zero"
#
# Usage:
#   bash run_reconstruction.sh <method> <dataset_name>
#
# Methods: spatialgt, sedr, knn
# =============================================================================

set -euo pipefail

# ============ Configuration ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths (modify as needed)
CACHE_DIR="${PROJECT_ROOT}/example_data/HBRC/HBRC_preprocessed"
CHECKPOINT_DIR="${PROJECT_ROOT}/model/pretrained/checkpoint"
OUTPUT_BASE="${PROJECT_ROOT}/output/reconstruction"

# Default parameters
SEED=42
N_SPOTS=20
MASK_MODE="patch"
MASK_INIT="zero"

# GPU settings
export CUDA_VISIBLE_DEVICES=0
GPU_ID=0

# ============ Parse Arguments ============
METHOD="${1:-spatialgt}"
DATASET_NAME="${2:-HBRC}"

# Validate method
if [[ ! "$METHOD" =~ ^(spatialgt|sedr|knn)$ ]]; then
    echo "Error: Invalid method '$METHOD'. Use: spatialgt, sedr, knn"
    exit 1
fi

# Output directory
OUTPUT_DIR="${OUTPUT_BASE}/${DATASET_NAME}/${METHOD}"
mkdir -p "$OUTPUT_DIR"

# ============ Run Reconstruction ============
echo "=== SpatialGT Reconstruction Evaluation ==="
echo "Method: $METHOD"
echo "Dataset: $DATASET_NAME"
echo "Cache: $CACHE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "N_spots: $N_SPOTS"
echo ""

case "$METHOD" in
    spatialgt)
        echo "Running SpatialGT reconstruction (10 steps)..."
        python "${SCRIPT_DIR}/spatialgt_reconstruction.py" \
            --ckpt "$CHECKPOINT_DIR" \
            --cache_dir "$CACHE_DIR" \
            --dataset_name "$DATASET_NAME" \
            --out_dir "$OUTPUT_DIR" \
            --mask_mode "$MASK_MODE" \
            --n_spots "$N_SPOTS" \
            --mask_init_mode "$MASK_INIT" \
            --steps 10 \
            --seed "$SEED" \
            --gpu_ids "$GPU_ID"
        ;;
    
    sedr)
        echo "Running SEDR reconstruction (1 step)..."
        python "${SCRIPT_DIR}/sedr_reconstruction.py" \
            --cache_dir "$CACHE_DIR" \
            --dataset_name "$DATASET_NAME" \
            --out_dir "$OUTPUT_DIR" \
            --mask_mode "$MASK_MODE" \
            --n_spots "$N_SPOTS" \
            --steps 1 \
            --seed "$SEED" \
            --device "cuda:$GPU_ID"
        ;;
    
    knn)
        echo "Running KNN reconstruction (10 steps)..."
        python "${SCRIPT_DIR}/knn_reconstruction.py" \
            --cache_dir "$CACHE_DIR" \
            --dataset_name "$DATASET_NAME" \
            --out_dir "$OUTPUT_DIR" \
            --mask_mode "$MASK_MODE" \
            --n_spots "$N_SPOTS" \
            --mask_init "$MASK_INIT" \
            --steps 10 \
            --seed "$SEED" \
            --gpu_id "$GPU_ID"
        ;;
esac

echo ""
echo "[DONE] Results saved to $OUTPUT_DIR"
