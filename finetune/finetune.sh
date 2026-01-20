#!/usr/bin/env bash
# =============================================================================
# SpatialGT Fine-tuning Launch Script
# =============================================================================
#
# This script fine-tunes a pretrained SpatialGT model on a new dataset.
#
# Default settings:
#   - Full fine-tuning: all 8 transformer layers unfrozen
#   - 100 training epochs
#   - Learning rate: 1e-4
#
# Usage:
#   Single GPU:     bash finetune.sh
#   Multi-GPU:      bash finetune.sh --multi
#   Background:     nohup bash finetune.sh > logs/finetune.log 2>&1 &
#
# Before running, make sure:
#   1. The virtual environment is activated
#   2. The target dataset has been preprocessed
#   3. A pretrained checkpoint is available
# =============================================================================

set -euo pipefail

# ============ Configuration ============
# GPU settings (modify based on your hardware)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Number of GPUs
NUM_GPUS=${NUM_GPUS:-$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)}

# ============ Paths ============
# Project root (parent of finetune/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Pretrained checkpoint (REQUIRED - modify this path)
BASE_CHECKPOINT="${PROJECT_ROOT}/model/pretrained/checkpoint"

# Dataset cache directory (REQUIRED - modify this path)
CACHE_DIR="${PROJECT_ROOT}/example_data/HBRC/HBRC_preprocessed"

# Output directory for fine-tuned model
OUTPUT_DIR="${PROJECT_ROOT}/output/finetune/HBRC"

# ============ Training Parameters ============
# Fine-tuning strategy
UNFREEZE_LAYERS=8       # Number of layers to unfreeze (8 = full fine-tuning)
NUM_EPOCHS=100          # Training epochs
LEARNING_RATE=1e-4      # Learning rate
BATCH_SIZE=64           # Batch size per GPU

# Checkpointing
CHECKPOINT_INTERVAL=500 # Save every N steps
SAVE_TOTAL_LIMIT=10     # Keep last N checkpoints

# Data loading
NUM_WORKERS=1           # Workers (use 1 for H5 mode)
CACHE_MODE="h5"         # Cache mode: h5 or lmdb

# ============ Log Directory ============
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune_$(date +%F_%H-%M-%S).log"

# ============ Parse Arguments ============
MULTI_GPU=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi)
            MULTI_GPU=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============ Validate Paths ============
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Cache directory not found: $CACHE_DIR"
    echo "Please preprocess your data first using pretrain/preprocess.py"
    exit 1
fi

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "WARNING: Checkpoint not found: $BASE_CHECKPOINT"
    echo "Please set BASE_CHECKPOINT to your pretrained model checkpoint"
    # For testing, we can skip this check
fi

# ============ Launch Fine-tuning ============
echo "=== SpatialGT Fine-tuning ==="
echo "Timestamp: $(date)"
echo "Log file: $LOG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS total)"
echo "Checkpoint: $BASE_CHECKPOINT"
echo "Cache: $CACHE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Unfreeze layers: $UNFREEZE_LAYERS"
echo ""

mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python ${SCRIPT_DIR}/finetune.py \
    --base_ckpt ${BASE_CHECKPOINT} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --unfreeze_last_n ${UNFREEZE_LAYERS} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --cache_mode ${CACHE_MODE} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --save_total_limit ${SAVE_TOTAL_LIMIT}"

if [ "$MULTI_GPU" = true ] || [ "$NUM_GPUS" -gt 1 ]; then
    echo "Mode: Multi-GPU (torchrun)"
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=127.0.0.1 \
        --master_port=29501 \
        ${SCRIPT_DIR}/finetune.py \
        --base_ckpt ${BASE_CHECKPOINT} \
        --cache_dir ${CACHE_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --unfreeze_last_n ${UNFREEZE_LAYERS} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --cache_mode ${CACHE_MODE} \
        --checkpoint_interval ${CHECKPOINT_INTERVAL} \
        --save_total_limit ${SAVE_TOTAL_LIMIT}
else
    echo "Mode: Single GPU"
    exec $CMD
fi
