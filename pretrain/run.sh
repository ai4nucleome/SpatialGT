#!/usr/bin/env bash
# =============================================================================
# SpatialGT Pretraining Launch Script
# =============================================================================
# 
# Usage:
#   Single GPU:     bash run.sh
#   Multi-GPU:      bash run.sh --multi
#   Background:     nohup bash run.sh > logs/train.log 2>&1 &
#
# Before running, make sure:
#   1. The virtual environment is activated
#   2. The data has been preprocessed using preprocess.py
#   3. Config.py paths are correctly configured
# =============================================================================

set -euo pipefail

# ============ Configuration ============
# GPU settings (modify based on your hardware)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Performance optimizations
export OMP_NUM_THREADS=16
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Number of GPUs to use (auto-detect if not set)
NUM_GPUS=${NUM_GPUS:-$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)}

# ============ Log Directory ============
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%F_%H-%M-%S).log"

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

# ============ Launch Training ============
echo "=== SpatialGT Pretraining ==="
echo "Timestamp: $(date)"
echo "Log file: $LOG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS total)"

if [ "$MULTI_GPU" = true ] || [ "$NUM_GPUS" -gt 1 ]; then
    echo "Mode: Multi-GPU (torchrun)"
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12398 \
        run_pretrain.py
else
    echo "Mode: Single GPU"
    exec python run_pretrain.py
fi
