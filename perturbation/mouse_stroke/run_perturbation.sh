#!/usr/bin/env bash
# =============================================================================
# SpatialGT Mouse Stroke Perturbation Evaluation Script
# =============================================================================
#
# This script runs perturbation evaluation on mouse stroke data.
#
# Default settings:
#   - Perturb ICA region
#   - Evaluate on ICA, PIA_P, PIA_D regions
#   - 10 iteration steps
#   - Seed: 42
#   - Metrics: PCC, Spearman, Cos, L1, L2
#   - Save expression for all steps
#
# Usage:
#   bash run_perturbation.sh --ckpt <checkpoint_path>
#
# =============================================================================

set -euo pipefail

# ============ Configuration ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default paths
CACHE_SHAM="${PROJECT_ROOT}/example_data/mouse_stroke/sham_1_preprocessed"
CACHE_PT="${PROJECT_ROOT}/example_data/mouse_stroke/pt_1_preprocessed"
DEG_CSV="${SCRIPT_DIR}/DEG/PT1-1_vs_Sham1-1_ICA_DEG.csv"
ROI_MANIFEST="${SCRIPT_DIR}/DEG/roi_manifest.json"
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Default parameters
STEPS=10
PATCH_SIZE=20
SEED=42
BATCH_SIZE=64
NUM_WORKERS=4

# ============ Parse Arguments ============
CKPT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --cache_sham)
            CACHE_SHAM="$2"
            shift 2
            ;;
        --cache_pt)
            CACHE_PT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --patch_size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============ Validate Arguments ============
if [[ -z "$CKPT" ]]; then
    echo "Error: --ckpt is required"
    echo ""
    echo "Usage: bash run_perturbation.sh --ckpt <checkpoint_path>"
    echo ""
    echo "Options:"
    echo "  --ckpt          Checkpoint directory (required)"
    echo "  --cache_sham    Sham cache directory"
    echo "  --cache_pt      PT cache directory"
    echo "  --output_dir    Output directory"
    echo "  --steps         Number of iteration steps (default: 10)"
    echo "  --patch_size    Number of spots to perturb (default: 20)"
    echo "  --seed          Random seed (default: 42)"
    exit 1
fi

# ============ Create Output Directory ============
mkdir -p "$OUTPUT_DIR"

# ============ Run Perturbation Evaluation ============
echo "=============================================="
echo "SpatialGT Mouse Stroke Perturbation Evaluation"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Checkpoint:   $CKPT"
echo "  Cache Sham:   $CACHE_SHAM"
echo "  Cache PT:     $CACHE_PT"
echo "  DEG CSV:      $DEG_CSV"
echo "  Output:       $OUTPUT_DIR"
echo "  Steps:        $STEPS"
echo "  Patch Size:   $PATCH_SIZE"
echo "  Seed:         $SEED"
echo ""

cd "$SCRIPT_DIR"

python spatialgt_perturb_eval.py \
    --ckpt "$CKPT" \
    --cache_sham "$CACHE_SHAM" \
    --cache_pt "$CACHE_PT" \
    --sham_dataset_name "Sham1-1" \
    --pt_dataset_name "PT1-1" \
    --roi_manifest "$ROI_MANIFEST" \
    --deg_csv "$DEG_CSV" \
    --steps "$STEPS" \
    --patch_size "$PATCH_SIZE" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --out_dir "$OUTPUT_DIR" \
    --show_progress

echo ""
echo "=============================================="
echo "Perturbation evaluation completed!"
echo "=============================================="
echo ""

# ============ Run Step-wise MSE Analysis ============
echo "Running step-wise MSE analysis for non-perturbed spots..."
echo ""

python analyze_step_mse.py \
    --expr_dirs "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_step "$STEPS"

echo ""
echo "=============================================="
echo "All analysis completed!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Results:     $OUTPUT_DIR/summary.csv"
echo "  Expression:  $OUTPUT_DIR/expression/"
echo "  MSE Plot:    $OUTPUT_DIR/step_mse_non_perturbed.png"
echo ""
