#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Activation Thresholds for MNP and Fibroblast Cells.

This script computes optimal activation thresholds using marker gene expression
and ROC analysis with Youden's J statistic.

Marker genes:
- Activated fibroblasts: TIMP1, IL1R1, CXCL14, CD44
- Activated inflammatory MNPs: S100A4, TIMP1, S100A9, CD80, ITGAX, LYZ, IL1B

Usage:
    python compute_activation_threshold.py --condition UC --output_dir ./output/activation_threshold
    python compute_activation_threshold.py --condition UC_VDZ --output_dir ./output/activation_threshold
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Matplotlib settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Script directory for relative paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Data path configuration
# Override with environment variable COLITIS_DATA_ROOT or pass via command line
DATA_ROOT = Path(os.environ.get("COLITIS_DATA_ROOT", _REPO_ROOT / "example_data" / "human_colitis"))
RAW_DATA_DIR = DATA_ROOT / "raw_data"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"

# Marker gene definitions
MARKER_GENES = {
    "MNP_activated": ["S100A4", "TIMP1", "S100A9", "CD80", "ITGAX", "LYZ", "IL1B"],
    "Fibroblast_activated": ["TIMP1", "IL1R1", "CXCL14", "CD44"],
}

# Cell type definitions
CELLTYPE_LABELS = {
    "MNP": {
        "positive": ["05B-MNP_activated_inf"],  # Activated
        "negative": ["05-MNP"],  # Non-activated
    },
    "Fibroblast": {
        "positive": ["03C-activated_fibroblast"],  # Activated
        "negative": ["03A-Myofibroblast", "03B-Fibroblast"],  # Non-activated
    },
}

# Condition to sample mapping
CONDITION_SAMPLES = {
    "UC": ["HS5_UC_R_0", "HS7_UC_L_0", "HS7_UC_R_1", "HS8_UC_L_1"],
    "UC_VDZ": ["HS10_VDZ_L_1", "HS10_VDZ_R_0", "HS11_VDZ_L_3", "HS11_VDZ_R_0", "HS12_VDZ_L_1"],
    "HC": ["HS1_HC_L_0_fov18", "HS1_HC_L_0_fov19", "HS1_HC_R_0", "HS2_HC_L_0", 
           "HS2_HC_R_0", "HS3_HC_R_0", "HS4_HC_L_0", "HS4_HC_R_0"],
}


def load_condition_data(condition: str, use_preprocessed: bool = True) -> ad.AnnData:
    """Load and merge all sample data for a given condition."""
    samples = CONDITION_SAMPLES.get(condition, [])
    if not samples:
        raise ValueError(f"Unknown condition: {condition}")
    
    adatas = []
    for sample in samples:
        if use_preprocessed:
            processed_path = PREPROCESSED_DIR / condition / sample / "processed.h5ad"
            if not processed_path.exists():
                processed_path = PREPROCESSED_DIR / sample / sample / "processed.h5ad"
            if not processed_path.exists():
                processed_path = RAW_DATA_DIR / sample / f"{sample}.h5ad"
        else:
            processed_path = RAW_DATA_DIR / sample / f"{sample}.h5ad"
        
        if processed_path.exists():
            adata = sc.read_h5ad(processed_path)
            adata.obs['sample'] = sample
            adata.obs['condition'] = condition
            adatas.append(adata)
            print(f"  Loaded {sample}: {adata.n_obs} cells")
        else:
            print(f"  [WARN] File not found: {processed_path}")
    
    if not adatas:
        raise FileNotFoundError(f"No data files found for condition: {condition}")
    
    combined = ad.concat(adatas, join='inner', merge='same')
    print(f"  Combined {condition}: {combined.n_obs} cells, {combined.n_vars} genes")
    
    return combined


def compute_marker_score(
    adata: ad.AnnData, 
    marker_genes: List[str], 
    use_layer: str = "X_log1p"
) -> np.ndarray:
    """
    Compute mean marker gene expression score for each cell.
    
    Args:
        adata: AnnData object
        marker_genes: List of marker genes
        use_layer: Layer to use (default: X_log1p)
    
    Returns:
        Score array of shape (n_cells,)
    """
    # Get expression matrix
    if use_layer in adata.layers:
        X = adata.layers[use_layer]
    else:
        X = adata.X
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)
    
    # Find available marker genes
    available_genes = []
    gene_indices = []
    for gene in marker_genes:
        if gene in adata.var_names:
            available_genes.append(gene)
            gene_indices.append(adata.var_names.get_loc(gene))
        else:
            print(f"  [WARN] Marker gene not found: {gene}")
    
    if len(available_genes) == 0:
        raise ValueError("No marker genes found in the dataset")
    
    print(f"  Using {len(available_genes)}/{len(marker_genes)} marker genes: {available_genes}")
    
    # Compute mean expression score
    scores = X[:, gene_indices].mean(axis=1)
    
    return scores


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """
    Find optimal threshold using ROC analysis (Youden's J statistic).
    
    Args:
        y_true: True labels (0/1)
        scores: Prediction scores
    
    Returns:
        optimal_threshold: Optimal threshold value
        metrics: Dictionary containing various metrics
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J = TPR - FPR maximum)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute metrics at optimal threshold
    y_pred = (scores >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    
    metrics = {
        "optimal_threshold": float(optimal_threshold),
        "auc_roc": float(roc_auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "youden_j": float(j_scores[optimal_idx]),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
    
    return optimal_threshold, metrics


def plot_roc_curve(metrics: Dict, celltype: str, condition: str, output_path: Path) -> None:
    """Plot ROC curve and metrics bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: ROC curve
    ax1 = axes[0]
    fpr = np.array(metrics["fpr_curve"])
    tpr = np.array(metrics["tpr_curve"])
    roc_auc = metrics["auc_roc"]
    optimal_threshold = metrics["optimal_threshold"]
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark optimal point
    optimal_idx = np.argmax(np.array(metrics["tpr_curve"]) - np.array(metrics["fpr_curve"]))
    ax1.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], 
                color='red', s=100, zorder=5, 
                label=f'Optimal (threshold={optimal_threshold:.3f})')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curve - {celltype} Activation\n({condition})', fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Metrics bar chart
    ax2 = axes[1]
    metric_names = ['AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'Accuracy']
    metric_values = [
        metrics["auc_roc"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["precision"],
        metrics["f1_score"],
        metrics["accuracy"],
    ]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metric_names)))
    bars = ax2.bar(metric_names, metric_values, color=colors, edgecolor='black')
    
    # Add values on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylim([0, 1.15])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Classification Metrics\n(Threshold = {optimal_threshold:.3f})', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved ROC plot: {output_path}")


def plot_score_distribution(
    scores: np.ndarray, 
    y_true: np.ndarray, 
    optimal_threshold: float, 
    celltype: str, 
    condition: str, 
    output_path: Path
) -> None:
    """Plot score distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate scores by activation status
    scores_pos = scores[y_true == 1]
    scores_neg = scores[y_true == 0]
    
    # Draw histograms
    bins = np.linspace(min(scores.min(), 0), scores.max() * 1.1, 50)
    ax.hist(scores_neg, bins=bins, alpha=0.6, label=f'Non-activated (n={len(scores_neg)})', 
            color='blue', edgecolor='black')
    ax.hist(scores_pos, bins=bins, alpha=0.6, label=f'Activated (n={len(scores_pos)})', 
            color='red', edgecolor='black')
    
    # Draw threshold line
    ax.axvline(x=optimal_threshold, color='green', linestyle='--', lw=2,
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    ax.set_xlabel('Activation Score (Mean Marker Expression)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{celltype} Activation Score Distribution\n({condition})', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved distribution plot: {output_path}")


def analyze_celltype_activation(
    adata: ad.AnnData, 
    celltype: str, 
    condition: str, 
    output_dir: Path,
    annotation_col: str = "fine_annotation"
) -> Dict:
    """
    Analyze activation status for a specific cell type.
    
    Args:
        adata: AnnData object
        celltype: Cell type (MNP or Fibroblast)
        condition: Condition name
        output_dir: Output directory
        annotation_col: Cell type annotation column name
    
    Returns:
        Analysis results dictionary
    """
    print(f"\n[Analyzing {celltype} activation in {condition}]")
    
    # Get cell type definition
    celltype_def = CELLTYPE_LABELS.get(celltype)
    if not celltype_def:
        raise ValueError(f"Unknown cell type: {celltype}")
    
    # Get marker genes
    marker_key = f"{celltype}_activated"
    marker_genes = MARKER_GENES.get(marker_key)
    if not marker_genes:
        raise ValueError(f"No marker genes defined for: {marker_key}")
    
    print(f"  Marker genes: {marker_genes}")
    
    # Filter cells
    positive_types = celltype_def["positive"]
    negative_types = celltype_def["negative"]
    all_types = positive_types + negative_types
    
    mask = adata.obs[annotation_col].isin(all_types)
    subset = adata[mask].copy()
    
    if subset.n_obs == 0:
        print(f"  [ERROR] No cells found for {celltype}")
        return None
    
    # Create labels
    y_true = subset.obs[annotation_col].isin(positive_types).astype(int).values
    
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    print(f"  Positive (activated): {n_pos}")
    print(f"  Negative (non-activated): {n_neg}")
    
    if n_pos == 0 or n_neg == 0:
        print(f"  [ERROR] Need both positive and negative samples")
        return None
    
    # Compute scores
    scores = compute_marker_score(subset, marker_genes)
    
    # Find optimal threshold
    optimal_threshold, metrics = find_optimal_threshold(y_true, scores)
    
    print(f"\n  Results:")
    print(f"    Optimal threshold: {optimal_threshold:.4f}")
    print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"    Specificity: {metrics['specificity']:.4f}")
    print(f"    F1 Score: {metrics['f1_score']:.4f}")
    
    # Add metadata
    metrics["celltype"] = celltype
    metrics["condition"] = condition
    metrics["n_positive"] = int(n_pos)
    metrics["n_negative"] = int(n_neg)
    metrics["marker_genes"] = marker_genes
    metrics["positive_types"] = positive_types
    metrics["negative_types"] = negative_types
    
    # Save scores
    scores_df = pd.DataFrame({
        "cell_id": subset.obs_names,
        "score": scores,
        "true_label": y_true,
        "predicted": (scores >= optimal_threshold).astype(int),
        "annotation": subset.obs[annotation_col].values,
        "sample": subset.obs["sample"].values if "sample" in subset.obs else "unknown",
    })
    
    scores_file = output_dir / f"{condition}_{celltype}_activation_scores.csv"
    scores_df.to_csv(scores_file, index=False)
    print(f"\n  [OK] Saved scores: {scores_file}")
    
    # Plot ROC curve
    roc_plot_file = output_dir / f"{condition}_{celltype}_ROC.png"
    plot_roc_curve(metrics, celltype, condition, roc_plot_file)
    
    # Plot score distribution
    dist_plot_file = output_dir / f"{condition}_{celltype}_distribution.png"
    plot_score_distribution(scores, y_true, optimal_threshold, celltype, condition, dist_plot_file)
    
    # Remove curve data for JSON saving
    metrics_save = {k: v for k, v in metrics.items() 
                    if k not in ["fpr_curve", "tpr_curve", "thresholds"]}
    
    return metrics_save


def main():
    parser = argparse.ArgumentParser(description="Compute activation thresholds for MNP and Fibroblast")
    
    parser.add_argument("--condition", type=str, required=True,
                       help="Condition to analyze (UC, UC_VDZ, HC)")
    parser.add_argument("--output_dir", type=str, default="./output/activation_threshold",
                       help="Output directory")
    parser.add_argument("--celltypes", type=str, nargs="+", 
                       default=["MNP", "Fibroblast"],
                       help="Cell types to analyze")
    parser.add_argument("--annotation_col", type=str, default="fine_annotation",
                       help="Cell type annotation column name")
    parser.add_argument("--use_raw", action="store_true",
                       help="Use raw data instead of preprocessed")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Compute Activation Thresholds")
    print("=" * 60)
    print(f"Condition: {args.condition}")
    print(f"Cell types: {args.celltypes}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[Loading {args.condition} data...]")
    adata = load_condition_data(args.condition, use_preprocessed=not args.use_raw)
    
    # Analyze each cell type
    all_results = {}
    for celltype in args.celltypes:
        result = analyze_celltype_activation(
            adata, celltype, args.condition, output_dir, args.annotation_col
        )
        if result:
            all_results[celltype] = result
    
    # Save summary results
    summary_file = output_dir / f"{args.condition}_activation_thresholds.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved summary: {summary_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary of Activation Thresholds")
    print("=" * 60)
    for celltype, metrics in all_results.items():
        print(f"\n{celltype}:")
        print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
