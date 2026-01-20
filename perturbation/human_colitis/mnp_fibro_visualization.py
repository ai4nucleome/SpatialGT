#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNP/Fibroblast Activation Status Spatial Visualization.

This script visualizes the spatial distribution of activated MNP and Fibroblast cells
based on perturbation experiment results.

Key features:
- Activated MNP: Orange (#F8AB61)
- Activated Fibroblast: Purple (#8F69C5)  
- Other cells: Light gray (#E0E0E0)
- MNP cells use expression from previous step for threshold judgment

Usage:
    python mnp_fibro_visualization.py --expr_dir path/to/expression --sample HS5_UC_R_0 --step 3 --out_dir path/to/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import anndata as ad

# Script directory for relative paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# ==============================================================================
# Constants
# ==============================================================================

# Override with environment variable COLITIS_DATA_ROOT or pass via command line
DATA_ROOT = Path(os.environ.get("COLITIS_DATA_ROOT", _REPO_ROOT / "example_data" / "human_colitis"))
RAW_DATA_ROOT = DATA_ROOT / "raw_data"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed"

# Default threshold directory
DEFAULT_THRESHOLD_DIR = _SCRIPT_DIR / "thresholds"

# Color scheme
COLOR_MNP_ACTIVATED = "#F8AB61"        # Orange for activated MNP
COLOR_FIBRO_ACTIVATED = "#8F69C5"      # Purple for activated Fibroblast
COLOR_OTHER = "#E0E0E0"                # Light gray for other cells
COLOR_PERTURBED_BORDER = "#FF0000"     # Red border for perturbed spots

# Marker genes for activation scoring
MARKER_GENES = {
    "MNP": ["S100A4", "TIMP1", "S100A9", "CD80", "ITGAX", "LYZ", "IL1B"],
    "Fibroblast": ["TIMP1", "IL1R1", "CXCL14", "CD44"],
}

# Cell type labels
CELLTYPE_LABELS = {
    "MNP": {
        "all": ["05-MNP", "05B-MNP_activated_inf"],
        "activated": ["05B-MNP_activated_inf"],
        "non_activated": ["05-MNP"],
    },
    "Fibroblast": {
        "all": ["03A-Myofibroblast", "03B-Fibroblast", "03C-activated_fibroblast"],
        "activated": ["03C-activated_fibroblast"],
        "non_activated": ["03A-Myofibroblast", "03B-Fibroblast"],
    },
}

# Default activation thresholds
DEFAULT_THRESHOLDS = {"MNP": 1.55, "Fibroblast": 1.93}


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_activation_thresholds(threshold_file: Optional[str] = None) -> Dict[str, float]:
    """Load activation thresholds from JSON file."""
    if threshold_file and Path(threshold_file).exists():
        with open(threshold_file, "r") as f:
            thresholds = json.load(f)
        return {
            "MNP": thresholds.get("MNP", {}).get("optimal_threshold", DEFAULT_THRESHOLDS["MNP"]),
            "Fibroblast": thresholds.get("Fibroblast", {}).get("optimal_threshold", DEFAULT_THRESHOLDS["Fibroblast"]),
        }
    return DEFAULT_THRESHOLDS


def load_spatial_coordinates(sample_name: str) -> pd.DataFrame:
    """Load spatial coordinates for all cells in a sample."""
    h5ad_path = RAW_DATA_ROOT / sample_name / f"{sample_name}.h5ad"
    if not h5ad_path.exists():
        h5ad_path = PREPROCESSED_ROOT / sample_name / sample_name / "processed.h5ad"
    
    adata = ad.read_h5ad(h5ad_path)
    
    coords_df = pd.DataFrame({
        "spot_idx": np.arange(adata.n_obs),
        "x": adata.obsm["spatial"][:, 0],
        "y": adata.obsm["spatial"][:, 1],
        "cell_type": adata.obs["fine_annotation"].values if "fine_annotation" in adata.obs else "unknown",
    })
    
    return coords_df


def load_expression_data(expr_dir: str, step: int) -> pd.DataFrame:
    """Load expression data for a specific step."""
    expr_file = Path(expr_dir) / f"pert{step}_expression.csv"
    if not expr_file.exists():
        raise FileNotFoundError(f"Expression file not found: {expr_file}")
    
    df = pd.read_csv(expr_file)
    return df


def compute_activation_scores(expr_df: pd.DataFrame, celltype: str) -> pd.DataFrame:
    """Compute activation scores for each cell based on marker gene expression."""
    markers = MARKER_GENES.get(celltype, [])
    markers_lower = [m.lower() for m in markers]
    
    # Find marker gene columns in expression dataframe
    expr_cols = [c for c in expr_df.columns if c.lower() in markers_lower]
    
    if not expr_cols:
        print(f"[WARN] No marker genes found for {celltype}")
        return pd.DataFrame({"spot_idx": expr_df["spot_idx"], "activation_score": 0.0})
    
    # Compute mean expression of marker genes
    marker_expr = expr_df[expr_cols].values
    scores = np.mean(marker_expr, axis=1)
    
    return pd.DataFrame({
        "spot_idx": expr_df["spot_idx"],
        "activation_score": scores,
    })


def determine_cell_colors(
    coords_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    expr_df_prev: Optional[pd.DataFrame],
    thresholds: Dict[str, float],
    use_prev_step_for_mnp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Determine cell colors based on activation status.
    
    For MNP cells, use previous step expression for threshold judgment.
    
    Returns:
        colors: Array of RGBA colors
        is_perturbed: Boolean array indicating perturbed spots
        stats: Dictionary with statistics
    """
    n_cells = len(coords_df)
    colors = np.array([to_rgba(COLOR_OTHER)] * n_cells)
    is_perturbed = np.zeros(n_cells, dtype=bool)
    
    # Get cell types from coordinates dataframe
    cell_types = coords_df["cell_type"].values
    
    # Get perturbed status from expression dataframe
    if "is_perturbed" in expr_df.columns:
        is_perturbed_series = expr_df.set_index("spot_idx")["is_perturbed"]
        for i, spot_idx in enumerate(coords_df["spot_idx"].values):
            if spot_idx in is_perturbed_series.index:
                is_perturbed[i] = is_perturbed_series[spot_idx]
    
    # Compute activation scores
    stats = {
        "MNP": {"n_all": 0, "n_activated": 0},
        "Fibroblast": {"n_all": 0, "n_activated": 0},
    }
    
    # Compute scores for current step
    mnp_scores = compute_activation_scores(expr_df, "MNP")
    fibro_scores = compute_activation_scores(expr_df, "Fibroblast")
    
    # For MNP: use previous step expression for threshold judgment
    if use_prev_step_for_mnp and expr_df_prev is not None:
        mnp_scores_for_judgment = compute_activation_scores(expr_df_prev, "MNP")
    else:
        mnp_scores_for_judgment = mnp_scores
    
    # Create score lookup tables
    mnp_score_map = dict(zip(mnp_scores_for_judgment["spot_idx"], mnp_scores_for_judgment["activation_score"]))
    fibro_score_map = dict(zip(fibro_scores["spot_idx"], fibro_scores["activation_score"]))
    
    mnp_threshold = thresholds.get("MNP", DEFAULT_THRESHOLDS["MNP"])
    fibro_threshold = thresholds.get("Fibroblast", DEFAULT_THRESHOLDS["Fibroblast"])
    
    # Assign colors
    for i, (spot_idx, cell_type) in enumerate(zip(coords_df["spot_idx"].values, cell_types)):
        # Check if MNP
        if cell_type in CELLTYPE_LABELS["MNP"]["all"]:
            stats["MNP"]["n_all"] += 1
            score = mnp_score_map.get(spot_idx, 0.0)
            if score >= mnp_threshold:
                colors[i] = to_rgba(COLOR_MNP_ACTIVATED)
                stats["MNP"]["n_activated"] += 1
        
        # Check if Fibroblast
        elif cell_type in CELLTYPE_LABELS["Fibroblast"]["all"]:
            stats["Fibroblast"]["n_all"] += 1
            score = fibro_score_map.get(spot_idx, 0.0)
            if score >= fibro_threshold:
                colors[i] = to_rgba(COLOR_FIBRO_ACTIVATED)
                stats["Fibroblast"]["n_activated"] += 1
    
    return colors, is_perturbed, stats


def plot_spatial_visualization(
    coords_df: pd.DataFrame,
    colors: np.ndarray,
    is_perturbed: np.ndarray,
    stats: Dict,
    title: str,
    out_path: str,
    figsize: Tuple[float, float] = (12, 10),
    spot_size: float = 20,
    show_perturbed_border: bool = True,
    dpi: int = 300,
):
    """Create spatial visualization plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    x = coords_df["x"].values
    y = coords_df["y"].values
    
    # Plot all spots
    ax.scatter(x, y, c=colors, s=spot_size, alpha=0.8, edgecolors='none')
    
    # Highlight perturbed spots with red border
    if show_perturbed_border and np.any(is_perturbed):
        perturbed_mask = is_perturbed
        ax.scatter(
            x[perturbed_mask], y[perturbed_mask],
            c='none', s=spot_size * 1.2,
            edgecolors=COLOR_PERTURBED_BORDER, linewidths=1.5,
            label='Perturbed'
        )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_MNP_ACTIVATED, edgecolor='none', 
                       label=f'Activated MNP ({stats["MNP"]["n_activated"]}/{stats["MNP"]["n_all"]})'),
        mpatches.Patch(facecolor=COLOR_FIBRO_ACTIVATED, edgecolor='none', 
                       label=f'Activated Fibroblast ({stats["Fibroblast"]["n_activated"]}/{stats["Fibroblast"]["n_all"]})'),
        mpatches.Patch(facecolor=COLOR_OTHER, edgecolor='none', label='Other cells'),
    ]
    
    if show_perturbed_border and np.any(is_perturbed):
        legend_elements.append(
            mpatches.Patch(facecolor='none', edgecolor=COLOR_PERTURBED_BORDER, 
                          linewidth=1.5, label=f'Perturbed Activated MNP ({np.sum(is_perturbed)} spots)')
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Legend at bottom center
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=2, fontsize=10, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved visualization to {out_path}")


def create_comparison_figure(
    coords_df: pd.DataFrame,
    expr_dir: str,
    steps: List[int],
    thresholds: Dict[str, float],
    title_prefix: str,
    out_path: str,
    prev_step_offset: int = 1,
    figsize_per_plot: Tuple[float, float] = (8, 7),
    spot_size: float = 10,
    dpi: int = 300,
):
    """Create a multi-panel comparison figure showing different steps."""
    n_steps = len(steps)
    n_cols = min(3, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    
    if n_steps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    x = coords_df["x"].values
    y = coords_df["y"].values
    
    all_stats = []
    has_perturbed_in_step0 = False
    
    for i, step in enumerate(steps):
        ax = axes[i]
        
        try:
            expr_df = load_expression_data(expr_dir, step)
        except FileNotFoundError:
            ax.set_title(f"Step {step} (not found)", fontsize=12)
            ax.set_visible(False)
            continue
        
        # Load previous step expression for MNP judgment
        expr_df_prev = None
        prev_step = step - prev_step_offset
        if prev_step >= 0:
            try:
                expr_df_prev = load_expression_data(expr_dir, prev_step)
            except FileNotFoundError:
                pass
        
        # Determine colors
        colors, is_perturbed, stats = determine_cell_colors(
            coords_df, expr_df, expr_df_prev, thresholds, use_prev_step_for_mnp=True
        )
        all_stats.append({"step": step, **stats})
        
        # Plot
        ax.scatter(x, y, c=colors, s=spot_size, alpha=0.8, edgecolors='none')
        
        # Only show perturbed spots border for step 0
        if step == 0 and np.any(is_perturbed):
            has_perturbed_in_step0 = True
            perturbed_mask = is_perturbed
            ax.scatter(
                x[perturbed_mask], y[perturbed_mask],
                c='none', s=spot_size * 2,
                edgecolors=COLOR_PERTURBED_BORDER, linewidths=1.0,
            )
        
        # Title with stats
        mnp_rate = stats["MNP"]["n_activated"] / stats["MNP"]["n_all"] * 100 if stats["MNP"]["n_all"] > 0 else 0
        fibro_rate = stats["Fibroblast"]["n_activated"] / stats["Fibroblast"]["n_all"] * 100 if stats["Fibroblast"]["n_all"] > 0 else 0
        
        ax.set_title(
            f'{title_prefix} Step {step}\n'
            f'MNP: {stats["MNP"]["n_activated"]}/{stats["MNP"]["n_all"]} ({mnp_rate:.1f}%)\n'
            f'Fibro: {stats["Fibroblast"]["n_activated"]}/{stats["Fibroblast"]["n_all"]} ({fibro_rate:.1f}%)',
            fontsize=10
        )
        
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Create shared legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_MNP_ACTIVATED, edgecolor='none', label='Activated MNP'),
        mpatches.Patch(facecolor=COLOR_FIBRO_ACTIVATED, edgecolor='none', label='Activated Fibroblast'),
        mpatches.Patch(facecolor=COLOR_OTHER, edgecolor='none', label='Other cells'),
    ]
    
    if has_perturbed_in_step0:
        legend_elements.append(
            mpatches.Patch(facecolor='none', edgecolor=COLOR_PERTURBED_BORDER, 
                          linewidth=1.5, label='Perturbed spots (step 0 only)')
        )
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_elements), fontsize=10, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved comparison figure to {out_path}")
    
    return all_stats


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="MNP/Fibroblast Activation Spatial Visualization")
    
    # Required arguments
    ap.add_argument("--expr_dir", required=True, help="Directory containing expression CSV files")
    ap.add_argument("--sample", required=True, help="Sample name (e.g., HS5_UC_R_0)")
    
    # Visualization options
    ap.add_argument("--step", type=int, default=3, help="Step to visualize (default: 3)")
    ap.add_argument("--prev_step_offset", type=int, default=1, 
                   help="Steps back for MNP judgment (default: 1, meaning use step-1 for MNP)")
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                   help="Multiple steps to compare (creates comparison figure)")
    
    # Threshold
    ap.add_argument("--threshold_file", type=str, default=None, help="JSON file with activation thresholds")
    ap.add_argument("--mnp_threshold", type=float, default=None, help="Override MNP threshold")
    ap.add_argument("--fibro_threshold", type=float, default=None, help="Override Fibroblast threshold")
    
    # Output
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--out_prefix", type=str, default="activation_spatial", help="Output file prefix")
    
    # Plot options
    ap.add_argument("--figsize", type=float, nargs=2, default=[12, 10], help="Figure size (width, height)")
    ap.add_argument("--spot_size", type=float, default=15, help="Spot size")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI")
    ap.add_argument("--no_perturbed_border", action="store_true", help="Do not show red border for perturbed spots")
    
    args = ap.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load thresholds
    thresholds = load_activation_thresholds(args.threshold_file)
    if args.mnp_threshold is not None:
        thresholds["MNP"] = args.mnp_threshold
    if args.fibro_threshold is not None:
        thresholds["Fibroblast"] = args.fibro_threshold
    
    print(f"[INFO] Thresholds: MNP={thresholds['MNP']:.4f}, Fibroblast={thresholds['Fibroblast']:.4f}")
    
    # Load spatial coordinates
    print(f"[INFO] Loading spatial coordinates for {args.sample}...")
    coords_df = load_spatial_coordinates(args.sample)
    print(f"[INFO] Loaded {len(coords_df)} spots")
    
    # Determine which steps to visualize
    if args.steps is not None:
        # Comparison figure with multiple steps
        out_path = out_dir / f"{args.out_prefix}_comparison.png"
        stats = create_comparison_figure(
            coords_df=coords_df,
            expr_dir=args.expr_dir,
            steps=args.steps,
            thresholds=thresholds,
            title_prefix=args.sample,
            out_path=str(out_path),
            prev_step_offset=args.prev_step_offset,
            spot_size=args.spot_size,
            dpi=args.dpi,
        )
        
        # Save stats
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(out_dir / f"{args.out_prefix}_stats.csv", index=False)
        print(f"[OK] Saved stats to {out_dir / f'{args.out_prefix}_stats.csv'}")
    else:
        # Single step visualization
        step = args.step
        print(f"[INFO] Visualizing step {step}...")
        
        # Load expression data
        expr_df = load_expression_data(args.expr_dir, step)
        
        # Load previous step expression for MNP judgment
        expr_df_prev = None
        prev_step = step - args.prev_step_offset
        if prev_step >= 0:
            try:
                expr_df_prev = load_expression_data(args.expr_dir, prev_step)
                print(f"[INFO] Using step {prev_step} expression for MNP threshold judgment")
            except FileNotFoundError:
                print(f"[WARN] Previous step {prev_step} not found, using current step for MNP judgment")
        
        # Determine colors
        colors, is_perturbed, stats = determine_cell_colors(
            coords_df, expr_df, expr_df_prev, thresholds, use_prev_step_for_mnp=True
        )
        
        # Print statistics
        print(f"\n=== Activation Statistics (Step {step}) ===")
        for ct in ["MNP", "Fibroblast"]:
            n_all = stats[ct]["n_all"]
            n_act = stats[ct]["n_activated"]
            rate = n_act / n_all * 100 if n_all > 0 else 0
            print(f"  {ct}: {n_act}/{n_all} activated ({rate:.1f}%)")
        print(f"  Perturbed spots: {np.sum(is_perturbed)}")
        
        # Plot
        out_path = out_dir / f"{args.out_prefix}_step{step}.png"
        title = f"{args.sample} - Step {step}"
        
        # Only show perturbed border for step 0
        show_perturbed = (step == 0) and (not args.no_perturbed_border)
        
        plot_spatial_visualization(
            coords_df=coords_df,
            colors=colors,
            is_perturbed=is_perturbed,
            stats=stats,
            title=title,
            out_path=str(out_path),
            figsize=tuple(args.figsize),
            spot_size=args.spot_size,
            show_perturbed_border=show_perturbed,
            dpi=args.dpi,
        )
        
        # Save stats
        stats_out = {
            "sample": args.sample,
            "step": step,
            "prev_step_for_mnp": prev_step,
            **{f"{ct}_{k}": v for ct, d in stats.items() for k, v in d.items()},
            "n_perturbed": int(np.sum(is_perturbed)),
        }
        with open(out_dir / f"{args.out_prefix}_step{step}_stats.json", "w") as f:
            json.dump(stats_out, f, indent=2)
        print(f"[OK] Saved stats to {out_dir / f'{args.out_prefix}_step{step}_stats.json'}")
    
    print("\n[DONE] Visualization complete!")


if __name__ == "__main__":
    main()
