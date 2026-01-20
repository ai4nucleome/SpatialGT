#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNP-Fibroblast Spatial Enrichment Analysis.

This script performs enrichment analysis between activated MNP and Fibroblast cells:
1. Nearest Neighbor (NN) distance visualization and statistics
2. Nearest neighbor type probability calculation
3. Neighborhood enrichment Z-score calculation

Usage:
    # For raw data
    python mnp_fibro_enrichment_analysis.py --mode raw --sample HS5_UC_R_0 --out_dir output/enrichment
    
    # For perturbation results
    python mnp_fibro_enrichment_analysis.py --mode perturbation --sample HS5_UC_R_0 \
        --model spatialgt --perturb_mode patch --dose_lambda 1.0 --step 3 --out_dir output/enrichment
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
from tqdm import tqdm

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

# Default output paths
DEFAULT_RESULTS_BASE = _SCRIPT_DIR / "output" / "perturbation_results"
DEFAULT_THRESHOLD_DIR = _SCRIPT_DIR / "thresholds"

# Spatial resolution: CosMx SMI platform uses 0.18 μm/pixel
PIXEL_TO_MICRON = 0.18

# Color scheme
COLOR_MNP_ACTIVATED = "#F8AB61"        # Orange for activated MNP
COLOR_FIBRO_ACTIVATED = "#8F69C5"      # Purple for activated Fibroblast
COLOR_OTHER = "#E0E0E0"                # Light gray for other cells
COLOR_NN_LINE = "#333A8C"              # Blue for NN connection lines
COLOR_SCALE_BAR = "#000000"            # Black for scale bar

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
    if threshold_file and os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            thresholds = json.load(f)
        return {
            "MNP": thresholds.get("MNP", {}).get("optimal_threshold", DEFAULT_THRESHOLDS["MNP"]),
            "Fibroblast": thresholds.get("Fibroblast", {}).get("optimal_threshold", DEFAULT_THRESHOLDS["Fibroblast"]),
        }
    return DEFAULT_THRESHOLDS


def load_raw_data(sample: str, condition: str = "UC") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load raw spatial data from preprocessed h5ad files.
    
    Returns:
        coords: (N, 2) spatial coordinates
        X: (N, G) expression matrix
        cell_types: (N,) cell type labels
        gene_names: list of gene names
    """
    if condition == "UC":
        h5ad_path = PREPROCESSED_ROOT / "UC" / sample / "processed.h5ad"
    elif condition == "UC_VDZ":
        h5ad_path = PREPROCESSED_ROOT / "UC_VDZ" / sample / "processed.h5ad"
    elif condition == "HC":
        h5ad_path = PREPROCESSED_ROOT / "HC" / sample / "processed.h5ad"
    else:
        h5ad_path = PREPROCESSED_ROOT / sample / sample / "processed.h5ad"
    
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Data not found: {h5ad_path}")
    
    adata = ad.read_h5ad(h5ad_path)
    
    coords = adata.obsm["spatial"]
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    ann_col = 'fine_annotation' if 'fine_annotation' in adata.obs.columns else 'cell_type'
    cell_types = adata.obs[ann_col].values
    gene_names = list(adata.var_names)
    
    return coords.astype(np.float32), X.astype(np.float32), cell_types, gene_names


def load_perturbation_expression(
    results_base: Path,
    model: str,
    sample: str,
    perturb_mode: str,
    dose_lambda: float,
    perturb_target: str,
    n_spots: int,
    seed: int,
    step: int,
    deg_suffix: str = "UC_MNP_normal_vs_MNP_activated_DEG"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load expression data from perturbation experiment."""
    target_suffix = perturb_target.replace(" ", "_")
    dir_name = f"{perturb_mode}_lambda{dose_lambda}_{target_suffix}_{n_spots}spots_seed{seed}"
    if deg_suffix:
        dir_name = f"{dir_name}_{deg_suffix}"
    
    expr_file = results_base / model / sample / dir_name / "expression" / f"pert{step}_expression.csv"
    
    if not expr_file.exists():
        print(f"[WARN] Expression file not found: {expr_file}")
        return None, str(expr_file)
    
    expr_df = pd.read_csv(expr_file)
    print(f"[OK] Loaded perturbation expression: {expr_file}")
    print(f"     Shape: {expr_df.shape}, Columns sample: {list(expr_df.columns[:5])}")
    
    return expr_df, str(expr_file)


def compute_activation_scores(
    X: np.ndarray,
    gene_names: List[str],
    celltype: str,
) -> np.ndarray:
    """Compute activation scores for all cells based on marker genes."""
    markers = MARKER_GENES.get(celltype, [])
    gene_names_upper = [g.upper() for g in gene_names]
    
    marker_indices = []
    for m in markers:
        if m.upper() in gene_names_upper:
            marker_indices.append(gene_names_upper.index(m.upper()))
    
    if not marker_indices:
        return np.zeros(X.shape[0])
    
    return X[:, marker_indices].mean(axis=1)


def get_activated_cells(
    coords: np.ndarray,
    X: np.ndarray,
    cell_types: np.ndarray,
    gene_names: List[str],
    thresholds: Dict[str, float],
) -> Dict[str, Dict]:
    """
    Get activated MNP and Fibroblast cells.
    
    Returns dict with:
        - "MNP": {"indices": [...], "coords": [...], "scores": [...]}
        - "Fibroblast": {"indices": [...], "coords": [...], "scores": [...]}
    """
    result = {}
    
    for celltype in ["MNP", "Fibroblast"]:
        all_labels = CELLTYPE_LABELS[celltype]["all"]
        mask = np.isin(cell_types, all_labels)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            result[celltype] = {
                "indices": np.array([], dtype=int),
                "coords": np.zeros((0, 2)),
                "scores": np.array([]),
                "n_total": 0,
                "n_activated": 0,
            }
            continue
        
        # Compute scores
        scores = compute_activation_scores(X, gene_names, celltype)
        cell_scores = scores[mask]
        
        # Filter activated cells
        threshold = thresholds[celltype]
        activated_mask = cell_scores >= threshold
        activated_indices = indices[activated_mask]
        
        result[celltype] = {
            "indices": activated_indices,
            "coords": coords[activated_indices],
            "scores": cell_scores[activated_mask],
            "n_total": len(indices),
            "n_activated": len(activated_indices),
        }
    
    return result


# ==============================================================================
# 1. Nearest Neighbor Distance Analysis
# ==============================================================================

def compute_nn_distances(
    actMNP_coords: np.ndarray,
    actFibro_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nearest neighbor distances from activated MNP to activated Fibroblast.
    
    Returns:
        distances: (N,) distances from each actMNP to nearest actFibro
        nn_indices: (N,) index of nearest actFibro for each actMNP
        nn_coords: (N, 2) coordinates of nearest actFibro
    """
    if len(actMNP_coords) == 0 or len(actFibro_coords) == 0:
        return np.array([]), np.array([], dtype=int), np.zeros((0, 2))
    
    tree = cKDTree(actFibro_coords)
    distances, nn_indices = tree.query(actMNP_coords, k=1)
    nn_coords = actFibro_coords[nn_indices]
    
    return distances, nn_indices, nn_coords


def add_scale_bar(
    ax,
    coords: np.ndarray,
    pixel_to_micron: float = PIXEL_TO_MICRON,
    scale_length_um: float = 100,
    location: str = 'lower right',
    bar_height_ratio: float = 0.002,
    padding: float = 0.02,
    fontsize: int = 10,
):
    """Add a scale bar to the plot."""
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    
    # Convert scale length to pixels
    scale_length_px = scale_length_um / pixel_to_micron
    
    # Determine position
    if 'right' in location:
        x_start = coords[:, 0].max() - padding * x_range - scale_length_px
    else:
        x_start = coords[:, 0].min() + padding * x_range
    
    if 'lower' in location:
        y_pos = coords[:, 1].max() - padding * y_range
    else:
        y_pos = coords[:, 1].min() + padding * y_range
    
    bar_height = bar_height_ratio * y_range
    
    # Draw scale bar
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x_start, y_pos + bar_height * 15),
        scale_length_px,
        bar_height,
        facecolor=COLOR_SCALE_BAR,
        edgecolor=COLOR_SCALE_BAR,
        zorder=10
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        x_start + scale_length_px / 2,
        y_pos - bar_height * 2,
        f'{int(scale_length_um)} μm',
        ha='center',
        va='top',
        fontsize=fontsize,
        fontweight='bold',
        zorder=10
    )


def plot_nn_visualization(
    coords: np.ndarray,
    cell_types: np.ndarray,
    activated: Dict[str, Dict],
    nn_distances_um: np.ndarray,
    nn_coords: np.ndarray,
    title: str,
    out_path: str,
    pixel_to_micron: float = PIXEL_TO_MICRON,
    figsize: Tuple[float, float] = (14, 12),
    spot_size: float = 15,
    line_alpha: float = 0.5,
    line_width: float = 1.5,
    scale_bar_length: float = 100,
    dpi: int = 300,
):
    """Create spatial visualization with NN connection lines and scale bar."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Assign colors to all cells
    n_cells = len(coords)
    colors = np.array([to_rgba(COLOR_OTHER)] * n_cells)
    
    # Color activated MNP
    for idx in activated["MNP"]["indices"]:
        colors[idx] = to_rgba(COLOR_MNP_ACTIVATED)
    
    # Color activated Fibroblast
    for idx in activated["Fibroblast"]["indices"]:
        colors[idx] = to_rgba(COLOR_FIBRO_ACTIVATED)
    
    # Plot all spots
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=spot_size, alpha=0.8, edgecolors='none', zorder=1)
    
    # Draw NN connection lines
    if len(activated["MNP"]["coords"]) > 0 and len(nn_coords) > 0:
        actMNP_coords = activated["MNP"]["coords"]
        
        lines = []
        for i in range(len(actMNP_coords)):
            lines.append([actMNP_coords[i], nn_coords[i]])
        
        lc = LineCollection(lines, colors=COLOR_NN_LINE, alpha=line_alpha, linewidths=line_width, zorder=2)
        ax.add_collection(lc)
    
    # Add scale bar
    add_scale_bar(ax, coords, pixel_to_micron, scale_bar_length)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_MNP_ACTIVATED, edgecolor='none', 
                       label=f'Activated MNP ({activated["MNP"]["n_activated"]}/{activated["MNP"]["n_total"]})'),
        mpatches.Patch(facecolor=COLOR_FIBRO_ACTIVATED, edgecolor='none', 
                       label=f'Activated Fibro ({activated["Fibroblast"]["n_activated"]}/{activated["Fibroblast"]["n_total"]})'),
        mpatches.Patch(facecolor=COLOR_OTHER, edgecolor='none', label='Other cells'),
    ]
    
    if len(nn_distances_um) > 0:
        legend_elements.append(
            plt.Line2D([0], [0], color=COLOR_NN_LINE, linewidth=2, alpha=line_alpha,
                      label=f'NN lines (median={np.median(nn_distances_um):.1f} μm)')
        )
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=10, frameon=True, fancybox=True)
    
    # Add statistics text
    if len(nn_distances_um) > 0:
        stats_text = (
            f"NN Distance Statistics (μm):\n"
            f"  Median: {np.median(nn_distances_um):.2f}\n"
            f"  Mean: {np.mean(nn_distances_um):.2f}\n"
            f"  Std: {np.std(nn_distances_um):.2f}\n"
            f"  Min: {np.min(nn_distances_um):.2f}\n"
            f"  Max: {np.max(nn_distances_um):.2f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved NN visualization to {out_path}")


# ==============================================================================
# 2. Nearest Neighbor Type Probability
# ==============================================================================

def compute_nn_type_probability(
    actMNP_coords: np.ndarray,
    all_coords: np.ndarray,
    cell_types: np.ndarray,
    actMNP_indices: np.ndarray,
) -> Dict[str, float]:
    """
    Compute probability that the nearest neighbor of activated MNP is:
    - activated Fibroblast
    - any Fibroblast
    - activated MNP (self-type)
    """
    if len(actMNP_coords) == 0:
        return {
            "p_nn_actFibro": 0.0,
            "p_nn_anyFibro": 0.0,
            "p_nn_actMNP": 0.0,
            "n_actMNP": 0,
        }
    
    tree = cKDTree(all_coords)
    _, nn_indices = tree.query(actMNP_coords, k=2)  # k=2 to exclude self
    
    nn_actFibro_count = 0
    nn_anyFibro_count = 0
    nn_actMNP_count = 0
    
    fibro_all = CELLTYPE_LABELS["Fibroblast"]["all"]
    fibro_act = CELLTYPE_LABELS["Fibroblast"]["activated"]
    mnp_act = CELLTYPE_LABELS["MNP"]["activated"]
    
    for i, mnp_idx in enumerate(actMNP_indices):
        nn_idx = nn_indices[i, 1] if nn_indices[i, 0] == mnp_idx else nn_indices[i, 0]
        nn_type = cell_types[nn_idx]
        
        if nn_type in fibro_act:
            nn_actFibro_count += 1
        if nn_type in fibro_all:
            nn_anyFibro_count += 1
        if nn_type in mnp_act:
            nn_actMNP_count += 1
    
    n = len(actMNP_coords)
    return {
        "p_nn_actFibro": nn_actFibro_count / n,
        "p_nn_anyFibro": nn_anyFibro_count / n,
        "p_nn_actMNP": nn_actMNP_count / n,
        "n_actMNP": n,
        "count_nn_actFibro": nn_actFibro_count,
        "count_nn_anyFibro": nn_anyFibro_count,
    }


# ==============================================================================
# 3. Neighborhood Enrichment Z-score
# ==============================================================================

def compute_neighborhood_enrichment(
    actMNP_coords: np.ndarray,
    actFibro_coords: np.ndarray,
    all_coords: np.ndarray,
    k: int = 10,
) -> Tuple[float, float]:
    """
    Compute neighborhood enrichment statistics.
    
    For each activated MNP, count how many of its k nearest neighbors are activated Fibroblast.
    """
    if len(actMNP_coords) == 0 or len(actFibro_coords) == 0:
        return 0.0, 0.0
    
    tree = cKDTree(all_coords)
    actFibro_tree = cKDTree(actFibro_coords)
    
    counts = []
    ratios = []
    
    for mnp_coord in actMNP_coords:
        _, nn_indices = tree.query(mnp_coord, k=k+1)
        
        count = 0
        for nn_idx in nn_indices[1:]:  # Skip first (self)
            nn_coord = all_coords[nn_idx]
            dist, _ = actFibro_tree.query(nn_coord, k=1)
            if dist < 1e-6:  # Same coordinate
                count += 1
        
        counts.append(count)
        ratios.append(count / k)
    
    return np.mean(counts), np.mean(ratios)


def compute_enrichment_zscore(
    coords: np.ndarray,
    cell_types: np.ndarray,
    X: np.ndarray,
    gene_names: List[str],
    thresholds: Dict[str, float],
    k: int = 10,
    n_permutations: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute neighborhood enrichment Z-score using permutation test.
    
    1. Compute observed enrichment (actFibro in k-neighborhood of actMNP)
    2. Permute activation labels B times and compute null distribution
    3. Calculate Z-score = (observed - mean_null) / std_null
    """
    np.random.seed(seed)
    
    activated = get_activated_cells(coords, X, cell_types, gene_names, thresholds)
    
    actMNP_coords = activated["MNP"]["coords"]
    actFibro_coords = activated["Fibroblast"]["coords"]
    
    if len(actMNP_coords) == 0 or len(actFibro_coords) == 0:
        return {
            "observed_count": 0.0,
            "observed_ratio": 0.0,
            "null_mean_count": 0.0,
            "null_std_count": 0.0,
            "null_mean_ratio": 0.0,
            "null_std_ratio": 0.0,
            "zscore_count": 0.0,
            "zscore_ratio": 0.0,
            "n_actMNP": len(actMNP_coords),
            "n_actFibro": len(actFibro_coords),
            "k": k,
            "n_permutations": n_permutations,
        }
    
    # Compute observed enrichment
    obs_count, obs_ratio = compute_neighborhood_enrichment(actMNP_coords, actFibro_coords, coords, k)
    
    # Permutation test
    mnp_mask = np.isin(cell_types, CELLTYPE_LABELS["MNP"]["all"])
    fibro_mask = np.isin(cell_types, CELLTYPE_LABELS["Fibroblast"]["all"])
    
    mnp_indices = np.where(mnp_mask)[0]
    fibro_indices = np.where(fibro_mask)[0]
    
    n_actMNP = len(actMNP_coords)
    n_actFibro = len(actFibro_coords)
    
    null_counts = []
    null_ratios = []
    
    for _ in tqdm(range(n_permutations), desc="Permutation test", leave=False):
        if len(mnp_indices) >= n_actMNP:
            perm_actMNP_indices = np.random.choice(mnp_indices, n_actMNP, replace=False)
        else:
            perm_actMNP_indices = mnp_indices
        
        if len(fibro_indices) >= n_actFibro:
            perm_actFibro_indices = np.random.choice(fibro_indices, n_actFibro, replace=False)
        else:
            perm_actFibro_indices = fibro_indices
        
        perm_actMNP_coords = coords[perm_actMNP_indices]
        perm_actFibro_coords = coords[perm_actFibro_indices]
        
        perm_count, perm_ratio = compute_neighborhood_enrichment(
            perm_actMNP_coords, perm_actFibro_coords, coords, k
        )
        null_counts.append(perm_count)
        null_ratios.append(perm_ratio)
    
    null_counts = np.array(null_counts)
    null_ratios = np.array(null_ratios)
    
    # Compute Z-scores
    null_mean_count = np.mean(null_counts)
    null_std_count = np.std(null_counts)
    null_mean_ratio = np.mean(null_ratios)
    null_std_ratio = np.std(null_ratios)
    
    zscore_count = (obs_count - null_mean_count) / null_std_count if null_std_count > 0 else 0.0
    zscore_ratio = (obs_ratio - null_mean_ratio) / null_std_ratio if null_std_ratio > 0 else 0.0
    
    return {
        "observed_count": obs_count,
        "observed_ratio": obs_ratio,
        "null_mean_count": null_mean_count,
        "null_std_count": null_std_count,
        "null_mean_ratio": null_mean_ratio,
        "null_std_ratio": null_std_ratio,
        "zscore_count": zscore_count,
        "zscore_ratio": zscore_ratio,
        "n_actMNP": n_actMNP,
        "n_actFibro": n_actFibro,
        "k": k,
        "n_permutations": n_permutations,
    }


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def run_enrichment_analysis(
    sample: str,
    condition: str = "UC",
    mode: str = "raw",
    results_base: Optional[Path] = None,
    model: Optional[str] = None,
    perturb_mode: Optional[str] = None,
    dose_lambda: Optional[float] = None,
    perturb_target: str = "MNP_activated",
    n_spots: int = 40,
    seed: int = 42,
    step: int = 3,
    deg_suffix: str = "UC_MNP_normal_vs_MNP_activated_DEG",
    k_neighbors: int = 10,
    n_permutations: int = 500,
    out_dir: str = "output/enrichment",
    save_plot: bool = True,
    figsize: Tuple[float, float] = (14, 12),
    spot_size: float = 15,
    pixel_to_micron: float = PIXEL_TO_MICRON,
    scale_bar_length: float = 100,
    dpi: int = 300,
    threshold_file: Optional[str] = None,
) -> Dict:
    """Run complete enrichment analysis for a sample."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if results_base is None:
        results_base = DEFAULT_RESULTS_BASE
    
    # Load data
    print(f"\n[INFO] Loading data for {sample} ({mode})...")
    coords, X, cell_types, gene_names = load_raw_data(sample, condition)
    thresholds = load_activation_thresholds(threshold_file)
    
    # If perturbation mode, override expression with perturbation results
    if mode == "perturbation" and model is not None:
        print(f"[INFO] Loading perturbation expression from {model}...")
        expr_df, expr_file = load_perturbation_expression(
            results_base, model, sample, perturb_mode, dose_lambda, perturb_target,
            n_spots, seed, step, deg_suffix
        )
        if expr_df is not None and len(expr_df) > 0:
            gene_columns = [c for c in expr_df.columns if c not in ['spot_idx', 'cell_type', 'is_perturbed']]
            expr_gene_map = {g.lower(): g for g in gene_columns}
            
            n_updated = 0
            for _, row in expr_df.iterrows():
                spot_idx = int(row["spot_idx"])
                if spot_idx >= len(X):
                    continue
                
                for gene_name in gene_names:
                    gene_lower = gene_name.lower()
                    if gene_lower in expr_gene_map:
                        expr_col = expr_gene_map[gene_lower]
                        gene_idx = gene_names.index(gene_name)
                        new_val = row[expr_col]
                        if pd.notna(new_val):
                            X[spot_idx, gene_idx] = float(new_val)
                n_updated += 1
            
            print(f"[OK] Updated {n_updated} spots from perturbation expression (step {step})")
    
    # Get activated cells
    print("[INFO] Identifying activated cells...")
    activated = get_activated_cells(coords, X, cell_types, gene_names, thresholds)
    
    print(f"  Activated MNP: {activated['MNP']['n_activated']}/{activated['MNP']['n_total']}")
    print(f"  Activated Fibroblast: {activated['Fibroblast']['n_activated']}/{activated['Fibroblast']['n_total']}")
    
    # 1. Nearest Neighbor Distance Analysis
    print("\n[1] Computing NN distances...")
    nn_distances_px, nn_indices, nn_coords = compute_nn_distances(
        activated["MNP"]["coords"],
        activated["Fibroblast"]["coords"]
    )
    
    nn_distances_um = nn_distances_px * pixel_to_micron
    
    nn_stats = {
        "nn_distance_median_um": float(np.median(nn_distances_um)) if len(nn_distances_um) > 0 else None,
        "nn_distance_mean_um": float(np.mean(nn_distances_um)) if len(nn_distances_um) > 0 else None,
        "nn_distance_std_um": float(np.std(nn_distances_um)) if len(nn_distances_um) > 0 else None,
        "nn_distance_min_um": float(np.min(nn_distances_um)) if len(nn_distances_um) > 0 else None,
        "nn_distance_max_um": float(np.max(nn_distances_um)) if len(nn_distances_um) > 0 else None,
        "n_nn_pairs": len(nn_distances_um),
        "pixel_to_micron": pixel_to_micron,
    }
    print(f"  Median NN distance: {nn_stats['nn_distance_median_um']:.2f} μm" if nn_stats['nn_distance_median_um'] else "  No NN pairs")
    
    # 2. Nearest Neighbor Type Probability
    print("\n[2] Computing NN type probabilities...")
    nn_probs = compute_nn_type_probability(
        activated["MNP"]["coords"],
        coords,
        cell_types,
        activated["MNP"]["indices"]
    )
    print(f"  P(NN = actFibro): {nn_probs['p_nn_actFibro']:.4f}")
    print(f"  P(NN = anyFibro): {nn_probs['p_nn_anyFibro']:.4f}")
    
    # 3. Neighborhood Enrichment Z-score
    print(f"\n[3] Computing enrichment Z-score (k={k_neighbors}, {n_permutations} permutations)...")
    zscore_results = compute_enrichment_zscore(
        coords, cell_types, X, gene_names, thresholds,
        k=k_neighbors, n_permutations=n_permutations, seed=seed
    )
    print(f"  Observed count: {zscore_results['observed_count']:.4f}")
    print(f"  Null mean: {zscore_results['null_mean_count']:.4f} ± {zscore_results['null_std_count']:.4f}")
    print(f"  Z-score (count): {zscore_results['zscore_count']:.4f}")
    print(f"  Z-score (ratio): {zscore_results['zscore_ratio']:.4f}")
    
    # Compile results
    results = {
        "sample": sample,
        "condition": condition,
        "mode": mode,
        "model": model,
        "perturb_mode": perturb_mode,
        "dose_lambda": dose_lambda,
        "perturb_target": perturb_target,
        "n_spots": n_spots,
        "seed": seed,
        "step": step if mode == "perturbation" else -1,
        "n_actMNP": activated["MNP"]["n_activated"],
        "n_total_MNP": activated["MNP"]["n_total"],
        "n_actFibro": activated["Fibroblast"]["n_activated"],
        "n_total_Fibro": activated["Fibroblast"]["n_total"],
        **nn_stats,
        **nn_probs,
        **zscore_results,
    }
    
    # Save NN visualization
    if save_plot:
        if mode == "raw":
            plot_name = f"nn_visualization_{sample}_{condition}_raw.png"
            title = f"{sample} ({condition}) - Raw Data"
        else:
            plot_name = f"nn_visualization_{sample}_{model}_{perturb_mode}_lambda{dose_lambda}_step{step}.png"
            title = f"{sample} - {model} ({perturb_mode}, λ={dose_lambda}, step{step})"
        
        plot_path = out_dir / plot_name
        plot_nn_visualization(
            coords=coords,
            cell_types=cell_types,
            activated=activated,
            nn_distances_um=nn_distances_um,
            nn_coords=nn_coords,
            title=title,
            out_path=str(plot_path),
            pixel_to_micron=pixel_to_micron,
            figsize=figsize,
            spot_size=spot_size,
            scale_bar_length=scale_bar_length,
            dpi=dpi,
        )
    
    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="MNP-Fibroblast Enrichment Analysis")
    
    # Mode selection
    ap.add_argument("--mode", type=str, default="raw", choices=["raw", "perturbation"],
                   help="Analysis mode: 'raw' for raw data, 'perturbation' for perturbation results")
    
    # Sample selection
    ap.add_argument("--sample", type=str, required=True, help="Sample name")
    ap.add_argument("--condition", type=str, default="UC", choices=["UC", "UC_VDZ", "HC"],
                   help="Sample condition")
    
    # Perturbation configuration
    ap.add_argument("--model", type=str, default="spatialgt", help="Model for perturbation mode")
    ap.add_argument("--perturb_mode", type=str, default="patch", help="Perturbation mode (random or patch)")
    ap.add_argument("--dose_lambda", type=float, default=1.0, help="Dose lambda value")
    ap.add_argument("--perturb_target", type=str, default="MNP_activated", help="Perturbation target")
    ap.add_argument("--n_spots", type=int, default=40, help="Number of spots")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--step", type=int, default=3, help="Step for perturbation")
    ap.add_argument("--deg_suffix", type=str, default="UC_MNP_normal_vs_MNP_activated_DEG", help="DEG suffix")
    ap.add_argument("--results_base", type=str, default=None, help="Base directory for perturbation results")
    
    # Analysis parameters
    ap.add_argument("--k_neighbors", type=int, default=10, help="Number of neighbors for enrichment analysis")
    ap.add_argument("--n_permutations", type=int, default=500, help="Number of permutations for Z-score")
    ap.add_argument("--threshold_file", type=str, default=None, help="Activation threshold file")
    
    # Output
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--out_prefix", type=str, default="enrichment", help="Output file prefix")
    
    # Visualization
    ap.add_argument("--no_plot", action="store_true", help="Skip visualization")
    ap.add_argument("--figsize", type=float, nargs=2, default=[14, 12], help="Figure size")
    ap.add_argument("--spot_size", type=float, default=15, help="Spot size")
    ap.add_argument("--dpi", type=int, default=600, help="DPI")
    
    # Spatial resolution
    ap.add_argument("--pixel_to_micron", type=float, default=PIXEL_TO_MICRON,
                   help=f"Conversion factor from pixels to microns (default: {PIXEL_TO_MICRON} for CosMx)")
    ap.add_argument("--scale_bar_length", type=float, default=50, help="Length of scale bar in microns")
    
    args = ap.parse_args()
    
    results_base = Path(args.results_base) if args.results_base else None
    
    # Run analysis
    results = run_enrichment_analysis(
        sample=args.sample,
        condition=args.condition,
        mode=args.mode,
        results_base=results_base,
        model=args.model if args.mode == "perturbation" else None,
        perturb_mode=args.perturb_mode if args.mode == "perturbation" else None,
        dose_lambda=args.dose_lambda if args.mode == "perturbation" else None,
        perturb_target=args.perturb_target,
        n_spots=args.n_spots,
        seed=args.seed,
        step=args.step,
        deg_suffix=args.deg_suffix,
        k_neighbors=args.k_neighbors,
        n_permutations=args.n_permutations,
        out_dir=args.out_dir,
        save_plot=not args.no_plot,
        figsize=tuple(args.figsize),
        spot_size=args.spot_size,
        pixel_to_micron=args.pixel_to_micron,
        scale_bar_length=args.scale_bar_length,
        dpi=args.dpi,
        threshold_file=args.threshold_file,
    )
    
    # Save results to CSV
    out_dir = Path(args.out_dir)
    if args.mode == "raw":
        csv_name = f"{args.out_prefix}_{args.sample}_{args.condition}_raw.csv"
    else:
        csv_name = f"{args.out_prefix}_{args.sample}_{args.model}_{args.perturb_mode}_lambda{args.dose_lambda}_step{args.step}.csv"
    
    csv_path = out_dir / csv_name
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"\n[OK] Saved results to {csv_path}")
    
    # Also save as JSON
    json_path = csv_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved results to {json_path}")
    
    print("\n[DONE] Enrichment analysis complete!")


if __name__ == "__main__":
    main()
