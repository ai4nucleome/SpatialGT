#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEDR Baseline for Niche Perturbation Evaluation.

This script uses SEDR (Spatial Embedded Deep Representation) as a baseline
for the perturbation prediction task. SEDR is trained on PT and Sham slices,
then used for iterative expression reconstruction with perturbation.

Supports:
- Two perturbation modes: patch (BFS-connected spots) and random (scattered spots)
- Two weighting schemes: gaussian (distance-based) and uniform (constant)
- Freeze perturbed spots during iteration
- Mean-vector similarity: PCC, Spearman, Cosine, L1, L2

Outputs:
- summary.csv/json: per-step, per-ROI similarity metrics
- roi_mean_vectors_step{t}.npz: mean expression vectors for downstream visualization
- perturb_manifest.json: reproducibility manifest
- expression/: full expression matrices for each step
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm
import anndata as ad

# Ensure repo root importability
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import SEDR
_SEDR_ROOT = _REPO_ROOT / "baseline" / "SEDR"
if str(_SEDR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEDR_ROOT))

from SEDR.SEDR_model import Sedr
from SEDR.graph_func import graph_construction


# ==============================================================================
# Utility Functions
# ==============================================================================

def _load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load gene vocabulary and return lowercase mapping."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {k.lower(): int(v) for k, v in vocab.items()}


def _barcodes_to_indices(obs_names: List[str], barcodes: List[str]) -> List[int]:
    """Convert barcodes to indices in obs_names."""
    pos = {b: i for i, b in enumerate(obs_names)}
    return [pos[b] for b in barcodes if b in pos]


def _indices_to_barcodes(obs_names: List[str], indices: List[int]) -> List[str]:
    """Convert indices to barcodes."""
    return [obs_names[i] for i in indices if 0 <= i < len(obs_names)]


# ==============================================================================
# Data Loading and Preparation
# ==============================================================================

def load_adata_and_prepare(
    cache_dir: Path,
    dataset_name: str,
    vocab: Dict[str, int],
) -> Tuple[ad.AnnData, np.ndarray, np.ndarray, List[int]]:
    """
    Load processed.h5ad and prepare expression matrix with vocab gene_ids.
    
    Returns:
        adata: AnnData object
        X: expression matrix [n_spots, n_genes] as numpy array
        gene_ids: array of gene_ids corresponding to columns of X
        valid_gene_indices: indices of genes that are in vocab
    """
    h5ad_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(h5ad_path))
    
    # Map gene names to vocab gene_ids
    gene_names = list(adata.var_names)
    gene_ids = []
    valid_gene_indices = []
    
    for i, gene in enumerate(gene_names):
        gid = vocab.get(gene.lower(), None)
        if gid is not None:
            gene_ids.append(gid)
            valid_gene_indices.append(i)
    
    gene_ids = np.array(gene_ids, dtype=np.int64)
    
    # Extract expression matrix (only valid genes)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X[:, valid_gene_indices].astype(np.float32)
    
    print(f"[DATA] Loaded {dataset_name}: {adata.n_obs} spots, {len(valid_gene_indices)} genes (of {adata.n_vars})")
    
    return adata, X, gene_ids, valid_gene_indices


def find_common_genes_between_slices(
    gene_ids_pt: np.ndarray,
    gene_ids_sham: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Find common genes between PT and Sham slices.
    
    Returns:
        common_gene_ids: sorted array of common gene_ids
        pt_to_common: mapping from PT gene index to common gene index
        sham_to_common: mapping from Sham gene index to common gene index
    """
    pt_set = set(gene_ids_pt.tolist())
    sham_set = set(gene_ids_sham.tolist())
    common = sorted(pt_set & sham_set)
    common_gene_ids = np.array(common, dtype=np.int64)
    
    # Create mappings
    common_to_idx = {gid: i for i, gid in enumerate(common)}
    
    pt_to_common = {}
    for i, gid in enumerate(gene_ids_pt):
        if gid in common_to_idx:
            pt_to_common[i] = common_to_idx[gid]
    
    sham_to_common = {}
    for i, gid in enumerate(gene_ids_sham):
        if gid in common_to_idx:
            sham_to_common[i] = common_to_idx[gid]
    
    return common_gene_ids, pt_to_common, sham_to_common


# ==============================================================================
# SEDR Wrapper Functions
# ==============================================================================

def fix_seed(seed: int = 42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_sedr(
    X: np.ndarray,
    adata: ad.AnnData,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    graph_knn_k: int = 6,
    rec_w: float = 10.0,
    gcn_w: float = 0.1,
    self_w: float = 1.0,
    device: str = "cuda:0",
    mode: str = "imputation",
    show_progress: bool = True,
    seed: int = 42,
) -> Sedr:
    """
    Train SEDR model on expression matrix.
    
    Args:
        X: expression matrix [n_spots, n_genes]
        adata: AnnData with spatial coordinates
        epochs: number of training epochs
        lr: learning rate
        weight_decay: weight decay
        graph_knn_k: number of neighbors for KNN graph
        rec_w: reconstruction loss weight
        gcn_w: GCN loss weight
        self_w: self-supervised loss weight
        device: torch device
        mode: "clustering" or "imputation"
        show_progress: whether to show progress bar
        seed: random seed for reproducibility
    
    Returns:
        Trained Sedr model
    """
    # Fix random seed for reproducibility
    fix_seed(seed)
    
    # Build graph
    graph_dict = graph_construction(adata, n=graph_knn_k, mode='KNN')
    
    # Create SEDR model
    sedr = Sedr(
        X=X,
        graph_dict=graph_dict,
        rec_w=rec_w,
        gcn_w=gcn_w,
        self_w=self_w,
        mode=mode,
        device=device,
    )
    
    # Train
    print(f"[SEDR] Training for {epochs} epochs...")
    sedr.train_without_dec(epochs=epochs, lr=lr, decay=weight_decay)
    
    return sedr


def sedr_reconstruct(
    sedr: Sedr,
    X: np.ndarray,
    adj_norm: torch.Tensor,
) -> np.ndarray:
    """
    Run SEDR reconstruction on expression matrix.
    
    Args:
        sedr: trained Sedr model
        X: input expression matrix [n_spots, n_genes]
        adj_norm: normalized adjacency matrix
    
    Returns:
        Reconstructed expression matrix [n_spots, n_genes]
    """
    sedr.model.eval()
    
    # Disable mask for deterministic inference
    sedr.model._mask_rate = 0.0
    
    X_tensor = torch.FloatTensor(X).to(sedr.device)
    
    with torch.no_grad():
        # Forward pass
        z, mu, logvar, de_feat, q, feat_x, gnn_z, loss = sedr.model(X_tensor, adj_norm)
    
    # Get reconstruction and clip to non-negative
    recon = de_feat.detach().cpu().numpy()
    recon = np.clip(recon, 0, None)
    
    return recon.astype(np.float32)


def sedr_reconstruct_with_roi_mask(
    sedr: Sedr,
    X: np.ndarray,
    adj_norm: torch.Tensor,
    mask_indices: List[int],
) -> np.ndarray:
    """
    Run SEDR reconstruction with specific ROI spots masked.
    
    This function masks the specified ROI spots (by adding the learned mask token)
    and lets SEDR reconstruct their expression based on the surrounding unmasked spots.
    
    Args:
        sedr: trained Sedr model
        X: input expression matrix [n_spots, n_genes]
        adj_norm: normalized adjacency matrix
        mask_indices: list of spot indices to mask (the ROI spots to predict)
    
    Returns:
        Reconstructed expression matrix [n_spots, n_genes]
    """
    sedr.model.eval()
    
    # Disable the random mask - we'll apply our own mask
    sedr.model._mask_rate = 0.0
    
    X_tensor = torch.FloatTensor(X.copy()).to(sedr.device)
    
    # Manually apply mask token to the specified ROI spots
    mask_indices_tensor = torch.LongTensor(mask_indices).to(sedr.device)
    X_tensor[mask_indices_tensor] = X_tensor[mask_indices_tensor] + sedr.model.enc_mask_token
    
    with torch.no_grad():
        # Manual forward pass (bypassing the random mask in forward)
        # Encoder
        feat_x = sedr.model.encoder(X_tensor)
        
        # GCN layers
        hidden1 = sedr.model.gc1(feat_x, adj_norm)
        mu = sedr.model.gc2(hidden1, adj_norm)
        logvar = sedr.model.gc3(hidden1, adj_norm)
        
        # Reparameterize (in eval mode, just return mu)
        gnn_z = mu
        
        # Concatenate features
        z = torch.cat((feat_x, gnn_z), 1)
        
        # Decoder
        if hasattr(sedr.model, 'decoder'):
            if hasattr(sedr.model.decoder, 'forward'):
                try:
                    # GraphConvolution decoder needs adj
                    de_feat = sedr.model.decoder(z, adj_norm)
                except TypeError:
                    # Sequential decoder doesn't need adj
                    de_feat = sedr.model.decoder(z)
            else:
                de_feat = sedr.model.decoder(z)
        else:
            de_feat = z
    
    # Get reconstruction and clip to non-negative
    recon = de_feat.detach().cpu().numpy()
    recon = np.clip(recon, 0, None)
    
    return recon.astype(np.float32)


# ==============================================================================
# Spot Selection: Patch (BFS) and Random
# ==============================================================================

def select_patch_spots(
    adata: ad.AnnData,
    roi_indices: List[int],
    patch_size: int,
    seed: int,
    graph_knn_k: int = 6,
) -> List[int]:
    """Select a contiguous patch of spots via BFS from a seed spot."""
    from sklearn import metrics
    
    rng = random.Random(seed)
    roi_set = set(roi_indices)
    if not roi_set:
        return []
    
    # Build adjacency from spatial coordinates
    spatial = adata.obsm['spatial']
    dist = metrics.pairwise_distances(spatial)
    
    # Get k nearest neighbors
    neighbors = {}
    for i in range(len(adata)):
        nb_indices = np.argsort(dist[i])[:graph_knn_k + 1]
        neighbors[i] = [int(j) for j in nb_indices if j != i]
    
    # Choose seed spot
    seed_spot = rng.choice(list(roi_set))
    
    # BFS expansion within ROI
    visited = {seed_spot}
    queue = deque([seed_spot])
    selected = [seed_spot]
    
    while queue and len(selected) < patch_size:
        curr = queue.popleft()
        for nb in neighbors.get(curr, []):
            if nb not in visited and nb in roi_set:
                visited.add(nb)
                selected.append(nb)
                queue.append(nb)
                if len(selected) >= patch_size:
                    break
    
    return selected[:patch_size]


def select_random_spots(
    roi_indices: List[int],
    n_spots: int,
    seed: int,
) -> List[int]:
    """Randomly sample n_spots from ROI."""
    rng = random.Random(seed)
    roi_list = list(set(roi_indices))
    if len(roi_list) <= n_spots:
        return roi_list
    return rng.sample(roi_list, n_spots)


# ==============================================================================
# Weight Computation: Gaussian and Uniform
# ==============================================================================

def compute_spot_weights(
    adata: ad.AnnData,
    selected_indices: List[int],
    weighting: str,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Compute weights for selected spots."""
    spatial = adata.obsm['spatial']
    coords = np.array([spatial[i] for i in selected_indices], dtype=np.float64)
    
    if len(coords) == 0:
        return np.array([], dtype=np.float32), {"weighting": weighting, "n_spots": 0}
    
    center = coords.mean(axis=0)
    
    meta = {
        "weighting": weighting,
        "n_spots": len(selected_indices),
        "center_xy": [float(center[0]), float(center[1])],
    }
    
    if weighting == "uniform":
        weights = np.ones(len(selected_indices), dtype=np.float32)
    else:  # gaussian
        d2 = ((coords - center) ** 2).sum(axis=1)
        if sigma is None:
            d = np.sqrt(d2)
            sigma = float(np.median(d)) if np.median(d) > 0 else float(np.max(d) if np.max(d) > 0 else 1.0)
        weights = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        meta["sigma"] = float(sigma)
    
    return weights, meta


# ==============================================================================
# DEG-based Perturbation
# ==============================================================================

def apply_deg_perturbation(
    X: np.ndarray,
    gene_ids: np.ndarray,
    selected_indices: List[int],
    weights: np.ndarray,
    deg_csv: Path,
    vocab: Dict[str, int],
    p_adj_thresh: float = 0.1,
    min_abs_logfc: float = 0.0,
    logfc_strength: float = 1.0,
    logfc_clip: float = 5.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply DEG-based perturbation to selected spots.
    
    Formula: new_val = old_val × 2^(logFC × strength × weight)
    """
    deg = pd.read_csv(deg_csv)
    if "avg_logFC" not in deg.columns:
        raise ValueError(f"DEG csv missing required column avg_logFC")
    
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col is not None:
        deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
        deg = deg[np.isfinite(deg[p_col])]
        deg = deg[deg[p_col].astype(float) < float(p_adj_thresh)]
    
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    # Build gene_id -> logFC mapping
    gene_logfc: Dict[int, float] = {}
    for _, r in deg.iterrows():
        gid = vocab.get(str(r["gene"]).lower(), None)
        if gid is None:
            continue
        logfc_val = float(r["avg_logFC"])
        logfc_clipped = np.clip(logfc_val, -float(logfc_clip), float(logfc_clip))
        gene_logfc[int(gid)] = logfc_clipped
    
    # Create gene_id to column index mapping
    gid_to_col = {int(gid): i for i, gid in enumerate(gene_ids)}
    
    # Apply perturbation
    X_perturbed = X.copy()
    n_hits_per_spot = []
    
    for i, spot_idx in enumerate(selected_indices):
        if i >= len(weights):
            break
        wi = float(weights[i])
        n_hits = 0
        
        for gid, logfc_val in gene_logfc.items():
            if gid in gid_to_col:
                col_idx = gid_to_col[gid]
                fold_change = np.power(2.0, logfc_val * logfc_strength * wi)
                X_perturbed[spot_idx, col_idx] *= fold_change
                n_hits += 1
        
        n_hits_per_spot.append(n_hits)
    
    # Ensure non-negative
    X_perturbed = np.maximum(X_perturbed, 0.0)
    
    meta = {
        "deg_csv": str(deg_csv),
        "p_adj_thresh": float(p_adj_thresh),
        "min_abs_logfc": float(min_abs_logfc),
        "logfc_strength": float(logfc_strength),
        "logfc_clip": float(logfc_clip),
        "n_deg_genes_mapped": len(gene_logfc),
        "n_hits_per_spot_mean": float(np.mean(n_hits_per_spot)) if n_hits_per_spot else 0,
    }
    
    return X_perturbed, meta


# ==============================================================================
# Similarity Metrics
# ==============================================================================

def compute_roi_mean_vector(
    X: np.ndarray,
    roi_indices: List[int],
) -> np.ndarray:
    """Compute mean expression vector for ROI spots."""
    if len(roi_indices) == 0:
        return np.zeros(X.shape[1], dtype=np.float32)
    return X[roi_indices].mean(axis=0).astype(np.float32)


def compute_mean_vector_similarities(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> Dict[str, float]:
    """Compute similarity metrics between two mean vectors."""
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.shape != vec_b.shape:
        return {"pcc": float("nan"), "spearman": float("nan"), "cos": float("nan"), "l1": float("nan"), "l2": float("nan")}
    
    # L1, L2
    l1 = float(np.sum(np.abs(vec_a - vec_b)))
    l2 = float(np.sqrt(np.sum((vec_a - vec_b) ** 2)))
    
    # Cosine
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a > 1e-8 and norm_b > 1e-8:
        cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    else:
        cos = float("nan")
    
    # PCC
    a_centered = vec_a - vec_a.mean()
    b_centered = vec_b - vec_b.mean()
    norm_ac = np.linalg.norm(a_centered)
    norm_bc = np.linalg.norm(b_centered)
    if norm_ac > 1e-8 and norm_bc > 1e-8:
        pcc = float(np.dot(a_centered, b_centered) / (norm_ac * norm_bc))
    else:
        pcc = float("nan")
    
    # Spearman (rank correlation)
    rank_a = np.argsort(np.argsort(vec_a)).astype(np.float32)
    rank_b = np.argsort(np.argsort(vec_b)).astype(np.float32)
    ra_centered = rank_a - rank_a.mean()
    rb_centered = rank_b - rank_b.mean()
    norm_ra = np.linalg.norm(ra_centered)
    norm_rb = np.linalg.norm(rb_centered)
    if norm_ra > 1e-8 and norm_rb > 1e-8:
        spearman = float(np.dot(ra_centered, rb_centered) / (norm_ra * norm_rb))
    else:
        spearman = float("nan")
    
    return {"pcc": pcc, "spearman": spearman, "cos": cos, "l1": l1, "l2": l2}


def align_to_common_genes(
    X: np.ndarray,
    gene_ids: np.ndarray,
    common_gene_ids: np.ndarray,
    gene_to_common: Dict[int, int],
) -> np.ndarray:
    """Align expression matrix to common gene order."""
    n_spots = X.shape[0]
    n_common = len(common_gene_ids)
    X_aligned = np.zeros((n_spots, n_common), dtype=np.float32)
    
    for col_idx, gid in enumerate(gene_ids):
        if col_idx in gene_to_common:
            common_idx = gene_to_common[col_idx]
            X_aligned[:, common_idx] = X[:, col_idx]
    
    return X_aligned


def save_full_expression_matrix_to_csv(
    X: np.ndarray,
    gene_ids: np.ndarray,
    obs_names: List[str],
    out_path: Path,
    vocab: Dict[str, int],
    perturbed_indices: Optional[Set[int]] = None,
) -> None:
    """Save full expression matrix to CSV."""
    if X.size == 0:
        print(f"[WARN] Empty expression matrix, skipping save to {out_path}")
        return
    
    # Reverse vocab: gene_id -> gene_name
    id_to_gene = {int(v): k for k, v in vocab.items()}
    
    # Get gene names
    gene_names = [id_to_gene.get(int(gid), f"gene_{gid}") for gid in gene_ids]
    
    # Create DataFrame
    df = pd.DataFrame(X, index=obs_names, columns=gene_names)
    df.index.name = "barcode"
    
    # Add is_perturbed column
    if perturbed_indices is not None:
        is_perturbed = [1 if i in perturbed_indices else 0 for i in range(len(obs_names))]
        df.insert(0, "is_perturbed", is_perturbed)
    
    df.to_csv(out_path)
    print(f"[OK] Saved full expression: {out_path} ({X.shape[0]} spots × {X.shape[1]} genes)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="SEDR baseline for niche perturbation evaluation")
    
    # Data paths
    ap.add_argument("--cache_pt", required=True, help="Cache directory for PT slice")
    ap.add_argument("--cache_sham", required=True, help="Cache directory for Sham slice")
    ap.add_argument("--pt_dataset_name", default="PT1-1")
    ap.add_argument("--sham_dataset_name", default="Sham1-1")
    ap.add_argument("--roi_manifest", required=True, help="JSON manifest with ROI barcodes")
    ap.add_argument("--deg_csv", required=True, help="DEG CSV with gene, avg_logFC columns")
    ap.add_argument("--vocab_file", default=None, help="Path to vocab.json")
    
    # Perturbation config
    ap.add_argument("--perturb_mode", choices=["patch", "random"], default="patch")
    ap.add_argument("--n_spots", type=int, default=20)
    ap.add_argument("--patch_size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weighting", choices=["gaussian", "uniform"], default="gaussian")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--p_adj_thresh", type=float, default=0.1)
    ap.add_argument("--min_abs_logfc", type=float, default=0.0)
    ap.add_argument("--logfc_strength", type=float, default=1.0)
    ap.add_argument("--logfc_clip", type=float, default=5.0)
    
    # Iteration settings
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--freeze_perturbed", action="store_true", default=True)
    
    # SEDR hyperparameters
    ap.add_argument("--sedr_epochs", type=int, default=200)
    ap.add_argument("--sedr_lr", type=float, default=0.01)
    ap.add_argument("--sedr_weight_decay", type=float, default=0.01)
    ap.add_argument("--sedr_graph_knn_k", type=int, default=6)
    ap.add_argument("--sedr_rec_w", type=float, default=10.0)
    ap.add_argument("--sedr_gcn_w", type=float, default=0.1)
    ap.add_argument("--sedr_self_w", type=float, default=1.0)
    ap.add_argument("--sedr_mode", choices=["clustering", "imputation"], default="imputation")
    
    # Device
    ap.add_argument("--device", default="cuda:0")
    
    # Output
    ap.add_argument("--out_dir", required=True, help="Output directory")
    
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine vocab file path
    if args.vocab_file:
        vocab_file = args.vocab_file
    else:
        vocab_file = str(_REPO_ROOT / "gene_embedding" / "vocab.json")
    
    # Load vocab
    print("[STEP] Loading vocab...")
    vocab = _load_vocab(vocab_file)
    
    # Load data
    print("[STEP] Loading PT data...")
    pt_adata, pt_X, pt_gene_ids, pt_valid_genes = load_adata_and_prepare(
        Path(args.cache_pt), args.pt_dataset_name, vocab
    )
    
    print("[STEP] Loading Sham data...")
    sham_adata, sham_X, sham_gene_ids, sham_valid_genes = load_adata_and_prepare(
        Path(args.cache_sham), args.sham_dataset_name, vocab
    )
    
    # Find common genes for evaluation
    common_gene_ids, pt_to_common, sham_to_common = find_common_genes_between_slices(
        pt_gene_ids, sham_gene_ids
    )
    print(f"[INFO] Common genes: {len(common_gene_ids)}")
    
    # Load ROI manifest
    manifest = _load_json(Path(args.roi_manifest))
    pt_key = "pt" if "pt" in manifest else "PT1_1"
    sham_key = "sham" if "sham" in manifest else "Sham1_1"
    pt_roi_barcodes = {k: manifest[pt_key][k]["barcodes"] for k in ["ICA", "PIA_P", "PIA_D"]}
    sham_roi_barcodes = {k: manifest[sham_key][k]["barcodes"] for k in ["ICA", "PIA_P", "PIA_D"]}
    
    pt_obs_names = list(pt_adata.obs_names)
    sham_obs_names = list(sham_adata.obs_names)
    
    pt_roi_indices = {k: _barcodes_to_indices(pt_obs_names, pt_roi_barcodes[k]) for k in ["ICA", "PIA_P", "PIA_D"]}
    sham_roi_indices = {k: _barcodes_to_indices(sham_obs_names, sham_roi_barcodes[k]) for k in ["ICA", "PIA_P", "PIA_D"]}
    
    # ===========================================================================
    # Train SEDR models
    # ===========================================================================
    print("\n" + "=" * 60)
    print("[SEDR] Training SEDR models...")
    print("=" * 60)
    
    # Train on PT (for PT reference expression)
    print("\n[SEDR] Training on PT slice...")
    pt_sedr = train_sedr(
        pt_X, pt_adata,
        epochs=args.sedr_epochs,
        lr=args.sedr_lr,
        weight_decay=args.sedr_weight_decay,
        graph_knn_k=args.sedr_graph_knn_k,
        rec_w=args.sedr_rec_w,
        gcn_w=args.sedr_gcn_w,
        self_w=args.sedr_self_w,
        device=args.device,
        mode=args.sedr_mode,
        seed=args.seed,
    )
    
    # Train on Sham (for ctrl0 reconstruction and pert0 baseline)
    print("\n[SEDR] Training on Sham slice...")
    sham_sedr = train_sedr(
        sham_X, sham_adata,
        epochs=args.sedr_epochs,
        lr=args.sedr_lr,
        weight_decay=args.sedr_weight_decay,
        graph_knn_k=args.sedr_graph_knn_k,
        rec_w=args.sedr_rec_w,
        gcn_w=args.sedr_gcn_w,
        self_w=args.sedr_self_w,
        device=args.device,
        mode=args.sedr_mode,
        seed=args.seed,
    )
    
    # ===========================================================================
    # Compute PT reference expression (reconstructed)
    # ===========================================================================
    print("\n[STEP] Computing PT reference expression...")
    pt_X_recon = sedr_reconstruct(pt_sedr, pt_X, pt_sedr.adj_norm)
    pt_X_common = align_to_common_genes(pt_X_recon, pt_gene_ids, common_gene_ids, pt_to_common)
    
    # ===========================================================================
    # Compute Sham control expression (normal recon, no mask)
    # ===========================================================================
    print("[STEP] Computing Sham control expression (no mask)...")
    sham_X_ctrl = sedr_reconstruct(sham_sedr, sham_X, sham_sedr.adj_norm)
    sham_X_ctrl_common = align_to_common_genes(sham_X_ctrl, sham_gene_ids, common_gene_ids, sham_to_common)
    
    # ===========================================================================
    # Select spots for perturbation (ICA region only)
    # ===========================================================================
    print("[STEP] Selecting spots for perturbation...")
    target_roi = "ICA"
    target_roi_indices = sham_roi_indices[target_roi]
    
    if args.perturb_mode == "patch":
        selected_indices = select_patch_spots(
            sham_adata, target_roi_indices, args.patch_size, args.seed, args.sedr_graph_knn_k
        )
    else:
        selected_indices = select_random_spots(target_roi_indices, args.n_spots, args.seed)
    
    print(f"[INFO] Selected {len(selected_indices)} spots for perturbation in {target_roi}")
    perturbed_set = set(selected_indices)
    
    # ===========================================================================
    # Compute weights and apply perturbation
    # ===========================================================================
    print("[STEP] Computing weights and applying perturbation...")
    weights, weight_meta = compute_spot_weights(sham_adata, selected_indices, args.weighting, args.sigma)
    
    sham_X_perturbed, perturb_meta = apply_deg_perturbation(
        sham_X.copy(), sham_gene_ids, selected_indices, weights,
        Path(args.deg_csv), vocab,
        args.p_adj_thresh, args.min_abs_logfc, args.logfc_strength, args.logfc_clip
    )
    
    # Write perturb manifest
    perturb_manifest = {
        "baseline_method": "SEDR",
        "perturb_target_roi": target_roi,
        "perturb_mode": args.perturb_mode,
        "n_spots_selected": len(selected_indices),
        "seed": args.seed,
        "selected_indices": selected_indices,
        "selected_barcodes": _indices_to_barcodes(sham_obs_names, selected_indices),
        "weights": weights.tolist(),
        **weight_meta,
        **perturb_meta,
        "steps": args.steps,
        "freeze_perturbed": args.freeze_perturbed,
        "sedr_epochs": args.sedr_epochs,
        "sedr_lr": args.sedr_lr,
        "sedr_graph_knn_k": args.sedr_graph_knn_k,
        "sedr_mode": args.sedr_mode,
    }
    
    with open(out_dir / "perturb_manifest.json", "w", encoding="utf-8") as f:
        json.dump(perturb_manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote perturb_manifest.json")
    
    # ===========================================================================
    # Train SEDR on perturbed Sham (for pert0 reconstruction)
    # ===========================================================================
    print("\n[SEDR] Training on Sham-perturbed slice (for pert0 reconstruction)...")
    sham_perturb_sedr = train_sedr(
        sham_X_perturbed, sham_adata,
        epochs=args.sedr_epochs,
        lr=args.sedr_lr,
        weight_decay=args.sedr_weight_decay,
        graph_knn_k=args.sedr_graph_knn_k,
        rec_w=args.sedr_rec_w,
        gcn_w=args.sedr_gcn_w,
        self_w=args.sedr_self_w,
        device=args.device,
        mode=args.sedr_mode,
        seed=args.seed,
    )
    
    # Reconstruct pert0 using sham_perturb_sedr (same scale as ctrl0/pt)
    print("[STEP] Computing pert0 expression (SEDR reconstructed)...")
    sham_X_pert0_recon = sedr_reconstruct(sham_perturb_sedr, sham_X_perturbed, sham_perturb_sedr.adj_norm)
    sham_X_pert0_common = align_to_common_genes(sham_X_pert0_recon, sham_gene_ids, common_gene_ids, sham_to_common)
    
    # ===========================================================================
    # SEDR ROI Mask Prediction (with frozen perturbed spots)
    # ===========================================================================
    print("\n[STEP] Running SEDR ROI mask prediction...")
    
    # Collect all ROI indices
    all_roi_indices = []
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        all_roi_indices.extend(sham_roi_indices[roi])
    all_roi_indices = list(set(all_roi_indices))
    
    # Frozen indices: the initially perturbed spots
    frozen_indices = set(selected_indices)
    
    # ROI spots to mask and predict (exclude frozen spots)
    mask_indices = [idx for idx in all_roi_indices if idx not in frozen_indices]
    
    print(f"[INFO] Total ROI spots: {len(all_roi_indices)}")
    print(f"[INFO] Frozen (perturbed) spots: {len(frozen_indices)}")
    print(f"[INFO] Spots to mask and predict: {len(mask_indices)}")
    
    # Save frozen values (perturbed values for selected spots)
    frozen_values = {idx: sham_X_perturbed[idx].copy() for idx in frozen_indices}
    
    step_expressions = {0: sham_X_pert0_common.copy()}
    step_raw_expressions = {0: sham_X_perturbed.copy()}
    
    # Start from the perturbed state
    X_current = sham_X_perturbed.copy()
    
    for t in range(1, args.steps + 1):
        print(f"[SEDR] Step {t}/{args.steps}: Mask {len(mask_indices)} spots...")
        
        # Only mask the non-frozen ROI spots
        X_recon = sedr_reconstruct_with_roi_mask(
            sham_sedr, X_current, sham_sedr.adj_norm,
            mask_indices=mask_indices
        )
        
        # Update state
        X_next = X_current.copy()
        
        # For frozen spots: keep their perturbed values
        for idx in frozen_indices:
            X_next[idx] = frozen_values[idx]
        
        # For non-frozen ROI spots: use reconstructed values
        for idx in mask_indices:
            X_next[idx] = X_recon[idx]
        
        # Align to common genes and save
        X_next_common = align_to_common_genes(X_next, sham_gene_ids, common_gene_ids, sham_to_common)
        step_expressions[t] = X_next_common.copy()
        step_raw_expressions[t] = X_next.copy()
        
        X_current = X_next
    
    # ===========================================================================
    # Save full expression for all steps
    # ===========================================================================
    print(f"\n[STEP] Saving full expression for all steps...")
    expr_out_dir = out_dir / "expression"
    expr_out_dir.mkdir(parents=True, exist_ok=True)
    
    for t in range(args.steps + 1):
        if t in step_raw_expressions:
            save_full_expression_matrix_to_csv(
                step_raw_expressions[t],
                sham_gene_ids,
                sham_obs_names,
                expr_out_dir / f"pert{t}_expression.csv",
                vocab,
                perturbed_set,
            )
    
    # ===========================================================================
    # Compute similarities
    # ===========================================================================
    print("[STEP] Computing similarities...")
    
    eval_rois = ["ICA", "PIA_P", "PIA_D"]
    results_rows: List[Dict] = []
    mean_vectors_by_step: Dict[int, Dict[str, Dict]] = {}
    
    # Step 0: ctrl0 and pert0
    mean_vectors_by_step[0] = {}
    for roi in eval_rois:
        pt_mean = compute_roi_mean_vector(pt_X_common, pt_roi_indices[roi])
        ctrl_mean = compute_roi_mean_vector(sham_X_ctrl_common, sham_roi_indices[roi])
        pert0_mean = compute_roi_mean_vector(step_expressions[0], sham_roi_indices[roi])
        
        mean_vectors_by_step[0][roi] = {
            "pt": pt_mean,
            "ctrl": ctrl_mean,
            "pert": pert0_mean,
            "gene_ids": common_gene_ids,
        }
        
        ctrl0_sim = compute_mean_vector_similarities(pt_mean, ctrl_mean)
        pert0_sim = compute_mean_vector_similarities(pt_mean, pert0_mean)
        
        for metric in ["pcc", "spearman", "cos", "l1", "l2"]:
            results_rows.append({
                "baseline_method": "SEDR",
                "perturb_target_roi": target_roi,
                "eval_roi": roi,
                "step": 0,
                "group": "ctrl0",
                "metric": metric,
                "value": ctrl0_sim[metric],
                "n_genes_common": len(common_gene_ids),
                "freeze_perturbed": args.freeze_perturbed,
            })
            results_rows.append({
                "baseline_method": "SEDR",
                "perturb_target_roi": target_roi,
                "eval_roi": roi,
                "step": 0,
                "group": "pert0",
                "metric": metric,
                "value": pert0_sim[metric],
                "n_genes_common": len(common_gene_ids),
                "freeze_perturbed": args.freeze_perturbed,
            })
    
    # Steps 1..T
    for t in range(1, args.steps + 1):
        mean_vectors_by_step[t] = {}
        
        for roi in eval_rois:
            pt_mean = compute_roi_mean_vector(pt_X_common, pt_roi_indices[roi])
            pert_mean = compute_roi_mean_vector(step_expressions[t], sham_roi_indices[roi])
            
            mean_vectors_by_step[t][roi] = {
                "pt": pt_mean,
                "pert": pert_mean,
                "gene_ids": common_gene_ids,
            }
            
            pert_sim = compute_mean_vector_similarities(pt_mean, pert_mean)
            
            for metric in ["pcc", "spearman", "cos", "l1", "l2"]:
                results_rows.append({
                    "baseline_method": "SEDR",
                    "perturb_target_roi": target_roi,
                    "eval_roi": roi,
                    "step": t,
                    "group": f"pert{t}",
                    "metric": metric,
                    "value": pert_sim[metric],
                    "n_genes_common": len(common_gene_ids),
                    "freeze_perturbed": args.freeze_perturbed,
                })
    
    # ===========================================================================
    # Write outputs
    # ===========================================================================
    print("[STEP] Writing outputs...")
    
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"rows": results_rows}, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Wrote summary.csv and summary.json")
    
    # Mean vectors per step
    for t, roi_data in mean_vectors_by_step.items():
        save_dict = {}
        for roi, data in roi_data.items():
            for key, val in data.items():
                save_dict[f"{roi}_{key}"] = val
        if save_dict:
            np.savez_compressed(out_dir / f"roi_mean_vectors_step{t}.npz", **save_dict)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMILARITY SUMMARY (SEDR Baseline)")
    print("=" * 60)
    
    for roi in eval_rois:
        print(f"\n[ROI: {roi}]")
        roi_rows = [r for r in results_rows if r["eval_roi"] == roi and r["metric"] == "pcc"]
        for r in sorted(roi_rows, key=lambda x: x["step"]):
            print(f"  step={r['step']} group={r['group']:6s} PCC={r['value']:.4f}")
    
    print("\n[DONE] All outputs written to:", out_dir)


if __name__ == "__main__":
    main()
