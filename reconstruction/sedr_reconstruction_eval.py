#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEDR reconstruction baseline.

This script evaluates patch-mask or random-mask reconstruction with zero-initialized
masked spots. It reports PCC and RMSE, matching the manuscript metrics.
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
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
import anndata as ad

# Ensure project modules are importable when running as a script.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRETRAIN_DIR = _PROJECT_ROOT / "pretrain"
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reconstruction_patch import select_seeded_patch_center_and_mask

# Import SEDR
_SEDR_ROOT = _PROJECT_ROOT / "baseline" / "SEDR"
if str(_SEDR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEDR_ROOT))

from SEDR.SEDR_model import Sedr
from SEDR.graph_func import graph_construction


# ==============================================================================
# Utility functions
# ==============================================================================

def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# ==============================================================================
# Data loading
# ==============================================================================

def load_adata_and_expression(cache_dir: Path, dataset_name: str) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Load processed.h5ad and extract log1p-normalized expression matrix.
    
    Uses adata.layers['X_log1p'] for consistency with SpatialGT evaluation.
    Falls back to adata.X if the layer is not available.
    
    Returns:
        adata: AnnData object
        X: expression matrix [n_spots, n_genes] as numpy array (log1p space)
    """
    h5ad_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(h5ad_path))
    
    if "X_log1p" in adata.layers:
        X = adata.layers["X_log1p"]
        data_source = "X_log1p"
    else:
        X = adata.X
        data_source = "adata.X (fallback)"
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    
    print(f"[DATA] Loaded {dataset_name}: {adata.n_obs} spots, {adata.n_vars} genes, source={data_source}")
    
    return adata, X


# ==============================================================================
# Spot selection: patch (BFS from center) and random
# ==============================================================================

def get_center_spot(adata: ad.AnnData) -> int:
    """Find the spot closest to the spatial center of the slice."""
    spatial = adata.obsm['spatial']
    center = spatial.mean(axis=0)
    dists = np.sqrt(((spatial - center) ** 2).sum(axis=1))
    return int(np.argmin(dists))


def build_knn_graph_dict(adata: ad.AnnData, k: int = 6) -> Dict[int, List[int]]:
    """Build k-NN graph from spatial coordinates."""
    spatial = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(spatial)
    _, indices = nbrs.kneighbors(spatial)
    
    neighbors = {}
    for i in range(len(adata)):
        neighbors[i] = [int(j) for j in indices[i] if j != i][:k]
    return neighbors


def select_patch_spots(
    neighbors: Dict[int, List[int]],
    seed_spot: int,
    patch_size: int,
) -> List[int]:
    """
    Select a contiguous patch of spots via BFS from a seed spot.
    """
    visited = {seed_spot}
    queue = deque([seed_spot])
    selected = [seed_spot]
    
    while queue and len(selected) < patch_size:
        curr = queue.popleft()
        for nb in neighbors.get(curr, []):
            if nb not in visited:
                visited.add(nb)
                selected.append(nb)
                queue.append(nb)
                if len(selected) >= patch_size:
                    break
    
    return selected[:patch_size]


def select_random_spots(
    n_total_spots: int,
    n_spots: int,
    seed: int,
) -> List[int]:
    """Randomly sample n_spots from all spots."""
    rng = random.Random(seed)
    all_spots = list(range(n_total_spots))
    if len(all_spots) <= n_spots:
        return all_spots
    return rng.sample(all_spots, n_spots)


# ==============================================================================
# SEDR Training and Reconstruction
# ==============================================================================

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
    """
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
    Run SEDR reconstruction on expression matrix (no masking).
    """
    sedr.model.eval()
    sedr.model._mask_rate = 0.0
    
    X_tensor = torch.FloatTensor(X).to(sedr.device)
    
    with torch.no_grad():
        z, mu, logvar, de_feat, q, feat_x, gnn_z, loss = sedr.model(X_tensor, adj_norm)
    
    recon = de_feat.detach().cpu().numpy()
    recon = np.clip(recon, 0, None)
    
    return recon.astype(np.float32)


def sedr_reconstruct_with_mask(
    sedr: Sedr,
    X: np.ndarray,
    adj_norm: torch.Tensor,
    mask_indices: List[int],
) -> np.ndarray:
    """
    Run SEDR reconstruction with specific spots masked.
    
    Masks the specified spots and lets SEDR reconstruct their expression
    based on the surrounding unmasked spots.
    """
    sedr.model.eval()
    sedr.model._mask_rate = 0.0
    
    X_tensor = torch.FloatTensor(X.copy()).to(sedr.device)
    
    # Apply mask token to specified spots
    mask_indices_tensor = torch.LongTensor(mask_indices).to(sedr.device)
    X_tensor[mask_indices_tensor] = X_tensor[mask_indices_tensor] + sedr.model.enc_mask_token
    
    with torch.no_grad():
        # Encoder
        feat_x = sedr.model.encoder(X_tensor)
        
        # GCN layers
        hidden1 = sedr.model.gc1(feat_x, adj_norm)
        mu = sedr.model.gc2(hidden1, adj_norm)
        
        # Reparameterize (in eval mode, just return mu)
        gnn_z = mu
        
        # Concatenate features
        z = torch.cat((feat_x, gnn_z), 1)
        
        # Decoder
        if hasattr(sedr.model, 'decoder'):
            try:
                de_feat = sedr.model.decoder(z, adj_norm)
            except TypeError:
                de_feat = sedr.model.decoder(z)
        else:
            de_feat = z
    
    recon = de_feat.detach().cpu().numpy()
    recon = np.clip(recon, 0, None)
    
    return recon.astype(np.float32)


# ==============================================================================
# Iterative reconstruction with frozen non-masked spots
# ==============================================================================

def run_iterative_reconstruction(
    sedr: Sedr,
    X_original: np.ndarray,
    mask_indices: List[int],
    steps: int,
    ema_alpha: float = 0.0,
) -> Dict[int, np.ndarray]:
    """
    Run iterative reconstruction, freezing non-masked spots.
    
    Args:
        sedr: trained SEDR model
        X_original: original expression matrix
        mask_indices: indices of spots to reconstruct
        steps: number of iterations
        ema_alpha: EMA coefficient (0 = direct replacement)
    
    Returns:
        step_reconstructions: {step: X_recon for masked spots}
    """
    n_spots, n_genes = X_original.shape
    mask_set = set(mask_indices)
    non_mask_indices = sorted(set(range(n_spots)) - mask_set)
    
    # Get adjacency matrix
    adj_norm = sedr.adj_norm
    
    # Initialize: zero out masked spots
    X_current = X_original.copy()
    for idx in mask_indices:
        X_current[idx] = 0.0
    
    step_recons = {}
    
    # Step 0: initial reconstruction with masked spots
    X_recon = sedr_reconstruct_with_mask(sedr, X_current, adj_norm, mask_indices)
    for idx in mask_indices:
        X_current[idx] = X_recon[idx]
    # Restore non-masked spots
    for idx in non_mask_indices:
        X_current[idx] = X_original[idx]
    
    step_recons[0] = X_current[mask_indices].copy()
    
    for t in range(1, steps + 1):
        # Reconstruct with current state
        X_recon = sedr_reconstruct_with_mask(sedr, X_current, adj_norm, mask_indices)
        
        # Apply EMA for masked spots
        X_next = X_current.copy()
        for idx in mask_indices:
            if ema_alpha > 0:
                X_next[idx] = ema_alpha * X_current[idx] + (1.0 - ema_alpha) * X_recon[idx]
            else:
                X_next[idx] = X_recon[idx]
        
        # Freeze non-masked spots
        for idx in non_mask_indices:
            X_next[idx] = X_original[idx]
        
        X_current = X_next
        step_recons[t] = X_current[mask_indices].copy()
    
    return step_recons


# ==============================================================================
# Reconstruction metrics
# ==============================================================================

def compute_spotwise_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute spot-wise PCC and RMSE for reconstructed masked spots."""
    pcc_list = []
    rmse_list = []

    for i in range(X_true.shape[0]):
        true_vec = X_true[i]
        pred_vec = X_pred[i]
        if np.std(true_vec) > 1e-8 and np.std(pred_vec) > 1e-8:
            pcc, _ = pearsonr(true_vec, pred_vec)
        else:
            pcc = 0.0
        rmse = float(np.sqrt(np.mean((true_vec - pred_vec) ** 2)))
        pcc_list.append(float(pcc))
        rmse_list.append(rmse)

    return {
        "pcc": float(np.nanmean(pcc_list)) if pcc_list else float("nan"),
        "rmse": float(np.nanmean(rmse_list)) if rmse_list else float("nan"),
        "pcc_std": float(np.nanstd(pcc_list)) if pcc_list else float("nan"),
        "rmse_std": float(np.nanstd(rmse_list)) if rmse_list else float("nan"),
    }

# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="SEDR reconstruction evaluation")
    
    # Data paths
    ap.add_argument("--cache_dir", type=str, required=True, help="Path to preprocessed cache directory")
    ap.add_argument("--dataset_name", type=str, required=True, help="Dataset name (subdirectory in cache_dir)")
    
    # Masking parameters
    ap.add_argument("--mask_mode", type=str, default="patch", choices=["patch", "random"],
                    help="Masking mode: patch (BFS from center) or random")
    ap.add_argument("--n_spots", type=int, default=20, help="Number of spots to mask")
    
    # SEDR parameters
    ap.add_argument("--sedr_epochs", type=int, default=200, help="SEDR training epochs")
    ap.add_argument("--sedr_lr", type=float, default=0.01, help="SEDR learning rate")
    ap.add_argument("--sedr_weight_decay", type=float, default=0.01, help="SEDR weight decay")
    ap.add_argument("--sedr_knn_k", type=int, default=6, help="SEDR graph KNN k")
    ap.add_argument("--sedr_rec_w", type=float, default=10.0, help="SEDR reconstruction weight")
    ap.add_argument("--sedr_gcn_w", type=float, default=0.1, help="SEDR GCN weight")
    ap.add_argument("--sedr_self_w", type=float, default=1.0, help="SEDR self-supervised weight")
    
    # Inference parameters
    ap.add_argument("--steps", type=int, default=10, help="Number of iteration steps")
    ap.add_argument("--ema_alpha", type=float, default=0.0, help="EMA coefficient (0 = direct replacement)")
    
    # Output
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    ap.add_argument("--save_expressions", action="store_true", default=False,
                    help="Save per-step reconstructed expressions NPZ "
                         "(disabled by default to save disk; enable for representative seed)")
    
    args = ap.parse_args()
    
    # Fix seed
    fix_seed(args.seed)
    
    # Paths
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Load data
    adata, X = load_adata_and_expression(cache_dir, args.dataset_name)
    n_spots, n_genes = X.shape
    
    # Build neighbor graph for spot selection
    print(f"[INFO] Building {args.sedr_knn_k}-NN graph...")
    neighbors = build_knn_graph_dict(adata, k=args.sedr_knn_k)
    
    # Select spots to mask
    patch_extra_meta: Dict[str, Any] = {}
    if args.mask_mode == "patch":
        allowed = set(range(n_spots))
        center_spot, mask_indices, patch_extra_meta = select_seeded_patch_center_and_mask(
            neighbors, args.n_spots, args.seed, allowed_vertices=allowed
        )
        w = patch_extra_meta.get("warning")
        if w:
            print(f"[WARN] Patch selection: {w}")
        print(
            f"[INFO] Patch mode: center={center_spot}, "
            f"masked {len(mask_indices)} spots (requested {args.n_spots})"
        )
    else:
        mask_indices = select_random_spots(n_spots, args.n_spots, args.seed)
        print(f"[INFO] Random mode: selected {len(mask_indices)} spots")
    
    # Ground truth for masked spots
    X_true = X[mask_indices].copy()
    
    # Train SEDR on full data
    print(f"\n[SEDR] Training on full slice...")
    sedr = train_sedr(
        X=X,
        adata=adata,
        epochs=args.sedr_epochs,
        lr=args.sedr_lr,
        weight_decay=args.sedr_weight_decay,
        graph_knn_k=args.sedr_knn_k,
        rec_w=args.sedr_rec_w,
        gcn_w=args.sedr_gcn_w,
        self_w=args.sedr_self_w,
        device=device,
        mode="imputation",
        seed=args.seed,
    )
    
    # Run iterative reconstruction
    print(f"\n[INFO] Running iterative reconstruction for {args.steps} steps...")
    step_recons = run_iterative_reconstruction(
        sedr=sedr,
        X_original=X,
        mask_indices=mask_indices,
        steps=args.steps,
        ema_alpha=args.ema_alpha,
    )
    
    # Compute metrics for each step
    results_rows = []
    
    for step in range(args.steps + 1):
        X_pred = step_recons[step]
        
        metrics = compute_spotwise_metrics(X_true, X_pred)
        for metric in ["pcc", "rmse"]:
            results_rows.append({
                "step": step,
                "metric": metric,
                "value": metrics[metric],
                "std": metrics.get(f"{metric}_std", float("nan")),
            })
        print(f"  Step {step}: PCC={metrics['pcc']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(out_dir / "summary.csv", index=False)
    
    # Save manifest
    manifest = {
        "task": "reconstruction",
        "eval_script": "sedr_reconstruction_eval.py",
        "baseline_method": "SEDR",
        "dataset": args.dataset_name,
        "mask_mode": args.mask_mode,
        "n_spots": len(mask_indices),
        "n_spots_requested": args.n_spots,
        "sedr_epochs": args.sedr_epochs,
        "sedr_lr": args.sedr_lr,
        "sedr_weight_decay": args.sedr_weight_decay,
        "sedr_knn_k": args.sedr_knn_k,
        "sedr_rec_w": args.sedr_rec_w,
        "sedr_gcn_w": args.sedr_gcn_w,
        "sedr_self_w": args.sedr_self_w,
        "steps": args.steps,
        "ema_alpha": args.ema_alpha,
        "device": device,
        "mask_indices": mask_indices,
        "n_genes": n_genes,
    }
    if patch_extra_meta:
        manifest["patch_selection"] = patch_extra_meta

    with open(out_dir / "reconstruction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save summary JSON
    summary_dict = {
        "final_step": args.steps,
        "final_metrics": {
            m: metrics[m] for m in ["pcc", "rmse"]
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    
    if args.save_expressions:
        expr_save = {
            "ground_truth": X_true,
            "mask_indices": np.array(mask_indices, dtype=np.int64),
        }
        for step, X_pred in step_recons.items():
            expr_save[f"step_{step}"] = X_pred
        np.savez(out_dir / "reconstruction_expressions.npz", **expr_save)

        gene_names = list(adata.var_names)
        with open(out_dir / "gene_names.json", "w") as f:
            json.dump(gene_names, f)

        print(f"[INFO] Saved per-step expressions ({len(step_recons)} steps, "
              f"{X_true.shape[0]} spots x {X_true.shape[1]} genes) to {out_dir}")
    else:
        print("[INFO] Skipped saving per-step expressions (use --save_expressions to enable)")
    
    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()

