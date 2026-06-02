#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN and ridge reconstruction baselines.

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
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.sparse import issparse
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_knn_graph_gpu(
    adata: ad.AnnData, 
    k: int = 6, 
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[int, List[int]], torch.Tensor]:
    """
    Build k-NN graph from spatial coordinates using GPU acceleration.
    
    Returns:
        neighbors: dict mapping spot index to list of neighbor indices
        knn_indices: tensor [n_spots, k] of neighbor indices
    """
    spatial = adata.obsm['spatial']
    spatial_t = torch.tensor(spatial, dtype=torch.float32, device=device)
    
    n_spots = spatial_t.shape[0]
    
    # Compute pairwise distances using GPU
    # dist[i, j] = ||spatial[i] - spatial[j]||^2
    # Use broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d) -> sum -> (n, n)
    # More memory efficient: compute in batches if needed
    
    if n_spots <= 10000:
        # Direct computation for small datasets
        dist_sq = torch.cdist(spatial_t, spatial_t, p=2)
        # Set self-distance to inf
        dist_sq.fill_diagonal_(float('inf'))
        # Get k nearest neighbors
        _, knn_indices = torch.topk(dist_sq, k, dim=1, largest=False)
    else:
        # Batch computation for large datasets
        batch_size = 1000
        knn_indices = torch.zeros((n_spots, k), dtype=torch.long, device=device)
        
        for i in range(0, n_spots, batch_size):
            end_i = min(i + batch_size, n_spots)
            batch_spatial = spatial_t[i:end_i]  # [batch, d]
            
            # Compute distances from batch to all points
            dist_sq = torch.cdist(batch_spatial, spatial_t, p=2)  # [batch, n_spots]
            
            # Set self-distance to inf
            for j in range(end_i - i):
                dist_sq[j, i + j] = float('inf')
            
            # Get k nearest
            _, batch_knn = torch.topk(dist_sq, k, dim=1, largest=False)
            knn_indices[i:end_i] = batch_knn
    
    # Convert to dict format
    knn_indices_cpu = knn_indices.cpu().numpy()
    neighbors = {}
    for i in range(n_spots):
        neighbors[i] = [int(j) for j in knn_indices_cpu[i]]
    
    print(f"[INFO] Built {k}-NN graph on {device} for {n_spots} spots")
    
    return neighbors, knn_indices


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
# GPU-accelerated KNN / Ridge Reconstruction
# ==============================================================================

class GPUReconstructor:
    """GPU-accelerated reconstruction using KNN average."""
    
    def __init__(
        self,
        X: np.ndarray,
        knn_indices: torch.Tensor,
        mask_indices: List[int],
        device: torch.device,
    ):
        """
        Initialize GPU reconstructor for KNN average.
        
        Args:
            X: expression matrix [n_spots, n_genes]
            knn_indices: tensor [n_spots, k] of neighbor indices
            mask_indices: list of spot indices to reconstruct
            device: torch device
        """
        self.device = device
        self.n_spots, self.n_genes = X.shape
        self.k = knn_indices.shape[1]
        
        # Move data to GPU
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.X_original = self.X.clone()
        self.knn_indices = knn_indices.to(device)
        
        # Create mask tensors
        self.mask_indices = torch.tensor(mask_indices, dtype=torch.long, device=device)
        self.mask_set = set(mask_indices)
        self.is_masked = torch.zeros(self.n_spots, dtype=torch.bool, device=device)
        self.is_masked[self.mask_indices] = True
        
        # Track current step
        self.current_step = 0
        
        self.X[self.mask_indices] = 0.0
        print("[KNN] Initialized masked spots to zero")
    
    def knn_average_step(self) -> None:
        """
        Perform one step of KNN average reconstruction for masked spots.
        Updates self.X in-place.
        
        All steps: Use ALL neighbors including masked ones.
        In step 0, masked neighbors have value 0 (initialized in __init__), which
        is included in the average calculation for fair comparison with SpatialGT.
        """
        # Get current expression of all neighbors
        neighbor_expr = self.X[self.knn_indices]  # [n_spots, k, n_genes]
        
        # Use ALL neighbors including masked ones (which are 0 in step 0)
        # This is fair for comparison with SpatialGT
        # Compute simple average: sum / k
        avg_expr = neighbor_expr.mean(dim=1)  # [n_spots, n_genes]
        
        # Only update masked spots
        self.X[self.mask_indices] = avg_expr[self.mask_indices]
    
    def reconstruct_step(self) -> None:
        """Perform one reconstruction step."""
        self.knn_average_step()
        # Increment step counter after each step
        self.current_step += 1
    
    def apply_ema(self, X_prev: torch.Tensor, ema_alpha: float) -> None:
        """Apply EMA update for masked spots."""
        if ema_alpha > 0:
            self.X[self.mask_indices] = (
                ema_alpha * X_prev[self.mask_indices] + 
                (1 - ema_alpha) * self.X[self.mask_indices]
            )
    
    def get_masked_expression(self) -> np.ndarray:
        """Get current expression values for masked spots."""
        return self.X[self.mask_indices].cpu().numpy()
    
    def reset_non_masked(self) -> None:
        """Reset non-masked spots to original values."""
        non_masked = ~self.is_masked
        self.X[non_masked] = self.X_original[non_masked]


class GPURidgeReconstructor:
    """
    GPU-accelerated Ridge regression for reconstruction.
    
    For each gene, trains a Ridge regression model:
        y_g = X_neighbors_g @ w_g + b_g
    
    where X_neighbors_g is the expression of k neighbors at gene g,
    and y_g is the center spot's expression at gene g.
    
    The Ridge solution is:
        w = (X^T X + lambda I)^-1 X^T y
    
    Training is performed on ALL spots in the slice (not just non-masked).
    """
    
    def __init__(
        self,
        X: np.ndarray,
        knn_indices: torch.Tensor,
        mask_indices: List[int],
        device: torch.device,
        ridge_alpha: float = 1.0,
    ):
        """
        Initialize GPU Ridge reconstructor.
        
        Args:
            X: expression matrix [n_spots, n_genes]
            knn_indices: tensor [n_spots, k] of neighbor indices
            mask_indices: list of spot indices to reconstruct
            device: torch device
            ridge_alpha: Ridge regularization parameter (lambda)
        """
        self.device = device
        self.n_spots, self.n_genes = X.shape
        self.k = knn_indices.shape[1]
        self.ridge_alpha = ridge_alpha
        
        # Move data to GPU
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.X_original = self.X.clone()
        self.knn_indices = knn_indices.to(device)
        
        # Create mask tensors
        self.mask_indices = torch.tensor(mask_indices, dtype=torch.long, device=device)
        self.mask_set = set(mask_indices)
        self.is_masked = torch.zeros(self.n_spots, dtype=torch.bool, device=device)
        self.is_masked[self.mask_indices] = True
        
        # Track current step
        self.current_step = 0
        
        # Train Ridge model on ALL spots
        self._train_ridge()
        
        self.X[self.mask_indices] = 0.0
        print("[RIDGE] Initialized masked spots to zero")
    
    def _train_ridge(self) -> None:
        """
        Train Ridge regression weights using ALL spots in the slice.
        
        For each gene g:
            - Features: neighbor expressions at gene g [n_spots, k]
            - Target: center spot expression at gene g [n_spots]
            - Ridge solution: w_g = (X^T X + lambda I)^-1 X^T y_g
        
        Trained parameters:
            - self.weights: [n_genes, k] - weights for each gene
            - self.bias: [n_genes] - bias for each gene
        """
        n_train = self.n_spots
        
        print(f"[RIDGE] Training Ridge regression on ALL {n_train} spots...")
        
        # Get neighbor expressions for ALL spots (use original values)
        neighbor_expr = self.X_original[self.knn_indices]  # [n_spots, k, n_genes]
        
        # Target: center spot expressions for ALL spots
        target = self.X_original  # [n_spots, n_genes]
        
        # Transpose for batch processing: [n_genes, n_spots, k]
        # For each gene, X is [n_spots, k], y is [n_spots, 1]
        X_batch = neighbor_expr.permute(2, 0, 1)  # [n_genes, n_spots, k]
        Y_batch = target.T.unsqueeze(2)  # [n_genes, n_spots, 1]
        
        # Compute X^T X for all genes: [n_genes, k, k]
        XTX = torch.bmm(X_batch.transpose(1, 2), X_batch)  # [n_genes, k, k]
        
        # Add regularization: (X^T X + lambda I)
        reg = self.ridge_alpha * torch.eye(self.k, device=self.device).unsqueeze(0)  # [1, k, k]
        XTX_reg = XTX + reg  # [n_genes, k, k]
        
        # Compute X^T Y for all genes: [n_genes, k, 1]
        XTY = torch.bmm(X_batch.transpose(1, 2), Y_batch)  # [n_genes, k, 1]
        
        # Solve linear system: (X^T X + lambda I) w = X^T y
        # weights: [n_genes, k]
        try:
            self.weights = torch.linalg.solve(XTX_reg, XTY).squeeze(2)  # [n_genes, k]
        except RuntimeError:
            # Fallback: use pseudo-inverse if singular
            print("[RIDGE] Using pseudo-inverse due to singular matrix")
            self.weights = torch.zeros(self.n_genes, self.k, device=self.device)
            for g in range(self.n_genes):
                self.weights[g] = torch.linalg.lstsq(XTX_reg[g], XTY[g]).solution.squeeze()
        
        # Compute bias: bias_g = mean(y_g) - w_g^T @ mean(X_g)
        X_mean = X_batch.mean(dim=1)  # [n_genes, k]
        Y_mean = Y_batch.mean(dim=1).squeeze()  # [n_genes]
        self.bias = Y_mean - (self.weights * X_mean).sum(dim=1)  # [n_genes]
        
        # Compute training R^2 for logging
        # Y_pred_train: [n_genes, n_spots], bias needs unsqueeze to broadcast correctly
        Y_pred_train = torch.bmm(X_batch, self.weights.unsqueeze(2)).squeeze(2) + self.bias.unsqueeze(1)  # [n_genes, n_spots]
        Y_train = target.T  # [n_genes, n_spots]
        ss_res = ((Y_train - Y_pred_train) ** 2).sum(dim=1)
        ss_tot = ((Y_train - Y_train.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        r2_per_gene = 1 - ss_res / (ss_tot + 1e-8)
        mean_r2 = r2_per_gene.mean().item()
        
        print(f"[RIDGE] Training complete. Alpha={self.ridge_alpha}, Mean R^2={mean_r2:.4f}")
    
    def predict(self, spot_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict expression for given spots using trained Ridge model.
        
        Args:
            spot_indices: indices of spots to predict [n]
        
        Returns:
            Predicted expression [n, n_genes]
        """
        # Get neighbor indices for these spots
        knn = self.knn_indices[spot_indices]  # [n, k]
        
        # Get current neighbor expressions
        neighbor_expr = self.X[knn]  # [n, k, n_genes]
        
        # Predict: for each gene g, pred_g = neighbor_expr[:, :, g] @ weights[g] + bias[g]
        # Using einsum: pred[i, g] = sum_j neighbor_expr[i, j, g] * weights[g, j] + bias[g]
        pred = torch.einsum('nkg,gk->ng', neighbor_expr, self.weights) + self.bias
        
        return pred
    
    def reconstruct_step(self) -> None:
        """Perform one step of Ridge reconstruction for masked spots."""
        pred = self.predict(self.mask_indices)
        self.X[self.mask_indices] = pred
        self.current_step += 1
    
    def apply_ema(self, X_prev: torch.Tensor, ema_alpha: float) -> None:
        """Apply EMA update for masked spots."""
        if ema_alpha > 0:
            self.X[self.mask_indices] = (
                ema_alpha * X_prev[self.mask_indices] + 
                (1 - ema_alpha) * self.X[self.mask_indices]
            )
    
    def get_masked_expression(self) -> np.ndarray:
        """Get current expression values for masked spots."""
        return self.X[self.mask_indices].cpu().numpy()
    
    def reset_non_masked(self) -> None:
        """Reset non-masked spots to original values."""
        non_masked = ~self.is_masked
        self.X[non_masked] = self.X_original[non_masked]


def run_iterative_reconstruction_gpu(
    X_original: np.ndarray,
    mask_indices: List[int],
    knn_indices: torch.Tensor,
    steps: int,
    device: torch.device,
    mode: str = "knn_avg",
    ridge_alpha: float = 1.0,
    ema_alpha: float = 0.0,
) -> Dict[int, np.ndarray]:
    """
    Run iterative reconstruction on GPU.
    
    Args:
        X_original: original expression matrix [n_spots, n_genes]
        mask_indices: indices of spots to reconstruct
        knn_indices: tensor [n_spots, k] of neighbor indices
        steps: number of iterations
        device: torch device
        mode: "knn_avg" or "ridge"
        ridge_alpha: Ridge regularization parameter (only for ridge mode)
        ema_alpha: EMA coefficient (0 = direct replacement)
    
    Returns:
        step_reconstructions: {step: X_recon for masked spots}
    """
    # Create appropriate reconstructor based on mode
    if mode == "ridge":
        reconstructor = GPURidgeReconstructor(
            X=X_original,
            knn_indices=knn_indices,
            mask_indices=mask_indices,
            device=device,
            ridge_alpha=ridge_alpha,
        )
    else:
        reconstructor = GPUReconstructor(
            X=X_original,
            knn_indices=knn_indices,
            mask_indices=mask_indices,
            device=device,
        )
    
    # Step 0: initial reconstruction
    reconstructor.reconstruct_step()
    reconstructor.reset_non_masked()
    
    step_recons = {}
    step_recons[0] = reconstructor.get_masked_expression()
    
    for t in range(1, steps + 1):
        X_prev = reconstructor.X.clone() if ema_alpha > 0 else None
        
        reconstructor.reconstruct_step()
        
        if ema_alpha > 0:
            reconstructor.apply_ema(X_prev, ema_alpha)
        
        reconstructor.reset_non_masked()
        step_recons[t] = reconstructor.get_masked_expression()
    
    return step_recons


# ==============================================================================
# Reconstruction metrics
# ==============================================================================

def compute_spotwise_metrics_gpu(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Compute spot-wise PCC and RMSE for reconstructed masked spots."""
    X_true_t = torch.tensor(X_true, dtype=torch.float32, device=device)
    X_pred_t = torch.tensor(X_pred, dtype=torch.float32, device=device)

    true_centered = X_true_t - X_true_t.mean(dim=1, keepdim=True)
    pred_centered = X_pred_t - X_pred_t.mean(dim=1, keepdim=True)
    true_std = true_centered.norm(dim=1)
    pred_std = pred_centered.norm(dim=1)
    valid_std = (true_std > 1e-8) & (pred_std > 1e-8)

    pcc = torch.zeros(X_true_t.shape[0], device=device)
    pcc[valid_std] = (true_centered[valid_std] * pred_centered[valid_std]).sum(dim=1) / (
        true_std[valid_std] * pred_std[valid_std]
    )

    rmse = ((X_true_t - X_pred_t) ** 2).mean(dim=1).sqrt()
    return {
        "pcc": float(pcc.mean().cpu().item()),
        "rmse": float(rmse.mean().cpu().item()),
        "pcc_std": float(pcc.std(unbiased=False).cpu().item()),
        "rmse_std": float(rmse.std(unbiased=False).cpu().item()),
    }

# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="KNN/Ridge reconstruction evaluation (GPU accelerated)")
    
    # Data paths
    ap.add_argument("--cache_dir", type=str, required=True, help="Path to preprocessed cache directory")
    ap.add_argument("--dataset_name", type=str, required=True, help="Dataset name (subdirectory in cache_dir)")
    
    # Masking parameters
    ap.add_argument("--mask_mode", type=str, default="patch", choices=["patch", "random"],
                    help="Masking mode: patch (BFS from center) or random")
    ap.add_argument("--n_spots", type=int, default=20, help="Number of spots to mask")
    ap.add_argument("--knn_k", type=int, default=6, help="Number of neighbors for KNN graph")
    
    # Inference parameters
    ap.add_argument("--inference_mode", type=str, default="knn_avg", choices=["knn_avg", "ridge"],
                    help="Inference mode: knn_avg or ridge")
    ap.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge regularization parameter")
    ap.add_argument("--steps", type=int, default=10, help="Number of iteration steps")
    ap.add_argument("--ema_alpha", type=float, default=0.0, help="EMA coefficient (0 = direct replacement)")
    # GPU settings
    ap.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    
    # Output
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--save_expressions", action="store_true", default=False,
                    help="Save per-step reconstructed expressions NPZ "
                         "(disabled by default to save disk; enable for representative seed)")
    
    args = ap.parse_args()
    
    # Fix seed
    fix_seed(args.seed)
    
    # Setup device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"[INFO] Using GPU: cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using CPU")
    
    # Paths
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata, X = load_adata_and_expression(cache_dir, args.dataset_name)
    n_spots, n_genes = X.shape
    
    # Build neighbor graph on GPU
    print(f"[INFO] Building {args.knn_k}-NN graph on {device}...")
    neighbors, knn_indices = build_knn_graph_gpu(adata, k=args.knn_k, device=device)
    
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
    
    # Run iterative reconstruction on GPU
    print(f"[INFO] Running iterative reconstruction for {args.steps} steps on {device}...")
    print("[INFO] Mask initialization: zero")
    step_recons = run_iterative_reconstruction_gpu(
        X_original=X,
        mask_indices=mask_indices,
        knn_indices=knn_indices,
        steps=args.steps,
        device=device,
        mode=args.inference_mode,
        ridge_alpha=args.ridge_alpha,
        ema_alpha=args.ema_alpha,
    )
    
    # Compute metrics for each step
    results_rows = []
    
    for step in range(args.steps + 1):
        X_pred = step_recons[step]
        
        metrics = compute_spotwise_metrics_gpu(X_true, X_pred, device)
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
        "eval_script": "knn_reconstruction_eval.py",
        "baseline_method": f"KNN_{args.inference_mode}",
        "dataset": args.dataset_name,
        "mask_mode": args.mask_mode,
        "n_spots": len(mask_indices),
        "n_spots_requested": args.n_spots,
        "knn_k": args.knn_k,
        "inference_mode": args.inference_mode,
        "ridge_alpha": args.ridge_alpha,
        "steps": args.steps,
        "ema_alpha": args.ema_alpha,
        "mask_indices": mask_indices,
        "n_genes": n_genes,
        "device": str(device),
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
