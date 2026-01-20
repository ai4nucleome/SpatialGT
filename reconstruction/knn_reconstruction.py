#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN/Ridge Baseline for Reconstruction Evaluation

This script evaluates reconstruction performance by masking a subset of spots
and using KNN average or Ridge regression to reconstruct their expression values.

Default settings:
- 10 iteration steps
- Seed: 42
- Mask init mode: "zero"
- Inference mode: "knn_avg"
- Adjustable number of spots to mask via --n_spots

Supports:
- Two masking modes: patch (BFS-connected spots from center) and random
- Two inference modes: knn_avg (KNN average) and ridge (Ridge regression)
- GPU acceleration for graph building and metrics computation
- Spot-wise and ROI-wise similarity metrics

Usage:
    python knn_reconstruction.py \\
        --cache_dir <cache_dir> \\
        --dataset_name <dataset_name> \\
        --out_dir <output_dir>
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
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
import anndata as ad

# Ensure repo root importability
import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
    Load processed.h5ad and extract expression matrix.
    
    Returns:
        adata: AnnData object
        X: expression matrix [n_spots, n_genes] as numpy array
    """
    h5ad_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(h5ad_path))
    
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    
    print(f"[DATA] Loaded {dataset_name}: {adata.n_obs} spots, {adata.n_vars} genes")
    
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
        mask_init: str = "zero",
    ):
        """
        Initialize GPU reconstructor for KNN average.
        
        Args:
            X: expression matrix [n_spots, n_genes]
            knn_indices: tensor [n_spots, k] of neighbor indices
            mask_indices: list of spot indices to reconstruct
            device: torch device
            mask_init: initialization for masked spots in step 0
                       "zero" - initialize to 0
                       "mean" - initialize to global mean expression
        """
        self.device = device
        self.n_spots, self.n_genes = X.shape
        self.k = knn_indices.shape[1]
        self.mask_init = mask_init
        
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
        
        # Initialize masked spots based on mask_init option
        if mask_init == "zero":
            # Initialize to ZERO (SpatialGT default behavior)
            self.X[self.mask_indices] = 0.0
            print(f"[KNN] Initialized masked spots to ZERO")
        else:
            # Initialize to global mean (alternative)
            global_mean = self.X_original[~self.is_masked].mean(dim=0)  # [n_genes]
            self.X[self.mask_indices] = global_mean
            print(f"[KNN] Initialized masked spots to GLOBAL MEAN")
    
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
        w = (X^T X + λI)^{-1} X^T y
    
    Training is performed on ALL spots in the slice (not just non-masked).
    """
    
    def __init__(
        self,
        X: np.ndarray,
        knn_indices: torch.Tensor,
        mask_indices: List[int],
        device: torch.device,
        ridge_alpha: float = 1.0,
        mask_init: str = "zero",
    ):
        """
        Initialize GPU Ridge reconstructor.
        
        Args:
            X: expression matrix [n_spots, n_genes]
            knn_indices: tensor [n_spots, k] of neighbor indices
            mask_indices: list of spot indices to reconstruct
            device: torch device
            ridge_alpha: Ridge regularization parameter (lambda)
            mask_init: initialization for masked spots in step 0
                       "zero" - initialize to 0
                       "mean" - initialize to global mean expression
        """
        self.device = device
        self.n_spots, self.n_genes = X.shape
        self.k = knn_indices.shape[1]
        self.ridge_alpha = ridge_alpha
        self.mask_init = mask_init
        
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
        
        # Initialize masked spots based on mask_init option
        if mask_init == "zero":
            # Initialize to ZERO (SpatialGT default behavior)
            self.X[self.mask_indices] = 0.0
            print(f"[RIDGE] Initialized masked spots to ZERO")
        else:
            # Initialize to global mean (alternative)
            global_mean = self.X_original[~self.is_masked].mean(dim=0)  # [n_genes]
            self.X[self.mask_indices] = global_mean
            print(f"[RIDGE] Initialized masked spots to GLOBAL MEAN")
    
    def _train_ridge(self) -> None:
        """
        Train Ridge regression weights using ALL spots in the slice.
        
        For each gene g:
            - Features: neighbor expressions at gene g [n_spots, k]
            - Target: center spot expression at gene g [n_spots]
            - Ridge solution: w_g = (X^T X + λI)^{-1} X^T y_g
        
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
        
        # Add regularization: (X^T X + λI)
        reg = self.ridge_alpha * torch.eye(self.k, device=self.device).unsqueeze(0)  # [1, k, k]
        XTX_reg = XTX + reg  # [n_genes, k, k]
        
        # Compute X^T Y for all genes: [n_genes, k, 1]
        XTY = torch.bmm(X_batch.transpose(1, 2), Y_batch)  # [n_genes, k, 1]
        
        # Solve linear system: (X^T X + λI) w = X^T y
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
    mask_init: str = "zero",
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
        mask_init: initialization for masked spots in step 0
                   "zero" - initialize to 0 (SpatialGT default)
                   "mean" - initialize to global mean expression
    
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
            mask_init=mask_init,
        )
    else:
        reconstructor = GPUReconstructor(
            X=X_original,
            knn_indices=knn_indices,
            mask_indices=mask_indices,
            device=device,
            mask_init=mask_init,
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
# GPU-accelerated similarity metrics
# ==============================================================================

def compute_spotwise_metrics_gpu(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    device: torch.device,
    n_projections: int = 100,
) -> Dict[str, float]:
    """
    Compute spot-wise similarity metrics using GPU acceleration.
    
    Args:
        X_true: ground truth expression [n_masked_spots, n_genes]
        X_pred: predicted expression [n_masked_spots, n_genes]
        device: torch device
        n_projections: number of projections for SWD
    
    Returns:
        Dictionary with aggregated metrics (mean over spots)
    
    Metrics include:
    - PCC, Spearman, Cosine: correlation/similarity metrics
    - L1/MAE: Mean Absolute Error
    - L2/RMSE: Root Mean Squared Error
    - MSE: Mean Squared Error
    - SWD: Sliced Wasserstein Distance
    """
    X_true_t = torch.tensor(X_true, dtype=torch.float32, device=device)
    X_pred_t = torch.tensor(X_pred, dtype=torch.float32, device=device)
    
    n_spots, n_genes = X_true_t.shape
    
    # Vectorized computation of metrics
    # PCC: pearson correlation per spot
    true_centered = X_true_t - X_true_t.mean(dim=1, keepdim=True)
    pred_centered = X_pred_t - X_pred_t.mean(dim=1, keepdim=True)
    
    true_std = true_centered.norm(dim=1)
    pred_std = pred_centered.norm(dim=1)
    
    valid_std = (true_std > 1e-8) & (pred_std > 1e-8)
    pcc = torch.zeros(n_spots, device=device)
    pcc[valid_std] = (true_centered[valid_std] * pred_centered[valid_std]).sum(dim=1) / (
        true_std[valid_std] * pred_std[valid_std]
    )
    
    # Cosine similarity
    true_norm = X_true_t.norm(dim=1)
    pred_norm = X_pred_t.norm(dim=1)
    valid_norm = (true_norm > 1e-8) & (pred_norm > 1e-8)
    cos = torch.zeros(n_spots, device=device)
    cos[valid_norm] = (X_true_t[valid_norm] * X_pred_t[valid_norm]).sum(dim=1) / (
        true_norm[valid_norm] * pred_norm[valid_norm]
    )
    
    # MSE, RMSE (L2), MAE (L1)
    diff = X_true_t - X_pred_t
    mse = (diff ** 2).mean(dim=1)
    l1 = diff.abs().mean(dim=1)
    l2 = mse.sqrt()
    
    # SWD using GPU
    swd = sliced_wasserstein_distance_gpu(X_true_t, X_pred_t, n_projections)
    
    # Spearman correlation (need to compute ranks, keep on GPU)
    spearman_list = []
    X_true_np = X_true_t.cpu().numpy()
    X_pred_np = X_pred_t.cpu().numpy()
    for i in range(n_spots):
        if np.std(X_true_np[i]) > 1e-8 and np.std(X_pred_np[i]) > 1e-8:
            sp, _ = spearmanr(X_true_np[i], X_pred_np[i])
        else:
            sp = 0.0
        spearman_list.append(sp)
    
    return {
        "pcc": float(pcc.mean().cpu().item()),
        "spearman": float(np.nanmean(spearman_list)),
        "cos": float(cos.mean().cpu().item()),
        "l1": float(l1.mean().cpu().item()),
        "l2": float(l2.mean().cpu().item()),
        "mse": float(mse.mean().cpu().item()),
        "mae": float(l1.mean().cpu().item()),  # alias for l1
        "rmse": float(l2.mean().cpu().item()),  # alias for l2
        "swd": float(swd),
        "pcc_std": float(pcc.std().cpu().item()),
        "spearman_std": float(np.nanstd(spearman_list)),
        "cos_std": float(cos.std().cpu().item()),
        "l1_std": float(l1.std().cpu().item()),
        "l2_std": float(l2.std().cpu().item()),
        "mse_std": float(mse.std().cpu().item()),
        "mae_std": float(l1.std().cpu().item()),
        "rmse_std": float(l2.std().cpu().item()),
    }


def compute_roi_metrics_gpu(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute ROI-wise (region of interest) similarity metrics using GPU.
    
    First averages expression across all spots in the ROI, then computes
    similarity metrics on the aggregated expression vectors.
    
    Args:
        X_true: ground truth expression [n_masked_spots, n_genes]
        X_pred: predicted expression [n_masked_spots, n_genes]
        device: torch device
    
    Returns:
        Dictionary with ROI-level metrics
    
    Metrics include:
    - PCC, Spearman, Cosine: correlation/similarity metrics
    - L1/MAE: Mean Absolute Error
    - L2/RMSE: Root Mean Squared Error
    - MSE: Mean Squared Error
    """
    X_true_t = torch.tensor(X_true, dtype=torch.float32, device=device)
    X_pred_t = torch.tensor(X_pred, dtype=torch.float32, device=device)
    
    # Compute ROI-level mean (average across spots)
    true_mean = X_true_t.mean(dim=0)  # [n_genes]
    pred_mean = X_pred_t.mean(dim=0)  # [n_genes]
    
    # PCC on ROI-averaged expression
    true_centered = true_mean - true_mean.mean()
    pred_centered = pred_mean - pred_mean.mean()
    true_std = true_centered.norm()
    pred_std = pred_centered.norm()
    
    if true_std > 1e-8 and pred_std > 1e-8:
        roi_pcc = float((true_centered * pred_centered).sum() / (true_std * pred_std))
    else:
        roi_pcc = 0.0
    
    # Spearman on ROI-averaged expression (needs CPU for ranking)
    true_mean_np = true_mean.cpu().numpy()
    pred_mean_np = pred_mean.cpu().numpy()
    if np.std(true_mean_np) > 1e-8 and np.std(pred_mean_np) > 1e-8:
        roi_spearman, _ = spearmanr(true_mean_np, pred_mean_np)
    else:
        roi_spearman = 0.0
    
    # Cosine on ROI-averaged expression
    true_norm = true_mean.norm()
    pred_norm = pred_mean.norm()
    if true_norm > 1e-8 and pred_norm > 1e-8:
        roi_cos = float((true_mean * pred_mean).sum() / (true_norm * pred_norm))
    else:
        roi_cos = 0.0
    
    # MSE, RMSE (L2), MAE (L1) on ROI-averaged expression
    diff = true_mean - pred_mean
    roi_mse = float((diff ** 2).mean().cpu().item())
    roi_l1 = float(diff.abs().mean().cpu().item())
    roi_l2 = float(np.sqrt(roi_mse))
    
    return {
        "roi_pcc": float(roi_pcc),
        "roi_spearman": float(roi_spearman),
        "roi_cos": float(roi_cos),
        "roi_l1": float(roi_l1),
        "roi_l2": float(roi_l2),
        "roi_mse": float(roi_mse),
        "roi_mae": float(roi_l1),  # alias for roi_l1
        "roi_rmse": float(roi_l2),  # alias for roi_l2
    }


def sliced_wasserstein_distance_gpu(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    n_projections: int = 100,
    seed: int = 42,
) -> float:
    """
    Compute Sliced Wasserstein Distance using GPU.
    
    Args:
        X: first point cloud [n, d] on GPU
        Y: second point cloud [m, d] on GPU
        n_projections: number of random projections
        seed: random seed
    
    Returns:
        SWD value
    """
    device = X.device
    
    if X.dim() == 1:
        X = X.unsqueeze(0)
    if Y.dim() == 1:
        Y = Y.unsqueeze(0)
    
    # Z-score normalize
    X_mean, X_std = X.mean(), X.std()
    Y_mean, Y_std = Y.mean(), Y.std()
    
    X_norm = (X - X_mean) / (X_std + 1e-8)
    Y_norm = (Y - Y_mean) / (Y_std + 1e-8)
    
    d = X_norm.shape[1]
    
    # Random projections
    torch.manual_seed(seed)
    directions = torch.randn(n_projections, d, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True)
    
    # Project all at once: [n_proj, n] and [n_proj, m]
    proj_X = X_norm @ directions.T  # [n, n_proj]
    proj_Y = Y_norm @ directions.T  # [m, n_proj]
    
    # Sort projections
    proj_X_sorted, _ = proj_X.sort(dim=0)  # [n, n_proj]
    proj_Y_sorted, _ = proj_Y.sort(dim=0)  # [m, n_proj]
    
    # Interpolate if different sizes
    n_x, n_y = proj_X_sorted.shape[0], proj_Y_sorted.shape[0]
    if n_x != n_y:
        n_max = max(n_x, n_y)
        # Interpolate to same size
        proj_X_sorted = F.interpolate(
            proj_X_sorted.T.unsqueeze(0), size=n_max, mode='linear', align_corners=True
        ).squeeze(0).T
        proj_Y_sorted = F.interpolate(
            proj_Y_sorted.T.unsqueeze(0), size=n_max, mode='linear', align_corners=True
        ).squeeze(0).T
    
    # Compute Wasserstein-1 distance for each projection
    swd = (proj_X_sorted - proj_Y_sorted).abs().mean()
    
    return float(swd.cpu().item())


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
    ap.add_argument("--mask_init", type=str, default="zero", choices=["zero", "mean"],
                    help="Initialization for masked spots in step 0: 'zero' (default, like SpatialGT) or 'mean' (global mean)")
    
    # Metrics
    ap.add_argument("--swd_n_projections", type=int, default=100, help="Number of projections for SWD")
    
    # GPU settings
    ap.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    
    # Output
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
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
    if args.mask_mode == "patch":
        center_spot = get_center_spot(adata)
        mask_indices = select_patch_spots(neighbors, center_spot, args.n_spots)
        print(f"[INFO] Patch mode: selected {len(mask_indices)} spots from center spot {center_spot}")
    else:
        mask_indices = select_random_spots(n_spots, args.n_spots, args.seed)
        print(f"[INFO] Random mode: selected {len(mask_indices)} spots")
    
    # Ground truth for masked spots
    X_true = X[mask_indices].copy()
    
    # Run iterative reconstruction on GPU
    print(f"[INFO] Running iterative reconstruction for {args.steps} steps on {device}...")
    print(f"[INFO] Mask initialization: {args.mask_init}")
    step_recons = run_iterative_reconstruction_gpu(
        X_original=X,
        mask_indices=mask_indices,
        knn_indices=knn_indices,
        steps=args.steps,
        device=device,
        mode=args.inference_mode,
        ridge_alpha=args.ridge_alpha,
        ema_alpha=args.ema_alpha,
        mask_init=args.mask_init,
    )
    
    # Compute metrics for each step
    results_rows = []
    
    for step in range(args.steps + 1):
        X_pred = step_recons[step]
        
        # Spot-wise metrics
        metrics = compute_spotwise_metrics_gpu(
            X_true, X_pred, device, n_projections=args.swd_n_projections
        )
        
        # ROI-wise metrics
        roi_metrics = compute_roi_metrics_gpu(X_true, X_pred, device)
        metrics.update(roi_metrics)
        
        # Add spot-wise metrics
        for metric in ["pcc", "spearman", "cos", "l1", "l2", "mse", "mae", "rmse", "swd"]:
            results_rows.append({
                "step": step,
                "metric": metric,
                "value": metrics[metric],
                "std": metrics.get(f"{metric}_std", float("nan")),
            })
        
        # Add ROI-wise metrics
        for metric in ["roi_pcc", "roi_spearman", "roi_cos", "roi_l1", "roi_l2", "roi_mse", "roi_mae", "roi_rmse"]:
            results_rows.append({
                "step": step,
                "metric": metric,
                "value": metrics[metric],
                "std": float("nan"),  # ROI metrics have no std (single value)
            })
        
        print(f"  Step {step}: PCC={metrics['pcc']:.4f}, Spearman={metrics['spearman']:.4f}, "
              f"Cos={metrics['cos']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, SWD={metrics['swd']:.4f}")
        print(f"           ROI: PCC={metrics['roi_pcc']:.4f}, Spearman={metrics['roi_spearman']:.4f}, "
              f"Cos={metrics['roi_cos']:.4f}, MSE={metrics['roi_mse']:.4f}, RMSE={metrics['roi_rmse']:.4f}, MAE={metrics['roi_mae']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(out_dir / "summary.csv", index=False)
    
    # Save manifest
    manifest = {
        "task": "reconstruction",
        "baseline_method": f"KNN_{args.inference_mode}",
        "dataset": args.dataset_name,
        "cache_dir": str(cache_dir),
        "mask_mode": args.mask_mode,
        "n_spots": len(mask_indices),
        "n_spots_requested": args.n_spots,
        "knn_k": args.knn_k,
        "inference_mode": args.inference_mode,
        "ridge_alpha": args.ridge_alpha,
        "steps": args.steps,
        "ema_alpha": args.ema_alpha,
        "mask_init": args.mask_init,
        "seed": args.seed,
        "mask_indices": mask_indices,
        "n_genes": n_genes,
        "swd_n_projections": args.swd_n_projections,
        "device": str(device),
    }
    
    with open(out_dir / "reconstruction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save summary JSON
    summary_dict = {
        "final_step": args.steps,
        "final_metrics": {
            m: metrics[m] for m in ["pcc", "spearman", "cos", "l1", "l2", "mse", "mae", "rmse", "swd"]
        },
        "final_roi_metrics": {
            m: metrics[m] for m in ["roi_pcc", "roi_spearman", "roi_cos", "roi_l1", "roi_l2", "roi_mse", "roi_mae", "roi_rmse"]
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    
    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
