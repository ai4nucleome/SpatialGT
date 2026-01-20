#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialGT Reconstruction Evaluation

This script evaluates reconstruction performance by masking a subset of spots
and using the SpatialGT model to reconstruct their expression values.

Default settings:
- 10 iteration steps
- Seed: 42
- Mask init mode: "zero"
- Adjustable number of spots to mask via --n_spots

Supports:
- Two masking modes: patch (BFS-connected spots from center) and random
- Iterative reconstruction with frozen non-masked spots
- Spot-wise and ROI-wise similarity metrics

Usage:
    python spatialgt_reconstruction.py \\
        --ckpt <checkpoint_path> \\
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
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
import anndata as ad

# Add project root to path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PRETRAIN_DIR = _PROJECT_ROOT / "pretrain"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PRETRAIN_DIR))

# Import from pretrain module
from pretrain.spatial_databank import SpatialDataBank
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils_train import process_batch, forward_pass

from Config import ReconstructionConfig


# ==============================================================================
# Utility Functions
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


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_info(
    cache_dir: Path, 
    dataset_name: str, 
    lmdb_manifest_path: Optional[str] = None
) -> Tuple[int, int]:
    """
    Get (start_idx, n_spots) for a dataset from metadata.
    
    Args:
        cache_dir: Path to cache directory
        dataset_name: Name of the dataset
        lmdb_manifest_path: Optional path to LMDB manifest
        
    Returns:
        Tuple of (start_idx, n_spots)
    """
    # Try LMDB manifest first
    if lmdb_manifest_path:
        manifest_path = Path(lmdb_manifest_path)
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            meta = manifest.get("metadata", {})
            datasets = meta.get("datasets", manifest.get("datasets", []))
            
            # Find dataset info
            n_spots = 0
            for ds_info in datasets:
                if ds_info.get("name") == dataset_name:
                    n_spots = ds_info.get("n_spots", 0)
                    break
            else:
                n_spots = manifest.get("total_spots", 0)
            
            # Find start index
            start_idx = 0
            for i, ds in enumerate(datasets):
                if ds.get("name") == dataset_name:
                    for info in meta.get("dataset_indices", []):
                        if info.get("dataset_idx") == i:
                            start_idx = int(info.get("start_idx", 0))
                            break
                    break
            
            return start_idx, n_spots
    
    # Fallback to metadata.json
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    
    meta = load_json(meta_path)
    datasets = meta.get("datasets", [])
    
    n_spots = 0
    for ds_info in datasets:
        if ds_info.get("name") == dataset_name:
            n_spots = ds_info.get("n_spots", 0)
            break
    
    start_idx = 0
    for i, ds in enumerate(datasets):
        if ds.get("name") == dataset_name:
            for info in meta.get("dataset_indices", []):
                if info.get("dataset_idx") == i:
                    start_idx = int(info.get("start_idx", 0))
                    break
            break
    
    return start_idx, n_spots


def load_state_dict_into_model(model: nn.Module, ckpt_dir: Path) -> None:
    """Load model weights from checkpoint directory."""
    # Try different file formats
    pth = ckpt_dir / "finetuned_state_dict.pth"
    if pth.exists():
        sd = torch.load(str(pth), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        return
    
    pth = ckpt_dir / "finetuned_model.pth"
    if pth.exists():
        sd = torch.load(str(pth), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        return
    
    safep = ckpt_dir / "model.safetensors"
    if safep.exists():
        from safetensors.torch import load_file
        sd = load_file(str(safep))
        model.load_state_dict(sd, strict=False)
        return
    
    binp = ckpt_dir / "pytorch_model.bin"
    if binp.exists():
        sd = torch.load(str(binp), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        return
    
    # Try any .pth file
    pth_files = list(ckpt_dir.glob("*.pth"))
    if pth_files:
        sd = torch.load(str(pth_files[0]), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        return
    
    raise FileNotFoundError(f"No weights found under {ckpt_dir}")


# ==============================================================================
# Spot Selection
# ==============================================================================

def get_center_spot(
    databank: SpatialDataBank, 
    cache_dir: Path, 
    dataset_name: str,
    lmdb_manifest_path: Optional[str] = None
) -> int:
    """Find the spot closest to the spatial center of the slice."""
    adata_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(adata_path))
    spatial = adata.obsm['spatial']
    center = spatial.mean(axis=0)
    dists = np.sqrt(((spatial - center) ** 2).sum(axis=1))
    local_idx = int(np.argmin(dists))
    
    # Convert to global index
    start_idx, _ = get_dataset_info(cache_dir, dataset_name, lmdb_manifest_path)
    return start_idx + local_idx


def select_patch_spots(
    databank: SpatialDataBank,
    center_global_idx: int,
    patch_size: int,
    all_global_indices: Set[int],
) -> List[int]:
    """Select a contiguous patch of spots via BFS from a center spot."""
    visited = {center_global_idx}
    queue = deque([center_global_idx])
    selected = [center_global_idx]
    
    while queue and len(selected) < patch_size:
        curr = queue.popleft()
        try:
            neighbors = databank.get_neighbors_for_spot(int(curr))
        except Exception:
            neighbors = []
        for nb in neighbors:
            nb = int(nb)
            if nb not in visited and nb in all_global_indices:
                visited.add(nb)
                selected.append(nb)
                queue.append(nb)
                if len(selected) >= patch_size:
                    break
    
    return selected[:patch_size]


def select_random_spots(
    all_global_indices: List[int],
    n_spots: int,
    seed: int,
) -> List[int]:
    """Randomly sample n_spots from all spots."""
    rng = random.Random(seed)
    if len(all_global_indices) <= n_spots:
        return list(all_global_indices)
    return rng.sample(all_global_indices, n_spots)


# ==============================================================================
# SpatialGT Inference
# ==============================================================================

def build_data_loader(
    databank: SpatialDataBank,
    config: ReconstructionConfig,
    batch_size: int,
    num_workers: int,
):
    """Build data loader for full-slice inference."""
    loader = databank.get_data_loader(
        split="train",
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        stratify_by_batch=False,
        validation_split=0.0,
        use_distributed=False,
        persistent_workers=config.persistent_workers and num_workers > 0,
        prefetch_factor=config.prefetch_factor if num_workers > 0 else 2,
        is_training=False,
    )
    return loader


def run_single_pass_inference(
    databank: SpatialDataBank,
    loader,
    model: nn.Module,
    device: torch.device,
    desc: str = "Inference",
    show_progress: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Run a single forward pass on the entire slice.
    
    Returns:
        overrides dict: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}
    """
    all_overrides: Dict[int, Dict[str, np.ndarray]] = {}
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    iterator = tqdm(loader, desc=desc, disable=not show_progress)
    
    for batch in iterator:
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            continue
        
        # Get centers_global from structure
        structure = batch.get("structure", {})
        centers_global = structure.get("centers_global_indices", None)
        
        if centers_global is None:
            centers_global = structure.get("center_global_indices", None)
        if centers_global is None:
            centers_global = batch.get("centers_global_indices", None)
        
        if isinstance(centers_global, torch.Tensor):
            centers_global = centers_global.cpu().numpy().astype(int).tolist()
        elif centers_global is None:
            continue
        
        batch_data = process_batch(batch, device, config=databank.config)
        
        with torch.no_grad():
            preds, _, _, _ = forward_pass(model, batch_data, config=databank.config)
        
        B = len(centers_global)
        center_gene_ids = batch_data["genes"][:B]
        center_pad_mask = batch_data["padding_attention_mask"][:B].to(torch.bool)
        
        gathered = preds.gather(1, center_gene_ids.clamp(min=0))
        gathered = gathered * center_pad_mask.to(gathered.dtype)
        
        gid_np = center_gene_ids.detach().cpu().numpy().astype(np.int64)
        val_np = gathered.detach().float().cpu().numpy().astype(np.float32)
        mask_np = center_pad_mask.detach().cpu().numpy()
        
        for i in range(B):
            gidx = int(centers_global[i])
            valid_mask = mask_np[i]
            valid_gids = gid_np[i][valid_mask]
            valid_vals = val_np[i][valid_mask]
            all_overrides[gidx] = {"gene_ids": valid_gids, "raw_normed_values": valid_vals}
    
    return all_overrides


def run_iterative_inference(
    databank: SpatialDataBank,
    loader_factory: callable,
    model: nn.Module,
    device: torch.device,
    steps: int,
    frozen_global_indices: Set[int],
    masked_global_indices: Set[int],
    ema_alpha: float = 0.0,
    show_progress: bool = True,
    mask_init_mode: str = "zero",
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    """
    Run iterative inference with proper masking.
    
    Args:
        databank: SpatialDataBank instance
        loader_factory: callable that creates a new DataLoader
        model: SpatialGT model
        device: torch device
        steps: number of iteration steps
        frozen_global_indices: set of non-masked spot indices
        masked_global_indices: set of masked spot indices
        ema_alpha: EMA coefficient
        show_progress: whether to show progress bar
        mask_init_mode: "zero" or "mean" for masked spots initialization
    
    Returns:
        step_overrides: {step: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}}
    """
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    current_state: Dict[int, Dict[str, np.ndarray]] = {}
    
    # Initialize frozen (non-masked) spots with original values
    frozen_values: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in frozen_global_indices:
        sd = databank.get_spot_data(int(gidx))
        frozen_values[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
        }
        current_state[int(gidx)] = frozen_values[int(gidx)].copy()
    
    # Compute mean expression if using "mean" init mode
    mean_expr = None
    if mask_init_mode == "mean" and len(frozen_values) > 0:
        all_expr = np.stack([v["raw_normed_values"] for v in frozen_values.values()], axis=0)
        mean_expr = all_expr.mean(axis=0).astype(np.float32)
    
    # Initialize masked spots
    for gidx in masked_global_indices:
        sd = databank.get_spot_data(int(gidx))
        gene_ids = np.asarray(sd["gene_ids"], dtype=np.int64)
        
        if mask_init_mode == "zero":
            init_expr = np.zeros_like(np.asarray(sd["raw_normed_values"], dtype=np.float32))
        elif mask_init_mode == "mean" and mean_expr is not None:
            init_expr = mean_expr.copy()
        else:
            init_expr = np.zeros_like(np.asarray(sd["raw_normed_values"], dtype=np.float32))
        
        current_state[int(gidx)] = {
            "gene_ids": gene_ids,
            "raw_normed_values": init_expr,
        }
    
    print(f"[INFO] Initialized {len(frozen_global_indices)} frozen spots, "
          f"{len(masked_global_indices)} masked spots (mode: {mask_init_mode})")
    
    # Set overrides
    databank.set_runtime_spot_overrides(current_state)
    
    # Step 0: Initial pass
    loader = loader_factory()
    initial_overrides = run_single_pass_inference(
        databank, loader, model, device, desc="Step 0", show_progress=show_progress
    )
    
    # Apply initial predictions for masked spots
    for gidx, ov in initial_overrides.items():
        if gidx in frozen_global_indices:
            current_state[gidx] = frozen_values[gidx].copy()
        elif gidx in masked_global_indices:
            current_state[gidx] = {
                "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
                "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32),
            }
    
    step_overrides[0] = {
        gidx: current_state[gidx].copy() 
        for gidx in masked_global_indices if gidx in current_state
    }
    
    # Iterative refinement
    for t in range(1, steps + 1):
        databank.set_runtime_spot_overrides(current_state)
        loader = loader_factory()
        
        pred_overrides = run_single_pass_inference(
            databank, loader, model, device, desc=f"Step {t}/{steps}", show_progress=show_progress
        )
        
        for gidx, ov in pred_overrides.items():
            if gidx in frozen_global_indices:
                current_state[gidx] = frozen_values[gidx].copy()
            elif gidx in masked_global_indices:
                new_val = np.asarray(ov["raw_normed_values"], dtype=np.float32)
                if ema_alpha > 0 and gidx in current_state:
                    old_val = current_state[gidx]["raw_normed_values"]
                    new_val = ema_alpha * old_val + (1.0 - ema_alpha) * new_val
                current_state[gidx] = {
                    "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
                    "raw_normed_values": new_val,
                }
        
        step_overrides[t] = {
            gidx: current_state[gidx].copy() 
            for gidx in masked_global_indices if gidx in current_state
        }
    
    return step_overrides


# ==============================================================================
# Metrics Computation
# ==============================================================================

def sliced_wasserstein_distance(
    X: np.ndarray, 
    Y: np.ndarray, 
    n_projections: int = 100
) -> float:
    """Compute Sliced Wasserstein Distance between two point clouds."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    
    X_mean, X_std = X.mean(), X.std()
    Y_mean, Y_std = Y.mean(), Y.std()
    
    X_norm = (X - X_mean) / (X_std + 1e-8)
    Y_norm = (Y - Y_mean) / (Y_std + 1e-8)
    
    d = X_norm.shape[1]
    
    np.random.seed(42)
    directions = np.random.randn(n_projections, d)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    swd = 0.0
    for direction in directions:
        proj_X = X_norm @ direction
        proj_Y = Y_norm @ direction
        
        proj_X_sorted = np.sort(proj_X)
        proj_Y_sorted = np.sort(proj_Y)
        
        if len(proj_X_sorted) != len(proj_Y_sorted):
            n = max(len(proj_X_sorted), len(proj_Y_sorted))
            proj_X_sorted = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(proj_X_sorted)),
                proj_X_sorted
            )
            proj_Y_sorted = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(proj_Y_sorted)),
                proj_Y_sorted
            )
        
        swd += np.mean(np.abs(proj_X_sorted - proj_Y_sorted))
    
    return swd / n_projections


def compute_spotwise_metrics(
    true_data: Dict[int, Dict[str, np.ndarray]],
    pred_data: Dict[int, Dict[str, np.ndarray]],
    n_projections: int = 100,
) -> Dict[str, float]:
    """
    Compute spot-wise similarity metrics.
    
    Metrics: PCC, Spearman, Cosine, L1/MAE, L2/RMSE, MSE, SWD
    """
    pcc_list, spearman_list, cos_list = [], [], []
    l1_list, l2_list, mse_list = [], [], []
    all_true, all_pred = [], []
    
    for gidx in true_data:
        if gidx not in pred_data:
            continue
        
        true_gids = true_data[gidx]["gene_ids"]
        true_vals = true_data[gidx]["raw_normed_values"]
        pred_gids = pred_data[gidx]["gene_ids"]
        pred_vals = pred_data[gidx]["raw_normed_values"]
        
        if len(true_gids) == len(pred_gids) and np.array_equal(true_gids, pred_gids):
            true_vec = np.asarray(true_vals, dtype=np.float32)
            pred_vec = np.asarray(pred_vals, dtype=np.float32)
        else:
            # Align by common genes
            true_gid_set = set(int(g) for g in true_gids if g >= 0)
            pred_gid_set = set(int(g) for g in pred_gids if g >= 0)
            common = sorted(true_gid_set & pred_gid_set)
            
            if len(common) == 0:
                continue
            
            true_pos = {int(g): i for i, g in enumerate(true_gids)}
            pred_pos = {int(g): i for i, g in enumerate(pred_gids)}
            
            true_vec = np.array([true_vals[true_pos[g]] for g in common], dtype=np.float32)
            pred_vec = np.array([pred_vals[pred_pos[g]] for g in common], dtype=np.float32)
        
        all_true.append(true_vec)
        all_pred.append(pred_vec)
        
        # PCC
        if np.std(true_vec) > 1e-8 and np.std(pred_vec) > 1e-8:
            pcc, _ = pearsonr(true_vec, pred_vec)
        else:
            pcc = 0.0
        pcc_list.append(pcc)
        
        # Spearman
        if np.std(true_vec) > 1e-8 and np.std(pred_vec) > 1e-8:
            sp, _ = spearmanr(true_vec, pred_vec)
        else:
            sp = 0.0
        spearman_list.append(sp)
        
        # Cosine
        norm_t = np.linalg.norm(true_vec)
        norm_p = np.linalg.norm(pred_vec)
        if norm_t > 1e-8 and norm_p > 1e-8:
            cos = float(np.dot(true_vec, pred_vec) / (norm_t * norm_p))
        else:
            cos = 0.0
        cos_list.append(cos)
        
        # MSE, RMSE (L2), MAE (L1)
        mse_val = float(np.mean((true_vec - pred_vec) ** 2))
        mse_list.append(mse_val)
        l1_list.append(float(np.mean(np.abs(true_vec - pred_vec))))
        l2_list.append(float(np.sqrt(mse_val)))
    
    # SWD
    if len(all_true) > 0:
        all_true_arr = np.vstack(all_true)
        all_pred_arr = np.vstack(all_pred)
        swd = sliced_wasserstein_distance(all_true_arr, all_pred_arr, n_projections)
    else:
        swd = float("nan")
    
    return {
        "pcc": float(np.nanmean(pcc_list)) if pcc_list else float("nan"),
        "spearman": float(np.nanmean(spearman_list)) if spearman_list else float("nan"),
        "cos": float(np.nanmean(cos_list)) if cos_list else float("nan"),
        "l1": float(np.nanmean(l1_list)) if l1_list else float("nan"),
        "l2": float(np.nanmean(l2_list)) if l2_list else float("nan"),
        "mse": float(np.nanmean(mse_list)) if mse_list else float("nan"),
        "mae": float(np.nanmean(l1_list)) if l1_list else float("nan"),
        "rmse": float(np.nanmean(l2_list)) if l2_list else float("nan"),
        "swd": float(swd),
        "pcc_std": float(np.nanstd(pcc_list)) if pcc_list else float("nan"),
        "spearman_std": float(np.nanstd(spearman_list)) if spearman_list else float("nan"),
        "cos_std": float(np.nanstd(cos_list)) if cos_list else float("nan"),
        "l1_std": float(np.nanstd(l1_list)) if l1_list else float("nan"),
        "l2_std": float(np.nanstd(l2_list)) if l2_list else float("nan"),
        "mse_std": float(np.nanstd(mse_list)) if mse_list else float("nan"),
        "mae_std": float(np.nanstd(l1_list)) if l1_list else float("nan"),
        "rmse_std": float(np.nanstd(l2_list)) if l2_list else float("nan"),
    }


def compute_roi_metrics(
    true_data: Dict[int, Dict[str, np.ndarray]],
    pred_data: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute ROI-wise similarity metrics.
    
    Averages expression across all spots in the ROI first.
    """
    all_true, all_pred = [], []
    
    for gidx in true_data:
        if gidx not in pred_data:
            continue
        
        true_gids = true_data[gidx]["gene_ids"]
        true_vals = true_data[gidx]["raw_normed_values"]
        pred_gids = pred_data[gidx]["gene_ids"]
        pred_vals = pred_data[gidx]["raw_normed_values"]
        
        if len(true_gids) == len(pred_gids) and np.array_equal(true_gids, pred_gids):
            true_vec = np.asarray(true_vals, dtype=np.float32)
            pred_vec = np.asarray(pred_vals, dtype=np.float32)
        else:
            true_gid_set = set(int(g) for g in true_gids if g >= 0)
            pred_gid_set = set(int(g) for g in pred_gids if g >= 0)
            common = sorted(true_gid_set & pred_gid_set)
            
            if len(common) == 0:
                continue
            
            true_pos = {int(g): i for i, g in enumerate(true_gids)}
            pred_pos = {int(g): i for i, g in enumerate(pred_gids)}
            
            true_vec = np.array([true_vals[true_pos[g]] for g in common], dtype=np.float32)
            pred_vec = np.array([pred_vals[pred_pos[g]] for g in common], dtype=np.float32)
        
        all_true.append(true_vec)
        all_pred.append(pred_vec)
    
    if len(all_true) == 0:
        return {k: float("nan") for k in [
            "roi_pcc", "roi_spearman", "roi_cos", "roi_l1", "roi_l2", 
            "roi_mse", "roi_mae", "roi_rmse"
        ]}
    
    true_arr = np.vstack(all_true)
    pred_arr = np.vstack(all_pred)
    
    true_mean = true_arr.mean(axis=0)
    pred_mean = pred_arr.mean(axis=0)
    
    # PCC
    if np.std(true_mean) > 1e-8 and np.std(pred_mean) > 1e-8:
        roi_pcc, _ = pearsonr(true_mean, pred_mean)
    else:
        roi_pcc = 0.0
    
    # Spearman
    if np.std(true_mean) > 1e-8 and np.std(pred_mean) > 1e-8:
        roi_spearman, _ = spearmanr(true_mean, pred_mean)
    else:
        roi_spearman = 0.0
    
    # Cosine
    norm_t = np.linalg.norm(true_mean)
    norm_p = np.linalg.norm(pred_mean)
    if norm_t > 1e-8 and norm_p > 1e-8:
        roi_cos = float(np.dot(true_mean, pred_mean) / (norm_t * norm_p))
    else:
        roi_cos = 0.0
    
    # MSE, RMSE, MAE
    roi_mse = float(np.mean((true_mean - pred_mean) ** 2))
    roi_l1 = float(np.mean(np.abs(true_mean - pred_mean)))
    roi_l2 = float(np.sqrt(roi_mse))
    
    return {
        "roi_pcc": float(roi_pcc),
        "roi_spearman": float(roi_spearman),
        "roi_cos": float(roi_cos),
        "roi_l1": float(roi_l1),
        "roi_l2": float(roi_l2),
        "roi_mse": float(roi_mse),
        "roi_mae": float(roi_l1),
        "roi_rmse": float(roi_l2),
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialGT Reconstruction Evaluation")
    
    # Model paths
    parser.add_argument("--ckpt", type=str, required=True, 
                        help="Path to SpatialGT checkpoint")
    
    # Data paths
    parser.add_argument("--cache_dir", type=str, required=True, 
                        help="Path to preprocessed cache directory")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Dataset name (subdirectory in cache_dir)")
    parser.add_argument("--cache_mode", type=str, default="h5", choices=["h5", "lmdb"])
    parser.add_argument("--lmdb_path", type=str, default=None)
    parser.add_argument("--lmdb_manifest", type=str, default=None)
    
    # Masking parameters
    parser.add_argument("--mask_mode", type=str, default="patch", choices=["patch", "random"],
                        help="Masking mode: patch (BFS from center) or random")
    parser.add_argument("--n_spots", type=int, default=20, 
                        help="Number of spots to mask")
    
    # Inference parameters
    parser.add_argument("--steps", type=int, default=10, 
                        help="Number of iteration steps (default: 10)")
    parser.add_argument("--ema_alpha", type=float, default=0.0, 
                        help="EMA coefficient (0 = direct replacement)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_init_mode", type=str, default="zero", choices=["zero", "mean"],
                        help="How to initialize masked spots: zero (default) or mean")
    
    # Metrics
    parser.add_argument("--swd_n_projections", type=int, default=100)
    
    # Output
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs (comma-separated)")
    
    args = parser.parse_args()
    
    # Fix seed
    fix_seed(args.seed)
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    
    # Paths
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt)
    
    # Setup config
    config = ReconstructionConfig()
    config.cache_dir = str(cache_dir)
    config.cache_mode = args.cache_mode.lower()
    config.strict_cache_only = True
    
    if args.cache_mode.lower() == "lmdb":
        if args.lmdb_path:
            config.lmdb_path = str(args.lmdb_path)
        if args.lmdb_manifest:
            config.lmdb_manifest_path = str(args.lmdb_manifest)
    
    # Initialize databank
    print(f"[INFO] Initializing databank...")
    dataset_paths = [str(cache_dir / args.dataset_name / "processed.h5ad")]
    databank = SpatialDataBank(
        dataset_paths=dataset_paths,
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False,
    )
    
    # Get all global indices
    start_idx, n_spots = get_dataset_info(
        cache_dir, args.dataset_name,
        lmdb_manifest_path=args.lmdb_manifest if args.cache_mode == "lmdb" else None
    )
    all_global_indices = list(range(start_idx, start_idx + n_spots))
    print(f"[INFO] Dataset has {n_spots} spots")
    
    # Select spots to mask
    lmdb_manifest = args.lmdb_manifest if args.cache_mode == "lmdb" else None
    if args.mask_mode == "patch":
        center_spot = get_center_spot(databank, cache_dir, args.dataset_name, lmdb_manifest)
        mask_indices = select_patch_spots(databank, center_spot, args.n_spots, set(all_global_indices))
        print(f"[INFO] Patch mode: selected {len(mask_indices)} spots from center {center_spot}")
    else:
        mask_indices = select_random_spots(all_global_indices, args.n_spots, args.seed)
        print(f"[INFO] Random mode: selected {len(mask_indices)} spots")
    
    mask_set = set(mask_indices)
    frozen_indices = set(all_global_indices) - mask_set
    
    # Get ground truth for masked spots
    true_data: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in mask_indices:
        sd = databank.get_spot_data(int(gidx))
        true_data[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
        }
    
    # Load model
    print(f"\n[INFO] Loading SpatialGT model from {ckpt_dir}...")
    model = SpatialNeighborTransformer(config)
    load_state_dict_into_model(model, ckpt_dir)
    model.to(device)
    model.eval()
    
    # Build loader factory
    def loader_factory():
        return build_data_loader(databank, config, args.batch_size, args.num_workers)
    
    # Run iterative reconstruction
    print(f"\n[INFO] Running iterative reconstruction for {args.steps} steps...")
    step_overrides = run_iterative_inference(
        databank=databank,
        loader_factory=loader_factory,
        model=model,
        device=device,
        steps=args.steps,
        frozen_global_indices=frozen_indices,
        masked_global_indices=mask_set,
        ema_alpha=args.ema_alpha,
        show_progress=True,
        mask_init_mode=args.mask_init_mode,
    )
    
    # Compute metrics for each step
    results_rows = []
    
    for step in range(args.steps + 1):
        pred_data = step_overrides[step]
        
        metrics = compute_spotwise_metrics(true_data, pred_data, n_projections=args.swd_n_projections)
        roi_metrics = compute_roi_metrics(true_data, pred_data)
        metrics.update(roi_metrics)
        
        for metric in ["pcc", "spearman", "cos", "l1", "l2", "mse", "mae", "rmse", "swd"]:
            results_rows.append({
                "step": step,
                "metric": metric,
                "value": metrics[metric],
                "std": metrics.get(f"{metric}_std", float("nan")),
            })
        
        for metric in ["roi_pcc", "roi_spearman", "roi_cos", "roi_l1", "roi_l2", "roi_mse", "roi_mae", "roi_rmse"]:
            results_rows.append({
                "step": step,
                "metric": metric,
                "value": metrics[metric],
                "std": float("nan"),
            })
        
        print(f"  Step {step}: PCC={metrics['pcc']:.4f}, Spearman={metrics['spearman']:.4f}, "
              f"Cos={metrics['cos']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(out_dir / "summary.csv", index=False)
    
    # Save manifest
    manifest = {
        "task": "reconstruction",
        "method": "SpatialGT",
        "dataset": args.dataset_name,
        "cache_dir": str(cache_dir),
        "checkpoint": str(ckpt_dir),
        "mask_mode": args.mask_mode,
        "n_spots": len(mask_indices),
        "steps": args.steps,
        "ema_alpha": args.ema_alpha,
        "seed": args.seed,
        "mask_init_mode": args.mask_init_mode,
        "mask_indices": mask_indices,
    }
    
    with open(out_dir / "reconstruction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save summary JSON
    final_metrics = metrics
    summary_dict = {
        "final_step": args.steps,
        "final_metrics": {m: final_metrics[m] for m in ["pcc", "spearman", "cos", "mse", "rmse", "mae", "swd"]},
        "final_roi_metrics": {m: final_metrics[m] for m in ["roi_pcc", "roi_spearman", "roi_cos", "roi_mse", "roi_rmse", "roi_mae"]},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    
    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
