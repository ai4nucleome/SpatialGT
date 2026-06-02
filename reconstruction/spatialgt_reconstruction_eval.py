#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialGT reconstruction evaluation.

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
import torch.nn as nn
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

from Config import Config
from spatial_databank import SpatialDataBank
from model_spatialpt import SpatialNeighborTransformer
from utils_train import process_batch, forward_pass


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


def _get_dataset_start_idx(cache_dir: Path, dataset_name: str, lmdb_manifest_path: Optional[str] = None) -> int:
    """Get dataset start index from metadata.json or LMDB manifest."""
    # Try LMDB manifest first
    if lmdb_manifest_path:
        manifest_path = Path(lmdb_manifest_path)
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            meta = manifest.get("metadata", {})
            datasets = meta.get("datasets", manifest.get("datasets", []))
            dataset_idx = None
            for i, ds in enumerate(datasets):
                if ds.get("name") == dataset_name:
                    dataset_idx = i
                    break
            if dataset_idx is None:
                dataset_idx = 0
            for info in meta.get("dataset_indices", []):
                if info.get("dataset_idx") == dataset_idx:
                    return int(info.get("start_idx", 0))
            return 0
    
    # Fallback to metadata.json
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return 0
    meta = _load_json(meta_path)
    datasets = meta.get("datasets", [])
    dataset_idx = None
    for i, ds in enumerate(datasets):
        if ds.get("name") == dataset_name:
            dataset_idx = i
            break
    if dataset_idx is None:
        dataset_idx = 0
    for info in meta.get("dataset_indices", []):
        if info.get("dataset_idx") == dataset_idx:
            return int(info.get("start_idx", 0))
    return 0


def _get_dataset_info(cache_dir: Path, dataset_name: str, lmdb_manifest_path: Optional[str] = None) -> Tuple[int, int]:
    """Get (start_idx, n_spots) for a dataset from metadata or LMDB manifest."""
    # Try LMDB manifest first
    if lmdb_manifest_path:
        manifest_path = Path(lmdb_manifest_path)
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            meta = manifest.get("metadata", {})
            datasets = meta.get("datasets", manifest.get("datasets", []))
            
            # Find dataset info
            for ds_info in datasets:
                if ds_info.get("name") == dataset_name:
                    n_spots = ds_info.get("n_spots", 0)
                    break
            else:
                n_spots = manifest.get("total_spots", 0)
            
            # Find start index
            dataset_idx = None
            for i, ds in enumerate(datasets):
                if ds.get("name") == dataset_name:
                    dataset_idx = i
                    break
            if dataset_idx is None:
                dataset_idx = 0
            start_idx = 0
            for info in meta.get("dataset_indices", []):
                if info.get("dataset_idx") == dataset_idx:
                    start_idx = int(info.get("start_idx", 0))
                    break
            
            return start_idx, n_spots
    
    # Fallback to metadata.json
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    meta = _load_json(meta_path)
    
    datasets = meta.get("datasets", [])
    n_spots = 0
    for ds_info in datasets:
        if ds_info.get("name") == dataset_name:
            n_spots = ds_info.get("n_spots", 0)
            break
    
    # Find start index
    dataset_idx = None
    for i, ds in enumerate(datasets):
        if ds.get("name") == dataset_name:
            dataset_idx = i
            break
    if dataset_idx is None:
        dataset_idx = 0
    start_idx = 0
    for info in meta.get("dataset_indices", []):
        if info.get("dataset_idx") == dataset_idx:
            start_idx = int(info.get("start_idx", 0))
            break
    
    return start_idx, n_spots


def _setup_config_for_cache(
    config: Config,
    cache_dir: str,
    dataset_name: str,
    cache_mode: str = "h5",
    lmdb_path: Optional[str] = None,
    lmdb_manifest_path: Optional[str] = None,
) -> Config:
    """Configure Config object for the specified cache mode."""
    config.cache_dir = str(cache_dir)
    config.cache_mode = cache_mode.lower()
    config.strict_cache_only = True
    config.max_seq_len = getattr(config, "max_seq_len", 3000)
    config.mask_value = getattr(config, "mask_value", -1)
    config.pad_value = getattr(config, "pad_value", -2)
    config.pad_token = getattr(config, "pad_token", "[PAD]")
    config.cls_token = getattr(config, "cls_token", "[CLS]")
    config.use_metadata_vocabs = getattr(config, "use_metadata_vocabs", False)
    config.log_vocab_messages = getattr(config, "log_vocab_messages", False)
    config.data_is_raw = getattr(config, "data_is_raw", True)
    config.input_style = getattr(config, "input_style", "log1p")
    config.include_zero_gene = getattr(config, "include_zero_gene", True)
    config.filter_gene_by_counts = getattr(config, "filter_gene_by_counts", False)
    config.subset_hvg = getattr(config, "subset_hvg", 3000)
    
    if cache_mode.lower() == "lmdb":
        if lmdb_path:
            config.lmdb_path = str(lmdb_path)
            config.runtime_lmdb_path = str(lmdb_path)
        if lmdb_manifest_path:
            config.lmdb_manifest_path = str(lmdb_manifest_path)
            config.runtime_lmdb_manifest_path = str(lmdb_manifest_path)
    
    return config


def _load_state_dict_into_model(model: SpatialNeighborTransformer, ckpt_dir: Path) -> None:
    pth = ckpt_dir / "finetuned_state_dict.pth"
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
    raise FileNotFoundError(f"No weights found under {ckpt_dir}")


# ==============================================================================
# Data loading
# ==============================================================================

def load_adata_and_expression(cache_dir: Path, dataset_name: str) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Load processed.h5ad and extract expression matrix.
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

def get_center_spot(databank: SpatialDataBank, cache_dir: Path, dataset_name: str, 
                    lmdb_manifest_path: Optional[str] = None) -> int:
    """Find the spot closest to the spatial center of the slice."""
    adata_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(adata_path))
    spatial = adata.obsm['spatial']
    center = spatial.mean(axis=0)
    dists = np.sqrt(((spatial - center) ** 2).sum(axis=1))
    local_idx = int(np.argmin(dists))
    
    # Convert to global index
    start_idx = _get_dataset_start_idx(cache_dir, dataset_name, lmdb_manifest_path)
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


def build_neighbor_dict_for_slice(
    databank: SpatialDataBank,
    all_global_indices: Set[int],
) -> Dict[int, List[int]]:
    """Directed neighbor lists restricted to spots in this slice (global indices)."""
    out: Dict[int, List[int]] = {}
    for g in all_global_indices:
        try:
            nbs = databank.get_neighbors_for_spot(int(g))
        except Exception:
            nbs = []
        out[int(g)] = [int(x) for x in nbs if int(x) in all_global_indices]
    return out


# ==============================================================================
# SpatialGT Inference
# ==============================================================================

def _build_all_loader(
    databank: SpatialDataBank,
    config: Config,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
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
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
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
    Returns overrides dict: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}
    """
    all_overrides: Dict[int, Dict[str, np.ndarray]] = {}
    
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    iterator = tqdm(loader, desc=desc, disable=not show_progress)
    
    skipped_batches = 0
    processed_spots = 0
    
    for batch in iterator:
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            skipped_batches += 1
            continue
        
        # Get centers_global from structure
        structure = batch.get("structure", {})
        centers_global = structure.get("centers_global_indices", None)
        
        if centers_global is None:
            # Try alternative key names
            centers_global = structure.get("center_global_indices", None)
        
        if centers_global is None:
            # Fallback: try to get from batch directly
            centers_global = batch.get("centers_global_indices", None)
        
        if isinstance(centers_global, torch.Tensor):
            centers_global = centers_global.cpu().numpy().astype(int).tolist()
        elif centers_global is None:
            # Skip batch if no global indices available
            skipped_batches += 1
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
            # Only keep valid (non-padding) positions
            valid_mask = mask_np[i]
            valid_gids = gid_np[i][valid_mask]
            valid_vals = val_np[i][valid_mask]
            all_overrides[gidx] = {"gene_ids": valid_gids, "raw_normed_values": valid_vals}
            processed_spots += 1
    
    if skipped_batches > 0:
        print(f"[WARN] {desc}: Skipped {skipped_batches} batches")
    print(f"[INFO] {desc}: Processed {processed_spots} spots, collected {len(all_overrides)} unique overrides")
    
    return all_overrides


def run_iterative_inference_with_freeze(
    databank: SpatialDataBank,
    loader_factory: callable,
    model: nn.Module,
    device: torch.device,
    steps: int,
    frozen_global_indices: Set[int],
    masked_global_indices: Set[int],
    ema_alpha: float = 0.0,
    show_progress: bool = True,
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    """
    Run iterative inference with proper masking.
    
    IMPORTANT: Masked spots handling during reconstruction:
    - When they are the center: their true expression is hidden (uses model's mask token)
    - When they are neighbors: initialized as zero vectors
    
    Args:
        databank: SpatialDataBank instance
        loader_factory: callable that creates a new DataLoader
        model: SpatialGT model
        device: torch device
        steps: number of iteration steps
        frozen_global_indices: set of global indices that keep original values (non-masked spots)
        masked_global_indices: set of global indices that are masked (to be reconstructed)
        ema_alpha: EMA coefficient
        show_progress: whether to show progress bar
    
    Returns:
        step_overrides: {step: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}}
    """
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    current_state: Dict[int, Dict[str, np.ndarray]] = {}
    
    # Step 1: Initialize frozen (non-masked) spots with their original values
    frozen_values: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in frozen_global_indices:
        sd = databank.get_spot_data(int(gidx))
        frozen_values[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
        }
        current_state[int(gidx)] = frozen_values[int(gidx)].copy()

    # Step 2: Initialize masked spots as zero-valued neighbors.
    for gidx in masked_global_indices:
        sd = databank.get_spot_data(int(gidx))
        gene_ids = np.asarray(sd["gene_ids"], dtype=np.int64)
        init_expr = np.zeros_like(np.asarray(sd["raw_normed_values"], dtype=np.float32))
        current_state[int(gidx)] = {
            "gene_ids": gene_ids,
            "raw_normed_values": init_expr,
        }
    
    print(f"[INFO] Initialized {len(frozen_global_indices)} frozen spots (keep original)")
    print(f"[INFO] Initialized {len(masked_global_indices)} masked spots as zero-valued neighbors")
    
    # Step 3: Set all spot overrides (both frozen and masked)
    # This ensures masked spots appear as zeros when they are neighbors
    databank.set_runtime_spot_overrides(current_state)
    
    # Step 4: Initial pass - model predicts based on zero-initialized masked spots
    loader = loader_factory()
    initial_overrides = run_single_pass_inference(
        databank, loader, model, device, desc="Step 0", show_progress=show_progress
    )
    
    # Check how many masked spots were predicted
    predicted_masked = sum(1 for gidx in masked_global_indices if gidx in initial_overrides)
    print(f"[DEBUG] Step 0: {predicted_masked}/{len(masked_global_indices)} masked spots have predictions")
    
    if predicted_masked == 0:
        print(f"[ERROR] No masked spots were predicted! Check if centers_global_indices is available in batch.")
        print(f"[DEBUG] Sample of masked_global_indices: {list(masked_global_indices)[:5]}")
        print(f"[DEBUG] Sample of initial_overrides keys: {list(initial_overrides.keys())[:5]}")
    
    # Apply initial predictions for masked spots, keep frozen for non-masked
    for gidx, ov in initial_overrides.items():
        if gidx in frozen_global_indices:
            current_state[gidx] = frozen_values[gidx].copy()
        elif gidx in masked_global_indices:
            # Update masked spots with their first prediction
            current_state[gidx] = {
                "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
                "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32),
            }
    
    # Record step 0 results (only masked spots)
    step_overrides[0] = {
        gidx: current_state[gidx].copy() 
        for gidx in masked_global_indices if gidx in current_state
    }
    
    # Step 5: Iterative refinement
    for t in range(1, steps + 1):
        # Update overrides with current state (including updated masked spot predictions)
        databank.set_runtime_spot_overrides(current_state)
        loader = loader_factory()
        
        pred_overrides = run_single_pass_inference(
            databank, loader, model, device, desc=f"Step {t}/{steps}", show_progress=show_progress
        )
        
        # Update state
        for gidx, ov in pred_overrides.items():
            if gidx in frozen_global_indices:
                # Keep frozen spots unchanged
                current_state[gidx] = frozen_values[gidx].copy()
            elif gidx in masked_global_indices:
                # Update masked spots with new prediction (with optional EMA)
                new_val = np.asarray(ov["raw_normed_values"], dtype=np.float32)
                if ema_alpha > 0 and gidx in current_state:
                    old_val = current_state[gidx]["raw_normed_values"]
                    new_val = ema_alpha * old_val + (1.0 - ema_alpha) * new_val
                current_state[gidx] = {
                    "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
                    "raw_normed_values": new_val,
                }
        
        # Record step t results (only masked spots)
        step_overrides[t] = {
            gidx: current_state[gidx].copy() 
            for gidx in masked_global_indices if gidx in current_state
        }
    
    return step_overrides


# ==============================================================================
# Reconstruction metrics
# ==============================================================================

def compute_spotwise_metrics(
    true_data: Dict[int, Dict[str, np.ndarray]],
    pred_data: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """Compute spot-wise PCC and RMSE for reconstructed masked spots."""
    pcc_list = []
    rmse_list = []

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
            true_gid_set = {int(g) for g in true_gids if g >= 0}
            pred_gid_set = {int(g) for g in pred_gids if g >= 0}
            common = sorted(true_gid_set & pred_gid_set)
            if not common:
                continue
            true_pos = {int(g): i for i, g in enumerate(true_gids)}
            pred_pos = {int(g): i for i, g in enumerate(pred_gids)}
            true_vec = np.array([true_vals[true_pos[g]] for g in common], dtype=np.float32)
            pred_vec = np.array([pred_vals[pred_pos[g]] for g in common], dtype=np.float32)

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
    ap = argparse.ArgumentParser(description="SpatialGT reconstruction evaluation")
    
    # Model paths
    ap.add_argument("--ckpt", type=str, required=True, help="Path to SpatialGT checkpoint")
    
    # Data paths
    ap.add_argument("--cache_dir", type=str, required=True, help="Path to preprocessed cache directory")
    ap.add_argument("--dataset_name", type=str, required=True, help="Dataset name (subdirectory in cache_dir)")
    ap.add_argument("--cache_mode", type=str, default="h5", choices=["h5", "lmdb"])
    ap.add_argument("--lmdb_path", type=str, default=None)
    ap.add_argument("--lmdb_manifest", type=str, default=None)
    
    # Masking parameters
    ap.add_argument("--mask_mode", type=str, default="patch", choices=["patch", "random"],
                    help="Masking mode: patch (BFS from center) or random")
    ap.add_argument("--n_spots", type=int, default=20, help="Number of spots to mask")
    
    # Inference parameters
    ap.add_argument("--steps", type=int, default=10, help="Number of iteration steps")
    ap.add_argument("--ema_alpha", type=float, default=0.0, help="EMA coefficient (0 = direct replacement)")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor")
    ap.add_argument("--persistent_workers", action="store_true", help="Use persistent workers")
    # Output
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs (comma-separated)")
    
    args = ap.parse_args()
    
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
    config = Config()
    config = _setup_config_for_cache(
        config, str(cache_dir), args.dataset_name, args.cache_mode,
        args.lmdb_path, args.lmdb_manifest
    )
    
    # Initialize databank
    print(f"[INFO] Initializing databank...")
    # SpatialDataBank expects (dataset_paths, cache_dir, config, ...)
    # In our preprocessed_cache layout: <cache_dir>/<dataset_name>/processed.h5ad
    dataset_paths = [str(cache_dir / args.dataset_name / "processed.h5ad")]
    databank = SpatialDataBank(
        dataset_paths=dataset_paths,
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False,
    )
    
    # Get all global indices for this dataset
    # Works with both h5 mode (metadata.json) and LMDB mode (manifest.json)
    start_idx, n_spots = _get_dataset_info(
        cache_dir, args.dataset_name, 
        lmdb_manifest_path=args.lmdb_manifest if args.cache_mode == "lmdb" else None
    )
    all_global_indices = list(range(start_idx, start_idx + n_spots))
    print(f"[INFO] Dataset has {n_spots} spots (global indices {start_idx} to {start_idx + n_spots - 1})")
    print(f"[INFO] Cache mode: {args.cache_mode}")
    
    # Select spots to mask
    lmdb_manifest = args.lmdb_manifest if args.cache_mode == "lmdb" else None
    patch_extra_meta: Dict[str, Any] = {}
    if args.mask_mode == "patch":
        allowed_g = set(all_global_indices)
        neigh_g = build_neighbor_dict_for_slice(databank, allowed_g)
        center_spot, mask_indices, patch_extra_meta = select_seeded_patch_center_and_mask(
            neigh_g, args.n_spots, args.seed, allowed_vertices=allowed_g
        )
        w = patch_extra_meta.get("warning")
        if w:
            print(f"[WARN] Patch selection: {w}")
        print(
            f"[INFO] Patch mode: center_global={center_spot}, "
            f"masked {len(mask_indices)} spots (requested {args.n_spots})"
        )
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
    _load_state_dict_into_model(model, ckpt_dir)
    model.to(device)
    model.eval()
    
    # Build loader factory
    def loader_factory():
        return _build_all_loader(
            databank, config, args.batch_size, args.num_workers,
            args.prefetch_factor, args.persistent_workers
        )
    
    # Run iterative reconstruction
    # Masked spots handling:
    # - When they are the center: their true expression is hidden (uses model's mask token)
    # - When they are neighbors: initialized as zero vectors
    print(f"\n[INFO] Running iterative reconstruction for {args.steps} steps...")
    print(f"[INFO] Masked spots ({len(mask_set)}) are zero-initialized")
    step_overrides = run_iterative_inference_with_freeze(
        databank=databank,
        loader_factory=loader_factory,
        model=model,
        device=device,
        steps=args.steps,
        frozen_global_indices=frozen_indices,
        masked_global_indices=mask_set,
        ema_alpha=args.ema_alpha,
        show_progress=True,
    )
    
    # Compute metrics for each step
    results_rows = []
    
    for step in range(args.steps + 1):
        pred_data = step_overrides[step]
        
        metrics = compute_spotwise_metrics(true_data, pred_data)
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
        "eval_script": "spatialgt_reconstruction_eval.py",
        "baseline_method": "SpatialGT",
        "dataset": args.dataset_name,
        "mask_mode": args.mask_mode,
        "n_spots": len(mask_indices),
        "n_spots_requested": args.n_spots,
        "steps": args.steps,
        "ema_alpha": args.ema_alpha,
        "device": str(device),
        "mask_indices": mask_indices,
        "masked_spot_value": "zero",
        "masking_note": "Masked spots are zero-initialized when acting as neighbors",
    }
    if patch_extra_meta:
        manifest["patch_selection"] = patch_extra_meta

    with open(out_dir / "reconstruction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save summary JSON
    final_metrics = metrics
    summary_dict = {
        "final_step": args.steps,
        "final_metrics": {
            m: final_metrics[m] for m in ["pcc", "rmse"]
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    
    # Save per-step reconstructed expressions for visualization
    # Convert dict-based format to matrices [n_masked, n_genes]
    ordered_mask = [gidx for gidx in mask_indices if gidx in true_data]
    X_true_matrix = np.stack(
        [true_data[gidx]["raw_normed_values"] for gidx in ordered_mask], axis=0
    )
    first_gidx = ordered_mask[0]
    gene_ids_arr = true_data[first_gidx]["gene_ids"]
    
    expr_save = {
        "ground_truth": X_true_matrix,
        "mask_indices": np.array(ordered_mask, dtype=np.int64),
        "gene_ids": gene_ids_arr,
    }
    for step in range(args.steps + 1):
        pred_data_step = step_overrides[step]
        X_pred_matrix = np.stack(
            [pred_data_step[gidx]["raw_normed_values"]
             for gidx in ordered_mask if gidx in pred_data_step],
            axis=0,
        )
        expr_save[f"step_{step}"] = X_pred_matrix
    np.savez(out_dir / "reconstruction_expressions.npz", **expr_save)
    
    # Save gene names from adata
    adata_for_names = ad.read_h5ad(str(cache_dir / args.dataset_name / "processed.h5ad"))
    gene_names = list(adata_for_names.var_names)
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    
    print(f"[INFO] Saved per-step expressions ({args.steps + 1} steps, "
          f"{X_true_matrix.shape[0]} spots x {X_true_matrix.shape[1]} genes) to {out_dir}")
    
    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()

