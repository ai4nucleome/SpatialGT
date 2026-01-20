#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialGT Perturbation Evaluation for Mouse Stroke Data

This script evaluates perturbation effects using SpatialGT's dual-line inference
with GAT error cancellation.

Algorithm (formerly V4):
1. Reconstruct raw data: X_0 = SpatialGT(raw_data)
2. Apply perturbation on reconstructed data: X'_0 = perturb(X_0)
3. For each step k:
   - X_k = SpatialGT(X_0) -- unperturbed reference line
   - X'_k_raw = SpatialGT(X'_{k-1}) -- perturbed line
   - delta_k = X'_k_raw - X_k -- net perturbation effect (GAT error cancelled)
   - X'_k = X'_0 + delta_k -- accumulate delta on perturbed base
   - Freeze perturbed spots

Default settings:
- Perturb ICA region, evaluate on ICA/PIA_P/PIA_D
- 10 iteration steps
- Seed: 42
- Metrics: PCC, Spearman, Cos, L1, L2
- Save expression for all steps

Usage:
    python spatialgt_perturb_eval.py \\
        --ckpt <checkpoint_path> \\
        --cache_sham <sham_cache> \\
        --cache_pt <pt_cache> \\
        --deg_csv <deg_file> \\
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

# Add project root to path
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_PRETRAIN_DIR = _PROJECT_ROOT / "pretrain"

# Import local config FIRST (before adding pretrain to path)
sys.path.insert(0, str(_SCRIPT_DIR))
from Config import PerturbationConfig

# Add pretrain to path
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PRETRAIN_DIR))

# Import from pretrain module
from pretrain.spatial_databank import SpatialDataBank
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils_train import process_batch, forward_pass


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


def get_dataset_start_idx(cache_dir: Path, dataset_name: str) -> int:
    """Get dataset start index from metadata."""
    meta = load_json(cache_dir / "metadata.json")
    datasets = meta.get("datasets", [])
    for i, ds in enumerate(datasets):
        if ds.get("name") == dataset_name:
            for info in meta.get("dataset_indices", []):
                if info.get("dataset_idx") == i:
                    return int(info.get("start_idx", 0))
    return 0


def load_state_dict_into_model(model: nn.Module, ckpt_dir: Path) -> None:
    """Load model weights from checkpoint directory."""
    for filename in ["finetuned_model.pth", "finetuned_state_dict.pth"]:
        pth = ckpt_dir / filename
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


def barcodes_to_global_idx(cache_dir: Path, dataset_name: str, barcodes: List[str]) -> List[int]:
    """Convert barcodes to global indices."""
    import anndata as ad
    start_idx = get_dataset_start_idx(cache_dir, dataset_name)
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    obs = list(proc.obs_names)
    pos = {b: i for i, b in enumerate(obs)}
    return [int(start_idx + pos[b]) for b in barcodes if b in pos]


def global_idx_to_barcode(cache_dir: Path, dataset_name: str, global_indices: List[int]) -> List[str]:
    """Convert global indices to barcodes."""
    import anndata as ad
    start_idx = get_dataset_start_idx(cache_dir, dataset_name)
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    obs = list(proc.obs_names)
    out = []
    for g in global_indices:
        local = int(g) - int(start_idx)
        out.append(str(obs[local]) if 0 <= local < len(obs) else f"unknown_{g}")
    return out


# ==============================================================================
# Data Loading
# ==============================================================================

def build_data_loader(
    databank: SpatialDataBank,
    config: PerturbationConfig,
    batch_size: int,
    num_workers: int,
):
    """Build data loader for full-slice inference."""
    return databank.get_data_loader(
        split="train",
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        stratify_by_batch=False,
        validation_split=0.0,
        use_distributed=False,
        persistent_workers=False,
        prefetch_factor=config.prefetch_factor if num_workers > 0 else 2,
        is_training=False,
    )


def setup_config_for_cache(
    config: PerturbationConfig,
    cache_dir: str,
    dataset_name: str,
    cache_mode: str = "h5",
    lmdb_path: Optional[str] = None,
    lmdb_manifest_path: Optional[str] = None,
) -> PerturbationConfig:
    """Configure for cache mode."""
    config.cache_dir = str(cache_dir)
    config.cache_mode = cache_mode.lower()
    config.strict_cache_only = True
    
    if cache_mode.lower() == "lmdb":
        if lmdb_path:
            config.lmdb_path = str(lmdb_path)
        if lmdb_manifest_path:
            config.lmdb_manifest_path = str(lmdb_manifest_path)
    
    return config


# ==============================================================================
# Spot Selection
# ==============================================================================

def select_patch_spots(
    databank: SpatialDataBank,
    roi_global_indices: List[int],
    patch_size: int,
    seed: int,
) -> List[int]:
    """Select a contiguous patch of spots via BFS."""
    rng = random.Random(seed)
    roi_set = set(int(x) for x in roi_global_indices)
    if not roi_set:
        return []
    
    seed_g = rng.choice(list(roi_set))
    
    visited = {seed_g}
    queue = deque([seed_g])
    selected = [seed_g]
    
    while queue and len(selected) < patch_size:
        curr = queue.popleft()
        try:
            neighbors = databank.get_neighbors_for_spot(int(curr))
        except Exception:
            neighbors = []
        for nb in neighbors:
            nb = int(nb)
            if nb not in visited and nb in roi_set:
                visited.add(nb)
                selected.append(nb)
                queue.append(nb)
                if len(selected) >= patch_size:
                    break
    
    return selected[:patch_size]


def select_random_spots(
    roi_global_indices: List[int],
    n_spots: int,
    seed: int,
) -> List[int]:
    """Randomly sample n_spots from ROI."""
    rng = random.Random(seed)
    roi_list = list(set(int(x) for x in roi_global_indices))
    if len(roi_list) <= n_spots:
        return roi_list
    return rng.sample(roi_list, n_spots)


# ==============================================================================
# Weight Computation
# ==============================================================================

def compute_spot_weights(
    cache_dir: Path,
    dataset_name: str,
    selected_global_indices: List[int],
    weighting: str,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Compute weights for selected spots."""
    import anndata as ad
    start_idx = get_dataset_start_idx(cache_dir, dataset_name)
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    
    coords = []
    valid_indices = []
    for g in selected_global_indices:
        local = int(g) - int(start_idx)
        if 0 <= local < proc.n_obs:
            coords.append(proc.obsm["spatial"][local])
            valid_indices.append(g)
    
    if not coords:
        return np.array([], dtype=np.float32), {"weighting": weighting, "n_spots": 0}
    
    coords = np.asarray(coords, dtype=np.float64)
    center = coords.mean(axis=0)
    
    meta = {"weighting": weighting, "n_spots": len(valid_indices)}
    
    if weighting == "uniform":
        weights = np.ones(len(valid_indices), dtype=np.float32)
    else:  # gaussian
        d2 = ((coords - center) ** 2).sum(axis=1)
        if sigma is None:
            d = np.sqrt(d2)
            sigma = float(np.median(d)) if np.median(d) > 0 else 1.0
        weights = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        meta["sigma"] = float(sigma)
    
    return weights, meta


# ==============================================================================
# DEG-based Perturbation
# ==============================================================================

def apply_deg_perturbation(
    databank: SpatialDataBank,
    selected_global_indices: List[int],
    weights: np.ndarray,
    deg_csv: Path,
    p_adj_thresh: float,
    min_abs_logfc: float,
    logfc_strength: float = 1.0,
    logfc_clip: float = 5.0,
    vocab_path: str = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Apply DEG-based perturbation using logFC fold change.
    Formula: new_val = old_val × 2^(logFC × strength × weight)
    """
    # Load DEG
    deg = pd.read_csv(deg_csv)
    if "avg_logFC" not in deg.columns:
        raise ValueError(f"DEG csv missing column avg_logFC")
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col:
        deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
        deg = deg[np.isfinite(deg[p_col])]
        deg = deg[deg[p_col].astype(float) < float(p_adj_thresh)]
    
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    # Load vocab
    if vocab_path is None:
        vocab_path = PerturbationConfig().vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {k.lower(): int(v) for k, v in vocab_json.items()}
    
    # Build gene_id -> logFC mapping
    gene_logfc: Dict[int, float] = {}
    n_up, n_down = 0, 0
    for _, r in deg.iterrows():
        gid = vocab.get(str(r["gene"]).lower(), None)
        if gid is None:
            continue
        logfc_val = float(r["avg_logFC"])
        logfc_clipped = np.clip(logfc_val, -float(logfc_clip), float(logfc_clip))
        gene_logfc[int(gid)] = logfc_clipped
        if logfc_val > 0:
            n_up += 1
        else:
            n_down += 1
    
    # Apply perturbation
    overrides: Dict[int, Dict[str, np.ndarray]] = {}
    n_hits_list = []
    
    for i, gidx in enumerate(selected_global_indices):
        if i >= len(weights):
            break
        sd = databank.get_spot_data(int(gidx))
        gene_ids = np.asarray(sd["gene_ids"], dtype=np.int64)
        vals = np.asarray(sd["raw_normed_values"], dtype=np.float32).copy()
        wi = float(weights[i])
        
        n_hits = 0
        for j, gid in enumerate(gene_ids):
            gid_int = int(gid)
            if gid_int in gene_logfc:
                logfc = gene_logfc[gid_int]
                fold_change = np.power(2.0, logfc * float(logfc_strength) * wi)
                vals[j] = vals[j] * fold_change
                n_hits += 1
        
        vals = np.maximum(vals, 0.0)
        n_hits_list.append(n_hits)
        overrides[int(gidx)] = {"gene_ids": gene_ids, "raw_normed_values": vals}
    
    meta = {
        "deg_csv": str(deg_csv),
        "p_adj_thresh": float(p_adj_thresh),
        "min_abs_logfc": float(min_abs_logfc),
        "logfc_strength": float(logfc_strength),
        "logfc_clip": float(logfc_clip),
        "n_deg_genes_mapped": int(len(gene_logfc)),
        "n_up_genes": n_up,
        "n_down_genes": n_down,
        "n_hits_mean": float(np.mean(n_hits_list)) if n_hits_list else 0,
    }
    
    return overrides, meta


# ==============================================================================
# Single-pass Inference
# ==============================================================================

def run_single_pass_inference(
    databank: SpatialDataBank,
    loader,
    model: nn.Module,
    device: torch.device,
    desc: str = "Inference",
    show_progress: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run a single forward pass on the entire slice."""
    all_overrides: Dict[int, Dict[str, np.ndarray]] = {}
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    iterator = tqdm(loader, desc=desc, disable=not show_progress)
    
    for batch in iterator:
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            continue
        
        centers_global = batch["structure"].get("centers_global_indices", None)
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
        
        for i in range(B):
            gidx = int(centers_global[i])
            all_overrides[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
    
    return all_overrides


# ==============================================================================
# Dual-line Iterative Inference
# ==============================================================================

def run_dual_line_inference(
    databank: SpatialDataBank,
    loader_factory: callable,
    model: nn.Module,
    device: torch.device,
    steps: int,
    frozen_indices: Set[int],
    x0_state: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    show_progress: bool = True,
) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], Dict[int, Dict[str, np.ndarray]]]:
    """
    Run dual-line iterative inference with GAT error cancellation.
    
    Each step k:
    1. X_k = SpatialGT(X_0) -- unperturbed line
    2. X'_k_raw = SpatialGT(X'_{k-1}) -- perturbed line
    3. delta_k = X'_k_raw - X_k
    4. X'_k = X'_0 + delta_k
    5. X'_k(frozen_spots) = X'_0(frozen_spots)
    """
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    
    # Deep copy X'_0
    x0_prime_base: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_prime_state.items():
        x0_prime_base[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    # Current X' state
    x_prime_current: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_prime_state.items():
        x_prime_current[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    for t in range(1, steps + 1):
        print(f"\n[Step {t}/{steps}] Dual-line inference...")
        
        # X line: SpatialGT(X_0)
        print(f"  [X line] Running inference...")
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(x0_state)
        
        x_k_pred: Dict[int, Dict[str, np.ndarray]] = {}
        loader = loader_factory()
        iterator = tqdm(loader, desc=f"X line {t}", disable=not show_progress)
        
        for batch in iterator:
            if isinstance(batch, dict) and batch.get("skip_batch", False):
                continue
            
            centers_global = batch["structure"].get("centers_global_indices", None)
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
            
            for i in range(B):
                gidx = int(centers_global[i])
                x_k_pred[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
        
        # X' line: SpatialGT(X'_{k-1})
        print(f"  [X' line] Running inference...")
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(x_prime_current)
        
        x_prime_k_raw: Dict[int, Dict[str, np.ndarray]] = {}
        loader = loader_factory()
        iterator = tqdm(loader, desc=f"X' line {t}", disable=not show_progress)
        
        for batch in iterator:
            if isinstance(batch, dict) and batch.get("skip_batch", False):
                continue
            
            centers_global = batch["structure"].get("centers_global_indices", None)
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
            
            for i in range(B):
                gidx = int(centers_global[i])
                x_prime_k_raw[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
        
        # Compute delta and X'_k
        print(f"  Computing delta and X'_{t}...")
        x_prime_k: Dict[int, Dict[str, np.ndarray]] = {}
        delta_stats = []
        
        for gidx in x_prime_k_raw.keys():
            gidx = int(gidx)
            
            if gidx in frozen_indices:
                x_prime_k[gidx] = {
                    "gene_ids": x0_prime_base[gidx]["gene_ids"].copy(),
                    "raw_normed_values": x0_prime_base[gidx]["raw_normed_values"].copy(),
                }
            else:
                gene_ids_pred = x_prime_k_raw[gidx]["gene_ids"]
                x_prime_raw_vals = x_prime_k_raw[gidx]["raw_normed_values"]
                
                if gidx in x_k_pred:
                    x_k_vals = x_k_pred[gidx]["raw_normed_values"]
                else:
                    x_k_vals = np.zeros_like(x_prime_raw_vals)
                
                delta = x_prime_raw_vals - x_k_vals
                
                # Map delta to X'_0's gene space
                gid_to_delta = {int(gid): delta[i] for i, gid in enumerate(gene_ids_pred)}
                
                if gidx in x0_prime_base:
                    x0_prime_gene_ids = x0_prime_base[gidx]["gene_ids"]
                    x0_prime_vals = x0_prime_base[gidx]["raw_normed_values"].copy()
                    
                    for i, gid in enumerate(x0_prime_gene_ids):
                        if int(gid) in gid_to_delta:
                            x0_prime_vals[i] += gid_to_delta[int(gid)]
                    
                    x_prime_k[gidx] = {
                        "gene_ids": x0_prime_gene_ids.copy(),
                        "raw_normed_values": x0_prime_vals.astype(np.float32),
                    }
                else:
                    x_prime_k[gidx] = {
                        "gene_ids": gene_ids_pred.copy(),
                        "raw_normed_values": x_prime_raw_vals.astype(np.float32),
                    }
                
                delta_stats.append(float(np.mean(np.abs(delta))))
        
        if delta_stats:
            print(f"  Delta abs_mean: {np.mean(delta_stats):.6f}")
        
        # Save step results
        step_overrides[t] = {
            int(k): {
                "gene_ids": v["gene_ids"].copy(),
                "raw_normed_values": v["raw_normed_values"].copy(),
            }
            for k, v in x_prime_k.items()
        }
        
        x_prime_current = x_prime_k
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  [OK] Step {t} completed: {len(x_prime_k)} spots")
    
    return step_overrides, x_prime_current


# ==============================================================================
# Metrics Computation
# ==============================================================================

def collect_expr_for_roi(
    overrides: Dict[int, Dict[str, np.ndarray]],
    roi_global: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Collect expression matrix for ROI spots."""
    keep = [int(g) for g in roi_global if int(g) in overrides]
    if not keep:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    
    # Find common genes
    gene_sets = []
    for g in keep:
        gids = np.asarray(overrides[g]["gene_ids"], dtype=np.int64)
        valid_gids = set(int(x) for x in gids.tolist() if int(x) >= 0)
        gene_sets.append(valid_gids)
    
    if not gene_sets:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    
    common_genes = gene_sets[0]
    for gs in gene_sets[1:]:
        common_genes &= gs
    
    if not common_genes:
        return np.zeros((len(keep), 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), keep
    
    common_gene_ids = np.array(sorted(common_genes), dtype=np.int64)
    
    # Build aligned expression matrix
    X = np.zeros((len(keep), len(common_gene_ids)), dtype=np.float32)
    for i, g in enumerate(keep):
        gids = np.asarray(overrides[g]["gene_ids"], dtype=np.int64)
        vals = np.asarray(overrides[g]["raw_normed_values"], dtype=np.float32)
        pos = {int(gid): idx for idx, gid in enumerate(gids.tolist())}
        for j, cg in enumerate(common_gene_ids.tolist()):
            if int(cg) in pos:
                X[i, j] = vals[pos[int(cg)]]
    
    return X, common_gene_ids, keep


def find_common_genes(gene_ids_list: List[np.ndarray]) -> np.ndarray:
    """Find intersection of gene_ids."""
    if not gene_ids_list:
        return np.zeros((0,), dtype=np.int64)
    
    common = set(int(g) for g in gene_ids_list[0].tolist() if int(g) >= 0)
    for gids in gene_ids_list[1:]:
        common &= set(int(g) for g in gids.tolist() if int(g) >= 0)
    
    return np.array(sorted(common), dtype=np.int64)


def align_and_compute_mean_vector(
    X: np.ndarray,
    gene_ids: np.ndarray,
    common_gene_ids: np.ndarray,
) -> np.ndarray:
    """Align expression matrix to common genes and compute mean vector."""
    if X.size == 0 or gene_ids.size == 0 or common_gene_ids.size == 0:
        return np.zeros(len(common_gene_ids), dtype=np.float32)
    
    pos = {int(g): i for i, g in enumerate(gene_ids.tolist())}
    aligned = np.zeros((X.shape[0], len(common_gene_ids)), dtype=np.float32)
    
    for j, g in enumerate(common_gene_ids.tolist()):
        if int(g) in pos:
            aligned[:, j] = X[:, pos[int(g)]]
    
    return aligned.mean(axis=0).astype(np.float32)


def compute_mean_vector_similarities(vec_a: np.ndarray, vec_b: np.ndarray) -> Dict[str, float]:
    """Compute similarity metrics between two mean vectors."""
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.shape != vec_b.shape:
        return {"pcc": float("nan"), "spearman": float("nan"), "cos": float("nan"), 
                "l1": float("nan"), "l2": float("nan")}
    
    # L1, L2
    l1 = float(np.sum(np.abs(vec_a - vec_b)))
    l2 = float(np.sqrt(np.sum((vec_a - vec_b) ** 2)))
    
    # Cosine
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 1e-8 and norm_b > 1e-8 else float("nan")
    
    # PCC
    a_centered = vec_a - vec_a.mean()
    b_centered = vec_b - vec_b.mean()
    norm_ac, norm_bc = np.linalg.norm(a_centered), np.linalg.norm(b_centered)
    pcc = float(np.dot(a_centered, b_centered) / (norm_ac * norm_bc)) if norm_ac > 1e-8 and norm_bc > 1e-8 else float("nan")
    
    # Spearman
    rank_a = np.argsort(np.argsort(vec_a)).astype(np.float32)
    rank_b = np.argsort(np.argsort(vec_b)).astype(np.float32)
    ra_centered = rank_a - rank_a.mean()
    rb_centered = rank_b - rank_b.mean()
    norm_ra, norm_rb = np.linalg.norm(ra_centered), np.linalg.norm(rb_centered)
    spearman = float(np.dot(ra_centered, rb_centered) / (norm_ra * norm_rb)) if norm_ra > 1e-8 and norm_rb > 1e-8 else float("nan")
    
    return {"pcc": pcc, "spearman": spearman, "cos": cos, "l1": l1, "l2": l2}


# ==============================================================================
# Expression Saving
# ==============================================================================

def save_expression_csv(
    state: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
    vocab_path: str,
    cache_dir: Path,
    dataset_name: str,
    perturbed_indices: Set[int] = None,
) -> None:
    """Save expression matrix to CSV."""
    import anndata as ad
    
    if not state:
        print(f"[WARN] Empty state, skipping save to {out_path}")
        return
    
    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_gene = {int(v): k for k, v in vocab.items()}
    
    # Get barcodes
    start_idx = get_dataset_start_idx(cache_dir, dataset_name)
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    obs_names = list(proc.obs_names)
    
    # Find all gene_ids
    all_gene_ids: Set[int] = set()
    for gidx, ov in state.items():
        for gid in ov["gene_ids"]:
            if int(gid) >= 0:
                all_gene_ids.add(int(gid))
    
    sorted_gene_ids = sorted(all_gene_ids)
    gene_names = [id_to_gene.get(gid, f"gene_{gid}") for gid in sorted_gene_ids]
    gid_to_col = {gid: i for i, gid in enumerate(sorted_gene_ids)}
    
    # Build matrix
    sorted_gidx = sorted(state.keys())
    n_spots = len(sorted_gidx)
    n_genes = len(sorted_gene_ids)
    
    expr_matrix = np.zeros((n_spots, n_genes), dtype=np.float32)
    barcodes = []
    is_perturbed = []
    
    for row_idx, gidx in enumerate(sorted_gidx):
        local_idx = int(gidx) - int(start_idx)
        barcodes.append(obs_names[local_idx] if 0 <= local_idx < len(obs_names) else f"spot_{gidx}")
        is_perturbed.append(1 if perturbed_indices and gidx in perturbed_indices else 0)
        
        ov = state[gidx]
        for j, gid in enumerate(ov["gene_ids"]):
            gid_int = int(gid)
            if gid_int >= 0 and gid_int in gid_to_col:
                expr_matrix[row_idx, gid_to_col[gid_int]] = ov["raw_normed_values"][j]
    
    # Create DataFrame
    df = pd.DataFrame(expr_matrix, index=barcodes, columns=gene_names)
    df.index.name = "barcode"
    df["is_perturbed"] = is_perturbed
    df.to_csv(out_path)
    print(f"[OK] Saved: {out_path} ({n_spots} spots × {n_genes} genes)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialGT Perturbation Evaluation")
    
    # Model paths
    parser.add_argument("--ckpt", required=True, help="Checkpoint for both Sham and PT models")
    parser.add_argument("--ckpt_pt", default=None, help="Separate checkpoint for PT (optional)")
    
    # Data paths
    parser.add_argument("--cache_sham", required=True, help="Cache directory for Sham slice")
    parser.add_argument("--cache_pt", required=True, help="Cache directory for PT slice")
    parser.add_argument("--sham_dataset_name", default="Sham1-1")
    parser.add_argument("--pt_dataset_name", default="PT1-1")
    parser.add_argument("--roi_manifest", required=True, help="JSON manifest with ROI barcodes")
    parser.add_argument("--deg_csv", required=True, help="DEG CSV file")
    
    # Cache mode
    parser.add_argument("--cache_mode", choices=["h5", "lmdb"], default="h5")
    parser.add_argument("--lmdb_path_sham", type=str, default=None)
    parser.add_argument("--lmdb_manifest_sham", type=str, default=None)
    parser.add_argument("--lmdb_path_pt", type=str, default=None)
    parser.add_argument("--lmdb_manifest_pt", type=str, default=None)
    
    # Perturbation config
    parser.add_argument("--patch_size", type=int, default=20, help="Number of spots to perturb (for patch mode)")
    parser.add_argument("--n_spots", type=int, default=20, help="Number of spots to perturb (for random mode)")
    parser.add_argument("--perturb_mode", choices=["patch", "random"], default="patch", help="Perturbation mode")
    parser.add_argument("--weighting", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--p_adj_thresh", type=float, default=0.1)
    parser.add_argument("--min_abs_logfc", type=float, default=0.0)
    parser.add_argument("--logfc_strength", type=float, default=1.0)
    parser.add_argument("--logfc_clip", type=float, default=5.0)
    
    # Iteration
    parser.add_argument("--steps", type=int, default=10, help="Number of iteration steps (default: 10)")
    
    # Performance
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--show_progress", action="store_true", default=True)
    
    # Output
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Setup
    fix_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load manifest
    manifest = load_json(Path(args.roi_manifest))
    pt_key = "PT1_1" if "PT1_1" in manifest else "pt"
    sham_key = "Sham1_1" if "Sham1_1" in manifest else "sham"
    
    sham_roi_barcodes = {k: manifest[sham_key][k]["barcodes"] for k in ["ICA", "PIA_P", "PIA_D"]}
    pt_roi_barcodes = {k: manifest[pt_key][k]["barcodes"] for k in ["ICA", "PIA_P", "PIA_D"]}
    
    # Config and model
    config = PerturbationConfig()
    
    model = SpatialNeighborTransformer(config).to(device)
    load_state_dict_into_model(model, Path(args.ckpt))
    model.eval()
    print(f"[INFO] Loaded model from: {args.ckpt}")
    
    # PT model (use separate or same)
    if args.ckpt_pt:
        model_pt = SpatialNeighborTransformer(config).to(device)
        load_state_dict_into_model(model_pt, Path(args.ckpt_pt))
        model_pt.eval()
        print(f"[INFO] Loaded PT model from: {args.ckpt_pt}")
    else:
        model_pt = model
    
    # Setup Sham databank
    sham_config = PerturbationConfig()
    sham_config = setup_config_for_cache(
        sham_config, args.cache_sham, args.sham_dataset_name,
        args.cache_mode, args.lmdb_path_sham, args.lmdb_manifest_sham
    )
    
    sham_bank = SpatialDataBank(
        dataset_paths=[str(Path(args.cache_sham) / args.sham_dataset_name / "processed.h5ad")],
        cache_dir=str(Path(args.cache_sham)),
        config=sham_config,
        force_rebuild=False,
    )
    
    sham_roi_global = {k: barcodes_to_global_idx(Path(args.cache_sham), args.sham_dataset_name, sham_roi_barcodes[k])
                       for k in ["ICA", "PIA_P", "PIA_D"]}
    
    # Setup PT databank
    pt_config = PerturbationConfig()
    pt_config = setup_config_for_cache(
        pt_config, args.cache_pt, args.pt_dataset_name,
        args.cache_mode, args.lmdb_path_pt, args.lmdb_manifest_pt
    )
    
    pt_bank = SpatialDataBank(
        dataset_paths=[str(Path(args.cache_pt) / args.pt_dataset_name / "processed.h5ad")],
        cache_dir=str(Path(args.cache_pt)),
        config=pt_config,
        force_rebuild=False,
    )
    
    pt_roi_global = {k: barcodes_to_global_idx(Path(args.cache_pt), args.pt_dataset_name, pt_roi_barcodes[k])
                     for k in ["ICA", "PIA_P", "PIA_D"]}
    
    print(f"[INFO] Sham ROIs: ICA={len(sham_roi_global['ICA'])}, PIA_P={len(sham_roi_global['PIA_P'])}, PIA_D={len(sham_roi_global['PIA_D'])}")
    print(f"[INFO] PT ROIs: ICA={len(pt_roi_global['ICA'])}, PIA_P={len(pt_roi_global['PIA_P'])}, PIA_D={len(pt_roi_global['PIA_D'])}")
    
    # Step 1: Reconstruct PT reference
    print("\n[Step 1] Computing PT reference expression...")
    pt_loader = build_data_loader(pt_bank, pt_config, args.batch_size, args.num_workers)
    pt_recon = run_single_pass_inference(pt_bank, pt_loader, model_pt, device, "PT recon", args.show_progress)
    
    pt_expr_by_roi = {}
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        pt_expr_by_roi[roi] = collect_expr_for_roi(pt_recon, pt_roi_global[roi])
    
    # Step 2: Reconstruct Sham control
    print("\n[Step 2] Computing Sham control expression...")
    sham_bank.clear_runtime_spot_overrides()
    sham_loader = build_data_loader(sham_bank, sham_config, args.batch_size, args.num_workers)
    sham_ctrl_recon = run_single_pass_inference(sham_bank, sham_loader, model, device, "Sham ctrl", args.show_progress)
    
    sham_ctrl_expr_by_roi = {}
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        sham_ctrl_expr_by_roi[roi] = collect_expr_for_roi(sham_ctrl_recon, sham_roi_global[roi])
    
    # Step 3: Select ICA spots for perturbation
    print("\n[Step 3] Selecting ICA spots for perturbation...")
    if args.perturb_mode == "random":
        selected_global = select_random_spots(sham_roi_global["ICA"], args.n_spots, args.seed)
        print(f"[INFO] Selected {len(selected_global)} random spots in ICA for perturbation")
    else:
        selected_global = select_patch_spots(sham_bank, sham_roi_global["ICA"], args.patch_size, args.seed)
        print(f"[INFO] Selected {len(selected_global)} patch spots in ICA for perturbation")
    
    # Step 4: Compute weights and apply perturbation
    print("\n[Step 4] Applying DEG-based perturbation...")
    weights, weight_meta = compute_spot_weights(
        Path(args.cache_sham), args.sham_dataset_name, selected_global,
        args.weighting, args.sigma
    )
    
    # Build X_0 = Sham reconstructed
    x0_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in sham_ctrl_recon.items():
        x0_state[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    # Apply perturbation on X_0 to get X'_0
    perturb_overrides, perturb_meta = apply_deg_perturbation(
        sham_bank, selected_global, weights, Path(args.deg_csv),
        args.p_adj_thresh, args.min_abs_logfc, args.logfc_strength, args.logfc_clip,
        config.vocab_file
    )
    
    # Re-apply perturbation on reconstructed X_0
    x0_prime_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_state.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # Load DEG info for re-applying
    deg = pd.read_csv(args.deg_csv)
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else None
    if p_col:
        deg = deg[pd.to_numeric(deg[p_col], errors="coerce") < args.p_adj_thresh]
    deg = deg[deg["avg_logFC"].abs() >= args.min_abs_logfc]
    
    with open(config.vocab_file, "r") as f:
        vocab = {k.lower(): int(v) for k, v in json.load(f).items()}
    
    gene_logfc = {}
    for _, r in deg.iterrows():
        gid = vocab.get(str(r["gene"]).lower())
        if gid:
            gene_logfc[int(gid)] = np.clip(float(r["avg_logFC"]), -args.logfc_clip, args.logfc_clip)
    
    for i, gidx in enumerate(selected_global):
        gidx = int(gidx)
        if gidx not in x0_state:
            continue
        
        gene_ids = x0_state[gidx]["gene_ids"]
        vals = x0_state[gidx]["raw_normed_values"].copy()
        wi = float(weights[i]) if i < len(weights) else 1.0
        
        for j, gid in enumerate(gene_ids):
            if int(gid) in gene_logfc:
                logfc = gene_logfc[int(gid)]
                vals[j] = vals[j] * np.power(2.0, logfc * args.logfc_strength * wi)
        
        x0_prime_state[gidx] = {"gene_ids": gene_ids.copy(), "raw_normed_values": np.maximum(vals, 0.0)}
    
    frozen_indices = set(int(x) for x in selected_global)
    print(f"[INFO] Perturbed {len(frozen_indices)} spots, {perturb_meta['n_deg_genes_mapped']} DEG genes mapped")
    
    # Save manifest
    manifest_data = {
        "perturb_target_roi": "ICA",
        "perturb_mode": args.perturb_mode,
        "n_spots_perturbed": len(selected_global),
        "seed": args.seed,
        "steps": args.steps,
        "selected_barcodes": global_idx_to_barcode(Path(args.cache_sham), args.sham_dataset_name, selected_global),
        **weight_meta,
        **perturb_meta,
        "ckpt": str(args.ckpt),
    }
    with open(out_dir / "perturb_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    # Collect pert0 expression
    sham_bank.clear_runtime_spot_overrides()
    sham_bank.set_runtime_spot_overrides(x0_prime_state)
    
    sham_pert0_expr_by_roi = {}
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        sham_pert0_expr_by_roi[roi] = collect_expr_for_roi(x0_prime_state, sham_roi_global[roi])
    
    # Step 5: Run dual-line iterative inference
    print(f"\n[Step 5] Running {args.steps}-step dual-line inference...")
    
    def sham_loader_factory():
        return build_data_loader(sham_bank, sham_config, args.batch_size, args.num_workers)
    
    step_overrides, final_state = run_dual_line_inference(
        sham_bank, sham_loader_factory, model, device,
        steps=args.steps,
        frozen_indices=frozen_indices,
        x0_state=x0_state,
        x0_prime_state=x0_prime_state,
        show_progress=args.show_progress,
    )
    
    # Step 6: Save expression for all steps
    print("\n[Step 6] Saving expression data...")
    expr_dir = out_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)
    
    # Save step 0 (X'_0)
    save_expression_csv(
        x0_prime_state, expr_dir / "pert0_expression.csv",
        config.vocab_file, Path(args.cache_sham), args.sham_dataset_name, frozen_indices
    )
    
    # Save steps 1..T
    for t in range(1, args.steps + 1):
        if t in step_overrides:
            save_expression_csv(
                step_overrides[t], expr_dir / f"pert{t}_expression.csv",
                config.vocab_file, Path(args.cache_sham), args.sham_dataset_name, frozen_indices
            )
    
    # Step 7: Compute similarities
    print("\n[Step 7] Computing similarities...")
    results_rows = []
    eval_rois = ["ICA", "PIA_P", "PIA_D"]
    
    # Step 0: ctrl0 and pert0
    for roi in eval_rois:
        pt_X, pt_gids, _ = pt_expr_by_roi[roi]
        ctrl_X, ctrl_gids, _ = sham_ctrl_expr_by_roi[roi]
        pert0_X, pert0_gids, _ = sham_pert0_expr_by_roi[roi]
        
        common_genes = find_common_genes([pt_gids, ctrl_gids, pert0_gids])
        if common_genes.size == 0:
            continue
        
        pt_mean = align_and_compute_mean_vector(pt_X, pt_gids, common_genes)
        ctrl_mean = align_and_compute_mean_vector(ctrl_X, ctrl_gids, common_genes)
        pert0_mean = align_and_compute_mean_vector(pert0_X, pert0_gids, common_genes)
        
        ctrl0_sim = compute_mean_vector_similarities(pt_mean, ctrl_mean)
        pert0_sim = compute_mean_vector_similarities(pt_mean, pert0_mean)
        
        for metric in ["pcc", "spearman", "cos", "l1", "l2"]:
            results_rows.append({
                "roi": roi, "step": 0, "group": "ctrl0", "metric": metric,
                "value": ctrl0_sim[metric], "n_genes": int(len(common_genes)),
            })
            results_rows.append({
                "roi": roi, "step": 0, "group": "pert0", "metric": metric,
                "value": pert0_sim[metric], "n_genes": int(len(common_genes)),
            })
    
    # Steps 1..T
    for t in range(1, args.steps + 1):
        step_ov = step_overrides.get(t, {})
        
        for roi in eval_rois:
            pt_X, pt_gids, _ = pt_expr_by_roi[roi]
            pert_X, pert_gids, _ = collect_expr_for_roi(step_ov, sham_roi_global[roi])
            
            if pert_X.size == 0:
                continue
            
            common_genes = find_common_genes([pt_gids, pert_gids])
            if common_genes.size == 0:
                continue
            
            pt_mean = align_and_compute_mean_vector(pt_X, pt_gids, common_genes)
            pert_mean = align_and_compute_mean_vector(pert_X, pert_gids, common_genes)
            
            pert_sim = compute_mean_vector_similarities(pt_mean, pert_mean)
            
            for metric in ["pcc", "spearman", "cos", "l1", "l2"]:
                results_rows.append({
                    "roi": roi, "step": t, "group": f"pert{t}", "metric": metric,
                    "value": pert_sim[metric], "n_genes": int(len(common_genes)),
                })
    
    # Save results
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"rows": results_rows}, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (PCC with PT reference)")
    print("=" * 60)
    
    for roi in eval_rois:
        print(f"\n[ROI: {roi}]")
        roi_df = df[(df["roi"] == roi) & (df["metric"] == "pcc")]
        for _, r in roi_df.iterrows():
            print(f"  step={r['step']:2d} {r['group']:8s} PCC={r['value']:.4f}")
    
    print(f"\n[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
