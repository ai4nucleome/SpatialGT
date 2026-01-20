#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialGT Virtual Treatment Perturbation for Human Colitis (UC/UC_VDZ).

Simulates VDZ treatment on UC patients by applying DEG-based perturbation
on MNP and/or Fibroblast cells and evaluating treatment effects through
activation status changes.

Default settings:
- Sample: HS5_UC_R_0
- Perturbation target: MNP_activated
- Mode: recon (V4) - uses reconstructed X_0
- Saves all step expression values

Outputs:
- activation_eval.csv: activated cell counts, scores, and rates per step
- expression/: full expression matrices for each step (pert0, pert1, ..., pertN)
- perturb_manifest.json: reproducibility manifest
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import anndata as ad
import scipy.sparse as sp

# Ensure repo root importability
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrain.Config import Config
from pretrain.spatial_databank import SpatialDataBank
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils_train import process_batch, forward_pass


# ==============================================================================
# Constants and Configuration
# ==============================================================================

# Data paths - Override with environment variable COLITIS_DATA_ROOT or pass via command line
DATA_ROOT = Path(os.environ.get("COLITIS_DATA_ROOT", _REPO_ROOT / "example_data" / "human_colitis"))
LMDB_ROOT = DATA_ROOT / "lmdb"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed"
RAW_DATA_ROOT = DATA_ROOT / "raw_data"

# Default model path
DEFAULT_MODEL_ROOT = _REPO_ROOT / "perturbation" / "human_colitis" / "model"

# DEG and threshold directories
DEFAULT_DEG_DIR = _SCRIPT_DIR / "DEG"
DEFAULT_THRESHOLD_DIR = _SCRIPT_DIR / "thresholds"

# Marker genes for activation scoring
MARKER_GENES = {
    "MNP": ["S100A4", "TIMP1", "S100A9", "CD80", "ITGAX", "LYZ", "IL1B"],
    "Fibroblast": ["TIMP1", "IL1R1", "CXCL14", "CD44"],
}

# Cell type labels for activation status
CELLTYPE_LABELS = {
    "MNP": {
        "all": ["05-MNP", "05B-MNP_activated_inf"],
        "activated": ["05B-MNP_activated_inf"],
        "non_activated": ["05-MNP"],
    },
    "MNP_activated": {
        "all": ["05B-MNP_activated_inf"],
        "activated": ["05B-MNP_activated_inf"],
        "non_activated": [],
    },
    "MNP_normal": {
        "all": ["05-MNP"],
        "activated": [],
        "non_activated": ["05-MNP"],
    },
    "Fibroblast": {
        "all": ["03A-Myofibroblast", "03B-Fibroblast", "03C-activated_fibroblast"],
        "activated": ["03C-activated_fibroblast"],
        "non_activated": ["03A-Myofibroblast", "03B-Fibroblast"],
    },
    "Fibroblast_activated": {
        "all": ["03C-activated_fibroblast"],
        "activated": ["03C-activated_fibroblast"],
        "non_activated": [],
    },
    "Fibroblast_normal": {
        "all": ["03A-Myofibroblast", "03B-Fibroblast"],
        "activated": [],
        "non_activated": ["03A-Myofibroblast", "03B-Fibroblast"],
    },
}

# Default activation thresholds
DEFAULT_THRESHOLDS = {
    "MNP": {"optimal_threshold": 1.55},
    "Fibroblast": {"optimal_threshold": 1.93},
}


# ==============================================================================
# Utility Functions
# ==============================================================================

def _load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_vocab(vocab_path: str = None) -> Dict[str, int]:
    """Load gene vocabulary."""
    if vocab_path is None:
        vocab_path = str(_REPO_ROOT / "gene_embedding" / "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {k.lower(): int(v) for k, v in vocab.items()}


def _load_state_dict_into_model(model: SpatialNeighborTransformer, ckpt_dir: Path) -> None:
    """Load model weights from checkpoint directory."""
    # Try finetuned model first
    pth = ckpt_dir / "finetuned_model.pth"
    if pth.exists():
        sd = torch.load(str(pth), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        return
    
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


def _build_all_loader(
    databank: SpatialDataBank,
    config: Config,
    batch_size: int,
    num_workers: int = 4,
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
        persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else 2,
        is_training=False,
    )
    return loader


# ==============================================================================
# DEG Loading and Perturbation Vector
# ==============================================================================

def load_deg_for_treatment(
    celltype: str,
    deg_dir: Path,
    p_thresh: float = 0.1,
    min_abs_logfc: float = 0.0,
    custom_deg_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load DEG file for perturbation.
    
    Args:
        celltype: "MNP", "Fibroblast", "MNP_activated", etc.
        deg_dir: directory containing DEG files
        p_thresh: p-value threshold for significance
        min_abs_logfc: minimum absolute logFC
        custom_deg_file: path to custom DEG file
    
    Returns:
        DataFrame with gene, logFC columns
    """
    if custom_deg_file:
        deg_file = custom_deg_file
    else:
        # Map subtypes to parent type for DEG file lookup
        celltype_for_deg = celltype
        if celltype in ["MNP_activated", "MNP_normal"]:
            celltype_for_deg = "MNP"
        elif celltype in ["Fibroblast_activated", "Fibroblast_normal"]:
            celltype_for_deg = "Fibroblast"
        
        # Try different naming conventions
        deg_file = deg_dir / f"UC_VDZ_vs_UC_{celltype_for_deg}_DEG.csv"
        if not deg_file.exists():
            deg_file = deg_dir / f"{celltype_for_deg}_DEG.csv"
    
    if not Path(deg_file).exists():
        raise FileNotFoundError(f"DEG file not found: {deg_file}")
    
    deg = pd.read_csv(deg_file)
    
    # Standardize column names
    if "avg_logFC" not in deg.columns and "logFC" in deg.columns:
        deg["avg_logFC"] = deg["logFC"]
    
    # Filter by significance
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else "p_val"
    if p_col in deg.columns:
        deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
        deg = deg[np.isfinite(deg[p_col])]
        deg = deg[deg[p_col].astype(float) < float(p_thresh)]
    
    if min_abs_logfc > 0:
        deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    print(f"[DEG] Loaded {len(deg)} genes for {celltype}")
    print(f"      Up-regulated: {(deg['avg_logFC'] > 0).sum()}, Down-regulated: {(deg['avg_logFC'] < 0).sum()}")
    
    return deg


def build_perturbation_vector(
    deg: pd.DataFrame,
    vocab: Dict[str, int],
    logfc_clip: float = 5.0,
) -> Dict[int, float]:
    """Build perturbation vector: gene_id -> logFC."""
    gene_logfc: Dict[int, float] = {}
    n_mapped = 0
    
    for _, row in deg.iterrows():
        gene_name = str(row["gene"]).lower()
        gid = vocab.get(gene_name, None)
        if gid is not None:
            logfc = float(row["avg_logFC"])
            logfc_clipped = np.clip(logfc, -logfc_clip, logfc_clip)
            gene_logfc[int(gid)] = logfc_clipped
            n_mapped += 1
    
    print(f"      Mapped {n_mapped} genes to vocab")
    return gene_logfc


# ==============================================================================
# Cell Type Information
# ==============================================================================

def load_sample_cell_info(
    sample_name: str,
    data_root: Path = DATA_ROOT,
    annotation_col: str = "fine_annotation",
) -> pd.DataFrame:
    """Load cell type annotation for a sample."""
    # Try raw data first
    h5ad_path = data_root / "raw_data" / sample_name / f"{sample_name}.h5ad"
    if not h5ad_path.exists():
        h5ad_path = data_root / "preprocessed_v3" / sample_name / sample_name / "processed.h5ad"
    
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found for {sample_name}")
    
    adata = ad.read_h5ad(h5ad_path)
    
    cell_info = pd.DataFrame({
        "cell_id": adata.obs_names,
        "cell_type": adata.obs[annotation_col].values if annotation_col in adata.obs else "unknown",
        "local_idx": np.arange(adata.n_obs),
    })
    
    return cell_info


def get_celltype_indices(
    cell_info: pd.DataFrame,
    celltype: str,
    include_all: bool = True,
) -> List[int]:
    """Get local indices of cells belonging to specified cell type."""
    labels = CELLTYPE_LABELS.get(celltype, {})
    if include_all:
        target_types = labels.get("all", [])
    else:
        target_types = labels.get("activated", []) + labels.get("non_activated", [])
    
    mask = cell_info["cell_type"].isin(target_types)
    return cell_info.loc[mask, "local_idx"].tolist()


# ==============================================================================
# Spot Selection and Weights
# ==============================================================================

def select_random_spots(
    cell_indices: List[int],
    n_spots: int,
    seed: int,
) -> List[int]:
    """Randomly sample spots from cell indices."""
    rng = random.Random(seed)
    if len(cell_indices) <= n_spots:
        return list(cell_indices)
    return rng.sample(cell_indices, n_spots)


def compute_spot_weights(
    databank: SpatialDataBank,
    selected_indices: List[int],
    weighting: str = "uniform",
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Compute weights for selected spots."""
    if len(selected_indices) == 0:
        return np.array([], dtype=np.float32), {"weighting": weighting, "n_spots": 0}
    
    # Get spatial coordinates
    coords = []
    for idx in selected_indices:
        try:
            sd = databank.get_spot_data(int(idx))
            if "spatial" in sd:
                coords.append(sd["spatial"])
            else:
                coords.append([0.0, 0.0])
        except Exception:
            coords.append([0.0, 0.0])
    
    coords = np.array(coords, dtype=np.float64)
    center = coords.mean(axis=0)
    
    meta = {
        "weighting": weighting,
        "n_spots": len(selected_indices),
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
# Perturbation Application
# ==============================================================================

def apply_perturbation(
    databank: SpatialDataBank,
    selected_indices: List[int],
    weights: np.ndarray,
    gene_logfc: Dict[int, float],
    dose_lambda: float = 1.0,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Apply DEG-based perturbation to selected spots.
    
    Formula: new_val = old_val × 2^(logFC × lambda × weight)
    """
    overrides: Dict[int, Dict[str, np.ndarray]] = {}
    n_hits_per_spot = []
    
    for i, gidx in enumerate(selected_indices):
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
                fold_change = np.power(2.0, logfc * dose_lambda * wi)
                vals[j] = vals[j] * fold_change
                n_hits += 1
        
        vals = np.maximum(vals, 0.0)
        n_hits_per_spot.append(n_hits)
        
        overrides[int(gidx)] = {"gene_ids": gene_ids, "raw_normed_values": vals}
    
    meta = {
        "dose_lambda": dose_lambda,
        "n_spots_perturbed": len(selected_indices),
        "n_genes_in_perturbation": len(gene_logfc),
        "n_hits_per_spot_mean": float(np.mean(n_hits_per_spot)) if n_hits_per_spot else 0,
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
        else:
            centers_global = None
        
        batch_data = process_batch(batch, device, config=databank.config)
        
        with torch.no_grad():
            preds, _, _, _ = forward_pass(model, batch_data, config=databank.config)
        
        B = len(centers_global) if centers_global is not None else preds.shape[0]
        center_gene_ids = batch_data["genes"][:B]
        center_pad_mask = batch_data["padding_attention_mask"][:B].to(torch.bool)
        
        gathered = preds.gather(1, center_gene_ids.clamp(min=0))
        gathered = gathered * center_pad_mask.to(gathered.dtype)
        
        gid_np = center_gene_ids.detach().cpu().numpy().astype(np.int64)
        val_np = gathered.detach().float().cpu().numpy().astype(np.float32)
        
        if centers_global is None:
            centers_global = list(range(B))
        
        for i in range(B):
            gidx = int(centers_global[i])
            all_overrides[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
    
    return all_overrides


# ==============================================================================
# Dual-line Iterative Inference with Error Cancellation
# ==============================================================================

def run_dual_line_iterative_inference(
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
    
    Algorithm:
    - X line (unperturbed): always starts from X_0, used for error calculation
    - X' line (perturbed): starts from previous step's true result
    
    Each step k:
    1. X_k = SpatialGT(X_0)
    2. X'_k_raw = SpatialGT(X'_{k-1}_true)
    3. delta_k = X'_k_raw - X_k  (error cancelled)
    4. X'_k_true = X'_0 + delta_k
    5. X'_k_true(perturb_spots) = X'_0(perturb_spots)  (frozen)
    """
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    
    # Deep copy X'_0 as the base
    x0_prime_base: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_prime_state.items():
        x0_prime_base[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    # Current state for X' line
    x_prime_current: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_prime_state.items():
        x_prime_current[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    for t in range(1, steps + 1):
        print(f"\n[STEP {t}/{steps}] Dual-line inference...")
        
        # Step 1: X_k = SpatialGT(X_0)
        print(f"  [X line] Running SpatialGT on X_0...")
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(x0_state)
        
        x_k_pred: Dict[int, Dict[str, np.ndarray]] = {}
        loader = loader_factory()
        iterator = tqdm(loader, desc=f"X line step {t}", disable=not show_progress)
        
        for batch in iterator:
            if isinstance(batch, dict) and batch.get("skip_batch", False):
                continue
            
            centers_global = batch["structure"].get("centers_global_indices", None)
            if isinstance(centers_global, torch.Tensor):
                centers_global = centers_global.cpu().numpy().astype(int).tolist()
            
            batch_data = process_batch(batch, device, config=databank.config)
            
            with torch.no_grad():
                preds, _, _, _ = forward_pass(model, batch_data, config=databank.config)
            
            B = len(centers_global) if centers_global is not None else preds.shape[0]
            center_gene_ids = batch_data["genes"][:B]
            center_pad_mask = batch_data["padding_attention_mask"][:B].to(torch.bool)
            
            gathered = preds.gather(1, center_gene_ids.clamp(min=0))
            gathered = gathered * center_pad_mask.to(gathered.dtype)
            
            gid_np = center_gene_ids.detach().cpu().numpy().astype(np.int64)
            val_np = gathered.detach().float().cpu().numpy().astype(np.float32)
            
            if centers_global is None:
                centers_global = list(range(B))
            
            for i in range(B):
                gidx = int(centers_global[i])
                x_k_pred[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
        
        # Step 2: X'_k_raw = SpatialGT(X'_{k-1}_true)
        print(f"  [X' line] Running SpatialGT on X'_{t-1}_true...")
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(x_prime_current)
        
        x_prime_k_raw: Dict[int, Dict[str, np.ndarray]] = {}
        loader = loader_factory()
        iterator = tqdm(loader, desc=f"X' line step {t}", disable=not show_progress)
        
        for batch in iterator:
            if isinstance(batch, dict) and batch.get("skip_batch", False):
                continue
            
            centers_global = batch["structure"].get("centers_global_indices", None)
            if isinstance(centers_global, torch.Tensor):
                centers_global = centers_global.cpu().numpy().astype(int).tolist()
            
            batch_data = process_batch(batch, device, config=databank.config)
            
            with torch.no_grad():
                preds, _, _, _ = forward_pass(model, batch_data, config=databank.config)
            
            B = len(centers_global) if centers_global is not None else preds.shape[0]
            center_gene_ids = batch_data["genes"][:B]
            center_pad_mask = batch_data["padding_attention_mask"][:B].to(torch.bool)
            
            gathered = preds.gather(1, center_gene_ids.clamp(min=0))
            gathered = gathered * center_pad_mask.to(gathered.dtype)
            
            gid_np = center_gene_ids.detach().cpu().numpy().astype(np.int64)
            val_np = gathered.detach().float().cpu().numpy().astype(np.float32)
            
            if centers_global is None:
                centers_global = list(range(B))
            
            for i in range(B):
                gidx = int(centers_global[i])
                x_prime_k_raw[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
        
        # Step 3-5: Compute delta and X'_k_true
        print(f"  Computing delta and X'_{t}_true...")
        x_prime_k_true: Dict[int, Dict[str, np.ndarray]] = {}
        
        delta_stats = {"mean": [], "abs_mean": []}
        
        for gidx in x_prime_k_raw.keys():
            gidx = int(gidx)
            
            if gidx in frozen_indices:
                # Frozen spots: keep X'_0 values
                x_prime_k_true[gidx] = {
                    "gene_ids": x0_prime_base[gidx]["gene_ids"].copy(),
                    "raw_normed_values": x0_prime_base[gidx]["raw_normed_values"].copy(),
                }
            else:
                # Non-perturbed spots: X'_k_true = X'_0 + delta_k
                gene_ids_pred = x_prime_k_raw[gidx]["gene_ids"]
                x_prime_raw_vals = x_prime_k_raw[gidx]["raw_normed_values"]
                
                if gidx in x_k_pred:
                    x_k_vals = x_k_pred[gidx]["raw_normed_values"]
                else:
                    x_k_vals = np.zeros_like(x_prime_raw_vals)
                
                delta = x_prime_raw_vals - x_k_vals
                gid_to_delta = {int(gid): delta[i] for i, gid in enumerate(gene_ids_pred)}
                
                if gidx in x0_prime_base:
                    x0_prime_gene_ids = x0_prime_base[gidx]["gene_ids"]
                    x0_prime_vals = x0_prime_base[gidx]["raw_normed_values"].copy()
                    
                    for i, gid in enumerate(x0_prime_gene_ids):
                        if int(gid) in gid_to_delta:
                            x0_prime_vals[i] += gid_to_delta[int(gid)]
                    
                    x_prime_true_vals = x0_prime_vals
                    gene_ids_out = x0_prime_gene_ids.copy()
                else:
                    x_prime_true_vals = x_prime_raw_vals
                    gene_ids_out = gene_ids_pred.copy()
                
                delta_stats["mean"].append(float(np.mean(delta)))
                delta_stats["abs_mean"].append(float(np.mean(np.abs(delta))))
                
                x_prime_k_true[gidx] = {
                    "gene_ids": gene_ids_out,
                    "raw_normed_values": x_prime_true_vals.astype(np.float32),
                }
        
        if delta_stats["mean"]:
            print(f"  Delta: mean={np.mean(delta_stats['mean']):.6f}, abs_mean={np.mean(delta_stats['abs_mean']):.6f}")
        
        # Capture step results
        step_overrides[t] = {
            int(k): {
                "gene_ids": v["gene_ids"].copy(),
                "raw_normed_values": v["raw_normed_values"].copy(),
            }
            for k, v in x_prime_k_true.items()
        }
        
        x_prime_current = x_prime_k_true
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  [OK] Step {t}: {len(x_prime_k_true)} spots")
    
    return step_overrides, x_prime_current


# ==============================================================================
# Activation Score Computation
# ==============================================================================

def compute_activation_scores(
    overrides: Dict[int, Dict[str, np.ndarray]],
    cell_info: pd.DataFrame,
    celltype: str,
    vocab: Dict[str, int],
) -> pd.DataFrame:
    """Compute activation scores for cells of specified type."""
    # Map subtypes to parent type for marker lookup
    marker_key = celltype
    if celltype in ["MNP_activated", "MNP_normal"]:
        marker_key = "MNP"
    elif celltype in ["Fibroblast_activated", "Fibroblast_normal"]:
        marker_key = "Fibroblast"
    
    markers = MARKER_GENES.get(marker_key, [])
    marker_ids = [vocab.get(g.lower()) for g in markers if g.lower() in vocab]
    marker_ids = [m for m in marker_ids if m is not None]
    
    if not marker_ids:
        print(f"[WARN] No marker genes found for {celltype}")
        return pd.DataFrame()
    
    target_types = CELLTYPE_LABELS.get(celltype, {}).get("all", [])
    mask = cell_info["cell_type"].isin(target_types)
    target_cells = cell_info[mask].copy()
    
    scores = []
    for _, row in target_cells.iterrows():
        local_idx = row["local_idx"]
        if local_idx in overrides:
            ov = overrides[local_idx]
            gene_ids = ov["gene_ids"]
            vals = ov["raw_normed_values"]
            
            marker_vals = []
            gid_to_idx = {int(g): i for i, g in enumerate(gene_ids)}
            for mid in marker_ids:
                if mid in gid_to_idx:
                    marker_vals.append(vals[gid_to_idx[mid]])
            
            scores.append(np.mean(marker_vals) if marker_vals else 0.0)
        else:
            scores.append(0.0)
    
    target_cells["activation_score"] = scores
    return target_cells


def evaluate_activation_change(
    scores_df: pd.DataFrame,
    threshold: float,
    celltype: str,
) -> Dict:
    """Evaluate activation status changes."""
    if scores_df.empty:
        return {
            "celltype": celltype,
            "n_total": 0,
            "n_activated": 0,
            "activation_rate": 0.0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
        }
    
    n_total = len(scores_df)
    n_activated = int((scores_df["activation_score"] >= threshold).sum())
    
    return {
        "celltype": celltype,
        "n_total": int(n_total),
        "n_activated": n_activated,
        "activation_rate": float(n_activated / n_total) if n_total > 0 else 0.0,
        "mean_score": float(scores_df["activation_score"].mean()),
        "median_score": float(scores_df["activation_score"].median()),
        "std_score": float(scores_df["activation_score"].std()),
        "min_score": float(scores_df["activation_score"].min()),
        "max_score": float(scores_df["activation_score"].max()),
    }


# ==============================================================================
# Expression Saving
# ==============================================================================

def save_step_expression(
    step_ov: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
    vocab: Dict[str, int],
    cell_info: pd.DataFrame,
    perturbed_indices: Set[int],
) -> None:
    """Save expression matrix for a step to CSV."""
    if not step_ov:
        print(f"[WARN] Empty overrides, skipping {out_path}")
        return
    
    # Reverse vocab
    id_to_gene = {v: k for k, v in vocab.items()}
    
    rows_data = []
    spot_indices = []
    
    for gidx, ov in sorted(step_ov.items()):
        spot_indices.append(gidx)
        gene_ids = ov["gene_ids"]
        values = ov["raw_normed_values"]
        
        spot_row = {}
        for gid, val in zip(gene_ids, values):
            gene_name = id_to_gene.get(int(gid), f"gene_{gid}")
            spot_row[gene_name] = val
        rows_data.append(spot_row)
    
    expr_df = pd.DataFrame(rows_data, index=spot_indices)
    expr_df.index.name = "spot_idx"
    
    # Add cell type info
    cell_types = []
    for gidx in spot_indices:
        ct = cell_info.loc[cell_info["local_idx"] == gidx, "cell_type"]
        cell_types.append(ct.values[0] if len(ct) > 0 else "unknown")
    expr_df.insert(0, "cell_type", cell_types)
    
    # Mark perturbed spots
    is_perturbed = [1 if gidx in perturbed_indices else 0 for gidx in spot_indices]
    expr_df.insert(1, "is_perturbed", is_perturbed)
    
    expr_df.to_csv(out_path)
    n_genes = len(expr_df.columns) - 2
    print(f"[OK] Saved: {out_path} ({len(spot_indices)} spots × {n_genes} genes)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="SpatialGT Virtual Treatment Perturbation for Colitis")
    
    # Sample selection
    ap.add_argument("--sample", type=str, default="HS5_UC_R_0",
                   help="UC sample name (default: HS5_UC_R_0)")
    ap.add_argument("--ckpt_dir", type=str, default=None,
                   help="Checkpoint directory")
    
    # Perturbation target
    ap.add_argument("--perturb_target", type=str, nargs="+", default=["MNP_activated"],
                   choices=["MNP", "Fibroblast", "MNP_activated", "MNP_normal", 
                            "Fibroblast_activated", "Fibroblast_normal"],
                   help="Cell types to perturb (default: MNP_activated)")
    
    # DEG file
    ap.add_argument("--deg_file", type=str, default=None,
                   help="Path to custom DEG file")
    ap.add_argument("--deg_dir", type=str, default=None,
                   help="DEG directory")
    
    # Perturbation parameters
    ap.add_argument("--n_spots", type=int, default=40,
                   help="Number of spots to perturb")
    ap.add_argument("--dose_lambda", type=float, default=1.0,
                   help="Dose parameter lambda")
    ap.add_argument("--weighting", type=str, default="uniform",
                   choices=["uniform", "gaussian"])
    
    # DEG parameters
    ap.add_argument("--p_thresh", type=float, default=0.1)
    ap.add_argument("--min_abs_logfc", type=float, default=0.0)
    ap.add_argument("--logfc_clip", type=float, default=5.0)
    
    # Iteration settings
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--freeze_perturbed", action="store_true", default=True)
    
    # Activation threshold
    ap.add_argument("--activation_threshold_file", type=str, default=None)
    
    # Data source
    ap.add_argument("--use_lmdb", action="store_true", default=True,
                   help="Use LMDB cache (default: True)")
    ap.add_argument("--data_root", type=str, default=None,
                   help="Data root directory")
    
    # Performance
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_progress", action="store_true", default=True)
    
    # Output
    ap.add_argument("--out_dir", type=str, required=True)
    
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data root
    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    
    print("=" * 60)
    print("SpatialGT Virtual Treatment Perturbation (Colitis)")
    print("=" * 60)
    print(f"Sample: {args.sample}")
    print(f"Perturb targets: {args.perturb_target}")
    print(f"N_spots: {args.n_spots}, Lambda: {args.dose_lambda}")
    print(f"Steps: {args.steps}")
    print("=" * 60)
    
    # Load vocab
    vocab = _load_vocab()
    
    # Load activation thresholds
    if args.activation_threshold_file and Path(args.activation_threshold_file).exists():
        with open(args.activation_threshold_file, "r") as f:
            activation_thresholds = json.load(f)
    else:
        activation_thresholds = DEFAULT_THRESHOLDS
    
    # Setup paths
    lmdb_dir = data_root / "lmdb" / args.sample
    deg_dir = Path(args.deg_dir) if args.deg_dir else DEFAULT_DEG_DIR
    
    # Checkpoint directory
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
    else:
        ckpt_dir = DEFAULT_MODEL_ROOT / args.sample
        if not ckpt_dir.exists():
            ckpt_dir = DEFAULT_MODEL_ROOT / "UC"
    
    print(f"[INFO] Checkpoint: {ckpt_dir}")
    
    # Setup config and databank
    config = Config()
    
    if args.use_lmdb:
        lmdb_path = lmdb_dir / "spatial_cache.lmdb"
        manifest_path = lmdb_dir / "spatial_cache.manifest.json"
        
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        
        config.cache_mode = "lmdb"
        config.strict_cache_only = True
        config.lmdb_path = str(lmdb_path)
        config.runtime_lmdb_path = str(lmdb_path)
        config.lmdb_manifest_path = str(manifest_path)
        config.runtime_lmdb_manifest_path = str(manifest_path)
        config.cache_dir = str(lmdb_dir)
        
        processed_path = data_root / "preprocessed_v3" / args.sample / args.sample / "processed.h5ad"
        
        databank = SpatialDataBank(
            dataset_paths=[str(processed_path)],
            cache_dir=str(lmdb_dir),
            config=config,
            force_rebuild=False,
        )
    else:
        processed_path = data_root / "preprocessed_v3" / args.sample / args.sample / "processed.h5ad"
        
        databank = SpatialDataBank(
            dataset_paths=[str(processed_path)],
            cache_dir=str(data_root / "preprocessed_v3" / args.sample),
            config=config,
            force_rebuild=False,
        )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialNeighborTransformer(config).to(device)
    _load_state_dict_into_model(model, ckpt_dir)
    model.eval()
    print(f"[OK] Loaded model from {ckpt_dir}")
    
    # Load cell info
    cell_info = load_sample_cell_info(args.sample, data_root)
    print(f"[OK] Loaded cell info: {len(cell_info)} cells")
    
    # Build DEG perturbation vectors
    perturbation_vectors = {}
    for celltype in args.perturb_target:
        deg = load_deg_for_treatment(
            celltype, deg_dir, args.p_thresh, args.min_abs_logfc, args.deg_file
        )
        gene_logfc = build_perturbation_vector(deg, vocab, args.logfc_clip)
        perturbation_vectors[celltype] = gene_logfc
    
    # Select cells to perturb
    all_perturb_indices = []
    for celltype in args.perturb_target:
        indices = get_celltype_indices(cell_info, celltype, include_all=True)
        selected = select_random_spots(indices, args.n_spots, args.seed)
        all_perturb_indices.extend(selected)
        print(f"[INFO] Selected {len(selected)} {celltype} cells")
    
    all_perturb_indices = list(set(all_perturb_indices))
    perturbed_spot_set = set(all_perturb_indices)
    print(f"[INFO] Total perturbed cells: {len(all_perturb_indices)}")
    
    # Compute weights
    weights, weight_meta = compute_spot_weights(databank, all_perturb_indices, args.weighting)
    
    # Merge perturbation vectors
    merged_gene_logfc: Dict[int, float] = {}
    for celltype in args.perturb_target:
        for gid, logfc in perturbation_vectors[celltype].items():
            if gid in merged_gene_logfc:
                merged_gene_logfc[gid] = (merged_gene_logfc[gid] + logfc) / 2
            else:
                merged_gene_logfc[gid] = logfc
    
    # Save manifest
    manifest = {
        "sample": args.sample,
        "perturb_target": args.perturb_target,
        "n_perturbed_cells": len(all_perturb_indices),
        "perturbed_indices": all_perturb_indices,
        "dose_lambda": args.dose_lambda,
        "steps": args.steps,
        "seed": args.seed,
        "ckpt_dir": str(ckpt_dir),
        "deg_file": args.deg_file,
        **weight_meta,
    }
    
    with open(out_dir / "perturb_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Build X_0 using reconstruction (V4 mode)
    print("\n[STEP] Building X_0 = SpatialGT(raw_data)...")
    databank.clear_runtime_spot_overrides()
    
    loader = _build_all_loader(databank, config, args.batch_size, args.num_workers)
    x0_state = run_single_pass_inference(
        databank, loader, model, device,
        desc="X_0 = SpatialGT(raw)", show_progress=args.show_progress
    )
    print(f"[OK] X_0: {len(x0_state)} spots")
    
    # Build X'_0 (apply perturbation on reconstructed X_0)
    print("\n[STEP] Building X'_0 (perturbation on X_0)...")
    x0_prime_state: Dict[int, Dict[str, np.ndarray]] = {}
    
    for gidx, ov in x0_state.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # Apply perturbation to selected spots
    for i, gidx in enumerate(all_perturb_indices):
        gidx = int(gidx)
        if gidx not in x0_state:
            continue
        
        gene_ids = x0_state[gidx]["gene_ids"]
        vals = x0_state[gidx]["raw_normed_values"].copy()
        wi = float(weights[i]) if i < len(weights) else 1.0
        
        for j, gid in enumerate(gene_ids):
            gid_int = int(gid)
            if gid_int in merged_gene_logfc:
                logfc = merged_gene_logfc[gid_int]
                fold_change = np.power(2.0, logfc * args.dose_lambda * wi)
                vals[j] = vals[j] * fold_change
        
        vals = np.maximum(vals, 0.0)
        x0_prime_state[gidx] = {
            "gene_ids": gene_ids.copy(),
            "raw_normed_values": vals.astype(np.float32),
        }
    
    print(f"[OK] X'_0: {len(perturbed_spot_set)} spots perturbed")
    
    # Run dual-line iterative inference
    print(f"\n[STEP] Running dual-line inference for {args.steps} steps...")
    frozen_indices = perturbed_spot_set if args.freeze_perturbed else set()
    
    def loader_factory():
        return _build_all_loader(databank, config, args.batch_size, args.num_workers)
    
    step_overrides, final_state = run_dual_line_iterative_inference(
        databank, loader_factory, model, device,
        steps=args.steps,
        frozen_indices=frozen_indices,
        x0_state=x0_state,
        x0_prime_state=x0_prime_state,
        show_progress=args.show_progress,
    )
    
    # Add step 0
    step_overrides[0] = x0_prime_state
    
    # Save all step expressions
    print("\n[STEP] Saving expression for all steps...")
    expr_dir = out_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)
    
    for t in range(args.steps + 1):
        if t in step_overrides:
            save_step_expression(
                step_overrides[t],
                expr_dir / f"pert{t}_expression.csv",
                vocab, cell_info, perturbed_spot_set
            )
    
    # Evaluate activation for each step
    print("\n[STEP] Evaluating activation status...")
    activation_rows = []
    
    eval_celltypes = ["MNP", "Fibroblast"]
    
    for celltype in eval_celltypes:
        threshold = activation_thresholds.get(celltype, {}).get("optimal_threshold", 1.5)
        print(f"[INFO] {celltype} threshold: {threshold:.4f}")
        
        for t in range(args.steps + 1):
            step_ov = step_overrides.get(t, {})
            
            scores = compute_activation_scores(step_ov, cell_info, celltype, vocab)
            eval_result = evaluate_activation_change(scores, threshold, celltype)
            eval_result["step"] = t
            eval_result["group"] = f"pert{t}"
            activation_rows.append(eval_result)
    
    # Save activation results
    df_activation = pd.DataFrame(activation_rows)
    df_activation.to_csv(out_dir / "activation_eval.csv", index=False)
    
    with open(out_dir / "activation_eval.json", "w") as f:
        json.dump(activation_rows, f, indent=2)
    
    print(f"\n[OK] Results saved to {out_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ACTIVATION SUMMARY")
    print("=" * 60)
    
    for celltype in eval_celltypes:
        print(f"\n[{celltype}]")
        ct_rows = [r for r in activation_rows if r["celltype"] == celltype]
        for r in sorted(ct_rows, key=lambda x: x["step"]):
            print(f"  Step {r['step']:2d}: n_activated={r['n_activated']:4d}/{r['n_total']:4d} "
                  f"({r['activation_rate']*100:5.1f}%), mean_score={r['mean_score']:.4f}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
