#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN/Linear Regression Baseline for Niche Perturbation Evaluation.

This script implements KNN averaging and GPU Ridge regression baselines
for the perturbation prediction task using dual-line inference with
error cancellation.

Algorithm (Dual-line inference):
  - X line (unperturbed): always starts from X_0 (raw data), used for error calculation
  - X' line (perturbed): starts from previous step's true result
  - delta_k = X'_k_raw - X_k (net perturbation effect, error cancelled)
  - X'_k_true = X'_0 + delta_k (add delta to original perturbed state)
  - Frozen spots: X'_k_true(perturb_spots) = X'_0(perturb_spots)

Supports:
- Two perturbation modes: patch (BFS-connected spots) and random (scattered spots)
- Two weighting schemes: gaussian (distance-based) and uniform (constant)
- Two inference modes: knn_avg (KNN average) and linear_reg_gpu (GPU Ridge)
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
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import Ridge

# Ensure repo root importability
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import from pretrain module
from pretrain.spatial_databank import SpatialDataBank
from pretrain.Config import Config


# ==============================================================================
# Utility Functions
# ==============================================================================

def _load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_dataset_start_idx(cache_dir: Path, dataset_name: str) -> int:
    """Get the starting global index for a dataset in the cache."""
    meta = _load_json(cache_dir / "metadata.json")
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


def _setup_config_for_cache(
    config: Config,
    cache_dir: str,
    dataset_name: str,
    cache_mode: str = "h5",
    lmdb_path: Optional[str] = None,
    lmdb_manifest_path: Optional[str] = None,
) -> Config:
    """Configure the config object for cache loading."""
    config.cache_dir = str(cache_dir)
    config.cache_mode = cache_mode.lower()
    config.strict_cache_only = True
    
    if cache_mode.lower() == "lmdb":
        if lmdb_path:
            config.lmdb_path = str(lmdb_path)
            config.runtime_lmdb_path = str(lmdb_path)
        else:
            lmdb_base = Path(cache_dir).parent / "lmdb_cache"
            config.lmdb_path = str(lmdb_base / "spatial_cache.lmdb")
            config.runtime_lmdb_path = config.lmdb_path
        
        if lmdb_manifest_path:
            config.lmdb_manifest_path = str(lmdb_manifest_path)
            config.runtime_lmdb_manifest_path = str(lmdb_manifest_path)
        else:
            config.lmdb_manifest_path = str(Path(config.lmdb_path).with_suffix(".manifest.json"))
            config.runtime_lmdb_manifest_path = config.lmdb_manifest_path
    
    return config


def _barcodes_to_global_idx(cache_dir: Path, dataset_name: str, barcodes: List[str]) -> List[int]:
    """Convert barcodes to global indices."""
    import anndata as ad
    meta = _load_json(cache_dir / "metadata.json")
    start_idx = None
    for info in meta.get("dataset_indices", []):
        if info.get("dataset_idx") == 0 or meta.get("datasets", [{}])[info.get("dataset_idx", 0)].get("name") == dataset_name:
            start_idx = info.get("start_idx")
            break
    if start_idx is None:
        start_idx = 0
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    obs = list(proc.obs_names)
    pos = {b: i for i, b in enumerate(obs)}
    out = []
    for b in barcodes:
        if b in pos:
            out.append(int(start_idx + pos[b]))
    return out


def _global_idx_to_barcode(cache_dir: Path, dataset_name: str, global_indices: List[int]) -> List[str]:
    """Convert global indices to barcodes."""
    import anndata as ad
    start_idx = _get_dataset_start_idx(cache_dir, dataset_name)
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    obs = list(proc.obs_names)
    out = []
    for g in global_indices:
        local = int(g) - int(start_idx)
        if 0 <= local < len(obs):
            out.append(str(obs[local]))
        else:
            out.append(f"unknown_{g}")
    return out


# ==============================================================================
# Spot Selection: Patch (BFS) and Random
# ==============================================================================

def select_patch_spots(
    databank: SpatialDataBank,
    roi_global_indices: List[int],
    patch_size: int,
    seed: int,
    seed_spot_idx: Optional[int] = None,
) -> List[int]:
    """Select a contiguous patch of spots via BFS from a seed spot."""
    rng = random.Random(seed)
    roi_set = set(int(x) for x in roi_global_indices)
    if not roi_set:
        return []
    
    if seed_spot_idx is not None and seed_spot_idx in roi_set:
        seed_g = seed_spot_idx
    else:
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
# Weight Computation: Gaussian and Uniform
# ==============================================================================

def compute_spot_weights(
    cache_dir: Path,
    dataset_name: str,
    selected_global_indices: List[int],
    weighting: str,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Compute weights for selected spots based on weighting scheme."""
    import anndata as ad
    start_idx = _get_dataset_start_idx(cache_dir, dataset_name)
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
    
    meta = {
        "weighting": weighting,
        "n_spots": len(valid_indices),
        "center_xy": [float(center[0]), float(center[1])],
    }
    
    if weighting == "uniform":
        weights = np.ones(len(valid_indices), dtype=np.float32)
    else:  # gaussian
        d2 = ((coords - center) ** 2).sum(axis=1)
        if sigma is None:
            d = np.sqrt(d2)
            sigma = float(np.median(d)) if np.median(d) > 0 else float(np.max(d) if np.max(d) > 0 else 1.0)
        weights = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        meta["sigma"] = float(sigma)
        meta["w_stats"] = {
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "median": float(np.median(weights)),
        }
    
    return weights, meta


# ==============================================================================
# DEG-based Perturbation Application
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
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """Apply DEG-based perturbation to selected spots."""
    deg = pd.read_csv(deg_csv)
    if "avg_logFC" not in deg.columns:
        raise ValueError(f"DEG csv missing required column avg_logFC: {deg.columns.tolist()}")
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col is not None:
        deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
        deg = deg[np.isfinite(deg[p_col])]
        deg = deg[deg[p_col].astype(float) < float(p_adj_thresh)]
    
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    vocab_path = Config().vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {k.lower(): int(v) for k, v in vocab_json.items()}
    
    gene_logfc: Dict[int, float] = {}
    n_up_genes = 0
    n_down_genes = 0
    for _, r in deg.iterrows():
        gid = vocab.get(str(r["gene"]).lower(), None)
        if gid is None:
            continue
        logfc_val = float(r["avg_logFC"])
        logfc_clipped = np.clip(logfc_val, -float(logfc_clip), float(logfc_clip))
        gene_logfc[int(gid)] = logfc_clipped
        if logfc_val > 0:
            n_up_genes += 1
        elif logfc_val < 0:
            n_down_genes += 1
    
    overrides: Dict[int, Dict[str, np.ndarray]] = {}
    n_hits_per_spot = []
    logfc_applied_stats = []
    
    for i, gidx in enumerate(selected_global_indices):
        if i >= len(weights):
            break
        sd = databank.get_spot_data(int(gidx))
        gene_ids = np.asarray(sd["gene_ids"], dtype=np.int64)
        vals = np.asarray(sd["raw_normed_values"], dtype=np.float32).copy()
        wi = float(weights[i])
        
        n_hits = 0
        logfc_vals_this_spot = []
        
        for j, gid in enumerate(gene_ids):
            gid_int = int(gid)
            if gid_int in gene_logfc:
                logfc_val = gene_logfc[gid_int]
                fold_change = np.power(2.0, logfc_val * float(logfc_strength) * wi)
                vals[j] = vals[j] * fold_change
                n_hits += 1
                logfc_vals_this_spot.append(logfc_val)
        
        vals = np.maximum(vals, 0.0)
        
        n_hits_per_spot.append(n_hits)
        if logfc_vals_this_spot:
            logfc_applied_stats.extend(logfc_vals_this_spot)
        
        overrides[int(gidx)] = {"gene_ids": gene_ids, "raw_normed_values": vals}
    
    meta = {
        "deg_csv": str(deg_csv),
        "p_col": p_col,
        "p_adj_thresh": float(p_adj_thresh),
        "min_abs_logfc": float(min_abs_logfc),
        "logfc_strength": float(logfc_strength),
        "logfc_clip": float(logfc_clip),
        "n_deg_rows_used": int(len(deg)),
        "n_deg_genes_mapped": int(len(gene_logfc)),
        "n_up_gene_ids": n_up_genes,
        "n_down_gene_ids": n_down_genes,
        "n_hits_per_spot_stats": {
            "min": int(np.min(n_hits_per_spot)) if n_hits_per_spot else 0,
            "max": int(np.max(n_hits_per_spot)) if n_hits_per_spot else 0,
            "mean": float(np.mean(n_hits_per_spot)) if n_hits_per_spot else 0,
        },
        "logfc_applied_stats": {
            "min": float(np.min(logfc_applied_stats)) if logfc_applied_stats else 0,
            "max": float(np.max(logfc_applied_stats)) if logfc_applied_stats else 0,
            "mean": float(np.mean(logfc_applied_stats)) if logfc_applied_stats else 0,
        },
    }
    
    return overrides, meta


# ==============================================================================
# KNN / Ridge Regression Inference
# ==============================================================================

def get_spot_expression(
    databank: SpatialDataBank,
    global_idx: int,
    current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get expression for a spot from current state or databank."""
    if current_state is not None and int(global_idx) in current_state:
        ov = current_state[int(global_idx)]
        return np.asarray(ov["gene_ids"], dtype=np.int64), np.asarray(ov["raw_normed_values"], dtype=np.float32)
    
    ov = databank.get_runtime_spot_override(int(global_idx)) if hasattr(databank, "get_runtime_spot_override") else None
    if ov is not None and ov.get("gene_ids") is not None:
        return np.asarray(ov["gene_ids"], dtype=np.int64), np.asarray(ov["raw_normed_values"], dtype=np.float32)
    
    sd = databank.get_spot_data(int(global_idx))
    return np.asarray(sd["gene_ids"], dtype=np.int64), np.asarray(sd["raw_normed_values"], dtype=np.float32)


def knn_average_predict(
    databank: SpatialDataBank,
    center_global_idx: int,
    current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """Predict expression using KNN averaging."""
    center_gids, center_vals = get_spot_expression(databank, center_global_idx, current_state)
    
    try:
        neighbors = databank.get_neighbors_for_spot(int(center_global_idx))
    except Exception:
        neighbors = []
    
    if len(neighbors) == 0:
        return {"gene_ids": center_gids, "raw_normed_values": center_vals}
    
    neighbor_exprs = []
    neighbor_gids_list = []
    
    for nb_idx in neighbors:
        nb_gids, nb_vals = get_spot_expression(databank, int(nb_idx), current_state)
        neighbor_gids_list.append(nb_gids)
        neighbor_exprs.append((nb_gids, nb_vals))
    
    common_genes = set(int(g) for g in center_gids if int(g) >= 0)
    for nb_gids in neighbor_gids_list:
        common_genes &= set(int(g) for g in nb_gids if int(g) >= 0)
    
    if len(common_genes) == 0:
        return {"gene_ids": center_gids, "raw_normed_values": center_vals}
    
    common_genes_arr = np.array(sorted(common_genes), dtype=np.int64)
    center_pos = {int(g): i for i, g in enumerate(center_gids)}
    common_to_center_idx = {int(g): center_pos[int(g)] for g in common_genes_arr if int(g) in center_pos}
    
    n_neighbors = len(neighbor_exprs)
    n_common = len(common_genes_arr)
    neighbor_matrix = np.zeros((n_neighbors, n_common), dtype=np.float32)
    
    for i, (nb_gids, nb_vals) in enumerate(neighbor_exprs):
        nb_pos = {int(g): j for j, g in enumerate(nb_gids)}
        for k, cg in enumerate(common_genes_arr):
            if int(cg) in nb_pos:
                neighbor_matrix[i, k] = nb_vals[nb_pos[int(cg)]]
    
    weights = np.ones(n_neighbors, dtype=np.float32) / n_neighbors
    predicted_common = np.sum(neighbor_matrix * weights[:, np.newaxis], axis=0)
    
    predicted_vals = center_vals.copy()
    for k, cg in enumerate(common_genes_arr):
        if int(cg) in common_to_center_idx:
            predicted_vals[common_to_center_idx[int(cg)]] = predicted_common[k]
    
    return {"gene_ids": center_gids, "raw_normed_values": predicted_vals}


class GPURidgeRegressionModel:
    """GPU-accelerated Ridge regression model using PyTorch."""
    
    def __init__(self, alpha: float = 1.0, max_neighbors: int = 8, device: str = "cuda"):
        self.alpha = alpha
        self.max_neighbors = max_neighbors
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.weights: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.gene_id_to_idx: Dict[int, int] = {}
        self.idx_to_gene_id: Dict[int, int] = {}
        self.n_genes: int = 0
        self.is_trained: bool = False
        self.n_training_samples: int = 0
        
        self.neighbor_table: Optional[torch.Tensor] = None
        self.neighbor_mask: Optional[torch.Tensor] = None
        self.global_to_local: Dict[int, int] = {}
        self.local_to_global: Dict[int, int] = {}
    
    def _collect_training_data_fast(
        self,
        databank: SpatialDataBank,
        all_global_indices: List[int],
        current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
        """Collect training data for Ridge regression."""
        all_gene_ids: Set[int] = set()
        spot_data_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        iterator = tqdm(all_global_indices, desc="Collecting gene vocabulary", 
                       disable=not show_progress)
        for idx in iterator:
            gids, vals = get_spot_expression(databank, idx, current_state)
            spot_data_cache[idx] = (gids, vals)
            valid_gids = set(int(g) for g in gids if int(g) >= 0)
            all_gene_ids.update(valid_gids)
        
        sorted_gene_ids = sorted(all_gene_ids)
        gene_id_to_idx = {gid: i for i, gid in enumerate(sorted_gene_ids)}
        n_genes = len(sorted_gene_ids)
        
        print(f"[GPU Ridge] Found {n_genes} unique genes")
        
        X_list = []
        Y_list = []
        mask_list = []
        
        iterator = tqdm(all_global_indices, desc="Collecting training data", 
                       disable=not show_progress)
        
        for center_idx in iterator:
            center_gids, center_vals = spot_data_cache[center_idx]
            
            try:
                neighbors = databank.get_neighbors_for_spot(int(center_idx))
            except Exception:
                continue
            
            if len(neighbors) < 2:
                continue
            
            neighbors = neighbors[:self.max_neighbors]
            
            neighbor_features = np.zeros((self.max_neighbors, n_genes), dtype=np.float32)
            
            for ni, nb_idx in enumerate(neighbors):
                if nb_idx not in spot_data_cache:
                    nb_gids, nb_vals = get_spot_expression(databank, int(nb_idx), current_state)
                    spot_data_cache[nb_idx] = (nb_gids, nb_vals)
                else:
                    nb_gids, nb_vals = spot_data_cache[nb_idx]
                
                for j, gid in enumerate(nb_gids):
                    gid_int = int(gid)
                    if gid_int >= 0 and gid_int in gene_id_to_idx:
                        neighbor_features[ni, gene_id_to_idx[gid_int]] = nb_vals[j]
            
            center_target = np.zeros(n_genes, dtype=np.float32)
            gene_mask = np.zeros(n_genes, dtype=np.float32)
            
            for j, gid in enumerate(center_gids):
                gid_int = int(gid)
                if gid_int >= 0 and gid_int in gene_id_to_idx:
                    center_target[gene_id_to_idx[gid_int]] = center_vals[j]
                    gene_mask[gene_id_to_idx[gid_int]] = 1.0
            
            X_list.append(neighbor_features)
            Y_list.append(center_target)
            mask_list.append(gene_mask)
        
        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)
        mask = np.stack(mask_list, axis=0)
        
        return X, Y, mask, gene_id_to_idx
    
    def train(
        self,
        databank: SpatialDataBank,
        all_global_indices: List[int],
        current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        min_samples: int = 10,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Train Ridge regression model."""
        print(f"[GPU Ridge] Training on {len(all_global_indices)} spots on {self.device}...")
        
        X, Y, mask, gene_id_to_idx = self._collect_training_data_fast(
            databank, all_global_indices, current_state, show_progress
        )
        
        n_samples, max_neighbors, n_genes = X.shape
        self.n_training_samples = n_samples
        self.gene_id_to_idx = gene_id_to_idx
        self.idx_to_gene_id = {v: k for k, v in gene_id_to_idx.items()}
        self.n_genes = n_genes
        
        print(f"[GPU Ridge] Training data shape: X={X.shape}, Y={Y.shape}")
        
        X_t = torch.from_numpy(X).to(self.device)
        Y_t = torch.from_numpy(Y).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)
        
        self.weights = torch.zeros(n_genes, max_neighbors, device=self.device)
        self.bias = torch.zeros(n_genes, device=self.device)
        
        reg_I = self.alpha * torch.eye(max_neighbors, device=self.device)
        
        n_trained = 0
        n_skipped = 0
        
        batch_size = 512
        n_batches = (n_genes + batch_size - 1) // batch_size
        
        iterator = tqdm(range(n_batches), desc="Training Ridge (GPU batch)", 
                       disable=not show_progress)
        
        for batch_idx in iterator:
            start_g = batch_idx * batch_size
            end_g = min((batch_idx + 1) * batch_size, n_genes)
            
            for g in range(start_g, end_g):
                X_g = X_t[:, :, g]
                y_g = Y_t[:, g]
                m_g = mask_t[:, g]
                
                valid_mask = m_g > 0.5
                n_valid = valid_mask.sum().item()
                
                if n_valid < min_samples:
                    n_skipped += 1
                    continue
                
                X_valid = X_g[valid_mask]
                y_valid = y_g[valid_mask]
                
                XtX = X_valid.T @ X_valid
                Xty = X_valid.T @ y_valid
                
                try:
                    w = torch.linalg.solve(XtX + reg_I, Xty)
                    y_mean = y_valid.mean()
                    X_mean = X_valid.mean(dim=0)
                    b = y_mean - torch.dot(w, X_mean)
                    
                    self.weights[g] = w
                    self.bias[g] = b
                    n_trained += 1
                except Exception:
                    n_skipped += 1
        
        self.is_trained = True
        
        stats = {
            "n_genes_trained": n_trained,
            "n_genes_skipped": n_skipped,
            "n_genes_total": n_genes,
            "n_training_samples": n_samples,
            "alpha": self.alpha,
            "max_neighbors": self.max_neighbors,
            "min_samples": min_samples,
            "device": str(self.device),
        }
        
        print(f"[GPU Ridge] Trained {n_trained} gene models, skipped {n_skipped}")
        
        return stats
    
    def build_neighbor_table(
        self,
        databank: SpatialDataBank,
        all_global_indices: List[int],
        show_progress: bool = True,
    ) -> None:
        """Build neighbor lookup table for batch inference."""
        n_spots = len(all_global_indices)
        idx_to_global = {i: g for i, g in enumerate(all_global_indices)}
        global_to_idx = {g: i for i, g in enumerate(all_global_indices)}
        
        neighbor_table = np.full((n_spots, self.max_neighbors), -1, dtype=np.int64)
        neighbor_mask = np.zeros((n_spots, self.max_neighbors), dtype=np.float32)
        
        iterator = tqdm(range(n_spots), desc="Building neighbor table", 
                       disable=not show_progress)
        
        for i in iterator:
            global_idx = idx_to_global[i]
            try:
                neighbors = databank.get_neighbors_for_spot(int(global_idx))
            except Exception:
                continue
            
            for ni, nb_global in enumerate(neighbors[:self.max_neighbors]):
                nb_global = int(nb_global)
                if nb_global in global_to_idx:
                    neighbor_table[i, ni] = global_to_idx[nb_global]
                    neighbor_mask[i, ni] = 1.0
        
        self.neighbor_table = torch.from_numpy(neighbor_table).to(self.device)
        self.neighbor_mask = torch.from_numpy(neighbor_mask).to(self.device)
        self.global_to_local = global_to_idx
        self.local_to_global = idx_to_global
    
    def predict_batch(
        self,
        expr_matrix: torch.Tensor,
        batch_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Predict expression for a batch of spots."""
        if not self.is_trained:
            return expr_matrix
        
        if batch_indices is None:
            batch_indices = list(range(expr_matrix.shape[0]))
        
        n_batch = len(batch_indices)
        batch_idx_t = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        
        nb_indices = self.neighbor_table[batch_idx_t]
        nb_mask = self.neighbor_mask[batch_idx_t]
        
        nb_indices_safe = nb_indices.clamp(min=0)
        nb_expr = expr_matrix[nb_indices_safe]
        nb_expr = nb_expr * nb_mask.unsqueeze(-1)
        
        W_t = self.weights.T
        predictions = torch.einsum('bkg,kg->bg', nb_expr, W_t) + self.bias.unsqueeze(0)
        predictions = F.relu(predictions)
        
        return predictions


def run_single_pass_inference(
    databank: SpatialDataBank,
    all_global_indices: List[int],
    infer_mode: str,
    current_state: Dict[int, Dict[str, np.ndarray]],
    ridge_model: Optional[GPURidgeRegressionModel] = None,
    show_progress: bool = True,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run single pass inference for all spots."""
    all_overrides = {}
    
    if infer_mode == "linear_reg_gpu" and ridge_model is not None and ridge_model.is_trained:
        # GPU batch inference
        print(f"[GPU Ridge] {desc} - using batch GPU inference...")
        
        n_spots = len(all_global_indices)
        expr_matrix = torch.zeros((n_spots, ridge_model.n_genes), device=ridge_model.device)
        gene_id_arrays: Dict[int, np.ndarray] = {}
        
        for i, global_idx in enumerate(all_global_indices):
            gids, vals = get_spot_expression(databank, global_idx, current_state)
            gene_id_arrays[i] = gids
            vals = np.asarray(vals, dtype=np.float32)
            
            for j, gid in enumerate(gids):
                gid_int = int(gid)
                if gid_int >= 0 and gid_int in ridge_model.gene_id_to_idx:
                    expr_matrix[i, ridge_model.gene_id_to_idx[gid_int]] = float(vals[j])
        
        if ridge_model.neighbor_table is None or ridge_model.neighbor_table.shape[0] != n_spots:
            ridge_model.build_neighbor_table(databank, all_global_indices, show_progress)
        
        predictions = ridge_model.predict_batch(expr_matrix)
        pred_np = predictions.cpu().numpy()
        
        for i, global_idx in enumerate(all_global_indices):
            gids = gene_id_arrays[i]
            vals = np.zeros(len(gids), dtype=np.float32)
            
            for j, gid in enumerate(gids):
                gid_int = int(gid)
                if gid_int >= 0 and gid_int in ridge_model.gene_id_to_idx:
                    vals[j] = pred_np[i, ridge_model.gene_id_to_idx[gid_int]]
            
            all_overrides[int(global_idx)] = {"gene_ids": gids, "raw_normed_values": vals}
    else:
        # CPU KNN inference
        iterator = tqdm(all_global_indices, desc=desc, disable=not show_progress)
        for gidx in iterator:
            pred = knn_average_predict(databank, gidx, current_state)
            all_overrides[int(gidx)] = pred
    
    return all_overrides


# ==============================================================================
# Dual-line Iterative Inference with Error Cancellation
# ==============================================================================

def run_dual_line_iterative_inference(
    databank: SpatialDataBank,
    all_global_indices: List[int],
    steps: int,
    frozen_indices: Set[int],
    x0_state: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    infer_mode: str = "knn_avg",
    ridge_model: Optional[GPURidgeRegressionModel] = None,
    show_progress: bool = True,
) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], Dict[int, Dict[str, np.ndarray]]]:
    """
    Run dual-line iterative inference with error cancellation.
    
    Algorithm:
    - X line (unperturbed): always starts from X_0 (raw), only used for error calculation
    - X' line (perturbed): starts from previous step's true result
    
    Each step k:
    1. X_k = Model(X_0)  -- unperturbed line from X_0
    2. X'_k_raw = Model(X'_{k-1}_true)  -- perturbed line from last true result
    3. delta_k = X'_k_raw - X_k  -- net perturbation effect (error cancelled)
    4. X'_k_true = X'_0 + delta_k  -- add delta to original perturbed state
    5. X'_k_true(perturb_spots) = X'_0(perturb_spots)  -- freeze perturbed spots
    """
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    
    # Deep copy X'_0 as the base for accumulating deltas
    x0_prime_base: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_prime_state.items():
        x0_prime_base[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    # Current state for X' line (starts as X'_0)
    x_prime_current: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_prime_state.items():
        x_prime_current[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }
    
    for t in range(1, steps + 1):
        print(f"\n[STEP {t}/{steps}] Dual-line inference...")
        
        # Step 1: X_k = Model(X_0) - Run unperturbed line from X_0
        print(f"  [X line] Running {infer_mode} on X_0...")
        x_k_pred = run_single_pass_inference(
            databank, all_global_indices, infer_mode, x0_state, ridge_model, show_progress,
            desc=f"X line step {t}"
        )
        
        # Step 2: X'_k_raw = Model(X'_{k-1}_true) - Run perturbed line
        print(f"  [X' line] Running {infer_mode} on X'_{t-1}_true...")
        x_prime_k_raw = run_single_pass_inference(
            databank, all_global_indices, infer_mode, x_prime_current, ridge_model, show_progress,
            desc=f"X' line step {t}"
        )
        
        # Steps 3-5: Compute delta and X'_k_true
        print(f"  Computing delta and X'_{t}_true...")
        x_prime_k_true: Dict[int, Dict[str, np.ndarray]] = {}
        
        delta_stats = {"mean": [], "abs_mean": []}
        
        for gidx in x_prime_k_raw.keys():
            gidx = int(gidx)
            
            if gidx in frozen_indices:
                # Frozen perturbed spots: keep X'_0 values
                x_prime_k_true[gidx] = {
                    "gene_ids": x0_prime_base[gidx]["gene_ids"].copy(),
                    "raw_normed_values": x0_prime_base[gidx]["raw_normed_values"].copy(),
                }
            else:
                # Non-perturbed spots: X'_k_true = X'_0 + delta_k
                gene_ids_pred = x_prime_k_raw[gidx]["gene_ids"]
                x_prime_raw_vals = x_prime_k_raw[gidx]["raw_normed_values"]
                
                # Get X_k values
                if gidx in x_k_pred:
                    x_k_vals = x_k_pred[gidx]["raw_normed_values"]
                else:
                    x_k_vals = np.zeros_like(x_prime_raw_vals)
                
                # delta_k = X'_k_raw - X_k
                delta = x_prime_raw_vals - x_k_vals
                
                # Create mapping from gene_id to delta value
                gid_to_delta = {int(gid): delta[i] for i, gid in enumerate(gene_ids_pred)}
                
                # Get X'_0 values and apply delta
                if gidx in x0_prime_base:
                    x0_prime_gene_ids = x0_prime_base[gidx]["gene_ids"]
                    x0_prime_vals = x0_prime_base[gidx]["raw_normed_values"].copy()
                    
                    # Apply delta to matching genes: X'_k_true = X'_0 + delta_k
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
        
        # Report delta statistics
        if delta_stats["mean"]:
            print(f"  Delta stats: mean={np.mean(delta_stats['mean']):.6f}, "
                  f"abs_mean={np.mean(delta_stats['abs_mean']):.6f}")
        
        # Capture step results
        step_overrides[t] = {
            int(k): {
                "gene_ids": v["gene_ids"].copy(),
                "raw_normed_values": v["raw_normed_values"].copy(),
            }
            for k, v in x_prime_k_true.items()
        }
        
        # Update current X' state for next iteration
        x_prime_current = x_prime_k_true
        
        print(f"  [INFO] Step {t} completed: {len(x_prime_k_true)} spots in X'_{t}_true")
    
    return step_overrides, x_prime_current


# ==============================================================================
# Similarity Metrics
# ==============================================================================

def _collect_expr_for_roi(
    overrides: Dict[int, Dict[str, np.ndarray]],
    roi_global: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Collect expression matrix for ROI spots."""
    keep = [int(g) for g in roi_global if int(g) in overrides]
    if not keep:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    
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
    
    X = np.zeros((len(keep), len(common_gene_ids)), dtype=np.float32)
    for i, g in enumerate(keep):
        gids = np.asarray(overrides[g]["gene_ids"], dtype=np.int64)
        vals = np.asarray(overrides[g]["raw_normed_values"], dtype=np.float32)
        pos = {int(gid): idx for idx, gid in enumerate(gids.tolist())}
        for j, cg in enumerate(common_gene_ids.tolist()):
            if int(cg) in pos:
                X[i, j] = vals[pos[int(cg)]]
    
    return X, common_gene_ids, keep


def _collect_input_expr_for_roi(
    databank: SpatialDataBank,
    roi_global: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Collect input expression matrix for ROI spots from databank."""
    keep: List[int] = []
    tmp: Dict[int, Dict[str, np.ndarray]] = {}
    
    for g in roi_global:
        gidx = int(g)
        try:
            base = databank.get_spot_data(gidx)
        except Exception:
            continue
        ov = databank.get_runtime_spot_override(gidx) if hasattr(databank, "get_runtime_spot_override") else None
        gene_ids = np.asarray(
            ov.get("gene_ids") if isinstance(ov, dict) and ov.get("gene_ids") is not None else base["gene_ids"],
            dtype=np.int64
        )
        vals = np.asarray(
            ov.get("raw_normed_values") if isinstance(ov, dict) and ov.get("raw_normed_values") is not None else base["raw_normed_values"],
            dtype=np.float32
        )
        keep.append(gidx)
        tmp[gidx] = {"gene_ids": gene_ids, "raw_normed_values": vals}
    
    return _collect_expr_for_roi(tmp, keep)


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


def compute_mean_vector_similarities(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> Dict[str, float]:
    """Compute similarity metrics between two mean vectors."""
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.shape != vec_b.shape:
        return {"pcc": float("nan"), "spearman": float("nan"), "cos": float("nan"), "l1": float("nan"), "l2": float("nan")}
    
    l1 = float(np.sum(np.abs(vec_a - vec_b)))
    l2 = float(np.sqrt(np.sum((vec_a - vec_b) ** 2)))
    
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a > 1e-8 and norm_b > 1e-8:
        cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    else:
        cos = float("nan")
    
    a_centered = vec_a - vec_a.mean()
    b_centered = vec_b - vec_b.mean()
    norm_ac = np.linalg.norm(a_centered)
    norm_bc = np.linalg.norm(b_centered)
    if norm_ac > 1e-8 and norm_bc > 1e-8:
        pcc = float(np.dot(a_centered, b_centered) / (norm_ac * norm_bc))
    else:
        pcc = float("nan")
    
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


def find_common_genes(gene_ids_list: List[np.ndarray]) -> np.ndarray:
    """Find common genes across multiple gene ID arrays."""
    if not gene_ids_list:
        return np.zeros((0,), dtype=np.int64)
    
    common = set(int(g) for g in gene_ids_list[0].tolist() if int(g) >= 0)
    for gids in gene_ids_list[1:]:
        common &= set(int(g) for g in gids.tolist() if int(g) >= 0)
    
    return np.array(sorted(common), dtype=np.int64)


def save_full_expression_to_csv(
    state: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
    vocab_path: str,
    cache_dir: Path,
    dataset_name: str,
    perturbed_indices: Optional[Set[int]] = None,
) -> None:
    """Save full expression matrix for all spots to CSV."""
    import anndata as ad
    
    if not state:
        print(f"[WARN] Empty state, skipping save to {out_path}")
        return
    
    # Load vocab for reverse mapping (gene_id -> gene_name)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_gene = {int(v): k for k, v in vocab.items()}
    
    # Get barcodes
    start_idx = _get_dataset_start_idx(cache_dir, dataset_name)
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    obs_names = list(proc.obs_names)
    
    # Find all gene_ids across all spots
    all_gene_ids: Set[int] = set()
    for gidx, ov in state.items():
        gids = ov["gene_ids"]
        for gid in gids:
            if int(gid) >= 0:
                all_gene_ids.add(int(gid))
    
    sorted_gene_ids = sorted(all_gene_ids)
    gene_names = [id_to_gene.get(gid, f"gene_{gid}") for gid in sorted_gene_ids]
    gid_to_col = {gid: i for i, gid in enumerate(sorted_gene_ids)}
    
    # Build expression matrix
    sorted_gidx = sorted(state.keys())
    n_spots = len(sorted_gidx)
    n_genes = len(sorted_gene_ids)
    
    expr_matrix = np.zeros((n_spots, n_genes), dtype=np.float32)
    barcodes = []
    is_perturbed_list = []
    
    for row_idx, gidx in enumerate(sorted_gidx):
        # Get barcode
        local_idx = int(gidx) - int(start_idx)
        if 0 <= local_idx < len(obs_names):
            barcodes.append(obs_names[local_idx])
        else:
            barcodes.append(f"spot_{gidx}")
        
        # Check if perturbed
        if perturbed_indices is not None:
            is_perturbed_list.append(1 if int(gidx) in perturbed_indices else 0)
        else:
            is_perturbed_list.append(0)
        
        # Fill expression values
        ov = state[gidx]
        for j, gid in enumerate(ov["gene_ids"]):
            gid_int = int(gid)
            if gid_int >= 0 and gid_int in gid_to_col:
                expr_matrix[row_idx, gid_to_col[gid_int]] = ov["raw_normed_values"][j]
    
    # Create DataFrame and save
    df = pd.DataFrame(expr_matrix, index=barcodes, columns=gene_names)
    df.index.name = "barcode"
    df["is_perturbed"] = is_perturbed_list
    
    # Reorder columns to put is_perturbed first
    cols = ["is_perturbed"] + [c for c in df.columns if c != "is_perturbed"]
    df = df[cols]
    
    df.to_csv(out_path)
    print(f"[OK] Saved full expression: {out_path} ({n_spots} spots × {n_genes} genes)")


def get_all_global_indices_from_databank(databank: SpatialDataBank) -> List[int]:
    """Get all global indices from databank."""
    try:
        n_spots = databank.num_spots if hasattr(databank, 'num_spots') else len(databank)
        return list(range(n_spots))
    except Exception:
        return list(range(1752))


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="KNN/Ridge baseline for niche perturbation evaluation")
    
    # Data paths
    ap.add_argument("--cache_pt", required=True, help="Cache directory for PT (lesion) slice")
    ap.add_argument("--cache_sham", required=True, help="Cache directory for Sham (healthy) slice")
    ap.add_argument("--pt_dataset_name", default="PT1-1")
    ap.add_argument("--sham_dataset_name", default="Sham1-1")
    ap.add_argument("--roi_manifest", required=True, help="JSON manifest with ROI barcodes")
    
    # Cache mode (h5 or lmdb)
    ap.add_argument("--cache_mode", choices=["h5", "lmdb"], default="h5")
    ap.add_argument("--lmdb_path_pt", type=str, default=None)
    ap.add_argument("--lmdb_manifest_pt", type=str, default=None)
    ap.add_argument("--lmdb_path_sham", type=str, default=None)
    ap.add_argument("--lmdb_manifest_sham", type=str, default=None)
    
    # Inference mode
    ap.add_argument("--infer_mode", choices=["knn_avg", "linear_reg_gpu"], default="knn_avg",
                    help="Inference method: knn_avg (KNN average), linear_reg_gpu (GPU Ridge)")
    
    # Ridge parameters
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--gpu_batch_size", type=int, default=256)
    ap.add_argument("--ridge_max_neighbors", type=int, default=8)
    ap.add_argument("--ridge_min_samples", type=int, default=10)
    
    # Perturbation config
    ap.add_argument("--perturb_mode", choices=["patch", "random"], default="patch")
    ap.add_argument("--n_spots", type=int, default=20)
    ap.add_argument("--patch_size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weighting", choices=["gaussian", "uniform"], default="gaussian")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--deg_csv", required=True)
    ap.add_argument("--p_adj_thresh", type=float, default=0.1)
    ap.add_argument("--min_abs_logfc", type=float, default=0.0)
    ap.add_argument("--logfc_strength", type=float, default=1.0)
    ap.add_argument("--logfc_clip", type=float, default=5.0)
    
    # Iteration settings
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--freeze_perturbed", action="store_true", default=True)
    
    # Performance
    ap.add_argument("--show_progress", action="store_true", default=True)
    
    # Output
    ap.add_argument("--out_dir", required=True)
    
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("KNN/Ridge Niche Perturbation Evaluation")
    print("=" * 70)
    print(f"Inference mode: {args.infer_mode}")
    print(f"Target ROI: ICA, Steps: {args.steps}, Mode: {args.perturb_mode}")
    print("=" * 70)
    
    # Load manifest
    manifest = _load_json(Path(args.roi_manifest))
    pt_key = "pt" if "pt" in manifest else "PT1_1"
    sham_key = "sham" if "sham" in manifest else "Sham1_1"
    pt_roi_barcodes = {k: manifest[pt_key][k]["barcodes"] for k in ["ICA", "PIA_P", "PIA_D"]}
    sham_roi_barcodes = {k: manifest[sham_key][k]["barcodes"] for k in ["ICA", "PIA_P", "PIA_D"]}
    
    # Build PT databank
    pt_config = Config()
    pt_config = _setup_config_for_cache(
        pt_config, args.cache_pt, args.pt_dataset_name, args.cache_mode,
        args.lmdb_path_pt, args.lmdb_manifest_pt
    )
    
    pt_bank = SpatialDataBank(
        dataset_paths=[str(Path(args.cache_pt) / args.pt_dataset_name / "processed.h5ad")],
        cache_dir=str(Path(args.cache_pt)),
        config=pt_config,
        force_rebuild=False,
    )
    
    pt_roi_global = {k: _barcodes_to_global_idx(Path(args.cache_pt), args.pt_dataset_name, pt_roi_barcodes[k])
                     for k in ["ICA", "PIA_P", "PIA_D"]}
    
    # Build Sham databank
    sham_config = Config()
    sham_config = _setup_config_for_cache(
        sham_config, args.cache_sham, args.sham_dataset_name, args.cache_mode,
        args.lmdb_path_sham, args.lmdb_manifest_sham
    )
    
    sham_bank = SpatialDataBank(
        dataset_paths=[str(Path(args.cache_sham) / args.sham_dataset_name / "processed.h5ad")],
        cache_dir=str(Path(args.cache_sham)),
        config=sham_config,
        force_rebuild=False,
    )
    
    sham_roi_global = {k: _barcodes_to_global_idx(Path(args.cache_sham), args.sham_dataset_name, sham_roi_barcodes[k])
                       for k in ["ICA", "PIA_P", "PIA_D"]}
    
    all_sham_global = get_all_global_indices_from_databank(sham_bank)
    
    # Train Ridge model if needed (on X_0)
    ridge_model = None
    if args.infer_mode == "linear_reg_gpu":
        print("\n[STEP] Training GPU Ridge model on X_0...")
        ridge_model = GPURidgeRegressionModel(
            alpha=args.ridge_alpha,
            max_neighbors=args.ridge_max_neighbors,
            device="cuda"
        )
        sham_bank.clear_runtime_spot_overrides()
        ridge_model.train(sham_bank, all_sham_global, current_state=None,
                         min_samples=args.ridge_min_samples, show_progress=args.show_progress)
    
    # Compute PT reference expression
    print("\n[STEP] Computing PT reference expression...")
    pt_bank.clear_runtime_spot_overrides() if hasattr(pt_bank, "clear_runtime_spot_overrides") else None
    
    pt_expr_by_roi: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        pt_expr_by_roi[roi] = _collect_input_expr_for_roi(pt_bank, pt_roi_global[roi])
    
    # Load X_0 (raw Sham data)
    print("\n[STEP] Loading X_0 (raw Sham data)...")
    sham_bank.clear_runtime_spot_overrides()
    
    x0_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in tqdm(all_sham_global, desc="Loading X_0", disable=not args.show_progress):
        try:
            sd = sham_bank.get_spot_data(gidx)
            x0_state[int(gidx)] = {
                "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64).copy(),
                "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32).copy(),
            }
        except Exception:
            pass
    print(f"[INFO] X_0: {len(x0_state)} spots loaded")
    
    # Collect Sham control expression
    sham_ctrl_expr_by_roi: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        sham_ctrl_expr_by_roi[roi] = _collect_expr_for_roi(x0_state, sham_roi_global[roi])
    
    # Select spots for perturbation (ICA region only)
    print("\n[STEP] Selecting spots for perturbation...")
    target_roi = "ICA"
    target_roi_global = sham_roi_global[target_roi]
    
    if args.perturb_mode == "patch":
        selected_global = select_patch_spots(sham_bank, target_roi_global, args.patch_size, args.seed)
    else:
        selected_global = select_random_spots(target_roi_global, args.n_spots, args.seed)
    
    print(f"[INFO] Selected {len(selected_global)} spots for perturbation in {target_roi}")
    
    # Compute weights
    weights, weight_meta = compute_spot_weights(
        Path(args.cache_sham), args.sham_dataset_name, selected_global,
        args.weighting, args.sigma
    )
    
    # Apply perturbation
    perturb_overrides, perturb_meta = apply_deg_perturbation(
        sham_bank, selected_global, weights, Path(args.deg_csv),
        args.p_adj_thresh, args.min_abs_logfc,
        args.logfc_strength, args.logfc_clip
    )
    
    perturbed_spot_set = set(int(x) for x in selected_global)
    
    # Write manifest
    perturb_manifest = {
        "baseline_method": args.infer_mode,
        "perturb_target_roi": target_roi,
        "perturb_mode": args.perturb_mode,
        "n_spots_selected": len(selected_global),
        "seed": args.seed,
        "selected_global_indices": selected_global,
        "selected_barcodes": _global_idx_to_barcode(Path(args.cache_sham), args.sham_dataset_name, selected_global),
        "weights": weights.tolist(),
        **weight_meta,
        **perturb_meta,
        "steps": args.steps,
        "freeze_perturbed": args.freeze_perturbed,
        "cache_mode": args.cache_mode,
        "ridge_alpha": args.ridge_alpha if args.infer_mode == "linear_reg_gpu" else None,
    }
    
    with open(out_dir / "perturb_manifest.json", "w", encoding="utf-8") as f:
        json.dump(perturb_manifest, f, ensure_ascii=False, indent=2)
    print("[OK] Wrote perturb_manifest.json")
    
    # Build X'_0 (perturbed initial state)
    print("\n[STEP] Building X'_0 (perturbed initial state)...")
    x0_prime_state: Dict[int, Dict[str, np.ndarray]] = {}
    
    # Copy X_0 first
    for gidx, ov in x0_state.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # Apply perturbation
    for gidx, ov in perturb_overrides.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32),
        }
    
    print(f"[INFO] X'_0: {len(perturbed_spot_set)} spots perturbed")
    
    # Collect pert0 expression
    sham_pert0_expr_by_roi: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    for roi in ["ICA", "PIA_P", "PIA_D"]:
        sham_pert0_expr_by_roi[roi] = _collect_expr_for_roi(x0_prime_state, sham_roi_global[roi])
    
    # Run dual-line iterative inference
    print(f"\n[STEP] Running dual-line inference for {args.steps} steps...")
    frozen_indices = set(selected_global) if args.freeze_perturbed else set()
    
    step_overrides, final_state = run_dual_line_iterative_inference(
        sham_bank, all_sham_global,
        steps=args.steps,
        frozen_indices=frozen_indices,
        x0_state=x0_state,
        x0_prime_state=x0_prime_state,
        infer_mode=args.infer_mode,
        ridge_model=ridge_model,
        show_progress=args.show_progress,
    )
    
    # Save full expression for all steps
    print(f"\n[STEP] Saving full expression for all steps...")
    expr_out_dir = out_dir / "expression"
    expr_out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = Config().vocab_file
    
    # Save step 0 (X'_0)
    save_full_expression_to_csv(
        x0_prime_state,
        expr_out_dir / "pert0_expression.csv",
        vocab_path,
        Path(args.cache_sham),
        args.sham_dataset_name,
        perturbed_spot_set,
    )
    
    # Save step 1..T
    for t in range(1, args.steps + 1):
        if t in step_overrides:
            save_full_expression_to_csv(
                step_overrides[t],
                expr_out_dir / f"pert{t}_expression.csv",
                vocab_path,
                Path(args.cache_sham),
                args.sham_dataset_name,
                perturbed_spot_set,
            )
    
    # Compute similarities
    print("\n[STEP] Computing similarities...")
    
    eval_rois = ["ICA", "PIA_P", "PIA_D"]
    results_rows: List[Dict] = []
    mean_vectors_by_step: Dict[int, Dict[str, Dict]] = {}
    
    # Step 0: ctrl0 and pert0
    mean_vectors_by_step[0] = {}
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
        
        mean_vectors_by_step[0][roi] = {
            "pt": pt_mean, "ctrl": ctrl_mean, "pert": pert0_mean, "gene_ids": common_genes,
        }
        
        ctrl0_sim = compute_mean_vector_similarities(pt_mean, ctrl_mean)
        pert0_sim = compute_mean_vector_similarities(pt_mean, pert0_mean)
        
        for metric in ["pcc", "spearman", "cos", "l1", "l2"]:
            results_rows.append({
                "baseline_method": args.infer_mode,
                "perturb_target_roi": target_roi, "eval_roi": roi,
                "step": 0, "group": "ctrl0", "metric": metric, "value": ctrl0_sim[metric],
                "n_genes_common": len(common_genes), "freeze_perturbed": args.freeze_perturbed,
            })
            results_rows.append({
                "baseline_method": args.infer_mode,
                "perturb_target_roi": target_roi, "eval_roi": roi,
                "step": 0, "group": "pert0", "metric": metric, "value": pert0_sim[metric],
                "n_genes_common": len(common_genes), "freeze_perturbed": args.freeze_perturbed,
            })
    
    # Steps 1..T
    for t in range(1, args.steps + 1):
        mean_vectors_by_step[t] = {}
        step_ov = step_overrides.get(t, {})
        
        for roi in eval_rois:
            pt_X, pt_gids, _ = pt_expr_by_roi[roi]
            pert_X, pert_gids, _ = _collect_expr_for_roi(step_ov, sham_roi_global[roi])
            
            if pert_X.size == 0:
                continue
            
            common_genes = find_common_genes([pt_gids, pert_gids])
            if common_genes.size == 0:
                continue
            
            pt_mean = align_and_compute_mean_vector(pt_X, pt_gids, common_genes)
            pert_mean = align_and_compute_mean_vector(pert_X, pert_gids, common_genes)
            
            mean_vectors_by_step[t][roi] = {"pt": pt_mean, "pert": pert_mean, "gene_ids": common_genes}
            
            pert_sim = compute_mean_vector_similarities(pt_mean, pert_mean)
            
            for metric in ["pcc", "spearman", "cos", "l1", "l2"]:
                results_rows.append({
                    "baseline_method": args.infer_mode,
                    "perturb_target_roi": target_roi, "eval_roi": roi,
                    "step": t, "group": f"pert{t}", "metric": metric, "value": pert_sim[metric],
                    "n_genes_common": len(common_genes), "freeze_perturbed": args.freeze_perturbed,
                })
    
    # Write outputs
    print("\n[STEP] Writing outputs...")
    
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"rows": results_rows}, f, ensure_ascii=False, indent=2)
    
    print("[OK] Wrote summary.csv and summary.json")
    
    for t, roi_data in mean_vectors_by_step.items():
        save_dict = {}
        for roi, data in roi_data.items():
            for key, val in data.items():
                save_dict[f"{roi}_{key}"] = val
        if save_dict:
            np.savez_compressed(out_dir / f"roi_mean_vectors_step{t}.npz", **save_dict)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"SIMILARITY SUMMARY ({args.infer_mode})")
    print("=" * 60)
    
    for roi in eval_rois:
        print(f"\n[ROI: {roi}]")
        roi_rows = [r for r in results_rows if r["eval_roi"] == roi and r["metric"] == "pcc"]
        for r in sorted(roi_rows, key=lambda x: x["step"]):
            print(f"  step={r['step']} group={r['group']:6s} PCC={r['value']:.4f}")
    
    print("\n[DONE] All outputs written to:", out_dir)


if __name__ == "__main__":
    main()
