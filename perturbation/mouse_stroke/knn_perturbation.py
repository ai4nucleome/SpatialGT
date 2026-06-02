#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN and ridge mouse stroke perturbation baselines.

This public script keeps only random-spot DEG perturbation and dual-line
iteration. The initial control state is the raw Sham expression state, matching
the original baseline implementation. Reference-slice output scoring and
exploratory selection branches have been removed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pretrain.Config import Config
from pretrain.spatial_databank import SpatialDataBank

def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_dataset_start_idx(cache_dir: Path, dataset_name: str) -> int:
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


def select_random_spots(roi_global_indices: List[int], n_spots: int, seed: int) -> List[int]:
    """Select scattered spots from the requested ROI."""
    rng = random.Random(seed)
    roi_list = sorted(set(int(x) for x in roi_global_indices))
    if n_spots <= 0 or len(roi_list) <= n_spots:
        return roi_list
    return rng.sample(roi_list, n_spots)


def compute_uniform_weights(selected_global_indices: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Use equal perturbation strength for all selected spots."""
    weights = np.ones(len(selected_global_indices), dtype=np.float32)
    return weights, {"weighting": "uniform"}


def load_deg_gene_logfc(
    deg_csv: Path,
    p_adj_thresh: float,
    min_abs_logfc: float,
    logfc_clip: Optional[float],
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Load a DEG table and map gene symbols to SpatialGT vocabulary IDs."""
    deg = pd.read_csv(deg_csv)
    if "avg_logFC" not in deg.columns:
        raise ValueError(f"DEG csv missing required column avg_logFC: {deg.columns.tolist()}")
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]

    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col is not None:
        deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
        deg = deg[np.isfinite(deg[p_col])]
        deg = deg[deg[p_col].astype(float) <= float(p_adj_thresh)]
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]

    vocab_path = Config().vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {k.lower(): int(v) for k, v in vocab_json.items()}

    gene_logfc: Dict[int, float] = {}
    n_up_genes = 0
    n_down_genes = 0
    for _, row in deg.iterrows():
        gid = vocab.get(str(row["gene"]).lower())
        if gid is None:
            continue
        logfc_val = float(row["avg_logFC"])
        if logfc_clip is not None:
            logfc_val = float(np.clip(logfc_val, -float(logfc_clip), float(logfc_clip)))
        gene_logfc[int(gid)] = logfc_val
        if logfc_val > 0:
            n_up_genes += 1
        elif logfc_val < 0:
            n_down_genes += 1

    meta = {
        "deg_csv": str(deg_csv),
        "p_col": p_col,
        "p_adj_thresh": float(p_adj_thresh),
        "min_abs_logfc": float(min_abs_logfc),
        "logfc_clip": float(logfc_clip) if logfc_clip is not None else None,
        "n_deg_rows_used": int(len(deg)),
        "n_deg_genes_mapped": int(len(gene_logfc)),
        "n_up_gene_ids": int(n_up_genes),
        "n_down_gene_ids": int(n_down_genes),
    }
    return gene_logfc, meta


def apply_deg_perturbation_to_state(
    base_state: Dict[int, Dict[str, np.ndarray]],
    selected_global_indices: List[int],
    weights: np.ndarray,
    gene_logfc: Dict[int, float],
    logfc_strength: float,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Apply a high-dimensional logFC perturbation to selected spots in log1p space."""
    out: Dict[int, Dict[str, np.ndarray]] = {
        int(g): {
            "gene_ids": np.asarray(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in base_state.items()
    }
    n_hits_per_spot = []
    logfc_applied = []
    for i, gidx in enumerate(selected_global_indices):
        gidx = int(gidx)
        if gidx not in out:
            continue
        weight = float(weights[i]) if i < len(weights) else 1.0
        gene_ids = out[gidx]["gene_ids"]
        vals = out[gidx]["raw_normed_values"].copy()
        n_hits = 0
        for j, gid in enumerate(gene_ids):
            logfc_val = gene_logfc.get(int(gid))
            if logfc_val is None:
                continue
            fold_change = np.power(2.0, logfc_val * float(logfc_strength) * weight)
            linear_val = np.expm1(vals[j])
            vals[j] = np.log1p(max(linear_val * fold_change, 0.0))
            n_hits += 1
            logfc_applied.append(logfc_val)
        vals = np.maximum(vals, 0.0)
        out[gidx] = {"gene_ids": gene_ids.copy(), "raw_normed_values": vals.astype(np.float32)}
        n_hits_per_spot.append(n_hits)

    meta = {
        "logfc_strength": float(logfc_strength),
        "n_hits_per_spot_stats": {
            "min": int(np.min(n_hits_per_spot)) if n_hits_per_spot else 0,
            "max": int(np.max(n_hits_per_spot)) if n_hits_per_spot else 0,
            "mean": float(np.mean(n_hits_per_spot)) if n_hits_per_spot else 0.0,
        },
        "logfc_applied_stats": {
            "min": float(np.min(logfc_applied)) if logfc_applied else 0.0,
            "max": float(np.max(logfc_applied)) if logfc_applied else 0.0,
            "mean": float(np.mean(logfc_applied)) if logfc_applied else 0.0,
        },
    }
    return out, meta


def load_roi_manifest(roi_manifest: Path) -> Dict[str, List[str]]:
    """Load Sham ROI barcodes from the manifest."""
    manifest = _load_json(roi_manifest)
    key = "sham" if "sham" in manifest else "Sham1_1"
    return {roi: manifest[key][roi]["barcodes"] for roi in ["ICA", "PIA_P", "PIA_D"]}


def get_all_global_indices(cache_dir: Path, dataset_name: str) -> List[int]:
    """Return all global spot indices for a single-section cache."""
    import anndata as ad
    start_idx = _get_dataset_start_idx(cache_dir, dataset_name)
    adata = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    return list(range(start_idx, start_idx + int(adata.n_obs)))
def get_spot_expression(
    databank: SpatialDataBank,
    global_idx: int,
    current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
        if not self.is_trained:
            return expr_matrix
        
        if batch_indices is None:
            nb_indices = self.neighbor_table
            nb_mask = self.neighbor_mask
        else:
            batch_idx_t = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
            nb_indices = self.neighbor_table[batch_idx_t]
            nb_mask = self.neighbor_mask[batch_idx_t]
        
        nb_indices_safe = nb_indices.clamp(min=0)
        nb_expr = expr_matrix[nb_indices_safe]   # [batch, max_neighbors, n_genes]
        nb_expr = nb_expr * nb_mask.unsqueeze(-1)
        
        actual_k = nb_expr.shape[1]
        trained_k = self.weights.shape[1]
        
        if actual_k < trained_k:
            W = self.weights[:, :actual_k]
        elif actual_k > trained_k:
            W = F.pad(self.weights, (0, actual_k - trained_k))
        else:
            W = self.weights
        
        W_t = W.T   # [actual_k, n_genes]
        predictions = torch.einsum('bkg,kg->bg', nb_expr, W_t) + self.bias.unsqueeze(0)
        predictions = F.relu(predictions)
        
        return predictions
    
    def use_engine_neighbor_table(self, engine: 'GPUKNNEngine') -> None:
        """Adopt the neighbor table from a GPUKNNEngine for GPU dual-line inference.

        Truncates or pads per-spot neighbor lists to self.max_neighbors so that
        the weight matrix dimension matches.
        """
        src_table = engine.neighbor_table   # [n_spots, engine_max_k]
        src_mask = engine.neighbor_mask
        engine_max_k = src_table.shape[1]
        
        if engine_max_k <= self.max_neighbors:
            pad_cols = self.max_neighbors - engine_max_k
            self.neighbor_table = F.pad(src_table, (0, pad_cols), value=-1)
            self.neighbor_mask = F.pad(src_mask, (0, pad_cols), value=0.0)
        else:
            self.neighbor_table = src_table[:, :self.max_neighbors]
            self.neighbor_mask = src_mask[:, :self.max_neighbors]
        
        self.global_to_local = dict(engine.global_to_local)
        self.local_to_global = dict(engine.local_to_global)


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
# GPU KNN Engine - Batch GPU-accelerated KNN average
# ==============================================================================

class GPUKNNEngine:
    """GPU-accelerated KNN average inference engine.

    Converts the per-spot Python loop into batched GPU matrix operations:
      predictions[i] = mean(expr[neighbors[i]])
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.neighbor_table: Optional[torch.Tensor] = None
        self.neighbor_mask: Optional[torch.Tensor] = None
        self.n_valid_neighbors: Optional[torch.Tensor] = None
        self.gene_id_to_col: Dict[int, int] = {}
        self.global_to_local: Dict[int, int] = {}
        self.local_to_global: Dict[int, int] = {}
        self.n_spots: int = 0
        self.n_genes: int = 0
        self._spot_col_idx: Dict[int, np.ndarray] = {}
        self._spot_pos_idx: Dict[int, np.ndarray] = {}
        self._spot_gids: Dict[int, np.ndarray] = {}
        self._spot_n_orig: Dict[int, int] = {}

    def build(
        self,
        databank: SpatialDataBank,
        all_global_indices: List[int],
        initial_state: Dict[int, Dict[str, np.ndarray]],
        show_progress: bool = True,
    ) -> None:
        """Build neighbor table, gene vocabulary, and per-spot mapping arrays."""
        import time as _time
        t0 = _time.time()

        self.n_spots = len(all_global_indices)
        self.global_to_local = {int(g): i for i, g in enumerate(all_global_indices)}
        self.local_to_global = {i: int(g) for i, g in enumerate(all_global_indices)}

        all_gene_ids: Set[int] = set()
        spot_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for gidx in tqdm(all_global_indices, desc="[GPU KNN] Collecting genes",
                         disable=not show_progress):
            gids, vals = get_spot_expression(databank, int(gidx), initial_state)
            gids = np.asarray(gids, dtype=np.int64)
            vals = np.asarray(vals, dtype=np.float32)
            spot_data[int(gidx)] = (gids, vals)
            all_gene_ids.update(int(g) for g in gids if int(g) >= 0)

        sorted_genes = sorted(all_gene_ids)
        self.gene_id_to_col = {gid: i for i, gid in enumerate(sorted_genes)}
        self.n_genes = len(sorted_genes)

        for i in range(self.n_spots):
            gidx = self.local_to_global[i]
            gids, _ = spot_data[gidx]
            cols, positions = [], []
            for j, gid in enumerate(gids):
                gid_int = int(gid)
                if gid_int >= 0 and gid_int in self.gene_id_to_col:
                    cols.append(self.gene_id_to_col[gid_int])
                    positions.append(j)
            self._spot_col_idx[i] = np.array(cols, dtype=np.int64)
            self._spot_pos_idx[i] = np.array(positions, dtype=np.int64)
            self._spot_gids[i] = gids.copy()
            self._spot_n_orig[i] = len(gids)

        max_k = 0
        nb_cache: Dict[int, List[int]] = {}
        for i in range(self.n_spots):
            gidx = self.local_to_global[i]
            try:
                nbs = databank.get_neighbors_for_spot(int(gidx))
                nbs = [int(n) for n in nbs if int(n) in self.global_to_local]
            except Exception:
                nbs = []
            nb_cache[i] = nbs
            if len(nbs) > max_k:
                max_k = len(nbs)

        nb_table = np.zeros((self.n_spots, max_k), dtype=np.int64)
        nb_mask = np.zeros((self.n_spots, max_k), dtype=np.float32)
        for i, nbs in nb_cache.items():
            for ni, nb_global in enumerate(nbs):
                nb_table[i, ni] = self.global_to_local[nb_global]
                nb_mask[i, ni] = 1.0

        self.neighbor_table = torch.from_numpy(nb_table).to(self.device)
        self.neighbor_mask = torch.from_numpy(nb_mask).to(self.device)
        self.n_valid_neighbors = self.neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)

        elapsed = _time.time() - t0
        print(f"[GPU KNN] Built engine: {self.n_spots} spots, {self.n_genes} genes, "
              f"max_k={max_k}, device={self.device}  ({elapsed:.1f}s)")

    def state_to_matrix(self, state: Dict[int, Dict[str, np.ndarray]]) -> torch.Tensor:
        """Convert state dict to dense [n_spots, n_genes] GPU tensor."""
        expr = np.zeros((self.n_spots, self.n_genes), dtype=np.float32)
        for i in range(self.n_spots):
            gidx = self.local_to_global[i]
            if gidx not in state:
                continue
            vals = np.asarray(state[gidx]["raw_normed_values"], dtype=np.float32)
            cols = self._spot_col_idx[i]
            pos = self._spot_pos_idx[i]
            if len(cols) > 0:
                expr[i, cols] = vals[pos]
        return torch.from_numpy(expr).to(self.device)

    def matrix_to_state(self, expr_matrix: torch.Tensor) -> Dict[int, Dict[str, np.ndarray]]:
        """Convert dense GPU tensor back to state dict."""
        expr_np = expr_matrix.cpu().numpy()
        state: Dict[int, Dict[str, np.ndarray]] = {}
        for i in range(self.n_spots):
            gidx = self.local_to_global[i]
            vals = np.zeros(self._spot_n_orig[i], dtype=np.float32)
            cols = self._spot_col_idx[i]
            pos = self._spot_pos_idx[i]
            if len(cols) > 0:
                vals[pos] = expr_np[i, cols]
            state[gidx] = {
                "gene_ids": self._spot_gids[i].copy(),
                "raw_normed_values": vals,
            }
        return state

    @torch.no_grad()
    def predict(self, expr_matrix: torch.Tensor) -> torch.Tensor:
        """KNN average on GPU: gather neighbor expressions and average."""
        nb_expr = expr_matrix[self.neighbor_table]
        nb_expr = nb_expr * self.neighbor_mask.unsqueeze(-1)
        return nb_expr.sum(dim=1) / self.n_valid_neighbors


def run_dual_line_gpu(
    engine: GPUKNNEngine,
    x0_state: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    frozen_indices: Set[int],
    steps: int = 10,
    ridge_model: Optional[GPURidgeRegressionModel] = None,
) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], Dict[int, Dict[str, np.ndarray]]]:
    """GPU-accelerated dual-line iterative inference.

    X_k = predict(X_0) is constant across all steps → computed once.
    Each step only runs one GPU pass on X'_{k-1}.

    When ridge_model is provided, uses Ridge regression instead of KNN average.
    Both models keep everything on GPU tensors throughout the loop.
    """
    import time as _time

    mode_name = "Ridge" if ridge_model is not None else "KNN"

    X0 = engine.state_to_matrix(x0_state)
    X0_prime = engine.state_to_matrix(x0_prime_state)

    frozen_mask = torch.zeros(engine.n_spots, dtype=torch.bool, device=engine.device)
    for gidx in frozen_indices:
        if gidx in engine.global_to_local:
            frozen_mask[engine.global_to_local[gidx]] = True

    if ridge_model is not None:
        X_k = ridge_model.predict_batch(X0)
    else:
        X_k = engine.predict(X0)
    print(f"[GPU {mode_name}] Pre-computed X_k (unperturbed line, constant across steps)")

    X_prime_current = X0_prime.clone()
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    for t in range(1, steps + 1):
        t0 = _time.time()

        if ridge_model is not None:
            X_prime_raw = ridge_model.predict_batch(X_prime_current)
        else:
            X_prime_raw = engine.predict(X_prime_current)
        delta = X_prime_raw - X_k
        X_prime_true = X0_prime + delta
        X_prime_true[frozen_mask] = X0_prime[frozen_mask]

        nf_mask = ~frozen_mask
        if nf_mask.any():
            delta_nf = delta[nf_mask]
            d_mean = delta_nf.mean().item()
            d_abs = delta_nf.abs().mean().item()
        else:
            d_mean = d_abs = 0.0

        X_prime_current = X_prime_true.clone()
        step_overrides[t] = engine.matrix_to_state(X_prime_true)

        elapsed = _time.time() - t0
        print(f"  [STEP {t}/{steps}] delta mean={d_mean:.6f}  abs_mean={d_abs:.6f}  ({elapsed:.2f}s)")

    final_state = engine.matrix_to_state(X_prime_current)
    return step_overrides, final_state


# ==============================================================================
# : Dual-line iterative inference with error cancellation (CPU fallback)
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
        print(f"\n[STEP {t}/{steps}] Dual-line inference ()...")
        
        # =====================================================================
        # Step 1: X_k = Model(X_0) - Run unperturbed line from X_0
        # =====================================================================
        print(f"  [X line] Running {infer_mode} on X_0...")
        x_k_pred = run_single_pass_inference(
            databank, all_global_indices, infer_mode, x0_state, ridge_model, show_progress,
            desc=f"X line step {t}"
        )
        
        # =====================================================================
        # Step 2: X'_k_raw = Model(X'_{k-1}_true) - Run perturbed line
        # =====================================================================
        print(f"  [X' line] Running {infer_mode} on X'_{t-1}_true...")
        x_prime_k_raw = run_single_pass_inference(
            databank, all_global_indices, infer_mode, x_prime_current, ridge_model, show_progress,
            desc=f"X' line step {t}"
        )
        
        # =====================================================================
        # Step 3-5: Compute delta and X'_k_true
        # =====================================================================
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

def save_step_npz(
    state: Dict[int, Dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Save one step's per-spot expression to compressed npz (same format as SpatialGT)."""
    spot_indices = sorted(state.keys())
    save_dict = {"spot_indices": np.array(spot_indices, dtype=np.int64)}
    for gidx in spot_indices:
        ov = state[gidx]
        save_dict[f"gids_{gidx}"] = ov["gene_ids"]
        save_dict[f"vals_{gidx}"] = ov["raw_normed_values"]
    np.savez_compressed(str(path), **save_dict)


def save_full_expression_to_csv(
    state: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
    vocab_path: str,
    cache_dir: Path,
    dataset_name: str,
) -> None:
    """
    Save full expression matrix for all spots to CSV.
    
    Args:
        state: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}
        out_path: output CSV file path
        vocab_path: path to vocab.json for gene name mapping
        cache_dir: cache directory for barcode lookup
        dataset_name: dataset name for barcode lookup
    """
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
    
    for row_idx, gidx in enumerate(sorted_gidx):
        # Get barcode
        local_idx = int(gidx) - int(start_idx)
        if 0 <= local_idx < len(obs_names):
            barcodes.append(obs_names[local_idx])
        else:
            barcodes.append(f"spot_{gidx}")
        
        # Fill expression values
        ov = state[gidx]
        for j, gid in enumerate(ov["gene_ids"]):
            gid_int = int(gid)
            if gid_int >= 0 and gid_int in gid_to_col:
                expr_matrix[row_idx, gid_to_col[gid_int]] = ov["raw_normed_values"][j]
    
    # Create DataFrame and save
    df = pd.DataFrame(expr_matrix, index=barcodes, columns=gene_names)
    df.index.name = "barcode"
    df.to_csv(out_path)
    print(f"[OK] Saved full expression: {out_path} ({n_spots} spots × {n_genes} genes)")


def parse_save_expr_steps(save_expr_steps: str, max_steps: int) -> Set[int]:
    """Parse --save_expr_steps argument into set of step numbers."""
    if not save_expr_steps or save_expr_steps.strip() == "":
        return set()
    
    save_expr_steps = save_expr_steps.strip().lower()
    if save_expr_steps == "all":
        return set(range(max_steps + 1))
    
    steps = set()
    for s in save_expr_steps.split(","):
        s = s.strip()
        if s.isdigit():
            steps.add(int(s))
    return steps


def get_all_global_indices_from_databank(databank: SpatialDataBank) -> List[int]:
    try:
        n_spots = databank.num_spots if hasattr(databank, 'num_spots') else len(databank)
        return list(range(n_spots))
    except Exception:
        return list(range(1752))


def main():
    parser = argparse.ArgumentParser(description="KNN/ridge mouse stroke perturbation baseline")
    parser.add_argument("--cache_sham", required=True)
    parser.add_argument("--sham_dataset_name", default="Sham1-1")
    parser.add_argument("--roi_manifest", required=True)
    parser.add_argument("--deg_csv", required=True)
    parser.add_argument("--cache_mode", choices=["h5", "lmdb"], default="h5")
    parser.add_argument("--lmdb_path_sham", default=None)
    parser.add_argument("--lmdb_manifest_sham", default=None)
    parser.add_argument("--infer_mode", choices=["knn_avg", "linear_reg_gpu"], default="knn_avg")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--ridge_max_neighbors", type=int, default=8)
    parser.add_argument("--ridge_min_samples", type=int, default=10)
    parser.add_argument("--perturb_target_roi", choices=["ICA", "PIA_P", "PIA_D"], default="ICA")
    parser.add_argument("--n_spots", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_adj_thresh", type=float, default=0.05)
    parser.add_argument("--min_abs_logfc", type=float, default=0.0)
    parser.add_argument("--logfc_strength", type=float, default=1.0)
    parser.add_argument("--logfc_clip", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no_freeze_perturbed", action="store_true")
    parser.add_argument("--save_expr_steps", default="")
    parser.add_argument("--show_progress", action="store_true", default=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expr_dir = out_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    config = _setup_config_for_cache(config, args.cache_sham, args.sham_dataset_name, args.cache_mode, args.lmdb_path_sham, args.lmdb_manifest_sham)
    databank = SpatialDataBank(
        dataset_paths=[str(Path(args.cache_sham) / args.sham_dataset_name / "processed.h5ad")],
        cache_dir=str(Path(args.cache_sham)),
        config=config,
        load_data=False,
    )
    all_global = get_all_global_indices(Path(args.cache_sham), args.sham_dataset_name)

    raw_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in tqdm(all_global, desc="Loading raw state", disable=not args.show_progress):
        sd = databank.get_spot_data(int(gidx))
        raw_state[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32).copy(),
        }

    ridge_model = None
    if args.infer_mode == "linear_reg_gpu":
        ridge_model = GPURidgeRegressionModel(alpha=args.ridge_alpha, max_neighbors=args.ridge_max_neighbors, device="cuda")
        ridge_model.train(databank, all_global, current_state=raw_state, min_samples=args.ridge_min_samples, show_progress=args.show_progress)

    print("[STEP] Using raw Sham expression as X_0")
    x0_state = raw_state

    roi_barcodes = load_roi_manifest(Path(args.roi_manifest))
    sham_roi_global = {
        roi: _barcodes_to_global_idx(Path(args.cache_sham), args.sham_dataset_name, barcodes)
        for roi, barcodes in roi_barcodes.items()
    }
    selected_global = select_random_spots(sham_roi_global[args.perturb_target_roi], args.n_spots, args.seed)
    weights, weight_meta = compute_uniform_weights(selected_global)
    gene_logfc, deg_meta = load_deg_gene_logfc(Path(args.deg_csv), args.p_adj_thresh, args.min_abs_logfc, args.logfc_clip)
    x0_prime_state, perturb_meta = apply_deg_perturbation_to_state(x0_state, selected_global, weights, gene_logfc, args.logfc_strength)
    frozen_indices = set(selected_global) if not args.no_freeze_perturbed else set()

    manifest = {
        "method": args.infer_mode,
        "algorithm": "dual_line_error_cancellation",
        "x0_source": "raw_sham_expression",
        "perturb_target_roi": args.perturb_target_roi,
        "spot_selection": "random",
        "n_spots_selected": len(selected_global),
        "seed": args.seed,
        "selected_global_indices": selected_global,
        "selected_barcodes": _global_idx_to_barcode(Path(args.cache_sham), args.sham_dataset_name, selected_global),
        "weights": weights.tolist(),
        "freeze_perturbed": not args.no_freeze_perturbed,
        **weight_meta,
        **deg_meta,
        **perturb_meta,
    }
    with open(out_dir / "perturb_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if torch.cuda.is_available():
        engine = GPUKNNEngine(device="cuda")
        engine.build(databank, all_global, x0_state, show_progress=args.show_progress)
        if args.infer_mode == "linear_reg_gpu" and ridge_model is not None:
            ridge_model.use_engine_neighbor_table(engine)
            step_overrides, final_state = run_dual_line_gpu(engine, x0_state, x0_prime_state, frozen_indices, args.steps, ridge_model)
        else:
            step_overrides, final_state = run_dual_line_gpu(engine, x0_state, x0_prime_state, frozen_indices, args.steps)
    else:
        step_overrides, final_state = run_dual_line_iterative_inference(
            databank,
            all_global,
            steps=args.steps,
            frozen_indices=frozen_indices,
            x0_state=x0_state,
            x0_prime_state=x0_prime_state,
            infer_mode=args.infer_mode,
            ridge_model=ridge_model,
            show_progress=args.show_progress,
        )

    save_step_npz(x0_state, expr_dir / "ctrl.npz")
    save_step_npz(x0_prime_state, expr_dir / "step0.npz")
    for step, state in step_overrides.items():
        save_step_npz(state, expr_dir / f"step{step}.npz")

    save_steps = parse_save_expr_steps(args.save_expr_steps, args.steps)
    if save_steps:
        vocab_path = Config().vocab_file
        if 0 in save_steps:
            save_full_expression_to_csv(x0_prime_state, expr_dir / "expression_step0.csv", vocab_path, Path(args.cache_sham), args.sham_dataset_name)
        for step, state in step_overrides.items():
            if step in save_steps:
                save_full_expression_to_csv(state, expr_dir / f"expression_step{step}.csv", vocab_path, Path(args.cache_sham), args.sham_dataset_name)

    print("[OK] Finished KNN/ridge perturbation baseline")


if __name__ == "__main__":
    main()
