#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker gene expression comparison: pred vs GT.

Compares predicted perturbation results (pred) from spatialgt/KNN/SEDR 
against ground truth (PT) relative to control (Sham).

Metrics:
1. Block 1 - GT vs pred direction & magnitude consistency (ICA / PIA_P)
   - PCC, R², RMSE_identity for top-k up/down genes
   - Scatter plot with y=x reference line and fitted line

2. Block 2 - Niche specificity
   - 2A: PIA_P_special genes (significant in PIA but not ICA)
   - 2B: Double-delta (ΔΔ) sign consistency

Outputs:
- Cached pred DEG CSVs for reuse
- metrics.json/csv with all computed metrics
- Scatter plots with unified scale
- TopK gene tables

Usage:
    python marker_gene_expr_compare.py \
        --model spatialgt \
        --manifest_dir /path/to/perturb_output \
        --step 10 \
        --roi_manifest /path/to/roi_manifest.json \
        --gt_deg_ica /path/to/PT1-1_vs_Sham1-1_ICA_DEG.csv \
        --gt_deg_pia /path/to/PT1-1_vs_Sham1-1_PIA_P_DEG.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Set non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Ensure repo root importability
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Config import Config
from spatial_databank_v6 import SpatialDataBank


# ==============================================================================
# Utility functions
# ==============================================================================

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


def _barcodes_to_global_idx(cache_dir: Path, dataset_name: str, barcodes: List[str]) -> List[int]:
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


def load_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load gene vocabulary."""
    vocab_path = Config().vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    name_to_id = {k.lower(): int(v) for k, v in vocab.items()}
    id_to_name = {int(v): k for k, v in vocab.items()}
    return name_to_id, id_to_name


# ==============================================================================
# Model adapters: Extract pred expression at step t
# ==============================================================================

def _run_single_pass_spatialgt(
    databank: SpatialDataBank,
    model,
    device,
    config,
    batch_size: int,
    num_workers: int,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run a single forward pass with SpatialGT model."""
    import torch
    from utils_train import process_batch, forward_pass
    
    loader = databank.get_data_loader(
        split="train", batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, stratify_by_batch=False, validation_split=0.0,
        use_distributed=False, persistent_workers=False, prefetch_factor=2,
        is_training=False,
    )
    
    all_overrides = {}
    for batch in tqdm(loader, desc=desc):
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            continue
        centers_global = batch["structure"].get("centers_global_indices", None)
        if isinstance(centers_global, torch.Tensor):
            centers_global = centers_global.cpu().numpy().astype(int).tolist()
        batch_data = process_batch(batch, device, config=config)
        with torch.no_grad():
            preds, _, _, _ = forward_pass(model, batch_data, config=config)
        B = len(centers_global) if centers_global else preds.shape[0]
        center_gene_ids = batch_data["genes"][:B]
        center_pad_mask = batch_data["padding_attention_mask"][:B].to(torch.bool)
        gathered = preds.gather(1, center_gene_ids.clamp(min=0))
        gathered = gathered * center_pad_mask.to(gathered.dtype)
        gid_np = center_gene_ids.cpu().numpy().astype(np.int64)
        val_np = gathered.cpu().float().numpy().astype(np.float32)
        if centers_global is None:
            centers_global = list(range(B))
        for i in range(B):
            all_overrides[int(centers_global[i])] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}
    
    return all_overrides


def run_spatialgt_inference(
    manifest_dir: Path,
    step: int,
    cache_sham: Path,
    sham_dataset_name: str,
    roi_global_indices: Dict[str, List[int]],
    cache_mode: str = "h5",
    lmdb_path_sham: Optional[str] = None,
    lmdb_manifest_sham: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_recon_as_x0: bool = True,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Run SpatialGT inference with V2 dual-line error cancellation.
    
    V2 Algorithm:
    - X line (unperturbed): always starts from X_0 (raw or reconstructed data)
    - X' line (perturbed): starts from previous step's true result
    - delta_k = X'_k_raw - X_k (net perturbation effect, error cancelled)
    - X'_k_true = X'_0 + delta_k (add delta to original perturbed state)
    - Frozen spots: X'_k_true(perturb_spots) = X'_0(perturb_spots)
    
    Args:
        use_recon_as_x0: If True, use reconstructed data as X_0 (V4 mode); 
                         if False, use raw data as X_0 (V2 mode). Default True.
    
    Returns: {roi_name: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}}
    """
    import torch
    import torch.nn as nn
    from model_spatialpt import SpatialNeighborTransformer
    from utils_train import process_batch, forward_pass
    
    # Load manifest
    manifest = _load_json(manifest_dir / "perturb_manifest.json")
    
    # Get checkpoint paths
    ckpt_sham_ctrl = Path(manifest.get("ckpt_sham_ctrl", ""))
    ckpt_diffusion = Path(manifest.get("ckpt_diffusion", ""))
    
    if not ckpt_sham_ctrl.exists():
        raise FileNotFoundError(f"Sham ctrl checkpoint not found: {ckpt_sham_ctrl}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    config = _setup_config_for_cache(
        config,
        cache_dir=str(cache_sham),
        dataset_name=sham_dataset_name,
        cache_mode=cache_mode,
        lmdb_path=lmdb_path_sham,
        lmdb_manifest_path=lmdb_manifest_sham,
    )
    
    sham_bank = SpatialDataBank(
        dataset_paths=[str(cache_sham / sham_dataset_name / "processed.h5ad")],
        cache_dir=str(cache_sham),
        config=config,
        force_rebuild=False,
    )
    
    def _load_state_dict_into_model(model, ckpt_dir):
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
    
    # Load models
    model_ctrl = SpatialNeighborTransformer(config).to(device)
    _load_state_dict_into_model(model_ctrl, ckpt_sham_ctrl)
    model_ctrl.eval()
    
    model_diff = SpatialNeighborTransformer(config).to(device)
    _load_state_dict_into_model(model_diff, ckpt_diffusion)
    model_diff.eval()
    
    # Get perturbation info from manifest
    selected_global = manifest["selected_global_indices"]
    weights = np.array(manifest["weights"], dtype=np.float32)
    deg_csv = Path(manifest["deg_csv"])
    p_adj_thresh = manifest.get("p_adj_thresh", 0.1)
    min_abs_logfc = manifest.get("min_abs_logfc", 0.0)
    logfc_strength = manifest.get("logfc_strength", 1.0)
    logfc_clip = manifest.get("logfc_clip", 5.0)
    frozen_indices = set(int(x) for x in selected_global) if manifest.get("freeze_perturbed", True) else set()
    
    # Load DEG for perturbation
    deg = pd.read_csv(deg_csv)
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col:
        deg = deg[deg[p_col].astype(float) < float(p_adj_thresh)]
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    name_to_id, _ = load_vocab()
    gene_logfc = {}
    for _, r in deg.iterrows():
        gid = name_to_id.get(str(r["gene"]).lower())
        if gid is not None:
            logfc_val = np.clip(float(r["avg_logFC"]), -float(logfc_clip), float(logfc_clip))
            gene_logfc[gid] = logfc_val
    
    # =========================================================================
    # Build X_0 (unperturbed baseline)
    # =========================================================================
    if use_recon_as_x0:
        # V4 mode: X_0 = SpatialGT(raw_data)
        print("[SpatialGT V2] Building X_0 = SpatialGT(raw_data)...")
        sham_bank.clear_runtime_spot_overrides()
        sham_bank.set_runtime_spot_overrides({})
        x0_state = _run_single_pass_spatialgt(
            sham_bank, model_ctrl, device, config, batch_size, num_workers,
            desc="X_0 = SpatialGT(raw)"
        )
    else:
        # V2 mode: X_0 = raw input data
        print("[SpatialGT V2] Building X_0 = raw input data...")
        sham_bank.clear_runtime_spot_overrides()
        x0_state = {}
        # Get total spots from metadata
        meta = _load_json(cache_sham / "metadata.json")
        n_spots = 0
        for info in meta.get("dataset_indices", []):
            n_spots = max(n_spots, info.get("end_idx", info.get("start_idx", 0) + 1))
        if n_spots == 0:
            n_spots = 2000  # fallback
        for gidx in tqdm(range(n_spots), desc="Loading X_0"):
            try:
                sd = sham_bank.get_spot_data(gidx)
                x0_state[int(gidx)] = {
                    "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64).copy(),
                    "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32).copy(),
                }
            except Exception:
                pass
    print(f"[SpatialGT V2] X_0: {len(x0_state)} spots loaded")
    
    # =========================================================================
    # Build X'_0 (perturbed initial state)
    # =========================================================================
    print("[SpatialGT V2] Building X'_0 (perturbed initial state)...")
    x0_prime_state = {}
    
    # Copy X_0 first
    for gidx, ov in x0_state.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # Apply perturbation to selected spots
    for i, gidx in enumerate(selected_global):
        gidx = int(gidx)
        if gidx not in x0_state:
            continue
        if i >= len(weights):
            break
        
        gene_ids = x0_state[gidx]["gene_ids"]
        vals = x0_state[gidx]["raw_normed_values"].copy()
        wi = float(weights[i])
        
        for j, gid in enumerate(gene_ids):
            if int(gid) in gene_logfc:
                fold_change = np.power(2.0, gene_logfc[int(gid)] * logfc_strength * wi)
                vals[j] = vals[j] * fold_change
        
        vals = np.maximum(vals, 0.0)
        x0_prime_state[gidx] = {"gene_ids": gene_ids.copy(), "raw_normed_values": vals}
    
    # Deep copy X'_0 as base for delta accumulation
    x0_prime_base = {}
    for gidx, ov in x0_prime_state.items():
        x0_prime_base[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # =========================================================================
    # Run V2 dual-line iterative inference
    # =========================================================================
    if step == 0:
        final_state = x0_prime_state
    else:
        x_prime_current = dict(x0_prime_state)
        
        for t in range(1, step + 1):
            print(f"\n[SpatialGT V2] Step {t}/{step} - Dual-line inference...")
            
            # Step 1: X_k = SpatialGT(X_0) - unperturbed line
            print(f"  [X line] Running SpatialGT on X_0...")
            sham_bank.clear_runtime_spot_overrides()
            sham_bank.set_runtime_spot_overrides(x0_state)
            x_k_pred = _run_single_pass_spatialgt(
                sham_bank, model_diff, device, config, batch_size, num_workers,
                desc=f"X line step {t}"
            )
            
            # Step 2: X'_k_raw = SpatialGT(X'_{k-1}_true) - perturbed line
            print(f"  [X' line] Running SpatialGT on X'_{t-1}_true...")
            sham_bank.clear_runtime_spot_overrides()
            sham_bank.set_runtime_spot_overrides(x_prime_current)
            x_prime_k_raw = _run_single_pass_spatialgt(
                sham_bank, model_diff, device, config, batch_size, num_workers,
                desc=f"X' line step {t}"
            )
            
            # Step 3-5: Compute delta and X'_k_true
            print(f"  Computing delta and X'_{t}_true...")
            x_prime_k_true = {}
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
                print(f"  Delta stats: mean={np.mean(delta_stats['mean']):.6f}, "
                      f"abs_mean={np.mean(delta_stats['abs_mean']):.6f}")
            
            x_prime_current = x_prime_k_true
            print(f"  [INFO] Step {t} completed: {len(x_prime_k_true)} spots")
        
        final_state = x_prime_current
    
    # Extract ROI data
    result = {}
    for roi_name, roi_indices in roi_global_indices.items():
        roi_data = {}
        for gidx in roi_indices:
            if gidx in final_state:
                roi_data[gidx] = final_state[gidx]
        result[roi_name] = roi_data
    
    return result


def _get_spot_expression_knn(
    databank: SpatialDataBank,
    global_idx: int,
    current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get expression for a spot from current_state or databank."""
    if current_state is not None and int(global_idx) in current_state:
        ov = current_state[int(global_idx)]
        return np.asarray(ov["gene_ids"], dtype=np.int64), np.asarray(ov["raw_normed_values"], dtype=np.float32)
    
    sd = databank.get_spot_data(int(global_idx))
    return np.asarray(sd["gene_ids"], dtype=np.int64), np.asarray(sd["raw_normed_values"], dtype=np.float32)


def _knn_average_predict(
    databank: SpatialDataBank,
    center_global_idx: int,
    current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """KNN average prediction for a single spot."""
    center_gids, center_vals = _get_spot_expression_knn(databank, center_global_idx, current_state)
    
    try:
        neighbors = databank.get_neighbors_for_spot(int(center_global_idx))
    except Exception:
        neighbors = []
    
    if len(neighbors) == 0:
        return {"gene_ids": center_gids, "raw_normed_values": center_vals}
    
    neighbor_exprs = []
    neighbor_gids_list = []
    
    for nb_idx in neighbors:
        nb_gids, nb_vals = _get_spot_expression_knn(databank, int(nb_idx), current_state)
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


def _run_single_pass_knn(
    databank: SpatialDataBank,
    all_global_indices: List[int],
    infer_mode: str,
    current_state: Dict[int, Dict[str, np.ndarray]],
    ridge_model = None,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run single pass inference for all spots using KNN/Ridge."""
    import torch
    import torch.nn.functional as F
    
    all_overrides = {}
    
    if infer_mode == "ridge_gpu" and ridge_model is not None and ridge_model.is_trained:
        # GPU batch inference
        n_spots = len(all_global_indices)
        expr_matrix = torch.zeros((n_spots, ridge_model.n_genes), device=ridge_model.device)
        gene_id_arrays: Dict[int, np.ndarray] = {}
        
        for i, global_idx in enumerate(all_global_indices):
            gids, vals = _get_spot_expression_knn(databank, global_idx, current_state)
            gene_id_arrays[i] = gids
            vals = np.asarray(vals, dtype=np.float32)
            
            for j, gid in enumerate(gids):
                gid_int = int(gid)
                if gid_int >= 0 and gid_int in ridge_model.gene_id_to_idx:
                    expr_matrix[i, ridge_model.gene_id_to_idx[gid_int]] = float(vals[j])
        
        if ridge_model.neighbor_table is None or ridge_model.neighbor_table.shape[0] != n_spots:
            ridge_model.build_neighbor_table(databank, all_global_indices, show_progress=True)
        
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
        for gidx in tqdm(all_global_indices, desc=desc):
            pred = _knn_average_predict(databank, gidx, current_state)
            all_overrides[int(gidx)] = pred
    
    return all_overrides


def run_knn_inference(
    manifest_dir: Path,
    step: int,
    cache_sham: Path,
    sham_dataset_name: str,
    roi_global_indices: Dict[str, List[int]],
    cache_mode: str = "h5",
    lmdb_path_sham: Optional[str] = None,
    lmdb_manifest_sham: Optional[str] = None,
    infer_mode: str = "ridge_gpu",
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Run KNN/Ridge inference with V2 dual-line error cancellation.
    
    V2 Algorithm:
    - X line (unperturbed): always starts from X_0 (raw data)
    - X' line (perturbed): starts from previous step's true result
    - delta_k = X'_k_raw - X_k (net perturbation effect, error cancelled)
    - X'_k_true = X'_0 + delta_k (add delta to original perturbed state)
    - Frozen spots: X'_k_true(perturb_spots) = X'_0(perturb_spots)
    
    Args:
        infer_mode: "knn_avg" (simple average of neighbors) or "ridge_gpu" (GPU Ridge regression)
    """
    import torch
    
    manifest = _load_json(manifest_dir / "perturb_manifest.json")
    
    config = Config()
    config = _setup_config_for_cache(
        config, str(cache_sham), sham_dataset_name, cache_mode,
        lmdb_path_sham, lmdb_manifest_sham,
    )
    
    sham_bank = SpatialDataBank(
        dataset_paths=[str(cache_sham / sham_dataset_name / "processed.h5ad")],
        cache_dir=str(cache_sham),
        config=config,
        force_rebuild=False,
    )
    
    # Get all global indices
    meta = _load_json(cache_sham / "metadata.json")
    start_idx = 0
    for info in meta.get("dataset_indices", []):
        start_idx = info.get("start_idx", 0)
        break
    proc = ad.read_h5ad(str(cache_sham / sham_dataset_name / "processed.h5ad"), backed="r")
    all_global_indices = [start_idx + i for i in range(proc.n_obs)]
    
    # Get perturbation info from manifest
    selected_global = manifest["selected_global_indices"]
    weights = np.array(manifest["weights"], dtype=np.float32)
    deg_csv = Path(manifest["deg_csv"])
    p_adj_thresh = manifest.get("p_adj_thresh", 0.1)
    min_abs_logfc = manifest.get("min_abs_logfc", 0.0)
    logfc_strength = manifest.get("logfc_strength", 1.0)
    logfc_clip = manifest.get("logfc_clip", 5.0)
    frozen_indices = set(int(x) for x in selected_global) if manifest.get("freeze_perturbed", True) else set()
    
    # Load DEG
    deg = pd.read_csv(deg_csv)
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col:
        deg = deg[deg[p_col].astype(float) < float(p_adj_thresh)]
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    name_to_id, _ = load_vocab()
    gene_logfc = {}
    for _, r in deg.iterrows():
        gid = name_to_id.get(str(r["gene"]).lower())
        if gid is not None:
            logfc_val = np.clip(float(r["avg_logFC"]), -float(logfc_clip), float(logfc_clip))
            gene_logfc[gid] = logfc_val
    
    print(f"[KNN V2] Inference mode: {infer_mode}")
    
    # =========================================================================
    # Build X_0 (unperturbed baseline - raw data)
    # =========================================================================
    print("[KNN V2] Building X_0 (raw data)...")
    x0_state = {}
    for gidx in tqdm(all_global_indices, desc="Loading X_0"):
        try:
            sd = sham_bank.get_spot_data(gidx)
            x0_state[int(gidx)] = {
                "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64).copy(),
                "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32).copy(),
            }
        except Exception:
            pass
    print(f"[KNN V2] X_0: {len(x0_state)} spots loaded")
    
    # =========================================================================
    # Build X'_0 (perturbed initial state)
    # =========================================================================
    print("[KNN V2] Building X'_0 (perturbed initial state)...")
    x0_prime_state = {}
    
    # Copy X_0 first
    for gidx, ov in x0_state.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # Apply perturbation to selected spots
    for i, gidx in enumerate(selected_global):
        gidx = int(gidx)
        if gidx not in x0_state:
            continue
        if i >= len(weights):
            break
        
        gene_ids = x0_state[gidx]["gene_ids"]
        vals = x0_state[gidx]["raw_normed_values"].copy()
        wi = float(weights[i])
        
        for j, gid in enumerate(gene_ids):
            if int(gid) in gene_logfc:
                fold_change = np.power(2.0, gene_logfc[int(gid)] * logfc_strength * wi)
                vals[j] = vals[j] * fold_change
        
        vals = np.maximum(vals, 0.0)
        x0_prime_state[gidx] = {"gene_ids": gene_ids.copy(), "raw_normed_values": vals}
    
    # Deep copy X'_0 as base for delta accumulation
    x0_prime_base = {}
    for gidx, ov in x0_prime_state.items():
        x0_prime_base[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
    
    # =========================================================================
    # Train Ridge model if needed
    # =========================================================================
    ridge_model = None
    if infer_mode == "ridge_gpu":
        from perturbation.mouse_stroke.knn_niche_perturb_eval_v2 import GPURidgeRegressionModel
        
        ridge_alpha = manifest.get("ridge_alpha")
        if ridge_alpha is None:
            ridge_alpha = 1.0
        
        ridge_model = GPURidgeRegressionModel(
            alpha=ridge_alpha,
            max_neighbors=8,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        print("[KNN V2] Training Ridge model on X_0...")
        ridge_model.train(sham_bank, all_global_indices, current_state=None, show_progress=True)
        ridge_model.build_neighbor_table(sham_bank, all_global_indices, show_progress=True)
    
    # =========================================================================
    # Run V2 dual-line iterative inference
    # =========================================================================
    if step == 0:
        final_state = x0_prime_state
    else:
        x_prime_current = dict(x0_prime_state)
        
        for t in range(1, step + 1):
            print(f"\n[KNN V2] Step {t}/{step} - Dual-line inference...")
            
            # Step 1: X_k = KNN/Ridge(X_0) - unperturbed line
            print(f"  [X line] Running {infer_mode} on X_0...")
            x_k_pred = _run_single_pass_knn(
                sham_bank, all_global_indices, infer_mode, x0_state, ridge_model,
                desc=f"X line step {t}"
            )
            
            # Step 2: X'_k_raw = KNN/Ridge(X'_{k-1}_true) - perturbed line
            print(f"  [X' line] Running {infer_mode} on X'_{t-1}_true...")
            x_prime_k_raw = _run_single_pass_knn(
                sham_bank, all_global_indices, infer_mode, x_prime_current, ridge_model,
                desc=f"X' line step {t}"
            )
            
            # Step 3-5: Compute delta and X'_k_true
            print(f"  Computing delta and X'_{t}_true...")
            x_prime_k_true = {}
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
                print(f"  Delta stats: mean={np.mean(delta_stats['mean']):.6f}, "
                      f"abs_mean={np.mean(delta_stats['abs_mean']):.6f}")
            
            x_prime_current = x_prime_k_true
            print(f"  [INFO] Step {t} completed: {len(x_prime_k_true)} spots")
        
        final_state = x_prime_current
    
    # Extract ROI data
    result = {}
    for roi_name, roi_indices in roi_global_indices.items():
        roi_data = {}
        for gidx in roi_indices:
            if gidx in final_state:
                roi_data[gidx] = final_state[gidx]
        result[roi_name] = roi_data
    
    return result


def run_sedr_inference(
    manifest_dir: Path,
    step: int,
    cache_sham: Path,
    sham_dataset_name: str,
    roi_global_indices: Dict[str, List[int]],
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Run SEDR inference to get predicted expression at specified step.
    """
    import torch
    
    manifest = _load_json(manifest_dir / "perturb_manifest.json")
    
    # Import SEDR functions
    from perturbation.mouse_stroke.sedr_niche_perturb_eval import (
        load_adata_and_prepare, train_sedr, sedr_reconstruct_with_override,
        apply_deg_perturbation, _load_vocab, graph_construction,
    )
    
    vocab = _load_vocab(Config().vocab_file)
    adata, X, gene_ids, valid_gene_indices = load_adata_and_prepare(cache_sham, sham_dataset_name, vocab)
    
    obs_names = list(adata.obs_names)
    
    # Build adjacency (n=8 neighbors by default)
    from SEDR.graph_func import graph_construction
    graph_dict = graph_construction(adata, n=8, mode='KNN')
    adj_norm = graph_dict['adj_norm']
    # adj_norm may be sparse tensor or dense tensor
    if hasattr(adj_norm, 'toarray'):
        adj_norm = torch.tensor(adj_norm.toarray(), dtype=torch.float32)
    elif hasattr(adj_norm, 'to_dense'):
        adj_norm = adj_norm.to_dense().float()
    elif not isinstance(adj_norm, torch.Tensor):
        adj_norm = torch.tensor(adj_norm, dtype=torch.float32)
    else:
        adj_norm = adj_norm.float()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adj_norm = adj_norm.to(device)
    
    # Train SEDR
    sedr = train_sedr(X, adata, epochs=200, device=device, mode="imputation", show_progress=True)
    
    # Get selected indices
    selected_barcodes = manifest.get("selected_barcodes", [])
    selected_indices = [obs_names.index(b) for b in selected_barcodes if b in obs_names]
    
    weights = np.array(manifest["weights"], dtype=np.float32)[:len(selected_indices)]
    deg_csv = Path(manifest["deg_csv"])
    
    X_pert, _ = apply_deg_perturbation(
        X.copy(), gene_ids, selected_indices, weights, deg_csv, vocab,
        p_adj_thresh=manifest.get("p_adj_thresh", 0.1),
        min_abs_logfc=manifest.get("min_abs_logfc", 0.0),
        logfc_strength=manifest.get("logfc_strength", 1.0),
        logfc_clip=manifest.get("logfc_clip", 5.0),
    )
    
    frozen_set = set(selected_indices) if manifest.get("freeze_perturbed", True) else set()
    frozen_values = X_pert[list(frozen_set), :] if frozen_set else None
    
    X_current = X_pert.copy()
    for t in range(step):
        X_current = sedr_reconstruct_with_override(
            sedr, X_current, adj_norm, frozen_set, 
            X_pert[list(frozen_set), :] if frozen_set else None,
            ema_alpha=manifest.get("ema_alpha", 0.0),
        )
    
    # Convert to overrides format
    result = {}
    for roi_name, roi_global_list in roi_global_indices.items():
        roi_data = {}
        start_idx = _get_dataset_start_idx(cache_sham, sham_dataset_name)
        for gidx in roi_global_list:
            local_idx = gidx - start_idx
            if 0 <= local_idx < X_current.shape[0]:
                roi_data[gidx] = {
                    "gene_ids": gene_ids.copy(),
                    "raw_normed_values": X_current[local_idx, :].astype(np.float32),
                }
        result[roi_name] = roi_data
    
    return result


# ==============================================================================
# Load expression from pre-computed CSV files
# ==============================================================================

def load_expression_from_csv(
    csv_path: Path,
    roi_barcodes: List[str],
    cache_dir: Path,
    dataset_name: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load expression data from a pre-computed expression CSV file.
    
    Args:
        csv_path: Path to expression_step*.csv file
        roi_barcodes: List of barcodes to extract
        cache_dir: Cache directory for getting global indices
        dataset_name: Dataset name for global index mapping
    
    Returns:
        {global_idx: {"gene_ids": np.array, "raw_normed_values": np.array}}
    """
    print(f"[INFO] Loading expression from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Get gene names (column names, excluding index)
    gene_names = list(df.columns)
    
    # Load vocab for gene name to ID mapping
    name_to_id, _ = load_vocab()
    
    # Build gene_ids array and valid column indices
    gene_ids = []
    valid_cols = []
    for i, gname in enumerate(gene_names):
        gid = name_to_id.get(str(gname).lower())
        if gid is not None:
            gene_ids.append(gid)
            valid_cols.append(i)
    
    gene_ids = np.array(gene_ids, dtype=np.int64)
    
    # Get global index mapping for ROI barcodes
    barcode_to_gidx = {}
    proc = ad.read_h5ad(str(cache_dir / dataset_name / "processed.h5ad"), backed="r")
    start_idx = _get_dataset_start_idx(cache_dir, dataset_name)
    for local_idx, barcode in enumerate(proc.obs_names):
        barcode_to_gidx[barcode] = start_idx + local_idx
    
    # Extract expression for ROI barcodes
    result = {}
    roi_set = set(roi_barcodes)
    
    for barcode in df.index:
        if barcode not in roi_set:
            continue
        if barcode not in barcode_to_gidx:
            continue
        
        gidx = barcode_to_gidx[barcode]
        vals = df.loc[barcode].values[valid_cols].astype(np.float32)
        
        result[gidx] = {
            "gene_ids": gene_ids.copy(),
            "raw_normed_values": vals,
        }
    
    print(f"[INFO] Loaded {len(result)} spots from expression CSV")
    return result


def check_expression_csv_exists(manifest_dir: Path, step: int) -> Optional[Path]:
    """
    Check if pre-computed expression CSV exists for the given step.
    
    Returns:
        Path to expression CSV if exists, None otherwise
    """
    expr_dir = manifest_dir / "expression"
    if not expr_dir.exists():
        return None
    
    csv_path = expr_dir / f"expression_step{step}.csv"
    if csv_path.exists():
        return csv_path
    
    return None


# ==============================================================================
# DEG computation and caching
# ==============================================================================

def compute_pred_deg(
    pred_expr: Dict[int, Dict[str, np.ndarray]],
    ctrl_expr: Dict[int, Dict[str, np.ndarray]],
    id_to_name: Dict[int, str],
) -> pd.DataFrame:
    """
    Compute DEG between pred and ctrl expression.
    Uses mean log expression difference as logFC.
    
    Returns DataFrame with columns: gene, avg_logFC
    """
    # Collect expression per gene
    pred_by_gene: Dict[int, List[float]] = {}
    ctrl_by_gene: Dict[int, List[float]] = {}
    
    for gidx, data in pred_expr.items():
        gene_ids = data["gene_ids"]
        vals = data["raw_normed_values"]
        for i, gid in enumerate(gene_ids):
            gid = int(gid)
            if gid not in pred_by_gene:
                pred_by_gene[gid] = []
            pred_by_gene[gid].append(float(vals[i]))
    
    for gidx, data in ctrl_expr.items():
        gene_ids = data["gene_ids"]
        vals = data["raw_normed_values"]
        for i, gid in enumerate(gene_ids):
            gid = int(gid)
            if gid not in ctrl_by_gene:
                ctrl_by_gene[gid] = []
            ctrl_by_gene[gid].append(float(vals[i]))
    
    # Compute mean difference (logFC)
    common_genes = set(pred_by_gene.keys()) & set(ctrl_by_gene.keys())
    
    results = []
    for gid in common_genes:
        pred_mean = np.mean(pred_by_gene[gid])
        ctrl_mean = np.mean(ctrl_by_gene[gid])
        # logFC = mean(pred) - mean(ctrl) (already in log space)
        logfc = pred_mean - ctrl_mean
        gene_name = id_to_name.get(gid, f"gene_{gid}")
        results.append({"gene": gene_name, "avg_logFC": logfc, "gene_id": gid})
    
    return pd.DataFrame(results)


def get_ctrl_expression(
    cache_sham: Path,
    sham_dataset_name: str,
    roi_barcodes: List[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Get control expression from Sham slice for specified ROI barcodes.
    """
    config = Config()
    config.cache_dir = str(cache_sham)
    config.cache_mode = "h5"
    config.strict_cache_only = True
    
    sham_bank = SpatialDataBank(
        dataset_paths=[str(cache_sham / sham_dataset_name / "processed.h5ad")],
        cache_dir=str(cache_sham),
        config=config,
        force_rebuild=False,
    )
    
    roi_global = _barcodes_to_global_idx(cache_sham, sham_dataset_name, roi_barcodes)
    
    result = {}
    for gidx in roi_global:
        try:
            sd = sham_bank.get_spot_data(gidx)
            result[gidx] = {
                "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
                "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
            }
        except Exception:
            continue
    
    return result


# ==============================================================================
# Metrics computation
# ==============================================================================

def compute_block1_metrics(
    gt_deg: pd.DataFrame,
    pred_deg: pd.DataFrame,
    topk: int = 50,
    p_sig: float = 0.05,
    gt_scale: float = 1.0,
    normalize_mode: str = "none",
    topk_by: str = "pvalue",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Block 1: GT vs pred direction & magnitude consistency.
    
    Args:
        topk_by: "pvalue" (select by smallest p-value) or "logfc" (select by largest |logFC|)
    
    Returns:
        metrics: {pcc, r2, rmse_identity, slope, intercept}
        topk_table: DataFrame with gene, logfc_gt, logfc_pred
    """
    # Filter GT by p-value if available
    if "p_val_adj" in gt_deg.columns:
        gt_sig = gt_deg[gt_deg["p_val_adj"] < p_sig].copy()
        p_col_for_sort = "p_val_adj"
    elif "p_val" in gt_deg.columns:
        gt_sig = gt_deg[gt_deg["p_val"] < p_sig].copy()
        p_col_for_sort = "p_val"
    else:
        gt_sig = gt_deg.copy()
        p_col_for_sort = None
    
    # Get top-k up and down genes based on topk_by criteria
    if topk_by == "pvalue" and p_col_for_sort is not None:
        # Select by smallest p-value (most significant)
        up_df = gt_sig[gt_sig["avg_logFC"] > 0].nsmallest(topk, p_col_for_sort)
        down_df = gt_sig[gt_sig["avg_logFC"] < 0].nsmallest(topk, p_col_for_sort)
    else:
        # Select by largest |logFC| (original behavior)
        up_df = gt_sig[gt_sig["avg_logFC"] > 0].nlargest(topk, "avg_logFC")
        down_df = gt_sig[gt_sig["avg_logFC"] < 0].nsmallest(topk, "avg_logFC")
    
    up_genes = up_df["gene"].tolist()
    down_genes = down_df["gene"].tolist()
    
    # If one side has fewer than topk, compensate from the other side
    target_total = 2 * topk
    if len(up_genes) < topk and len(down_genes) >= topk:
        # Need more down genes to compensate
        shortfall = topk - len(up_genes)
        if topk_by == "pvalue" and p_col_for_sort is not None:
            extra_down = gt_sig[gt_sig["avg_logFC"] < 0].nsmallest(topk + shortfall, p_col_for_sort)["gene"].tolist()
        else:
            extra_down = gt_sig[gt_sig["avg_logFC"] < 0].nsmallest(topk + shortfall, "avg_logFC")["gene"].tolist()
        down_genes = extra_down
    elif len(down_genes) < topk and len(up_genes) >= topk:
        # Need more up genes to compensate
        shortfall = topk - len(down_genes)
        if topk_by == "pvalue" and p_col_for_sort is not None:
            extra_up = gt_sig[gt_sig["avg_logFC"] > 0].nsmallest(topk + shortfall, p_col_for_sort)["gene"].tolist()
        else:
            extra_up = gt_sig[gt_sig["avg_logFC"] > 0].nlargest(topk + shortfall, "avg_logFC")["gene"].tolist()
        up_genes = extra_up
    
    selected_genes = up_genes + down_genes
    
    # Align with pred DEG (apply gt_scale to GT logFC for comparison)
    gt_map = {str(g).lower(): float(lfc) * gt_scale for g, lfc in zip(gt_deg["gene"], gt_deg["avg_logFC"])}
    pred_map = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg["gene"], pred_deg["avg_logFC"])}
    
    # Build p-value map from GT DEG
    p_col = "p_val_adj" if "p_val_adj" in gt_deg.columns else ("p_val" if "p_val" in gt_deg.columns else None)
    if p_col:
        pval_map = {str(g).lower(): float(p) for g, p in zip(gt_deg["gene"], gt_deg[p_col])}
    else:
        pval_map = {}
    
    rows = []
    for g in selected_genes:
        g_lower = str(g).lower()
        if g_lower in gt_map and g_lower in pred_map:
            rows.append({
                "gene": g,
                "logfc_gt": gt_map[g_lower],  # Already scaled
                "logfc_pred": pred_map[g_lower],
                "p_val": pval_map.get(g_lower, np.nan),
            })
    
    if len(rows) == 0:
        return {"pcc": np.nan, "spearman": np.nan, "r2": np.nan, "rmse_identity": np.nan, 
                "rmse_identity_norm": np.nan, "slope": np.nan, "intercept": np.nan}, pd.DataFrame()
    
    df = pd.DataFrame(rows)
    x = df["logfc_gt"].values
    y = df["logfc_pred"].values
    
    # Apply normalization for RMSE and fitting (optional)
    # Note: PCC and Spearman are computed on original values (scale-invariant anyway)
    if normalize_mode == "zscore":
        # Z-score: both normalized to mean=0, std=1
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
        y_norm = (y - y_mean) / y_std if y_std > 0 else y - y_mean
    elif normalize_mode == "zscore_shared":
        # Use GT's statistics to normalize both (pred deviation from GT reference)
        x_mean, x_std = np.mean(x), np.std(x)
        x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
        y_norm = (y - x_mean) / x_std if x_std > 0 else y - x_mean
    elif normalize_mode == "unit_var":
        # Scale both to unit variance, keep original mean
        x_std = np.std(x)
        y_std = np.std(y)
        x_norm = x / x_std if x_std > 0 else x
        y_norm = y / y_std if y_std > 0 else y
    else:
        # none - no normalization
        x_norm = x
        y_norm = y
    
    # Store normalized values in df for plotting
    df["logfc_gt_norm"] = x_norm
    df["logfc_pred_norm"] = y_norm
    
    # PCC
    if len(x) > 1:
        pcc, _ = stats.pearsonr(x, y)
    else:
        pcc = np.nan
    
    # Spearman correlation (rank-based, also scale-invariant but measures monotonic relationship)
    if len(x) > 1:
        spearman, _ = stats.spearmanr(x, y)
    else:
        spearman = np.nan
    
    # Linear regression for R²
    if len(x) > 1:
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        y_pred = lr.predict(x.reshape(-1, 1))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        slope = lr.coef_[0]
        intercept = lr.intercept_
    else:
        r2 = np.nan
        slope = np.nan
        intercept = np.nan
    
    # RMSE identity (y=x) on original values
    rmse_identity = np.sqrt(np.mean((y - x) ** 2))
    
    # RMSE identity on normalized values (more comparable across experiments)
    rmse_identity_norm = np.sqrt(np.mean((y_norm - x_norm) ** 2))
    
    # MAE (Mean Absolute Error) - on both original and normalized values
    mae = np.mean(np.abs(y - x))
    mae_norm = np.mean(np.abs(y_norm - x_norm))
    
    # CCC (Concordance Correlation Coefficient) - measures agreement, not just correlation
    # CCC = (2 * ρ * σx * σy) / (σx² + σy² + (μx - μy)²)
    # Range: [-1, 1], equals 1 only when perfect agreement (values are equal, not just correlated)
    def compute_ccc(x_vals, y_vals):
        if len(x_vals) < 2:
            return np.nan
        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x, var_y = np.var(x_vals, ddof=0), np.var(y_vals, ddof=0)
        covar = np.mean((x_vals - mean_x) * (y_vals - mean_y))
        ccc = (2 * covar) / (var_x + var_y + (mean_x - mean_y) ** 2)
        return ccc
    
    ccc = compute_ccc(x, y)
    ccc_norm = compute_ccc(x_norm, y_norm)
    
    # Linear regression on normalized values for fitting line
    if len(x_norm) > 1:
        lr_norm = LinearRegression()
        lr_norm.fit(x_norm.reshape(-1, 1), y_norm)
        slope_norm = lr_norm.coef_[0]
        intercept_norm = lr_norm.intercept_
    else:
        slope_norm = np.nan
        intercept_norm = np.nan
    
    metrics = {
        "pcc": float(pcc),
        "spearman": float(spearman),
        "ccc": float(ccc),
        "ccc_norm": float(ccc_norm),
        "r2": float(r2),
        "rmse_identity": float(rmse_identity),
        "rmse_identity_norm": float(rmse_identity_norm),
        "mae": float(mae),
        "mae_norm": float(mae_norm),
        "slope": float(slope) if not np.isnan(slope) else None,
        "intercept": float(intercept) if not np.isnan(intercept) else None,
        "slope_norm": float(slope_norm) if not np.isnan(slope_norm) else None,
        "intercept_norm": float(intercept_norm) if not np.isnan(intercept_norm) else None,
        "normalize_mode": normalize_mode,
        "n_genes": len(rows),
    }
    
    return metrics, df


def compute_block2a_metrics(
    gt_deg_pia: pd.DataFrame,
    gt_deg_ica: pd.DataFrame,
    pred_deg_pia: pd.DataFrame,
    topk: int = 50,
    p_sig: float = 0.05,
    logfc_a: float = 0.25,
    p_nonsig: float = 0.1,
    ica_noise_abs: float = 0.1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Block 2A: PIA_P_special genes - significant in PIA but not ICA.
    """
    # Build lookup maps
    p_col_pia = "p_val_adj" if "p_val_adj" in gt_deg_pia.columns else "p_val"
    p_col_ica = "p_val_adj" if "p_val_adj" in gt_deg_ica.columns else "p_val"
    
    pia_map = {str(g).lower(): {"logfc": float(lfc), "p": float(p)} 
               for g, lfc, p in zip(gt_deg_pia["gene"], gt_deg_pia["avg_logFC"], gt_deg_pia.get(p_col_pia, [1]*len(gt_deg_pia)))}
    ica_map = {str(g).lower(): {"logfc": float(lfc), "p": float(p)} 
               for g, lfc, p in zip(gt_deg_ica["gene"], gt_deg_ica["avg_logFC"], gt_deg_ica.get(p_col_ica, [1]*len(gt_deg_ica)))}
    pred_pia_map = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg_pia["gene"], pred_deg_pia["avg_logFC"])}
    
    special_genes = []
    for g, data in pia_map.items():
        # PIA significant: p < p_sig and |logFC| > logfc_a
        if data["p"] >= p_sig or abs(data["logfc"]) < logfc_a:
            continue
        
        # ICA not significant or opposite direction
        if g in ica_map:
            ica_data = ica_map[g]
            # ICA not significant (p > p_nonsig) OR opposite direction with |logfc| > noise
            ica_not_sig = ica_data["p"] > p_nonsig
            opposite_dir = (data["logfc"] * ica_data["logfc"] < 0) and (abs(ica_data["logfc"]) > ica_noise_abs)
            if ica_not_sig or opposite_dir:
                special_genes.append((g, abs(data["logfc"])))
    
    # Sort by |logFC| and take topK
    special_genes.sort(key=lambda x: -x[1])
    selected = [g for g, _ in special_genes[:topk]]
    
    if len(selected) == 0:
        return {"acc_pia_spec": np.nan, "n_special_genes": 0}, pd.DataFrame()
    
    # Compute direction consistency
    correct = 0
    rows = []
    for g in selected:
        gt_logfc = pia_map[g]["logfc"]
        pred_logfc = pred_pia_map.get(g, 0)
        sign_match = np.sign(gt_logfc) == np.sign(pred_logfc)
        correct += int(sign_match)
        rows.append({
            "gene": g,
            "logfc_gt_pia": gt_logfc,
            "logfc_pred_pia": pred_logfc,
            "sign_match": sign_match,
        })
    
    acc = correct / len(selected)
    
    return {"acc_pia_spec": float(acc), "n_special_genes": len(selected)}, pd.DataFrame(rows)


def compute_block2a_ica_metrics(
    gt_deg_ica: pd.DataFrame,
    gt_deg_pia: pd.DataFrame,
    pred_deg_ica: pd.DataFrame,
    topk: int = 50,
    p_sig: float = 0.05,
    logfc_a: float = 0.25,
    p_nonsig: float = 0.1,
    pia_noise_abs: float = 0.1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Block 2A for ICA: ICA_special genes - significant in ICA but not PIA.
    """
    # Build lookup maps
    p_col_ica = "p_val_adj" if "p_val_adj" in gt_deg_ica.columns else "p_val"
    p_col_pia = "p_val_adj" if "p_val_adj" in gt_deg_pia.columns else "p_val"
    
    ica_map = {str(g).lower(): {"logfc": float(lfc), "p": float(p)} 
               for g, lfc, p in zip(gt_deg_ica["gene"], gt_deg_ica["avg_logFC"], gt_deg_ica.get(p_col_ica, [1]*len(gt_deg_ica)))}
    pia_map = {str(g).lower(): {"logfc": float(lfc), "p": float(p)} 
               for g, lfc, p in zip(gt_deg_pia["gene"], gt_deg_pia["avg_logFC"], gt_deg_pia.get(p_col_pia, [1]*len(gt_deg_pia)))}
    pred_ica_map = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg_ica["gene"], pred_deg_ica["avg_logFC"])}
    
    special_genes = []
    for g, data in ica_map.items():
        # ICA significant: p < p_sig and |logFC| > logfc_a
        if data["p"] >= p_sig or abs(data["logfc"]) < logfc_a:
            continue
        
        # PIA not significant or opposite direction
        if g in pia_map:
            pia_data = pia_map[g]
            # PIA not significant (p > p_nonsig) OR opposite direction with |logfc| > noise
            pia_not_sig = pia_data["p"] > p_nonsig
            opposite_dir = (data["logfc"] * pia_data["logfc"] < 0) and (abs(pia_data["logfc"]) > pia_noise_abs)
            if pia_not_sig or opposite_dir:
                special_genes.append((g, abs(data["logfc"])))
    
    # Sort by |logFC| and take topK
    special_genes.sort(key=lambda x: -x[1])
    selected = [g for g, _ in special_genes[:topk]]
    
    if len(selected) == 0:
        return {"acc_ica_spec": np.nan, "n_special_genes": 0}, pd.DataFrame()
    
    # Compute direction consistency
    correct = 0
    rows = []
    for g in selected:
        gt_logfc = ica_map[g]["logfc"]
        pred_logfc = pred_ica_map.get(g, 0)
        sign_match = np.sign(gt_logfc) == np.sign(pred_logfc)
        correct += int(sign_match)
        rows.append({
            "gene": g,
            "logfc_gt_ica": gt_logfc,
            "logfc_pred_ica": pred_logfc,
            "sign_match": sign_match,
        })
    
    acc = correct / len(selected)
    
    return {"acc_ica_spec": float(acc), "n_special_genes": len(selected)}, pd.DataFrame(rows)


def compute_block2b_ica_metrics(
    gt_deg_ica: pd.DataFrame,
    gt_deg_pia: pd.DataFrame,
    pred_deg_ica: pd.DataFrame,
    pred_deg_pia: pd.DataFrame,
    topk: int = 50,
    normalize_mode: str = "none",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Block 2B for ICA: Double-delta from ICA perspective.
    ΔΔgt = ΔICAgt - ΔPIAgt (ICA - PIA, genes stronger in ICA)
    ΔΔpred = ΔICApred - ΔPIApred
    
    Args:
        normalize_mode: "none", "zscore", etc. for normalizing ΔΔ values
    """
    # Build maps
    gt_ica = {str(g).lower(): float(lfc) for g, lfc in zip(gt_deg_ica["gene"], gt_deg_ica["avg_logFC"])}
    gt_pia = {str(g).lower(): float(lfc) for g, lfc in zip(gt_deg_pia["gene"], gt_deg_pia["avg_logFC"])}
    pred_ica = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg_ica["gene"], pred_deg_ica["avg_logFC"])}
    pred_pia = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg_pia["gene"], pred_deg_pia["avg_logFC"])}
    
    # Find common genes
    common = set(gt_ica.keys()) & set(gt_pia.keys()) & set(pred_ica.keys()) & set(pred_pia.keys())
    
    if len(common) == 0:
        return {"acc_delta_delta_ica": np.nan, "n_genes": 0}, pd.DataFrame()
    
    # Compute ΔΔ (ICA - PIA)
    delta_delta = []
    for g in common:
        dd_gt = gt_ica[g] - gt_pia[g]
        dd_pred = pred_ica[g] - pred_pia[g]
        delta_delta.append((g, dd_gt, dd_pred, abs(dd_gt)))
    
    # Sort by |ΔΔgt| and take topK
    delta_delta.sort(key=lambda x: -x[3])
    selected = delta_delta[:topk]
    
    # Compute direction consistency
    correct = 0
    rows = []
    for g, dd_gt, dd_pred, _ in selected:
        sign_match = np.sign(dd_gt) == np.sign(dd_pred)
        correct += int(sign_match)
        rows.append({
            "gene": g,
            "delta_delta_gt_ica": dd_gt,
            "delta_delta_pred_ica": dd_pred,
            "sign_match": sign_match,
        })
    
    acc = correct / len(selected) if selected else np.nan
    df = pd.DataFrame(rows)
    
    # Apply normalization if requested
    if normalize_mode != "none" and not df.empty:
        x = df["delta_delta_gt_ica"].values
        y = df["delta_delta_pred_ica"].values
        
        if normalize_mode == "zscore":
            x_mean, x_std = np.mean(x), np.std(x)
            y_mean, y_std = np.mean(y), np.std(y)
            x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
            y_norm = (y - y_mean) / y_std if y_std > 0 else y - y_mean
        elif normalize_mode == "zscore_shared":
            x_mean, x_std = np.mean(x), np.std(x)
            x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
            y_norm = (y - x_mean) / x_std if x_std > 0 else y - x_mean
        else:
            x_norm = x
            y_norm = y
        
        df["delta_delta_gt_ica_norm"] = x_norm
        df["delta_delta_pred_ica_norm"] = y_norm
    
    return {"acc_delta_delta_ica": float(acc), "n_genes": len(selected)}, df


def compute_block2b_metrics(
    gt_deg_pia: pd.DataFrame,
    gt_deg_ica: pd.DataFrame,
    pred_deg_pia: pd.DataFrame,
    pred_deg_ica: pd.DataFrame,
    topk: int = 50,
    normalize_mode: str = "none",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Block 2B: Double-delta (ΔΔ) sign consistency.
    ΔΔgt = ΔPIAgt - ΔICAgt
    ΔΔpred = ΔPIApred - ΔICApred
    
    Args:
        normalize_mode: "none", "zscore", etc. for normalizing ΔΔ values
    """
    # Build maps
    gt_pia = {str(g).lower(): float(lfc) for g, lfc in zip(gt_deg_pia["gene"], gt_deg_pia["avg_logFC"])}
    gt_ica = {str(g).lower(): float(lfc) for g, lfc in zip(gt_deg_ica["gene"], gt_deg_ica["avg_logFC"])}
    pred_pia = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg_pia["gene"], pred_deg_pia["avg_logFC"])}
    pred_ica = {str(g).lower(): float(lfc) for g, lfc in zip(pred_deg_ica["gene"], pred_deg_ica["avg_logFC"])}
    
    # Find common genes
    common = set(gt_pia.keys()) & set(gt_ica.keys()) & set(pred_pia.keys()) & set(pred_ica.keys())
    
    if len(common) == 0:
        return {"acc_delta_delta": np.nan, "n_genes": 0}, pd.DataFrame()
    
    # Compute ΔΔ
    delta_delta = []
    for g in common:
        dd_gt = gt_pia[g] - gt_ica[g]
        dd_pred = pred_pia[g] - pred_ica[g]
        delta_delta.append((g, dd_gt, dd_pred, abs(dd_gt)))
    
    # Sort by |ΔΔgt| and take topK
    delta_delta.sort(key=lambda x: -x[3])
    selected = delta_delta[:topk]
    
    # Compute direction consistency
    correct = 0
    rows = []
    for g, dd_gt, dd_pred, _ in selected:
        sign_match = np.sign(dd_gt) == np.sign(dd_pred)
        correct += int(sign_match)
        rows.append({
            "gene": g,
            "delta_delta_gt": dd_gt,
            "delta_delta_pred": dd_pred,
            "sign_match": sign_match,
        })
    
    acc = correct / len(selected) if selected else np.nan
    df = pd.DataFrame(rows)
    
    # Apply normalization if requested
    if normalize_mode != "none" and not df.empty:
        x = df["delta_delta_gt"].values
        y = df["delta_delta_pred"].values
        
        if normalize_mode == "zscore":
            x_mean, x_std = np.mean(x), np.std(x)
            y_mean, y_std = np.mean(y), np.std(y)
            x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
            y_norm = (y - y_mean) / y_std if y_std > 0 else y - y_mean
        elif normalize_mode == "zscore_shared":
            x_mean, x_std = np.mean(x), np.std(x)
            x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
            y_norm = (y - x_mean) / x_std if x_std > 0 else y - x_mean
        else:
            x_norm = x
            y_norm = y
        
        df["delta_delta_gt_norm"] = x_norm
        df["delta_delta_pred_norm"] = y_norm
    
    return {"acc_delta_delta": float(acc), "n_genes": len(selected)}, df


# ==============================================================================
# Plotting
# ==============================================================================

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    axis_label_x: str = "PT-1 vs Sham DEG logFC",
    axis_label_y: str = "Perturbed Prediction vs Sham DEG logFC",
    axis_limit: Optional[float] = None,
    annotate_genes: bool = False,
    n_annotate_per_direction: int = 5,
    use_normalized: bool = False,
    low_response_threshold: float = 0.05,
):
    """
    Plot scatter with y=x reference line and fitted line.
    Color logic:
    - Q2/Q4 (direction mismatch): dark blue (#333A8C) - Incorrect Direction Prediction
    - Q1/Q3 (same direction): |pred| < threshold -> light blue (#78A9CD) - Low Response Prediction
    - Q1/Q3 (same direction): |pred| >= threshold -> orange (#F8AB61) - High Response Prediction
    
    Args:
        annotate_genes: Whether to annotate gene names on the plot
        n_annotate_per_direction: Number of genes to annotate per direction (up/down)
        use_normalized: If True, use normalized columns (x_col + "_norm", y_col + "_norm") if available
        low_response_threshold: Threshold for low response classification (default 0.05, use 0.5 for z-score)
    """
    if df.empty:
        print(f"[WARN] Empty data for scatter plot: {out_path}")
        return
    
    # Determine which columns to use
    actual_x_col = x_col
    actual_y_col = y_col
    if use_normalized:
        norm_x_col = x_col + "_norm"
        norm_y_col = y_col + "_norm"
        if norm_x_col in df.columns and norm_y_col in df.columns:
            actual_x_col = norm_x_col
            actual_y_col = norm_y_col
            print(f"  [INFO] Using normalized columns for plotting: {norm_x_col}, {norm_y_col}")
    
    x = df[actual_x_col].values
    y = df[actual_y_col].values
    
    # Determine axis limits
    if axis_limit is None:
        max_val = max(abs(x.max()), abs(x.min()), abs(y.max()), abs(y.min()))
        axis_limit = max_val * 1.1
    
    # Wider figure to accommodate color legend on the right
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # Remove spines (outer frame)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Color logic:
    # - Quadrant 2 (x<0, y>0) or Quadrant 4 (x>0, y<0): direction mismatch -> #333A8C (dark blue)
    # - Quadrant 1 or 3 (same sign): |pred| < threshold -> #78A9CD (light blue), >= threshold -> #F8AB61 (orange)
    THRESHOLD = low_response_threshold
    abs_y = np.abs(y)
    
    # Check if in Q2 or Q4 (direction mismatch: signs differ)
    direction_mismatch = (x * y) < 0  # True if in Q2 or Q4
    
    # Base colors for Q1/Q3 based on threshold
    base_colors = np.where(abs_y < THRESHOLD, '#78A9CD', '#F8AB61')
    
    # Override with dark blue for direction mismatch (Q2/Q4)
    colors = np.where(direction_mismatch, '#333A8C', base_colors)
    
    # Scatter with conditional colors
    scatter = ax.scatter(x, y, alpha=0.8, s=35, c=colors,
                         edgecolors="white", linewidths=0.3)
    
    # y=x reference line
    ax.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 
            color='#666666', linestyle='--', linewidth=1.2, label='y=x')
    
    # Fitted line with custom color (temporarily disabled)
    # if len(x) > 1:
    #     lr = LinearRegression()
    #     lr.fit(x.reshape(-1, 1), y)
    #     x_fit = np.linspace(-axis_limit, axis_limit, 100)
    #     y_fit = lr.predict(x_fit.reshape(-1, 1))
    #     ax.plot(x_fit, y_fit, color='#C93735', linewidth=1, 
    #             label=f'y={lr.coef_[0]:.2f}x+{lr.intercept_:.2f}')
    
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_xlabel(axis_label_x, fontsize=11)
    ax.set_ylabel(axis_label_y, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axhline(y=0, color='#CCCCCC', linewidth=0.8, linestyle='-', zorder=0)
    ax.axvline(x=0, color='#CCCCCC', linewidth=0.8, linestyle='-', zorder=0)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Tick styling
    ax.tick_params(axis='both', which='both', length=0)
    
    # Annotate gene names if enabled
    # Use original x_col for direction determination, but actual_x_col for positions
    if annotate_genes and "gene" in df.columns and n_annotate_per_direction > 0:
        # Get up-regulated and down-regulated genes separately (based on original GT direction)
        # Use the original x_col for direction, not the normalized one
        direction_col = x_col if x_col in df.columns else actual_x_col
        
        # Filter genes that are within axis_limit (so annotations are visible)
        df_visible = df[
            (df[actual_x_col].abs() <= axis_limit) & 
            (df[actual_y_col].abs() <= axis_limit)
        ].copy()
        
        df_up = df_visible[df_visible[direction_col] > 0].copy()
        df_down = df_visible[df_visible[direction_col] < 0].copy()
        
        # Sort by p-value (if available) or by |logFC| and select top n genes from each direction
        genes_to_annotate = []
        if "p_val" in df_visible.columns:
            # Sort by p-value (most significant) among visible genes
            if not df_up.empty:
                df_up_sorted = df_up.nsmallest(n_annotate_per_direction, "p_val")
                genes_to_annotate.extend(df_up_sorted["gene"].tolist())
            if not df_down.empty:
                df_down_sorted = df_down.nsmallest(n_annotate_per_direction, "p_val")
                genes_to_annotate.extend(df_down_sorted["gene"].tolist())
        else:
            # Fallback: sort by |logFC| (largest absolute value) among visible genes
            if not df_up.empty:
                df_up_sorted = df_up.nlargest(n_annotate_per_direction, direction_col)
                genes_to_annotate.extend(df_up_sorted["gene"].tolist())
            if not df_down.empty:
                df_down_sorted = df_down.nsmallest(n_annotate_per_direction, direction_col)
                genes_to_annotate.extend(df_down_sorted["gene"].tolist())
        
        # Create text annotations
        texts = []
        for gene in genes_to_annotate:
            gene_row = df[df["gene"] == gene].iloc[0]
            gene_x = gene_row[actual_x_col]
            gene_y = gene_row[actual_y_col]
            texts.append(ax.text(gene_x, gene_y, gene, fontsize=8, ha='center', va='bottom',
                                 color='#333333', fontweight='normal'))
        
        # Adjust text positions to avoid overlap (requires adjustText package)
        if texts:
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5))
            except ImportError:
                # adjustText not installed, use simple offset
                for t in texts:
                    t.set_position((t.get_position()[0], t.get_position()[1] + axis_limit * 0.03))
            except Exception:
                pass  # If adjustText fails, use original positions
    
    # Create combined legend below the plot, arranged horizontally
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Combine line elements and color patches
    legend_elements = [
        Line2D([0], [0], color='#666666', linestyle='--', linewidth=1.2, label='y=x'),
        Patch(facecolor='#F8AB61', edgecolor='white', label='High Response'),
        Patch(facecolor='#78A9CD', edgecolor='white', label='Low Response'),
        Patch(facecolor='#333A8C', edgecolor='white', label='Incorrect Direction'),
    ]
    
    # Create legend below the plot, horizontally arranged
    ax.legend(handles=legend_elements, 
              loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=9, frameon=False,
              handletextpad=0.4, columnspacing=1.2)
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Marker gene expression comparison: pred vs GT")
    
    # Model selection
    ap.add_argument("--model", choices=["spatialgt", "knn", "sedr"], required=True)
    ap.add_argument("--manifest_dir", required=True, help="Directory with perturb_manifest.json")
    ap.add_argument("--step", type=int, default=10, help="Which step to evaluate")
    
    # Data paths
    ap.add_argument("--roi_manifest", required=True, help="ROI manifest JSON")
    ap.add_argument("--gt_deg_ica", required=True, help="GT DEG CSV for ICA")
    ap.add_argument("--gt_deg_pia", required=True, help="GT DEG CSV for PIA_P")
    ap.add_argument("--cache_sham", required=True, help="Cache dir for Sham slice")
    ap.add_argument("--sham_dataset_name", default="Sham1-1")
    
    # Cache mode
    ap.add_argument("--cache_mode", choices=["h5", "lmdb"], default="lmdb")
    ap.add_argument("--lmdb_path_sham", type=str, default=None)
    ap.add_argument("--lmdb_manifest_sham", type=str, default=None)
    
    # Metrics parameters
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--p_sig", type=float, default=0.05)
    ap.add_argument("--p_nonsig", type=float, default=0.1)
    ap.add_argument("--logfc_a", type=float, default=0.25)
    ap.add_argument("--ica_noise_abs", type=float, default=0.1)
    ap.add_argument("--axis_limit", type=float, default=None, help="Fixed axis limit for scatter plots")
    ap.add_argument("--gt_scale", type=float, default=1.0, 
                    help="Scale factor for GT logFC in Block 1 (0-1, default=1). Does not affect Block 2.")
    ap.add_argument("--normalize_block1", choices=["none", "zscore", "zscore_shared", "unit_var"], 
                    default="none",
                    help="Normalization mode for Block 1 RMSE/fitting: "
                         "none=no normalization, "
                         "zscore=both normalized to mean=0/std=1, "
                         "zscore_shared=use GT stats to normalize both, "
                         "unit_var=scale both to unit variance")
    
    # Inference params
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--knn_infer_mode", choices=["knn_avg", "ridge_gpu"], default="ridge_gpu",
                    help="KNN inference mode: knn_avg (simple avg) or ridge_gpu (GPU Ridge regression)")
    
    # V2 dual-line options
    ap.add_argument("--use_recon_as_x0", action="store_true", default=True,
                    help="Use reconstructed data as X_0 (V4 mode). If False, use raw data (V2 mode). Default: True for spatialgt.")
    ap.add_argument("--no_recon_as_x0", dest="use_recon_as_x0", action="store_false",
                    help="Use raw data as X_0 (V2 mode) instead of reconstructed data.")
    
    # TopK selection options
    ap.add_argument("--topk_by", choices=["pvalue", "logfc"], default="pvalue",
                    help="TopK gene selection criteria: pvalue (smallest p-value) or logfc (largest |logFC|)")
    
    # Plotting options
    ap.add_argument("--plot_normalized", action="store_true",
                    help="Use normalized (z-score) values for scatter plots when normalization is enabled")
    ap.add_argument("--low_response_threshold", type=float, default=0.05,
                    help="Threshold for low response classification in scatter plots (default 0.05, use 0.5 for z-score)")
    ap.add_argument("--annotate_genes", action="store_true",
                    help="Whether to annotate gene names on scatter plots")
    ap.add_argument("--n_annotate_per_direction", type=int, default=5,
                    help="Number of genes to annotate per direction (up/down), based on smallest p-value")
    
    # Options
    ap.add_argument("--force_recompute", action="store_true")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: manifest_dir/marker_compare)")
    
    args = ap.parse_args()
    
    manifest_dir = Path(args.manifest_dir)
    cache_sham = Path(args.cache_sham)
    
    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else manifest_dir / "marker_compare" / f"step{args.step}" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = manifest_dir / "marker_compare_cache" / f"step{args.step}" / args.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Model: {args.model}, Step: {args.step}")
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] Cache: {cache_dir}")
    
    # Load ROI manifest
    roi_manifest = _load_json(Path(args.roi_manifest))
    # Support both "sham" and "Sham1_1" key formats
    sham_key = "Sham1_1" if "Sham1_1" in roi_manifest else "sham"
    sham_barcodes_ica = roi_manifest[sham_key]["ICA"]["barcodes"]
    sham_barcodes_pia = roi_manifest[sham_key]["PIA_P"]["barcodes"]
    
    # Convert to global indices
    roi_global_ica = _barcodes_to_global_idx(cache_sham, args.sham_dataset_name, sham_barcodes_ica)
    roi_global_pia = _barcodes_to_global_idx(cache_sham, args.sham_dataset_name, sham_barcodes_pia)
    
    roi_global_indices = {"ICA": roi_global_ica, "PIA_P": roi_global_pia}
    
    print(f"[INFO] ICA: {len(roi_global_ica)} spots, PIA_P: {len(roi_global_pia)} spots")
    
    # Load GT DEG
    gt_deg_ica = pd.read_csv(args.gt_deg_ica)
    gt_deg_pia = pd.read_csv(args.gt_deg_pia)
    
    # Load vocab
    name_to_id, id_to_name = load_vocab()
    
    # Check cache for pred DEG
    pred_deg_ica_path = cache_dir / "ICA_Pred_vs_Sham_DEG.csv"
    pred_deg_pia_path = cache_dir / "PIA_P_Pred_vs_Sham_DEG.csv"
    
    if pred_deg_ica_path.exists() and pred_deg_pia_path.exists() and not args.force_recompute:
        print("[INFO] Loading cached pred DEG...")
        pred_deg_ica = pd.read_csv(pred_deg_ica_path)
        pred_deg_pia = pd.read_csv(pred_deg_pia_path)
    else:
        # Check if pre-computed expression CSV exists
        expr_csv_path = check_expression_csv_exists(manifest_dir, args.step)
        
        if expr_csv_path is not None:
            # =========================================================
            # Fast path: Load from pre-computed expression CSV
            # =========================================================
            print(f"[INFO] Found pre-computed expression: {expr_csv_path}")
            print("[INFO] Loading expression from CSV (skipping model inference)...")
            
            # Load expression for each ROI
            pred_expr_ica = load_expression_from_csv(
                expr_csv_path, sham_barcodes_ica, cache_sham, args.sham_dataset_name
            )
            pred_expr_pia = load_expression_from_csv(
                expr_csv_path, sham_barcodes_pia, cache_sham, args.sham_dataset_name
            )
            
            pred_expr = {"ICA": pred_expr_ica, "PIA_P": pred_expr_pia}
            
        else:
            # =========================================================
            # Slow path: Run model inference
            # =========================================================
            print(f"[INFO] No pre-computed expression found, running {args.model} inference for step {args.step}...")
            
            # Run model-specific inference
            if args.model == "spatialgt":
                pred_expr = run_spatialgt_inference(
                    manifest_dir, args.step, cache_sham, args.sham_dataset_name,
                    roi_global_indices, args.cache_mode, args.lmdb_path_sham, 
                    args.lmdb_manifest_sham, args.batch_size, args.num_workers,
                    use_recon_as_x0=args.use_recon_as_x0,
                )
            elif args.model == "knn":
                pred_expr = run_knn_inference(
                    manifest_dir, args.step, cache_sham, args.sham_dataset_name,
                    roi_global_indices, args.cache_mode, args.lmdb_path_sham,
                    args.lmdb_manifest_sham, args.knn_infer_mode,
                )
            else:  # sedr
                pred_expr = run_sedr_inference(
                    manifest_dir, args.step, cache_sham, args.sham_dataset_name,
                    roi_global_indices,
                )
        
        # Get control expression
        ctrl_expr_ica = get_ctrl_expression(cache_sham, args.sham_dataset_name, sham_barcodes_ica)
        ctrl_expr_pia = get_ctrl_expression(cache_sham, args.sham_dataset_name, sham_barcodes_pia)
        
        # Compute pred DEG
        print("[INFO] Computing pred DEG...")
        pred_deg_ica = compute_pred_deg(pred_expr["ICA"], ctrl_expr_ica, id_to_name)
        pred_deg_pia = compute_pred_deg(pred_expr["PIA_P"], ctrl_expr_pia, id_to_name)
        
        # Save cache
        pred_deg_ica.to_csv(pred_deg_ica_path, index=False)
        pred_deg_pia.to_csv(pred_deg_pia_path, index=False)
        print(f"[OK] Cached pred DEG to: {cache_dir}")
    
    # ===========================================================================
    # Block 1: GT vs pred for ICA and PIA_P
    # ===========================================================================
    print("\n[STEP] Computing Block 1 metrics (GT vs pred)...")
    if args.gt_scale != 1.0:
        print(f"  [INFO] Using GT scale factor: {args.gt_scale}")
    if args.normalize_block1 != "none":
        print(f"  [INFO] Using normalization mode: {args.normalize_block1}")
    print(f"  [INFO] TopK selection by: {args.topk_by}")
    
    metrics_ica, topk_ica = compute_block1_metrics(
        gt_deg_ica, pred_deg_ica, args.topk, args.p_sig, args.gt_scale, args.normalize_block1, args.topk_by)
    metrics_pia, topk_pia = compute_block1_metrics(
        gt_deg_pia, pred_deg_pia, args.topk, args.p_sig, args.gt_scale, args.normalize_block1, args.topk_by)
    
    print(f"  ICA: PCC={metrics_ica['pcc']:.4f}, Spearman={metrics_ica['spearman']:.4f}, CCC={metrics_ica['ccc']:.4f}, R²={metrics_ica['r2']:.4f}")
    print(f"        RMSE={metrics_ica['rmse_identity']:.4f}, MAE={metrics_ica['mae']:.4f}, MAE_norm={metrics_ica['mae_norm']:.4f}")
    print(f"  PIA_P: PCC={metrics_pia['pcc']:.4f}, Spearman={metrics_pia['spearman']:.4f}, CCC={metrics_pia['ccc']:.4f}, R²={metrics_pia['r2']:.4f}")
    print(f"         RMSE={metrics_pia['rmse_identity']:.4f}, MAE={metrics_pia['mae']:.4f}, MAE_norm={metrics_pia['mae_norm']:.4f}")
    
    # Save topk tables
    topk_dir = out_dir / "topk_tables"
    topk_dir.mkdir(exist_ok=True)
    topk_ica.to_csv(topk_dir / "ICA_topk.csv", index=False)
    topk_pia.to_csv(topk_dir / "PIA_P_topk.csv", index=False)
    
    # Determine whether to use normalized values for plotting
    use_norm_plot = args.plot_normalized and args.normalize_block1 != "none"
    low_resp_thresh = args.low_response_threshold
    if use_norm_plot:
        print(f"  [INFO] Plotting with normalized values, low_response_threshold={low_resp_thresh}")
    
    # Plot scatter
    plot_scatter(topk_ica, "logfc_gt", "logfc_pred", f"ICA: GT vs Pred (n={len(topk_ica)})",
                 out_dir / "scatter_ICA.png", axis_limit=args.axis_limit if not use_norm_plot else None,
                 annotate_genes=args.annotate_genes, 
                 n_annotate_per_direction=args.n_annotate_per_direction,
                 use_normalized=use_norm_plot, low_response_threshold=low_resp_thresh)
    plot_scatter(topk_pia, "logfc_gt", "logfc_pred", f"PIA_P: GT vs Pred (n={len(topk_pia)})",
                 out_dir / "scatter_PIA_P.png", axis_limit=args.axis_limit if not use_norm_plot else None,
                 annotate_genes=args.annotate_genes,
                 n_annotate_per_direction=args.n_annotate_per_direction,
                 use_normalized=use_norm_plot, low_response_threshold=low_resp_thresh)
    
    # ===========================================================================
    # Block 2A: PIA_P_special genes (significant in PIA but not ICA)
    # ===========================================================================
    print("\n[STEP] Computing Block 2A metrics (PIA_P_special)...")
    
    metrics_2a_pia, special_genes_pia = compute_block2a_metrics(
        gt_deg_pia, gt_deg_ica, pred_deg_pia,
        args.topk, args.p_sig, args.logfc_a, args.p_nonsig, args.ica_noise_abs,
    )
    
    print(f"  PIA_P_special: Acc={metrics_2a_pia['acc_pia_spec']:.4f}, n={metrics_2a_pia['n_special_genes']}")
    
    if not special_genes_pia.empty:
        special_genes_pia.to_csv(topk_dir / "PIA_P_special_genes.csv", index=False)
        # Plot scatter for PIA_P special genes
        plot_scatter(special_genes_pia, "logfc_gt_pia", "logfc_pred_pia", 
                     f"PIA_P Special Genes: GT vs Pred (n={len(special_genes_pia)})",
                     out_dir / "scatter_PIA_P_special.png",
                     axis_label_x="GT logFC (PIA_P)", axis_label_y="Pred logFC (PIA_P)",
                     axis_limit=args.axis_limit,
                     annotate_genes=args.annotate_genes,
                     n_annotate_per_direction=args.n_annotate_per_direction,
                     low_response_threshold=low_resp_thresh)
    
    # ===========================================================================
    # Block 2A for ICA: ICA_special genes (significant in ICA but not PIA)
    # ===========================================================================
    print("\n[STEP] Computing Block 2A metrics (ICA_special)...")
    
    metrics_2a_ica, special_genes_ica = compute_block2a_ica_metrics(
        gt_deg_ica, gt_deg_pia, pred_deg_ica,
        args.topk, args.p_sig, args.logfc_a, args.p_nonsig, args.ica_noise_abs,
    )
    
    print(f"  ICA_special: Acc={metrics_2a_ica['acc_ica_spec']:.4f}, n={metrics_2a_ica['n_special_genes']}")
    
    if not special_genes_ica.empty:
        special_genes_ica.to_csv(topk_dir / "ICA_special_genes.csv", index=False)
        # Plot scatter for ICA special genes
        plot_scatter(special_genes_ica, "logfc_gt_ica", "logfc_pred_ica", 
                     f"ICA Special Genes: GT vs Pred (n={len(special_genes_ica)})",
                     out_dir / "scatter_ICA_special.png",
                     axis_label_x="GT logFC (ICA)", axis_label_y="Pred logFC (ICA)",
                     axis_limit=args.axis_limit,
                     annotate_genes=args.annotate_genes,
                     n_annotate_per_direction=args.n_annotate_per_direction,
                     low_response_threshold=low_resp_thresh)
    
    # ===========================================================================
    # Block 2B: Double-delta for PIA (ΔΔ = PIA - ICA)
    # ===========================================================================
    print("\n[STEP] Computing Block 2B metrics (ΔΔ PIA-ICA)...")
    
    metrics_2b_pia, dd_table_pia = compute_block2b_metrics(
        gt_deg_pia, gt_deg_ica, pred_deg_pia, pred_deg_ica, args.topk, args.normalize_block1,
    )
    
    print(f"  ΔΔ (PIA-ICA): Acc={metrics_2b_pia['acc_delta_delta']:.4f}, n={metrics_2b_pia['n_genes']}")
    
    if not dd_table_pia.empty:
        dd_table_pia.to_csv(topk_dir / "delta_delta_pia_topk.csv", index=False)
        plot_scatter(dd_table_pia, "delta_delta_gt", "delta_delta_pred", 
                     f"ΔΔ(PIA-ICA): GT vs Pred (n={len(dd_table_pia)})",
                     out_dir / "scatter_DeltaDelta_PIA.png",
                     axis_label_x="ΔΔ GT (PIA-ICA)", axis_label_y="ΔΔ Pred (PIA-ICA)",
                     axis_limit=args.axis_limit if not use_norm_plot else None,
                     annotate_genes=args.annotate_genes,
                     n_annotate_per_direction=args.n_annotate_per_direction,
                     use_normalized=use_norm_plot, low_response_threshold=low_resp_thresh)
    
    # ===========================================================================
    # Block 2B for ICA: Double-delta (ΔΔ = ICA - PIA)
    # ===========================================================================
    print("\n[STEP] Computing Block 2B metrics (ΔΔ ICA-PIA)...")
    
    metrics_2b_ica, dd_table_ica = compute_block2b_ica_metrics(
        gt_deg_ica, gt_deg_pia, pred_deg_ica, pred_deg_pia, args.topk, args.normalize_block1,
    )
    
    print(f"  ΔΔ (ICA-PIA): Acc={metrics_2b_ica['acc_delta_delta_ica']:.4f}, n={metrics_2b_ica['n_genes']}")
    
    if not dd_table_ica.empty:
        dd_table_ica.to_csv(topk_dir / "delta_delta_ica_topk.csv", index=False)
        plot_scatter(dd_table_ica, "delta_delta_gt_ica", "delta_delta_pred_ica", 
                     f"ΔΔ(ICA-PIA): GT vs Pred (n={len(dd_table_ica)})",
                     out_dir / "scatter_DeltaDelta_ICA.png",
                     axis_label_x="ΔΔ GT (ICA-PIA)", axis_label_y="ΔΔ Pred (ICA-PIA)",
                     axis_limit=args.axis_limit if not use_norm_plot else None,
                     annotate_genes=args.annotate_genes,
                     n_annotate_per_direction=args.n_annotate_per_direction,
                     use_normalized=use_norm_plot, low_response_threshold=low_resp_thresh)
    
    # ===========================================================================
    # Save all metrics
    # ===========================================================================
    all_metrics = {
        "model": args.model,
        "step": args.step,
        "topk": args.topk,
        "gt_scale": args.gt_scale,
        "normalize_block1": args.normalize_block1,
        "block1_ICA": metrics_ica,
        "block1_PIA_P": metrics_pia,
        "block2a_PIA_special": metrics_2a_pia,
        "block2a_ICA_special": metrics_2a_ica,
        "block2b_delta_delta_pia": metrics_2b_pia,
        "block2b_delta_delta_ica": metrics_2b_ica,
    }
    
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    # Also save as flat CSV
    flat_metrics = {
        "model": args.model,
        "step": args.step,
        "gt_scale": args.gt_scale,
        "normalize_block1": args.normalize_block1,
        "ICA_pcc": metrics_ica["pcc"],
        "ICA_spearman": metrics_ica["spearman"],
        "ICA_ccc": metrics_ica["ccc"],
        "ICA_ccc_norm": metrics_ica["ccc_norm"],
        "ICA_r2": metrics_ica["r2"],
        "ICA_rmse_id": metrics_ica["rmse_identity"],
        "ICA_rmse_norm": metrics_ica["rmse_identity_norm"],
        "ICA_mae": metrics_ica["mae"],
        "ICA_mae_norm": metrics_ica["mae_norm"],
        "PIA_P_pcc": metrics_pia["pcc"],
        "PIA_P_spearman": metrics_pia["spearman"],
        "PIA_P_ccc": metrics_pia["ccc"],
        "PIA_P_ccc_norm": metrics_pia["ccc_norm"],
        "PIA_P_r2": metrics_pia["r2"],
        "PIA_P_rmse_id": metrics_pia["rmse_identity"],
        "PIA_P_rmse_norm": metrics_pia["rmse_identity_norm"],
        "PIA_P_mae": metrics_pia["mae"],
        "PIA_P_mae_norm": metrics_pia["mae_norm"],
        "acc_pia_spec": metrics_2a_pia["acc_pia_spec"],
        "acc_ica_spec": metrics_2a_ica["acc_ica_spec"],
        "acc_delta_delta_pia": metrics_2b_pia["acc_delta_delta"],
        "acc_delta_delta_ica": metrics_2b_ica["acc_delta_delta_ica"],
    }
    pd.DataFrame([flat_metrics]).to_csv(out_dir / "metrics.csv", index=False)
    
    print(f"\n[DONE] All metrics saved to: {out_dir}")


if __name__ == "__main__":
    main()

