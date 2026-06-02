#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialGT mouse stroke perturbation inference.

This public script keeps only the dual-line perturbation iteration used for the
mouse stroke analysis. The control state is always initialized from a one-pass
SpatialGT reconstruction, and the perturbation is applied to that reconstructed
state before propagation. Reference-slice output scoring and exploratory
spot-selection branches have been removed.

Outputs:
- expression/ctrl.npz: reconstructed control state used as X_0
- expression/step0.npz: perturbed initial state X'_0
- expression/step*.npz: propagated perturbed states
- convergence/step_mse.csv and convergence_report.json
- perturb_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils_train import process_batch, forward_pass

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


def _build_all_loader(
    databank: SpatialDataBank,
    config: Config,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    """Build data loader for full-slice inference with optimized settings."""
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


def _setup_config_for_cache(
    config: Config,
    cache_dir: str,
    dataset_name: str,
    cache_mode: str = "h5",
    lmdb_path: Optional[str] = None,
    lmdb_manifest_path: Optional[str] = None,
) -> Config:
    """
    Configure Config object for the specified cache mode (h5 or lmdb).
    """
    config.cache_dir = str(cache_dir)
    config.cache_mode = cache_mode.lower()
    config.strict_cache_only = True
    
    if cache_mode.lower() == "lmdb":
        # For LMDB mode, set paths
        if lmdb_path:
            config.lmdb_path = str(lmdb_path)
            config.runtime_lmdb_path = str(lmdb_path)
        else:
            # Default LMDB path based on cache_dir
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


def _wrap_model_for_multi_gpu(model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Wrap model with DataParallel for multi-GPU inference if multiple GPUs are available.
    """
    if not torch.cuda.is_available():
        return model
    
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return model
    
    if device_ids is None:
        device_ids = list(range(n_gpus))
    
    print(f"[INFO] Using DataParallel with {len(device_ids)} GPUs: {device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)

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
def run_single_pass_inference(
    databank: SpatialDataBank,
    loader,
    model: nn.Module,
    device: torch.device,
    desc: str = "Inference",
    show_progress: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Run a single forward pass on the entire slice with progress bar.
    Supports both single-GPU and DataParallel wrapped models.
    Returns overrides dict: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}
    """
    all_overrides: Dict[int, Dict[str, np.ndarray]] = {}
    
    # Get the actual model (unwrap DataParallel if needed)
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

def compute_step_mse(
    current_state: Dict[int, Dict[str, np.ndarray]],
    prev_state: Dict[int, Dict[str, np.ndarray]],
    frozen_indices: Set[int],
) -> Tuple[float, int]:
    """Mean MSE between two states for non-frozen spots."""
    mse_values = []
    for gidx in current_state:
        if gidx in frozen_indices:
            continue
        if gidx not in prev_state:
            continue
        curr_vals = current_state[gidx]["raw_normed_values"]
        prev_vals = prev_state[gidx]["raw_normed_values"]
        n = min(len(curr_vals), len(prev_vals))
        mse_values.append(float(np.mean((curr_vals[:n] - prev_vals[:n]) ** 2)))

    if not mse_values:
        return 0.0, 0
    return float(np.mean(mse_values)), len(mse_values)


def save_step_npz(
    state: Dict[int, Dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Save one step's expression to compressed npz."""
    spot_indices = sorted(state.keys())
    save_dict = {"spot_indices": np.array(spot_indices, dtype=np.int64)}
    for gidx in spot_indices:
        ov = state[gidx]
        save_dict[f"gids_{gidx}"] = ov["gene_ids"]
        save_dict[f"vals_{gidx}"] = ov["raw_normed_values"]
    np.savez_compressed(str(path), **save_dict)

def run_dual_line_with_convergence(
    databank: SpatialDataBank,
    loader_factory: callable,
    model: nn.Module,
    device: torch.device,
    max_steps: int,
    frozen_indices: Set[int],
    x0_prime_base: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    x_line: Dict[int, Dict[str, np.ndarray]],
    roi_global_for_capture: Optional[Set[int]] = None,
    show_progress: bool = True,
    stopping_mode: str = "patience",
    patience: int = 3,
    emergency_div_factor: float = 3.0,
    expr_dir: Optional[Path] = None,
    conv_dir: Optional[Path] = None,
) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Dual-line iterative inference with MSE-minimum stopping.

    The X line (SpatialGT(X_0)) is precomputed and passed as x_line,
    so only the X' line runs each step. Step-wise MSE between consecutive
    X'_k_true states follows a U-shaped curve; the optimal output is
    at the MSE trough.

    Stopping modes:
    - "patience": stop after ``patience`` steps without MSE improvement;
      recommend the step with minimum MSE (U-curve trough).
    - "retrospective": run all ``max_steps``; recommend minimum MSE step.

    Both modes include emergency protection: immediate stop if
    step MSE > emergency_div_factor * best_mse.

    Returns (step_overrides, final_state, convergence_report).
    """
    step_overrides: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    xp_current = {
        int(g): {
            "gene_ids": np.array(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.array(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in x0_prime_state.items()
    }

    mse_history: List[float] = []
    mse_records: List[Dict] = []
    prev_state = xp_current
    actual_steps = 0
    best_mse = float("inf")
    best_step = 0
    steps_since_improvement = 0
    final_status = "completed"
    final_message = ""

    for t in range(1, max_steps + 1):
        step_start = time.time()
        print(f"\n{'='*50}")
        print(f"[STEP {t}/{max_steps}] Dual-line inference")
        print(f"{'='*50}")

        # === X' line: SpatialGT(X'_{t-1}_true) ===
        print(f"  [X' line] Running SpatialGT on X'_{t-1}_true...")
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(xp_current)

        xp_k_raw: Dict[int, Dict[str, np.ndarray]] = {}
        loader = loader_factory()
        iterator = tqdm(loader, desc=f"X' line step {t}", disable=not show_progress)

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
                xp_k_raw[gidx] = {"gene_ids": gid_np[i], "raw_normed_values": val_np[i]}

        # === Delta and X'_k_true (using precomputed X line) ===
        print(f"  Computing delta and X'_{t}_true...")
        xp_k_true: Dict[int, Dict[str, np.ndarray]] = {}
        delta_stats = {"mean": [], "abs_mean": []}

        for gidx in xp_k_raw.keys():
            gidx = int(gidx)

            if gidx in frozen_indices:
                xp_k_true[gidx] = {
                    "gene_ids": x0_prime_base[gidx]["gene_ids"].copy(),
                    "raw_normed_values": x0_prime_base[gidx]["raw_normed_values"].copy(),
                }
                continue

            gids_pred = xp_k_raw[gidx]["gene_ids"]
            xp_raw_vals = xp_k_raw[gidx]["raw_normed_values"]

            x_k_vals = (
                x_line[gidx]["raw_normed_values"]
                if gidx in x_line
                else np.zeros_like(xp_raw_vals)
            )
            delta = xp_raw_vals - x_k_vals
            gid_to_delta = {int(g): delta[i] for i, g in enumerate(gids_pred)}

            if gidx in x0_prime_base:
                base_gids = x0_prime_base[gidx]["gene_ids"]
                base_vals = x0_prime_base[gidx]["raw_normed_values"].copy()
                for i, g in enumerate(base_gids):
                    d = gid_to_delta.get(int(g))
                    if d is not None:
                        base_vals[i] += d
                xp_k_true[gidx] = {
                    "gene_ids": base_gids.copy(),
                    "raw_normed_values": base_vals.astype(np.float32),
                }
            else:
                xp_k_true[gidx] = {
                    "gene_ids": gids_pred.copy(),
                    "raw_normed_values": xp_raw_vals.astype(np.float32),
                }

            delta_stats["mean"].append(float(np.mean(delta)))
            delta_stats["abs_mean"].append(float(np.mean(np.abs(delta))))

        if delta_stats["mean"]:
            print(f"  Delta stats: mean={np.mean(delta_stats['mean']):.6f}, "
                  f"abs_mean={np.mean(delta_stats['abs_mean']):.6f}")

        # === Step MSE convergence tracking ===
        step_mse, n_compared = compute_step_mse(xp_k_true, prev_state, frozen_indices)
        mse_history.append(step_mse)

        if step_mse < best_mse:
            best_mse = step_mse
            best_step = t
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        step_elapsed = time.time() - step_start
        mse_records.append({
            "step": t,
            "mse": step_mse,
            "n_spots_compared": n_compared,
            "elapsed_sec": round(step_elapsed, 1),
            "is_best": steps_since_improvement == 0,
            "steps_since_improvement": steps_since_improvement,
        })

        status_mark = (
            "★ BEST" if steps_since_improvement == 0
            else f"patience {steps_since_improvement}/{patience}"
        )
        print(f"  MSE(step {t-1}→{t}) = {step_mse:.8f}  "
              f"[best={best_mse:.8f} @ step {best_step}]  {status_mark}")

        # === Save step expression as npz ===
        if expr_dir is not None:
            save_step_npz(xp_k_true, expr_dir / f"step{t}.npz")
            print(f"  Saved step{t}.npz ({len(xp_k_true)} spots)")

        # === Capture step data for requested output ===
        if roi_global_for_capture is not None:
            step_overrides[t] = {
                int(k): {
                    "gene_ids": v["gene_ids"].copy(),
                    "raw_normed_values": v["raw_normed_values"].copy(),
                }
                for k, v in xp_k_true.items() if int(k) in roi_global_for_capture
            }
        else:
            step_overrides[t] = {
                int(k): {
                    "gene_ids": v["gene_ids"].copy(),
                    "raw_normed_values": v["raw_normed_values"].copy(),
                }
                for k, v in xp_k_true.items()
            }

        prev_state = xp_k_true
        xp_current = xp_k_true
        actual_steps = t

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  [INFO] Step {t} completed: {len(xp_k_true)} spots in X'_{t}_true")

        # === Emergency divergence protection ===
        if best_mse > 0 and step_mse > emergency_div_factor * best_mse:
            final_status = "emergency_stop"
            final_message = (
                f"Emergency stop: MSE {step_mse:.6f} exceeded "
                f"{emergency_div_factor}x best MSE {best_mse:.6f}"
            )
            print(f"\n[EMERGENCY STOP] {final_message}")
            break

        # === Patience-based early stop ===
        if stopping_mode == "patience" and steps_since_improvement >= patience:
            final_status = "stopped_patience"
            final_message = (
                f"Patience exhausted: no improvement for {patience} steps "
                f"(best @ step {best_step}, MSE={best_mse:.8f})"
            )
            print(f"\n[PATIENCE STOP] {final_message}")
            break
    else:
        if stopping_mode == "retrospective":
            final_status = "completed_retrospective"
            final_message = (
                f"All {max_steps} steps completed; "
                f"best @ step {best_step} (MSE={best_mse:.8f})"
            )
        else:
            final_status = "max_steps_reached"
            final_message = (
                f"Reached max {max_steps} steps; "
                f"best @ step {best_step} (MSE={best_mse:.8f})"
            )

    print(f"\n{'='*50}")
    print(f"[RESULT] Optimal step: step {best_step} "
          f"(MSE = {best_mse:.8f})")
    print(f"{'='*50}")

    # === Save convergence report ===
    conv_report = {
        "stopping_mode": stopping_mode,
        "status": final_status,
        "message": final_message,
        "actual_steps": actual_steps,
        "max_steps": max_steps,
        "best_step": best_step,
        "best_mse": best_mse,
        "mse_history": mse_history,
        "patience": patience,
        "emergency_div_factor": emergency_div_factor,
    }

    if conv_dir is not None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        mse_df = pd.DataFrame(mse_records)
        mse_df.to_csv(conv_dir / "step_mse.csv", index=False)
        with open(conv_dir / "convergence_report.json", "w") as f:
            json.dump(conv_report, f, indent=2)

    return step_overrides, xp_current, conv_report

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


def main():
    parser = argparse.ArgumentParser(description="SpatialGT mouse stroke perturbation inference")
    parser.add_argument("--ckpt_control", required=True, help="Checkpoint used to reconstruct the Sham control state")
    parser.add_argument("--ckpt_diffusion", default=None, help="Checkpoint used for dual-line propagation; defaults to --ckpt_control")
    parser.add_argument("--cache_sham", required=True, help="Cache directory for the Sham slice")
    parser.add_argument("--sham_dataset_name", default="Sham1-1")
    parser.add_argument("--roi_manifest", required=True, help="JSON manifest with Sham ROI barcodes")
    parser.add_argument("--deg_csv", required=True, help="DEG CSV with gene and avg_logFC columns")
    parser.add_argument("--cache_mode", choices=["h5", "lmdb"], default="h5")
    parser.add_argument("--lmdb_path_sham", default=None)
    parser.add_argument("--lmdb_manifest_sham", default=None)
    parser.add_argument("--perturb_target_roi", choices=["ICA", "PIA_P", "PIA_D"], default="ICA")
    parser.add_argument("--n_spots", type=int, default=20, help="Number of randomly selected target ROI spots; <=0 uses all target ROI spots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_adj_thresh", type=float, default=0.05)
    parser.add_argument("--min_abs_logfc", type=float, default=0.0)
    parser.add_argument("--logfc_strength", type=float, default=1.0)
    parser.add_argument("--logfc_clip", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--no_freeze_perturbed", action="store_true", help="Allow directly perturbed spots to update during propagation")
    parser.add_argument("--stopping_mode", choices=["patience", "retrospective"], default="patience")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--emergency_div_factor", type=float, default=3.0)
    parser.add_argument("--save_expr_steps", default="", help="Comma-separated steps to save as CSV, or 'all'")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated CUDA device IDs")
    parser.add_argument("--show_progress", action="store_true", default=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expr_dir = out_dir / "expression"
    conv_dir = out_dir / "convergence"
    expr_dir.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    config = _setup_config_for_cache(
        config,
        cache_dir=args.cache_sham,
        dataset_name=args.sham_dataset_name,
        cache_mode=args.cache_mode,
        lmdb_path=args.lmdb_path_sham,
        lmdb_manifest_path=args.lmdb_manifest_sham,
    )
    databank = SpatialDataBank(
        dataset_paths=[str(Path(args.cache_sham) / args.sham_dataset_name / "processed.h5ad")],
        cache_dir=str(Path(args.cache_sham)),
        config=config,
        load_data=False,
    )

    model_control = SpatialNeighborTransformer(config).to(device)
    _load_state_dict_into_model(model_control, Path(args.ckpt_control))
    model_control.eval()

    diffusion_ckpt = Path(args.ckpt_diffusion) if args.ckpt_diffusion else Path(args.ckpt_control)
    model_diffusion = SpatialNeighborTransformer(config).to(device)
    _load_state_dict_into_model(model_diffusion, diffusion_ckpt)
    model_diffusion.eval()

    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None
        model_control = _wrap_model_for_multi_gpu(model_control, gpu_ids)
        model_diffusion = _wrap_model_for_multi_gpu(model_diffusion, gpu_ids)

    def loader_factory():
        return _build_all_loader(
            databank,
            config,
            args.batch_size,
            args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=False,
        )

    roi_barcodes = load_roi_manifest(Path(args.roi_manifest))
    sham_roi_global = {
        roi: _barcodes_to_global_idx(Path(args.cache_sham), args.sham_dataset_name, barcodes)
        for roi, barcodes in roi_barcodes.items()
    }
    target_roi_global = sham_roi_global[args.perturb_target_roi]
    selected_global = select_random_spots(target_roi_global, args.n_spots, args.seed)
    weights, weight_meta = compute_uniform_weights(selected_global)

    print("[STEP] Building reconstructed control state X_0")
    databank.clear_runtime_spot_overrides()
    x0_state = run_single_pass_inference(
        databank,
        loader_factory(),
        model_control,
        device,
        desc="X_0 reconstruction",
        show_progress=args.show_progress,
    )
    print(f"[INFO] X_0 contains {len(x0_state)} spots")

    gene_logfc, deg_meta = load_deg_gene_logfc(
        Path(args.deg_csv),
        args.p_adj_thresh,
        args.min_abs_logfc,
        args.logfc_clip,
    )
    x0_prime_state, perturb_meta = apply_deg_perturbation_to_state(
        x0_state,
        selected_global,
        weights,
        gene_logfc,
        args.logfc_strength,
    )
    frozen_indices = set(selected_global) if not args.no_freeze_perturbed else set()

    perturb_manifest = {
        "method": "SpatialGT",
        "algorithm": "dual_line_error_cancellation",
        "x0_source": "reconstructed_control",
        "perturb_target_roi": args.perturb_target_roi,
        "spot_selection": "random",
        "n_spots_selected": len(selected_global),
        "seed": args.seed,
        "selected_global_indices": selected_global,
        "selected_barcodes": _global_idx_to_barcode(Path(args.cache_sham), args.sham_dataset_name, selected_global),
        "weights": weights.tolist(),
        "freeze_perturbed": not args.no_freeze_perturbed,
        "ckpt_control": str(args.ckpt_control),
        "ckpt_diffusion": str(diffusion_ckpt),
        **weight_meta,
        **deg_meta,
        **perturb_meta,
    }
    with open(out_dir / "perturb_manifest.json", "w", encoding="utf-8") as f:
        json.dump(perturb_manifest, f, indent=2)

    print("[STEP] Precomputing unperturbed line from X_0")
    databank.clear_runtime_spot_overrides()
    databank.set_runtime_spot_overrides(x0_state)
    x_line = run_single_pass_inference(
        databank,
        loader_factory(),
        model_diffusion,
        device,
        desc="Unperturbed line",
        show_progress=args.show_progress,
    )

    save_step_npz(x0_state, expr_dir / "ctrl.npz")
    save_step_npz(x0_prime_state, expr_dir / "step0.npz")

    x0_prime_base = {
        int(g): {
            "gene_ids": np.asarray(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in x0_prime_state.items()
    }
    step_overrides, final_state, conv_report = run_dual_line_with_convergence(
        databank,
        loader_factory,
        model_diffusion,
        device,
        max_steps=args.max_steps,
        frozen_indices=frozen_indices,
        x0_prime_base=x0_prime_base,
        x0_prime_state=x0_prime_state,
        x_line=x_line,
        roi_global_for_capture=None,
        show_progress=args.show_progress,
        stopping_mode=args.stopping_mode,
        patience=args.patience,
        emergency_div_factor=args.emergency_div_factor,
        expr_dir=expr_dir,
        conv_dir=conv_dir,
    )

    save_steps = parse_save_expr_steps(args.save_expr_steps, conv_report["actual_steps"])
    if save_steps:
        vocab_path = Config().vocab_file
        if 0 in save_steps:
            save_full_expression_to_csv(x0_prime_state, expr_dir / "expression_step0.csv", vocab_path, Path(args.cache_sham), args.sham_dataset_name)
        for step, state in step_overrides.items():
            if step in save_steps:
                save_full_expression_to_csv(state, expr_dir / f"expression_step{step}.csv", vocab_path, Path(args.cache_sham), args.sham_dataset_name)

    print(f"[OK] Finished SpatialGT perturbation inference. Best step: {conv_report['best_step']}")


if __name__ == "__main__":
    main()
