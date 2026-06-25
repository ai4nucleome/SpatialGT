#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microenvironment perturbation experiment for cancer/healthy niche transitions.

Experiment design:
  For each tissue slice, select N paired spots (healthy center + cancer center).
  Systematically replace k neighbors (out of 8 for 1-hop, 16 for 2-hop) with
  linearly interpolated expressions between source and target niche, then run
  dual-line iterative inference to observe how the center spot responds.

Two directions per pair:
  - cancer_to_healthy: cancer center spot, gradually make neighbors healthier
  - healthy_to_cancer: healthy center spot, gradually make neighbors more cancerous

Perturbation formula (linear interpolation):
  perturbed_expr = (1 - alpha) * source_neighbor_expr + alpha * target_neighbor_expr

Default dual-line base (--x0_source=recon, Algorithm 1):
  The dual-line base state is the DENOISED expression X_0 = M(X) = recon1,
  not the raw input. Perturbation is interpolated on recon1 neighbors, and the
  fixed control prediction (X line) is X_hat = M(X_0) = recon2. This removes the
  raw reconstruction residual from the comparison. Pass --x0_source=raw to run
  the raw-input legacy mode (X_0 = raw input, X line = recon1).

Dual-line error cancellation:
  X line:  X_k = SpatialGT(X_0)        # X_0 = recon1 (recon) or raw (legacy)
  X' line: X'_k_raw = SpatialGT(X'_{k-1}_true)
  delta_k = X'_k_raw - X_k
  X'_k_true = X'_0 + delta_k
  X'_k_true(frozen) = X'_0(frozen)

Metrics (center spot vs target center spot):
  Pearson correlation coefficient (PCC) and RMSE

Output structure:
  {out_dir}/{dataset}/{direction}/pair{i}/hop{h}_k{k}_alpha{a}/
    |-- step_expressions/        # per-step gene expression
    |-- convergence_report.json  # step RMSE convergence
    |-- metrics.json             # similarity metrics
    `-- manifest.json            # experiment metadata
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
for _path in (_REPO_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pretrain.Config import Config
from pretrain.spatial_databank import SpatialDataBank
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils_train import process_batch, forward_pass


# -----------------------------------------------------------------------------
# Annotation loading
# -----------------------------------------------------------------------------

# Category mapping: dataset-specific label -> canonical {cancer, healthy, border, other}
_LABEL_MAP = {
    "1142243F": {
        "Invasive cancer + stroma + lymphocytes": "cancer",
        "Necrosis": "other",
        "Stroma": "healthy",
        "Artefact": "other",
        "Lymphocytes": "healthy",
        "TLS": "healthy",
    },
    "1160920F": {
        "Invasive cancer + stroma + lymphocytes": "cancer",
        "Stroma": "healthy",
        "Normal glands + lymphocytes": "healthy",
        "Lymphocytes": "healthy",
        "Adipose tissue": "other",
        "Artefact": "other",
        "DCIS": "cancer",
        "Cancer trapped in lymphocyte aggregation": "cancer",
    },
    "HBRC": {},  # dynamically built from ground_truth column
    "PDAC": {
        "Cancer": "cancer",
        "Stroma": "healthy",
        "Duct Epithelium": "healthy",
        "Pancreatic": "healthy",
    },
}


def _build_hbrc_label_map(labels: List[str]) -> Dict[str, str]:
    """Build label->canonical map for HBRC based on prefix."""
    m = {}
    for lbl in set(labels):
        if lbl.startswith("IDC") or lbl.startswith("DCIS") or lbl.startswith("LCIS"):
            m[lbl] = "cancer"
        elif lbl.startswith("Healthy"):
            m[lbl] = "healthy"
        elif lbl.startswith("Tumor_edge"):
            m[lbl] = "border"
        else:
            m[lbl] = "other"
    return m


def load_spot_annotations(
    dataset_name: str,
    cache_dir: str,
    truth_csv: Optional[str] = None,
) -> Dict[int, str]:
    """
    Load per-spot canonical labels {global_idx: 'cancer'|'healthy'|'border'|'other'}.
    """
    import anndata as ad

    cache_path = Path(cache_dir)
    meta_path = cache_path / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    start_idx = 0
    for info in meta.get("dataset_indices", []):
        ds_idx = info.get("dataset_idx", 0)
        ds_meta = meta.get("datasets", [{}])[ds_idx]
        if ds_meta.get("name") == dataset_name:
            start_idx = info.get("start_idx", 0)
            break

    proc_path = cache_path / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(proc_path), backed="r")
    barcodes = list(adata.obs_names)
    n_obs = adata.n_obs

    annotations: Dict[int, str] = {}

    if dataset_name in ("1142243F", "1160920F"):
        lmap = _LABEL_MAP[dataset_name]
        cls_col = adata.obs["Classification"]
        for i in range(n_obs):
            raw = str(cls_col.iloc[i])
            annotations[start_idx + i] = lmap.get(raw, "other")

    elif dataset_name == "PDAC":
        lmap = _LABEL_MAP["PDAC"]
        gt_col = adata.obs["Ground Truth"]
        for i in range(n_obs):
            raw = str(gt_col.iloc[i])
            annotations[start_idx + i] = lmap.get(raw, "other")

    elif dataset_name == "HBRC":
        if truth_csv is None:
            truth_csv = str(Path(cache_dir).parent / "hbrc_truth.csv")
        truth_df = pd.read_csv(truth_csv)
        bc_to_gt = dict(zip(truth_df["ID"], truth_df["ground_truth"]))
        all_labels = list(bc_to_gt.values())
        lmap = _build_hbrc_label_map(all_labels)
        for i, bc in enumerate(barcodes):
            gt = bc_to_gt.get(bc, "unknown")
            annotations[start_idx + i] = lmap.get(gt, "other")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return annotations


# -----------------------------------------------------------------------------
# Spot pair selection
# -----------------------------------------------------------------------------

def get_hop_neighbors(
    databank: SpatialDataBank,
    center_global_idx: int,
    hop: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Get 1-hop and optionally 2-hop neighbors.
    Returns (hop1_neighbors, hop2_neighbors).
    hop2_neighbors only populated when hop >= 2.
    """
    hop1 = databank.get_neighbors_for_spot(center_global_idx)
    hop1 = [int(n) for n in hop1 if int(n) >= 0]

    hop2 = []
    if hop >= 2:
        hop1_set = set(hop1) | {center_global_idx}
        for nb in hop1:
            nb_neighbors = databank.get_neighbors_for_spot(nb)
            for nn in nb_neighbors:
                nn = int(nn)
                if nn >= 0 and nn not in hop1_set and nn not in set(hop2):
                    hop2.append(nn)

    return hop1, hop2


def _score_spot_purity(
    global_idx: int,
    annotations: Dict[int, str],
    neighbors: List[int],
    target_label: str,
) -> float:
    """Score how 'pure' a spot's neighborhood is w.r.t. target_label."""
    if global_idx not in annotations:
        return -1.0
    if annotations[global_idx] != target_label:
        return -1.0
    if not neighbors:
        return 0.0
    match = sum(1 for n in neighbors if annotations.get(n) == target_label)
    return match / len(neighbors)


def _get_nhop_neighborhood(databank: SpatialDataBank, center: int, n_hops: int = 2) -> Set[int]:
    """Get all spot indices within n_hops of center (inclusive)."""
    hood: Set[int] = {center}
    frontier: Set[int] = {center}
    for _ in range(n_hops):
        next_frontier: Set[int] = set()
        for node in frontier:
            nbs = databank.get_neighbors_for_spot(node)
            for nb in nbs:
                nb = int(nb)
                if nb >= 0 and nb not in hood:
                    next_frontier.add(nb)
        hood.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return hood


def select_spot_pairs(
    databank: SpatialDataBank,
    annotations: Dict[int, str],
    n_pairs: int,
    seed: int,
    min_healthy_purity: float = 0.5,
    min_cancer_purity: float = 0.5,
    min_neighbors: int = 4,
    exclusion_hops: int = 4,
) -> List[Dict[str, Any]]:
    """
    Select N pairs of (healthy_center, cancer_center) spots.
    Each center should have a 'pure' neighborhood of the same type.
    Spatial exclusion: healthy center must NOT fall within cancer center's
    ``exclusion_hops``-hop neighborhood (and vice versa), ensuring that even
    2-hop perturbation experiments have no spatial overlap between pairs.
    """
    rng = random.Random(seed)

    healthy_candidates = []
    cancer_candidates = []

    all_indices = sorted(annotations.keys())
    for gidx in all_indices:
        label = annotations[gidx]
        if label not in ("cancer", "healthy"):
            continue
        hop1 = databank.get_neighbors_for_spot(gidx)
        hop1 = [int(n) for n in hop1 if int(n) >= 0]
        if len(hop1) < min_neighbors:
            continue
        purity = _score_spot_purity(gidx, annotations, hop1, label)
        if label == "healthy" and purity >= min_healthy_purity:
            healthy_candidates.append((gidx, purity, hop1))
        elif label == "cancer" and purity >= min_cancer_purity:
            cancer_candidates.append((gidx, purity, hop1))

    healthy_candidates.sort(key=lambda x: -x[1])
    cancer_candidates.sort(key=lambda x: -x[1])

    n_avail = min(len(healthy_candidates), len(cancer_candidates), n_pairs * 3)
    if n_avail < n_pairs:
        print(f"[WARN] Only {n_avail} candidates (need {n_pairs}). "
              f"Healthy={len(healthy_candidates)}, Cancer={len(cancer_candidates)}. "
              f"Reducing purity thresholds.")
        healthy_candidates_all = []
        cancer_candidates_all = []
        fallback_min_nb = max(min_neighbors // 2, 2)
        for gidx in all_indices:
            label = annotations[gidx]
            if label not in ("cancer", "healthy"):
                continue
            hop1 = databank.get_neighbors_for_spot(gidx)
            hop1 = [int(n) for n in hop1 if int(n) >= 0]
            if len(hop1) < fallback_min_nb:
                continue
            purity = _score_spot_purity(gidx, annotations, hop1, label)
            if purity < 0:
                continue
            if label == "healthy":
                healthy_candidates_all.append((gidx, purity, hop1))
            else:
                cancer_candidates_all.append((gidx, purity, hop1))
        healthy_candidates_all.sort(key=lambda x: -x[1])
        cancer_candidates_all.sort(key=lambda x: -x[1])
        healthy_candidates = healthy_candidates_all
        cancer_candidates = cancer_candidates_all

    top_healthy = healthy_candidates[:n_pairs * 5]
    top_cancer = cancer_candidates[:n_pairs * 5]

    rng.shuffle(top_healthy)
    rng.shuffle(top_cancer)

    pairs = []
    used_healthy = set()
    used_cancer = set()
    for h, c in zip(top_healthy, top_cancer):
        if len(pairs) >= n_pairs:
            break
        if h[0] in used_healthy or c[0] in used_cancer:
            continue

        c_hood = _get_nhop_neighborhood(databank, c[0], exclusion_hops)
        h_hood = _get_nhop_neighborhood(databank, h[0], exclusion_hops)
        if h[0] in c_hood or c[0] in h_hood:
            continue

        pairs.append({
            "healthy_center": h[0],
            "healthy_purity": h[1],
            "cancer_center": c[0],
            "cancer_purity": c[1],
        })
        used_healthy.add(h[0])
        used_cancer.add(c[0])

    if len(pairs) < n_pairs:
        print(f"[WARN] Could only find {len(pairs)} valid pairs out of {n_pairs} requested")

    return pairs


# -----------------------------------------------------------------------------
# Perturbation: neighbor expression interpolation
# -----------------------------------------------------------------------------

def create_interpolated_perturbation(
    databank: SpatialDataBank,
    source_neighbors: List[int],
    target_neighbors: List[int],
    k: int,
    alpha: float,
    seed: int,
    value_source: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Replace k source neighbors with interpolated expressions.

    perturbed_expr = (1-alpha)*source_expr + alpha*target_expr

    target_neighbors are used as the "template" for what the neighbors would
    look like in the target niche. We cycle through target_neighbors if
    len(target_neighbors) < k.

    value_source: optional precomputed expression state (e.g. recon1 = M(X)).
      When provided, the source/target neighbor expressions are taken from this
      state instead of the raw databank, so the perturbation is applied in the
      denoised (reconstructed) space (Algorithm 1, X_0 = M(X)).

    Returns overrides dict: {global_idx: {"gene_ids": ..., "raw_normed_values": ...}}
    """
    if k == 0 or alpha == 0.0:
        return {}

    rng = random.Random(seed)
    if k > len(source_neighbors):
        k = len(source_neighbors)

    selected = rng.sample(source_neighbors, k)

    def _get_expr(idx: int):
        if value_source is not None and int(idx) in value_source:
            d = value_source[int(idx)]
            return (np.asarray(d["gene_ids"], dtype=np.int64),
                    np.asarray(d["raw_normed_values"], dtype=np.float32))
        d = databank.get_spot_data(int(idx))
        return (np.asarray(d["gene_ids"], dtype=np.int64),
                np.asarray(d["raw_normed_values"], dtype=np.float32))

    overrides: Dict[int, Dict[str, np.ndarray]] = {}
    for i, src_idx in enumerate(selected):
        src_gids, src_vals = _get_expr(src_idx)

        tgt_idx = target_neighbors[i % len(target_neighbors)]
        tgt_gids, tgt_vals = _get_expr(tgt_idx)

        # Align gene spaces: build interpolated values on source gene space,
        # incorporating target values for matching genes
        tgt_map = {int(g): float(v) for g, v in zip(tgt_gids, tgt_vals)}
        new_vals = src_vals.copy()
        for j, gid in enumerate(src_gids):
            gid_int = int(gid)
            if gid_int in tgt_map:
                new_vals[j] = (1 - alpha) * src_vals[j] + alpha * tgt_map[gid_int]

        overrides[src_idx] = {
            "gene_ids": src_gids.copy(),
            "raw_normed_values": np.maximum(new_vals, 0.0).astype(np.float32),
        }

    return overrides


# -----------------------------------------------------------------------------
# Model loading utilities
# -----------------------------------------------------------------------------

def load_model(config: Config, ckpt_dir: str, device: torch.device) -> nn.Module:
    model = SpatialNeighborTransformer(config).to(device)
    ckpt_path = Path(ckpt_dir)

    for fname in ["finetuned_state_dict.pth", "model.safetensors", "pytorch_model.bin"]:
        fpath = ckpt_path / fname
        if fpath.exists():
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(str(fpath))
            else:
                sd = torch.load(str(fpath), map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
            model.eval()
            return model

    raise FileNotFoundError(f"No model weights in {ckpt_dir}")


def setup_databank(
    config: Config,
    cache_dir: str,
    dataset_name: str,
    lmdb_path: Optional[str] = None,
    lmdb_manifest: Optional[str] = None,
    cache_mode: str = "lmdb",
) -> SpatialDataBank:
    config.cache_dir = cache_dir
    config.cache_mode = cache_mode
    config.strict_cache_only = True

    if cache_mode == "lmdb" and lmdb_path:
        config.lmdb_path = lmdb_path
        config.runtime_lmdb_path = lmdb_path
        if lmdb_manifest:
            config.lmdb_manifest_path = lmdb_manifest
            config.runtime_lmdb_manifest_path = lmdb_manifest

    proc_path = str(Path(cache_dir) / dataset_name / "processed.h5ad")
    bank = SpatialDataBank(
        dataset_paths=[proc_path],
        cache_dir=cache_dir,
        config=config,
        force_rebuild=False,
    )
    return bank


def build_loader(
    databank: SpatialDataBank,
    config: Config,
    batch_size: int = 64,
    num_workers: int = 4,
):
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
        prefetch_factor=4 if num_workers > 0 else 2,
        is_training=False,
    )


# -----------------------------------------------------------------------------
# Single-pass inference (full slice)
# -----------------------------------------------------------------------------

def run_full_inference(
    databank: SpatialDataBank,
    loader,
    model: nn.Module,
    device: torch.device,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    """Full-slice forward pass, returns {global_idx: {gene_ids, raw_normed_values}}."""
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for batch in tqdm(loader, desc=desc):
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            continue
        centers = batch["structure"].get("centers_global_indices", None)
        if isinstance(centers, torch.Tensor):
            centers = centers.cpu().numpy().astype(int).tolist()

        batch_data = process_batch(batch, device, config=databank.config)
        with torch.no_grad():
            preds, _, _, _ = forward_pass(model, batch_data, config=databank.config)

        B = len(centers) if centers else preds.shape[0]
        gids = batch_data["genes"][:B]
        mask = batch_data["padding_attention_mask"][:B].to(torch.bool)
        gathered = preds.gather(1, gids.clamp(min=0)) * mask.to(preds.dtype)

        gid_np = gids.detach().cpu().numpy().astype(np.int64)
        val_np = gathered.detach().float().cpu().numpy().astype(np.float32)

        if centers is None:
            centers = list(range(B))
        for i in range(B):
            results[int(centers[i])] = {
                "gene_ids": gid_np[i],
                "raw_normed_values": val_np[i],
            }
    return results


def get_or_compute_recon1(
    databank: SpatialDataBank,
    model: nn.Module,
    device: torch.device,
    config: Config,
    cache_path: Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Run a single-pass reconstruction on raw data and cache results.
    If cache file exists, load from disk instead of recomputing.
    Returns {global_idx: {"gene_ids": ndarray, "raw_normed_values": ndarray}}.
    """
    if cache_path.exists():
        print(f"[RECON1] Loading cached reconstruction from {cache_path}")
        data = np.load(str(cache_path), allow_pickle=True)
        recon1: Dict[int, Dict[str, np.ndarray]] = {}
        for key in data.files:
            if key.startswith("gids_"):
                gidx = int(key[5:])
                recon1[gidx] = {
                    "gene_ids": data[f"gids_{gidx}"],
                    "raw_normed_values": data[f"vals_{gidx}"],
                }
        print(f"[RECON1] Loaded {len(recon1)} spots from cache")
        return recon1

    print(f"[RECON1] Running single-pass reconstruction on full slice...")
    databank.clear_runtime_spot_overrides()
    loader = build_loader(databank, config, batch_size, num_workers)
    recon1 = run_full_inference(databank, loader, model, device,
                                desc="Recon1 (full slice)")
    print(f"[RECON1] Reconstructed {len(recon1)} spots")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for gidx, ov in recon1.items():
        save_dict[f"gids_{gidx}"] = ov["gene_ids"]
        save_dict[f"vals_{gidx}"] = ov["raw_normed_values"]
    np.savez_compressed(str(cache_path), **save_dict)
    print(f"[RECON1] Cached to {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return recon1


def get_or_compute_recon2(
    databank: SpatialDataBank,
    model: nn.Module,
    device: torch.device,
    config: Config,
    recon1_cache: Dict[int, Dict[str, np.ndarray]],
    cache_path: Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute X_hat = M(X_0) = SpatialGT(recon1), i.e. a second reconstruction pass
    where the full-slice input is the denoised state X_0 = recon1 = M(X).

    This is the fixed control prediction (the "X line") for recon-space dual-line
    inference (Algorithm 1). Caches to disk like recon1.
    Returns {global_idx: {"gene_ids": ndarray, "raw_normed_values": ndarray}}.
    """
    if cache_path.exists():
        print(f"[RECON2] Loading cached X_hat=M(X_0) from {cache_path}")
        data = np.load(str(cache_path), allow_pickle=True)
        recon2: Dict[int, Dict[str, np.ndarray]] = {}
        for key in data.files:
            if key.startswith("gids_"):
                gidx = int(key[5:])
                recon2[gidx] = {
                    "gene_ids": data[f"gids_{gidx}"],
                    "raw_normed_values": data[f"vals_{gidx}"],
                }
        print(f"[RECON2] Loaded {len(recon2)} spots from cache")
        return recon2

    print(f"[RECON2] Running X_hat = SpatialGT(X_0 = recon1) on full slice...")
    databank.clear_runtime_spot_overrides()
    databank.set_runtime_spot_overrides(recon1_cache)
    loader = build_loader(databank, config, batch_size, num_workers)
    recon2 = run_full_inference(databank, loader, model, device,
                                desc="Recon2 = M(X_0) (full slice)")
    databank.clear_runtime_spot_overrides()
    print(f"[RECON2] Reconstructed {len(recon2)} spots")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for gidx, ov in recon2.items():
        save_dict[f"gids_{gidx}"] = ov["gene_ids"]
        save_dict[f"vals_{gidx}"] = ov["raw_normed_values"]
    np.savez_compressed(str(cache_path), **save_dict)
    print(f"[RECON2] Cached to {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return recon2


# -----------------------------------------------------------------------------
# Dual-line iterative inference
# -----------------------------------------------------------------------------

def compute_step_rmse(
    current_state: Dict[int, Dict[str, np.ndarray]],
    prev_state: Dict[int, Dict[str, np.ndarray]],
    frozen_indices: Set[int],
) -> Tuple[float, int]:
    """Mean RMSE between two consecutive states for non-frozen spots."""
    rmse_values = []
    for gidx in current_state:
        if gidx in frozen_indices:
            continue
        if gidx not in prev_state:
            continue
        curr_vals = current_state[gidx]["raw_normed_values"]
        prev_vals = prev_state[gidx]["raw_normed_values"]
        n = min(len(curr_vals), len(prev_vals))
        rmse_values.append(float(np.sqrt(np.mean((curr_vals[:n] - prev_vals[:n]) ** 2))))
    if not rmse_values:
        return 0.0, 0
    return float(np.mean(rmse_values)), len(rmse_values)


def run_dual_line_inference(
    databank: SpatialDataBank,
    loader_factory,
    model: nn.Module,
    device: torch.device,
    max_steps: int,
    frozen_indices: Set[int],
    x0_state: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    capture_indices: Optional[Set[int]] = None,
    precomputed_x_line: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    stopping_mode: str = "patience",
    patience: int = 3,
    emergency_div_factor: float = 3.0,
) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Dual-line iterative inference with patience-based early stopping.

    Stopping modes:
    - "patience": stop after ``patience`` steps without RMSE improvement.
    - "retrospective": run all ``max_steps``; recommend minimum RMSE step.

    Both modes include emergency protection: immediate stop if
    step RMSE > emergency_div_factor * best_rmse.

    Returns (step_results, final_state, convergence_info).
    """
    x0p_base = {
        int(g): {
            "gene_ids": np.array(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.array(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in x0_prime_state.items()
    }

    xp_current = {
        int(g): {
            "gene_ids": np.array(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.array(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in x0_prime_state.items()
    }

    step_results: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    if precomputed_x_line is not None:
        x_k = precomputed_x_line
    else:
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(x0_state)
        loader = loader_factory()
        x_k = run_full_inference(databank, loader, model, device, desc="X line (once)")

    prev_state = xp_current
    best_rmse = float("inf")
    best_step = 0
    steps_since_improvement = 0
    actual_steps = 0
    rmse_records: List[Dict] = []
    final_status = "completed"

    for t in range(1, max_steps + 1):
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(xp_current)
        loader = loader_factory()
        xp_k_raw = run_full_inference(databank, loader, model, device, desc=f"X' step {t}")

        xp_k_true: Dict[int, Dict[str, np.ndarray]] = {}
        for gidx in xp_k_raw:
            gidx = int(gidx)
            if gidx in frozen_indices:
                xp_k_true[gidx] = {
                    "gene_ids": x0p_base[gidx]["gene_ids"].copy(),
                    "raw_normed_values": x0p_base[gidx]["raw_normed_values"].copy(),
                }
            else:
                gids_pred = xp_k_raw[gidx]["gene_ids"]
                xp_raw_vals = xp_k_raw[gidx]["raw_normed_values"]
                x_k_vals = x_k[gidx]["raw_normed_values"] if gidx in x_k else np.zeros_like(xp_raw_vals)
                delta = xp_raw_vals - x_k_vals

                gid_to_delta = {int(g): delta[i] for i, g in enumerate(gids_pred)}

                if gidx in x0p_base:
                    base_gids = x0p_base[gidx]["gene_ids"]
                    base_vals = x0p_base[gidx]["raw_normed_values"].copy()
                    for i, g in enumerate(base_gids):
                        if int(g) in gid_to_delta:
                            base_vals[i] += gid_to_delta[int(g)]
                    xp_k_true[gidx] = {
                        "gene_ids": base_gids.copy(),
                        "raw_normed_values": base_vals.astype(np.float32),
                    }
                else:
                    xp_k_true[gidx] = {
                        "gene_ids": gids_pred.copy(),
                        "raw_normed_values": xp_raw_vals.astype(np.float32),
                    }

        # Step RMSE convergence tracking (full-slice, non-frozen)
        step_rmse, n_compared = compute_step_rmse(xp_k_true, prev_state, frozen_indices)

        if n_compared > 0:
            if step_rmse < best_rmse:
                best_rmse = step_rmse
                best_step = t
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            status_mark = (
                "BEST" if steps_since_improvement == 0
                else f"patience {steps_since_improvement}/{patience}"
            )
            print(f"  RMSE(step {t-1}->{t}) = {step_rmse:.8f}  "
                  f"[best={best_rmse:.8f} @ step {best_step}]  {status_mark}  "
                  f"(n={n_compared})")
        else:
            print(f"  RMSE(step {t-1}->{t}) = N/A  "
                  f"(no non-frozen spots to compare, warm-up step)")

        rmse_records.append({
            "step": t, "rmse": step_rmse if n_compared > 0 else None,
            "n_spots_compared": n_compared,
            "is_best": n_compared > 0 and steps_since_improvement == 0,
            "steps_since_improvement": steps_since_improvement,
        })

        # Capture
        if capture_indices is not None:
            step_results[t] = {
                int(k): {
                    "gene_ids": v["gene_ids"].copy(),
                    "raw_normed_values": v["raw_normed_values"].copy(),
                }
                for k, v in xp_k_true.items() if int(k) in capture_indices
            }
        else:
            step_results[t] = {
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

        # Emergency divergence protection (only when we have valid comparisons)
        if n_compared > 0 and best_rmse > 0 and step_rmse > emergency_div_factor * best_rmse:
            final_status = "emergency_stop"
            print(f"\n[EMERGENCY STOP] RMSE {step_rmse:.6f} > "
                  f"{emergency_div_factor}x best {best_rmse:.6f}")
            break

        # Patience-based early stop (only after at least one valid RMSE)
        if (n_compared > 0 and stopping_mode == "patience"
                and steps_since_improvement >= patience
                and best_rmse < float("inf")):
            final_status = "stopped_patience"
            print(f"\n[PATIENCE STOP] No improvement for {patience} steps "
                  f"(best @ step {best_step})")
            break
    else:
        if stopping_mode == "retrospective":
            final_status = "completed_retrospective"
        else:
            final_status = "max_steps_reached"

    convergence_info = {
        "final_status": final_status,
        "actual_steps": actual_steps,
        "best_step": best_step,
        "best_rmse": best_rmse,
        "stopping_mode": stopping_mode,
        "patience": patience,
        "rmse_records": rmse_records,
    }
    print(f"  [{final_status}] actual_steps={actual_steps}, best_step={best_step}, "
          f"best_rmse={best_rmse:.8f}")

    return step_results, xp_current, convergence_info


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_spot_similarity(
    vec_a: np.ndarray,
    gids_a: np.ndarray,
    vec_b: np.ndarray,
    gids_b: np.ndarray,
) -> Dict[str, float]:
    """Compute PCC and RMSE between two spots on their common gene space."""
    set_a = {int(g): i for i, g in enumerate(gids_a) if int(g) >= 0}
    set_b = {int(g): i for i, g in enumerate(gids_b) if int(g) >= 0}
    common = sorted(set(set_a.keys()) & set(set_b.keys()))

    if len(common) < 10:
        return {
            "pearson": float("nan"),
            "rmse": float("nan"),
            "n_common_genes": len(common),
        }

    a = np.array([vec_a[set_a[g]] for g in common], dtype=np.float32)
    b = np.array([vec_b[set_b[g]] for g in common], dtype=np.float32)

    diff = a - b
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    a_c, b_c = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a_c), np.linalg.norm(b_c)
    pearson = float(np.dot(a_c, b_c) / (na * nb)) if na > 1e-8 and nb > 1e-8 else float("nan")

    return {
        "pearson": pearson,
        "rmse": rmse,
        "n_common_genes": len(common),
    }


def compute_step_convergence(
    step_results: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    center_idx: int,
) -> List[Dict[str, float]]:
    """Compute RMSE between adjacent steps for center spot expression."""
    sorted_steps = sorted(step_results.keys())
    convergence = []
    for i in range(1, len(sorted_steps)):
        t_prev, t_curr = sorted_steps[i - 1], sorted_steps[i]
        if center_idx not in step_results[t_prev] or center_idx not in step_results[t_curr]:
            continue
        prev = step_results[t_prev][center_idx]["raw_normed_values"]
        curr = step_results[t_curr][center_idx]["raw_normed_values"]
        n = min(len(prev), len(curr))
        diff = prev[:n] - curr[:n]
        convergence.append({
            "step_from": t_prev,
            "step_to": t_curr,
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
        })
    return convergence


def save_step_expression(
    step_results: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    center_idx: int,
    out_dir: Path,
    vocab_path: str,
):
    """Save per-step center spot expression to npz files."""
    step_dir = out_dir / "step_expressions"
    step_dir.mkdir(parents=True, exist_ok=True)

    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_gene = {int(v): k for k, v in vocab.items()}

    for step, data in step_results.items():
        if center_idx not in data:
            continue
        gids = data[center_idx]["gene_ids"]
        vals = data[center_idx]["raw_normed_values"]
        gene_names = [id_to_gene.get(int(g), f"gene_{g}") for g in gids]
        np.savez_compressed(
            step_dir / f"step{step}.npz",
            gene_ids=gids,
            expression=vals,
            gene_names=np.array(gene_names, dtype=object),
        )


# -----------------------------------------------------------------------------
# Patch existing metrics with missing reference type
# -----------------------------------------------------------------------------

def patch_metrics(
    existing: Dict[str, Any],
    exp_out: Path,
    databank: SpatialDataBank,
    recon1_cache: Optional[Dict[int, Dict[str, np.ndarray]]],
) -> Dict[str, Any]:
    """
    Supplement an existing metrics.json with whichever reference type is missing
    (vs_recon or vs_raw). Reads step expressions from disk to recompute.
    """
    center_idx = existing["center_idx"]
    target_center_idx = existing["target_center_idx"]

    has_vs_recon = ("metrics_vs_recon" in existing
                    and existing["metrics_vs_recon"] is not None)
    has_vs_raw = ("metrics_vs_raw" in existing
                  and existing["metrics_vs_raw"] is not None)

    if has_vs_recon and has_vs_raw:
        return existing

    # Load step expressions from saved npz files
    step_dir = exp_out / "step_expressions"
    step_data: Dict[int, Dict[str, np.ndarray]] = {}
    if step_dir.exists():
        for f in sorted(step_dir.glob("step*.npz")):
            step_num = int(f.stem.replace("step", ""))
            data = np.load(str(f), allow_pickle=True)
            step_data[step_num] = {
                "gene_ids": data["gene_ids"],
                "raw_normed_values": data["expression"],
            }

    # --- Supplement vs_raw ---
    if not has_vs_raw:
        raw_center = databank.get_spot_data(center_idx)
        raw_orig_gids = np.asarray(raw_center["gene_ids"], dtype=np.int64)
        raw_orig_vals = np.asarray(raw_center["raw_normed_values"], dtype=np.float32)
        raw_target = databank.get_spot_data(target_center_idx)
        raw_target_gids = np.asarray(raw_target["gene_ids"], dtype=np.int64)
        raw_target_vals = np.asarray(raw_target["raw_normed_values"], dtype=np.float32)

        baseline = compute_spot_similarity(
            raw_orig_vals, raw_orig_gids, raw_target_vals, raw_target_gids)
        baseline["step"] = 0

        step_metrics = []
        for step in sorted(step_data.keys()):
            pred = step_data[step]
            sim = compute_spot_similarity(
                pred["raw_normed_values"], pred["gene_ids"],
                raw_target_vals, raw_target_gids,
            )
            sim["step"] = step
            step_metrics.append(sim)

        existing["metrics_vs_raw"] = [baseline] + step_metrics

    # --- Supplement vs_recon ---
    if not has_vs_recon:
        can_recon = (recon1_cache is not None
                     and center_idx in recon1_cache
                     and target_center_idx in recon1_cache)
        if can_recon:
            recon_orig_gids = recon1_cache[center_idx]["gene_ids"]
            recon_orig_vals = recon1_cache[center_idx]["raw_normed_values"]
            recon_target_gids = recon1_cache[target_center_idx]["gene_ids"]
            recon_target_vals = recon1_cache[target_center_idx]["raw_normed_values"]

            baseline = compute_spot_similarity(
                recon_orig_vals, recon_orig_gids,
                recon_target_vals, recon_target_gids)
            baseline["step"] = 0

            step_metrics = []
            for step in sorted(step_data.keys()):
                pred = step_data[step]
                sim = compute_spot_similarity(
                    pred["raw_normed_values"], pred["gene_ids"],
                    recon_target_vals, recon_target_gids,
                )
                sim["step"] = step
                step_metrics.append(sim)

            existing["metrics_vs_recon"] = [baseline] + step_metrics
        elif "metrics" in existing and not has_vs_recon:
            existing["metrics_vs_recon"] = existing["metrics"]

    # Keep backward-compat "metrics" field
    if "metrics_vs_recon" in existing and existing["metrics_vs_recon"]:
        existing["metrics"] = existing["metrics_vs_recon"]

    with open(exp_out / "metrics.json", "w") as f:
        json.dump(existing, f, indent=2, default=str)

    return existing


# -----------------------------------------------------------------------------
# Single perturbation experiment
# -----------------------------------------------------------------------------

def run_single_perturbation(
    databank: SpatialDataBank,
    model: nn.Module,
    device: torch.device,
    config: Config,
    center_idx: int,
    target_center_idx: int,
    source_neighbors: List[int],
    target_neighbors: List[int],
    k: int,
    alpha: float,
    hop: int,
    max_steps: int,
    seed: int,
    out_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    recon1_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    x_line_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    use_recon_as_x0: bool = True,
    stopping_mode: str = "patience",
    patience: int = 3,
    emergency_div_factor: float = 3.0,
) -> Dict[str, Any]:
    """
    Run a single perturbation experiment with patience-based early stopping.
    Metrics are computed against BOTH recon1 expression (metrics_vs_recon)
    and raw ground-truth expression (metrics_vs_raw).

    Two dual-line bases (Algorithm 1):
      use_recon_as_x0=True (default): X_0 = recon1 = M(X). Perturbation is
        applied in the denoised space (interpolate on recon1 source/target
        neighbors), and the X line is X_hat = M(X_0) supplied via x_line_cache
        (recon2). The unperturbed control is therefore the model's own fixed
        point, removing the raw reconstruction residual from the comparison.
      use_recon_as_x0=False (legacy raw-input mode): X_0 = raw full-slice expression, the X
        line is recon1, perturbation is applied on raw neighbor expressions.

    Returns metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    databank.clear_runtime_spot_overrides()

    raw_center = databank.get_spot_data(center_idx)
    raw_orig_gids = np.asarray(raw_center["gene_ids"], dtype=np.int64)
    raw_orig_vals = np.asarray(raw_center["raw_normed_values"], dtype=np.float32)
    raw_target = databank.get_spot_data(target_center_idx)
    raw_target_gids = np.asarray(raw_target["gene_ids"], dtype=np.int64)
    raw_target_vals = np.asarray(raw_target["raw_normed_values"], dtype=np.float32)

    has_recon = (recon1_cache is not None
                 and center_idx in recon1_cache
                 and target_center_idx in recon1_cache)
    if has_recon:
        recon_orig_gids = recon1_cache[center_idx]["gene_ids"]
        recon_orig_vals = recon1_cache[center_idx]["raw_normed_values"]
        recon_target_gids = recon1_cache[target_center_idx]["gene_ids"]
        recon_target_vals = recon1_cache[target_center_idx]["raw_normed_values"]

    if recon1_cache is None:
        raise ValueError("recon1_cache is required for full-state dual-line inference.")

    # ---------------------------------------------------------------------
    # Dual-line base state selection (Algorithm 1):
    #   use_recon_as_x0=True  -> X_0 = recon1 = M(X)  (denoised base, default)
    #   use_recon_as_x0=False -> X_0 = raw           (legacy raw-input mode)
    #
    # The X line (fixed control prediction X_hat = M(X_0)) is supplied via
    # x_line_cache: recon2 = M(recon1) in recon mode, recon1 = M(raw) in raw
    # mode. Perturbation overrides are interpolated in the SAME space as X_0,
    # so X'_0 = P(X_0) is consistent and the residual cancels in X'_0 + (M(X')-X_hat).
    # ---------------------------------------------------------------------
    value_source = recon1_cache if use_recon_as_x0 else None

    perturb_overrides = create_interpolated_perturbation(
        databank, source_neighbors, target_neighbors, k, alpha, seed,
        value_source=value_source,
    )
    frozen_indices = set(perturb_overrides.keys())

    x0_state: Dict[int, Dict[str, np.ndarray]] = {}
    if use_recon_as_x0:
        # X_0 = recon1 = M(X): copy the full-slice denoised state.
        for gidx, ov in recon1_cache.items():
            x0_state[int(gidx)] = {
                "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
                "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
            }
    else:
        # X_0 = raw full-slice expression (legacy raw-input mode).
        for gidx in sorted(recon1_cache.keys()):
            raw_spot = databank.get_spot_data(int(gidx))
            x0_state[int(gidx)] = {
                "gene_ids": np.asarray(raw_spot["gene_ids"], dtype=np.int64).copy(),
                "raw_normed_values": np.asarray(raw_spot["raw_normed_values"], dtype=np.float32).copy(),
            }

    # X'_0 = P(X_0): copy X_0 and override the perturbed (frozen) neighbors.
    x0_prime_state: Dict[int, Dict[str, np.ndarray]] = {
        int(gidx): {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }
        for gidx, ov in x0_state.items()
    }
    for gidx, ov in perturb_overrides.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32).copy(),
        }

    # X line (fixed control): X_hat = M(X_0). Defaults to recon1 when no
    # separate cache is supplied (i.e. raw-input legacy mode).
    precomputed_x_line = x_line_cache if x_line_cache is not None else recon1_cache

    capture_set = {center_idx, target_center_idx}
    capture_set.update(source_neighbors)

    def loader_factory():
        return build_loader(databank, config, batch_size, num_workers)

    step_results, final_state, convergence_info = run_dual_line_inference(
        databank, loader_factory, model, device,
        max_steps=max_steps,
        frozen_indices=frozen_indices,
        x0_state=x0_state,
        x0_prime_state=x0_prime_state,
        capture_indices=capture_set,
        precomputed_x_line=precomputed_x_line,
        stopping_mode=stopping_mode,
        patience=patience,
        emergency_div_factor=emergency_div_factor,
    )

    save_step_expression(step_results, center_idx, out_dir, config.vocab_file)

    convergence = compute_step_convergence(step_results, center_idx)

    step_metrics_vs_raw = []
    step_metrics_vs_recon = []
    for step in sorted(step_results.keys()):
        if center_idx not in step_results[step]:
            continue
        pred = step_results[step][center_idx]

        sim_raw = compute_spot_similarity(
            pred["raw_normed_values"], pred["gene_ids"],
            raw_target_vals, raw_target_gids,
        )
        sim_raw["step"] = step
        step_metrics_vs_raw.append(sim_raw)

        if has_recon:
            sim_recon = compute_spot_similarity(
                pred["raw_normed_values"], pred["gene_ids"],
                recon_target_vals, recon_target_gids,
            )
            sim_recon["step"] = step
            step_metrics_vs_recon.append(sim_recon)

    baseline_raw = compute_spot_similarity(
        raw_orig_vals, raw_orig_gids, raw_target_vals, raw_target_gids)
    baseline_raw["step"] = 0
    metrics_vs_raw = [baseline_raw] + step_metrics_vs_raw

    if has_recon:
        baseline_recon = compute_spot_similarity(
            recon_orig_vals, recon_orig_gids, recon_target_vals, recon_target_gids)
        baseline_recon["step"] = 0
        metrics_vs_recon = [baseline_recon] + step_metrics_vs_recon
    else:
        metrics_vs_recon = None

    result = {
        "center_idx": center_idx,
        "target_center_idx": target_center_idx,
        "k": k,
        "alpha": alpha,
        "hop": hop,
        "max_steps": max_steps,
        "actual_steps": convergence_info["actual_steps"],
        "best_step": convergence_info["best_step"],
        "stopping_mode": stopping_mode,
        "final_status": convergence_info["final_status"],
        "seed": seed,
        "x0_source": "recon" if use_recon_as_x0 else "raw",
        "n_perturbed_neighbors": len(frozen_indices),
        "n_source_neighbors": len(source_neighbors),
        "n_target_neighbors": len(target_neighbors),
        "metrics_vs_recon": metrics_vs_recon,
        "metrics_vs_raw": metrics_vs_raw,
        "metrics": metrics_vs_recon if metrics_vs_recon else metrics_vs_raw,
        "convergence": convergence,
        "convergence_info": convergence_info,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    conv_report = {
        "final_status": convergence_info["final_status"],
        "best_step": convergence_info["best_step"],
        "best_rmse": convergence_info["best_rmse"],
        "actual_steps": convergence_info["actual_steps"],
        "stopping_mode": stopping_mode,
        "patience": patience,
        "rmse_records": convergence_info["rmse_records"],
        "center_spot_convergence": convergence,
    }
    with open(out_dir / "convergence_report.json", "w") as f:
        json.dump(conv_report, f, indent=2, default=str)

    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Microenvironment perturbation experiment")

    ap.add_argument("--dataset", required=True,
                    choices=["1142243F", "1160920F", "HBRC", "PDAC"])
    ap.add_argument("--ckpt_dir", required=True, help="Model checkpoint directory")
    ap.add_argument("--cache_dir", required=True, help="Preprocessed cache directory")
    ap.add_argument("--lmdb_path", default=None)
    ap.add_argument("--lmdb_manifest", default=None)
    ap.add_argument("--cache_mode", default="lmdb", choices=["h5", "lmdb"])
    ap.add_argument("--truth_csv", default=None,
                    help="External truth CSV for HBRC annotations")

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=5)
    ap.add_argument("--hops", type=str, default="1,2",
                    help="Comma-separated hop distances (e.g. '1,2')")
    ap.add_argument("--k_values_1hop", type=str, default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--k_values_2hop", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    ap.add_argument("--alpha_values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--max_steps", type=int, default=10,
                    help="Maximum dual-line iteration steps")
    ap.add_argument("--stopping_mode", default="patience",
                    choices=["patience", "retrospective"],
                    help="Stopping strategy: patience (early stop) or retrospective (run all)")
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop after N steps without RMSE improvement (patience mode)")
    ap.add_argument("--emergency_div_factor", type=float, default=3.0,
                    help="Emergency stop if step RMSE > factor * best RMSE")
    ap.add_argument("--x0_source", default="recon", choices=["recon", "raw"],
                    help="Dual-line base state X_0. 'recon' (default): "
                         "X_0=M(X), perturb in denoised space, X line = M(X_0). "
                         "'raw' (legacy): X_0=raw, X line = M(X)=recon1.")
    ap.add_argument("--seed", type=int, default=42)

    # Pair selection parameters
    ap.add_argument("--min_purity_healthy", type=float, default=0.5,
                    help="Minimum neighborhood purity for healthy center (0-1)")
    ap.add_argument("--min_purity_cancer", type=float, default=0.5,
                    help="Minimum neighborhood purity for cancer center (0-1)")
    ap.add_argument("--min_neighbors", type=int, default=4,
                    help="Minimum 1-hop neighbor count for a spot to be considered")
    ap.add_argument("--exclusion_hops", type=int, default=4,
                    help="Paired centers must be >N hops apart (4 = safe for 2-hop perturbation)")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--directions", type=str,
                    default="cancer_to_healthy,healthy_to_cancer",
                    help="Comma-separated directions")
    ap.add_argument("--pair_idx", type=int, default=None,
                    help="Run only this pair index (0-based). None = run all.")

    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    hops = [int(h) for h in args.hops.split(",")]
    k_values_1hop = [int(x) for x in args.k_values_1hop.split(",")]
    k_values_2hop = [int(x) for x in args.k_values_2hop.split(",")]
    alpha_values = [float(x) for x in args.alpha_values.split(",")]
    directions = [d.strip() for d in args.directions.split(",")]

    config = Config()
    config.device = str(device)

    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Checkpoint: {args.ckpt_dir}")
    print(f"[INFO] Cache: {args.cache_dir} (mode={args.cache_mode})")
    print(f"[INFO] Pairs: {args.n_pairs}, Hops: {hops}, Max steps: {args.max_steps}")
    print(f"[INFO] Stopping: {args.stopping_mode}, patience={args.patience}, "
          f"emergency_div={args.emergency_div_factor}")
    print(f"[INFO] Directions: {directions}")

    # Load model
    model = load_model(config, args.ckpt_dir, device)
    print(f"[OK] Model loaded from {args.ckpt_dir}")

    # Setup databank
    databank = setup_databank(
        config, args.cache_dir, args.dataset,
        lmdb_path=args.lmdb_path,
        lmdb_manifest=args.lmdb_manifest,
        cache_mode=args.cache_mode,
    )
    print(f"[OK] DataBank initialized")

    # Load annotations
    annotations = load_spot_annotations(
        args.dataset, args.cache_dir, truth_csv=args.truth_csv
    )
    n_cancer = sum(1 for v in annotations.values() if v == "cancer")
    n_healthy = sum(1 for v in annotations.values() if v == "healthy")
    print(f"[OK] Annotations: {len(annotations)} spots, {n_cancer} cancer, {n_healthy} healthy")

    # Select pairs
    pairs = select_spot_pairs(
        databank, annotations, args.n_pairs, args.seed,
        min_healthy_purity=args.min_purity_healthy,
        min_cancer_purity=args.min_purity_cancer,
        min_neighbors=args.min_neighbors,
        exclusion_hops=args.exclusion_hops,
    )
    print(f"[OK] Selected {len(pairs)} spot pairs")

    # Save pairs manifest
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "pairs_manifest.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "seed": args.seed,
            "n_pairs": len(pairs),
            "pairs": pairs,
        }, f, indent=2)

    # Determine which pairs to run
    if args.pair_idx is not None:
        pair_range = [args.pair_idx]
    else:
        pair_range = list(range(len(pairs)))

    # Compute or load recon1 cache (single-pass reconstruction = M(X) = X_0).
    recon1_cache_path = out_base / f"recon1_cache_{args.dataset}.npz"
    recon1_cache = get_or_compute_recon1(
        databank, model, device, config,
        cache_path=recon1_cache_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Dual-line base selection (Algorithm 1).
    #   recon mode (default): X_0 = recon1 = M(X), X line = recon2 = M(X_0).
    #   raw mode (legacy):     X_0 = raw,           X line = recon1 = M(X).
    use_recon_as_x0 = (args.x0_source == "recon")
    if use_recon_as_x0:
        recon2_cache_path = out_base / f"recon2_cache_{args.dataset}.npz"
        x_line_cache = get_or_compute_recon2(
            databank, model, device, config,
            recon1_cache=recon1_cache,
            cache_path=recon2_cache_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"[INFO] x0_source=recon: X_0=M(X), X line=M(X_0) (recon2)")
    else:
        x_line_cache = recon1_cache
        print(f"[INFO] x0_source=raw: X_0=raw, X line=M(X) (recon1) [legacy]")

    # Count total experiments
    total_exps = 0
    for h in hops:
        k_vals = k_values_1hop if h == 1 else k_values_2hop
        total_exps += len(pair_range) * len(directions) * len(k_vals) * len(alpha_values)
    print(f"[INFO] Total experiments to run: {total_exps}")

    all_results = []
    exp_count = 0

    for pi in pair_range:
        pair = pairs[pi]
        healthy_center = pair["healthy_center"]
        cancer_center = pair["cancer_center"]

        for direction in directions:
            if direction == "cancer_to_healthy":
                center_idx = cancer_center
                target_idx = healthy_center
            else:
                center_idx = healthy_center
                target_idx = cancer_center

            for hop in hops:
                # Get neighbors for source center
                hop1, hop2 = get_hop_neighbors(databank, center_idx, hop)
                source_neighbors = hop1 if hop == 1 else hop1 + hop2

                # Get neighbors for target center (as template)
                t_hop1, t_hop2 = get_hop_neighbors(databank, target_idx, hop)
                target_neighbors = t_hop1 if hop == 1 else t_hop1 + t_hop2

                k_vals = k_values_1hop if hop == 1 else k_values_2hop

                for k in k_vals:
                    if k > len(source_neighbors):
                        continue

                    for alpha in alpha_values:
                        exp_count += 1
                        exp_tag = (f"pair{pi}/{direction}/hop{hop}_k{k}_"
                                   f"alpha{alpha:.1f}")
                        exp_out = out_base / args.dataset / direction / f"pair{pi}" / f"hop{hop}_k{k}_alpha{alpha:.1f}"

                        if (exp_out / "metrics.json").exists():
                            with open(exp_out / "metrics.json") as f:
                                existing = json.load(f)
                            _has_recon = ("metrics_vs_recon" in existing
                                          and existing["metrics_vs_recon"] is not None)
                            _has_raw = ("metrics_vs_raw" in existing
                                        and existing["metrics_vs_raw"] is not None)
                            if _has_recon and _has_raw:
                                print(f"[SKIP] {exp_tag} (complete)")
                                all_results.append(existing)
                                continue
                            print(f"[PATCH] {exp_tag} - supplementing "
                                  f"{'vs_raw' if not _has_raw else 'vs_recon'}")
                            existing = patch_metrics(
                                existing, exp_out, databank, recon1_cache)
                            all_results.append(existing)
                            continue

                        print(f"\n[{exp_count}/{total_exps}] {exp_tag}")

                        result = run_single_perturbation(
                            databank=databank,
                            model=model,
                            device=device,
                            config=config,
                            center_idx=center_idx,
                            target_center_idx=target_idx,
                            source_neighbors=source_neighbors,
                            target_neighbors=target_neighbors,
                            k=k,
                            alpha=alpha,
                            hop=hop,
                            max_steps=args.max_steps,
                            seed=args.seed,
                            out_dir=exp_out,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            recon1_cache=recon1_cache,
                            x_line_cache=x_line_cache,
                            use_recon_as_x0=use_recon_as_x0,
                            stopping_mode=args.stopping_mode,
                            patience=args.patience,
                            emergency_div_factor=args.emergency_div_factor,
                        )

                        result["pair_idx"] = pi
                        result["direction"] = direction
                        result["dataset"] = args.dataset
                        all_results.append(result)

                        # Save manifest for this experiment
                        with open(exp_out / "manifest.json", "w") as f:
                            json.dump({
                                "dataset": args.dataset,
                                "direction": direction,
                                "pair_idx": pi,
                                "hop": hop, "k": k, "alpha": alpha,
                                "center_idx": center_idx,
                                "target_center_idx": target_idx,
                                "n_source_neighbors": len(source_neighbors),
                                "n_target_neighbors": len(target_neighbors),
                                "ckpt_dir": args.ckpt_dir,
                                "seed": args.seed,
                                "max_steps": args.max_steps,
                                "stopping_mode": args.stopping_mode,
                                "patience": args.patience,
                                "x0_source": args.x0_source,
                            }, f, indent=2)

    # Save combined results
    summary_path = out_base / f"summary_{args.dataset}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Saved summary to {summary_path}")

    # Build summary CSV for dose-response curves (both reference types)
    rows = []
    for r in all_results:
        base_info = {
            "dataset": r.get("dataset", args.dataset),
            "direction": r.get("direction", ""),
            "pair_idx": r.get("pair_idx", -1),
            "hop": r.get("hop", -1),
            "k": r.get("k", -1),
            "alpha": r.get("alpha", -1),
        }
        for ref_key, ref_label in [("metrics_vs_recon", "recon"),
                                    ("metrics_vs_raw", "raw")]:
            for m in r.get(ref_key, []) or []:
                row = dict(base_info)
                row["reference"] = ref_label
                row["step"] = m.get("step", -1)
                row["pearson"] = m.get("pearson")
                row["rmse"] = m.get("rmse")
                row["n_common_genes"] = m.get("n_common_genes")
                rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_base / f"dose_response_{args.dataset}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved dose-response CSV to {csv_path}")


if __name__ == "__main__":
    main()
