#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN/Ridge baseline for cancer/healthy microenvironment perturbation.

Uses the same experimental design as microenv_perturbation.py (SpatialGT version):
  - Select N paired spots (healthy center + cancer center)
  - Linearly interpolate k neighbors toward target niche
  - Run dual-line inference with error cancellation

    Inference is performed via GPU KNN average or GPU Ridge regression instead of SpatialGT.
Output format is identical to the SpatialGT version for unified comparison.
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
import torch.nn.functional as F
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
for _path in (_REPO_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pretrain.Config import Config
from pretrain.spatial_databank import SpatialDataBank

from microenv_perturbation import (
    load_spot_annotations,
    get_hop_neighbors,
    select_spot_pairs,
    create_interpolated_perturbation,
    compute_spot_similarity,
    compute_step_convergence,
    save_step_expression,
    setup_databank,
    build_loader,
)


# -----------------------------------------------------------------------------
# Spot expression helpers
# -----------------------------------------------------------------------------

def get_spot_expression(
    databank: SpatialDataBank,
    global_idx: int,
    current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if current_state is not None and int(global_idx) in current_state:
        ov = current_state[int(global_idx)]
        return np.asarray(ov["gene_ids"], dtype=np.int64), np.asarray(ov["raw_normed_values"], dtype=np.float32)
    sd = databank.get_spot_data(int(global_idx))
    return np.asarray(sd["gene_ids"], dtype=np.int64), np.asarray(sd["raw_normed_values"], dtype=np.float32)


# -----------------------------------------------------------------------------
# KNN average prediction
# -----------------------------------------------------------------------------

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

    predicted_common = neighbor_matrix.mean(axis=0)
    predicted_vals = center_vals.copy()
    for k, cg in enumerate(common_genes_arr):
        if int(cg) in common_to_center_idx:
            predicted_vals[common_to_center_idx[int(cg)]] = predicted_common[k]
    return {"gene_ids": center_gids, "raw_normed_values": predicted_vals}


# -----------------------------------------------------------------------------
# GPU Ridge Regression
# -----------------------------------------------------------------------------

class GPURidgeRegressionModel:
    """GPU-accelerated Ridge regression: predict center expression from neighbors."""

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
        self.neighbor_table: Optional[torch.Tensor] = None
        self.neighbor_mask: Optional[torch.Tensor] = None
        self.global_to_local: Dict[int, int] = {}
        self.local_to_global: Dict[int, int] = {}

    def train(
        self,
        databank: SpatialDataBank,
        all_global_indices: List[int],
        current_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        print(f"[GPU Ridge] Training on {len(all_global_indices)} spots on {self.device}...")
        all_gene_ids: Set[int] = set()
        spot_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for idx in tqdm(all_global_indices, desc="Collecting vocabulary"):
            gids, vals = get_spot_expression(databank, idx, current_state)
            spot_cache[idx] = (gids, vals)
            all_gene_ids.update(int(g) for g in gids if int(g) >= 0)

        sorted_gids = sorted(all_gene_ids)
        gene_id_to_idx = {gid: i for i, gid in enumerate(sorted_gids)}
        n_genes = len(sorted_gids)
        print(f"[GPU Ridge] {n_genes} unique genes")

        X_list, Y_list, mask_list = [], [], []
        for center_idx in tqdm(all_global_indices, desc="Building training data"):
            center_gids, center_vals = spot_cache[center_idx]
            try:
                neighbors = databank.get_neighbors_for_spot(int(center_idx))
            except Exception:
                continue
            if len(neighbors) < 2:
                continue
            neighbors = neighbors[:self.max_neighbors]

            nb_feat = np.zeros((self.max_neighbors, n_genes), dtype=np.float32)
            for ni, nb_idx in enumerate(neighbors):
                nb_idx = int(nb_idx)
                if nb_idx not in spot_cache:
                    gids, vals = get_spot_expression(databank, nb_idx, current_state)
                    spot_cache[nb_idx] = (gids, vals)
                else:
                    gids, vals = spot_cache[nb_idx]
                for j, gid in enumerate(gids):
                    gid_int = int(gid)
                    if gid_int >= 0 and gid_int in gene_id_to_idx:
                        nb_feat[ni, gene_id_to_idx[gid_int]] = vals[j]

            target = np.zeros(n_genes, dtype=np.float32)
            gmask = np.zeros(n_genes, dtype=np.float32)
            for j, gid in enumerate(center_gids):
                gid_int = int(gid)
                if gid_int >= 0 and gid_int in gene_id_to_idx:
                    target[gene_id_to_idx[gid_int]] = center_vals[j]
                    gmask[gene_id_to_idx[gid_int]] = 1.0
            X_list.append(nb_feat)
            Y_list.append(target)
            mask_list.append(gmask)

        X = torch.from_numpy(np.stack(X_list)).to(self.device)
        Y = torch.from_numpy(np.stack(Y_list)).to(self.device)
        mask = torch.from_numpy(np.stack(mask_list)).to(self.device)

        n_samples, max_nb, _ = X.shape
        self.gene_id_to_idx = gene_id_to_idx
        self.idx_to_gene_id = {v: k for k, v in gene_id_to_idx.items()}
        self.n_genes = n_genes

        self.weights = torch.zeros(n_genes, max_nb, device=self.device)
        self.bias = torch.zeros(n_genes, device=self.device)
        reg_I = self.alpha * torch.eye(max_nb, device=self.device)

        n_trained = 0
        batch_sz = 512
        for start_g in tqdm(range(0, n_genes, batch_sz), desc="Training Ridge (GPU)"):
            for g in range(start_g, min(start_g + batch_sz, n_genes)):
                valid = mask[:, g] > 0.5
                n_valid = valid.sum().item()
                if n_valid < min_samples:
                    continue
                X_v = X[valid, :, g]
                y_v = Y[valid, g]
                try:
                    w = torch.linalg.solve(X_v.T @ X_v + reg_I, X_v.T @ y_v)
                    b = y_v.mean() - torch.dot(w, X_v.mean(dim=0))
                    self.weights[g] = w
                    self.bias[g] = b
                    n_trained += 1
                except Exception:
                    pass

        self.is_trained = True
        print(f"[GPU Ridge] Trained {n_trained}/{n_genes} gene models")
        return {"n_trained": n_trained, "n_genes": n_genes, "n_samples": n_samples}

    def build_neighbor_table(self, databank: SpatialDataBank, all_global_indices: List[int]):
        n_spots = len(all_global_indices)
        g2l = {g: i for i, g in enumerate(all_global_indices)}
        l2g = {i: g for i, g in enumerate(all_global_indices)}

        nb_table = np.full((n_spots, self.max_neighbors), -1, dtype=np.int64)
        nb_mask = np.zeros((n_spots, self.max_neighbors), dtype=np.float32)

        for i in range(n_spots):
            try:
                neighbors = databank.get_neighbors_for_spot(int(l2g[i]))
            except Exception:
                continue
            for ni, nb_g in enumerate(neighbors[:self.max_neighbors]):
                nb_g = int(nb_g)
                if nb_g in g2l:
                    nb_table[i, ni] = g2l[nb_g]
                    nb_mask[i, ni] = 1.0

        self.neighbor_table = torch.from_numpy(nb_table).to(self.device)
        self.neighbor_mask = torch.from_numpy(nb_mask).to(self.device)
        self.global_to_local = g2l
        self.local_to_global = l2g

    def predict_batch(self, expr_matrix: torch.Tensor) -> torch.Tensor:
        if not self.is_trained:
            return expr_matrix
        nb_idx = self.neighbor_table.clamp(min=0)
        nb_expr = expr_matrix[nb_idx]
        nb_expr = nb_expr * self.neighbor_mask.unsqueeze(-1)
        W_t = self.weights.T  # [max_nb, n_genes]
        preds = torch.einsum('bkg,kg->bg', nb_expr, W_t) + self.bias.unsqueeze(0)
        return F.relu(preds)


# -----------------------------------------------------------------------------
# FastRidgeEngine - GPU-accelerated dense-space dual-line inference
# -----------------------------------------------------------------------------

class FastRidgeEngine:
    """
    Pre-loads all spot data into dense GPU tensors at init.
    The dual-line loop runs entirely as GPU tensor operations,
    eliminating ~100M Python loop iterations per experiment.

    Typical speedup: 100-500x over the dict-based approach.
    """

    def __init__(
        self,
        ridge_model: GPURidgeRegressionModel,
        databank: SpatialDataBank,
        all_global_indices: List[int],
    ):
        self.ridge = ridge_model
        self.n_spots = len(all_global_indices)
        self.n_genes = ridge_model.n_genes
        self.device = ridge_model.device
        self.all_global_indices = all_global_indices
        self.g2l = {int(g): i for i, g in enumerate(all_global_indices)}

        self.spot_gids: List[np.ndarray] = []
        self.spot_valid_pos: List[np.ndarray] = []
        self.spot_dense_col: List[np.ndarray] = []

        self.x0_dense = torch.zeros(
            self.n_spots, self.n_genes, device=self.device)

        t0 = time.time()
        gid_to_idx = ridge_model.gene_id_to_idx
        for i, gidx in enumerate(
                tqdm(all_global_indices, desc="[FastEngine] Pre-loading")):
            sd = databank.get_spot_data(int(gidx))
            gids = np.asarray(sd["gene_ids"], dtype=np.int64)
            vals = np.asarray(sd["raw_normed_values"], dtype=np.float32)
            self.spot_gids.append(gids)

            valid_pos, dense_col = [], []
            for j, g in enumerate(gids):
                c = gid_to_idx.get(int(g), -1)
                if c >= 0:
                    valid_pos.append(j)
                    dense_col.append(c)
            vp = np.array(valid_pos, dtype=np.int64)
            dc = np.array(dense_col, dtype=np.int64)
            self.spot_valid_pos.append(vp)
            self.spot_dense_col.append(dc)

            if len(dc) > 0:
                self.x0_dense[i, dc] = torch.from_numpy(
                    vals[vp]).to(self.device)

        if (ridge_model.neighbor_table is None
                or ridge_model.neighbor_table.shape[0] != self.n_spots):
            ridge_model.build_neighbor_table(databank, all_global_indices)

        self.x_line_dense = ridge_model.predict_batch(self.x0_dense)

        elapsed = time.time() - t0
        mem_mb = (self.x0_dense.element_size() * self.x0_dense.nelement()
                  * 3 / 1024 / 1024)
        print(f"[FastEngine] Ready in {elapsed:.1f}s. "
              f"[{self.n_spots}, {self.n_genes}], ~{mem_mb:.0f} MB GPU")

    def run_dual_line(
        self,
        perturb_overrides: Dict[int, Dict[str, np.ndarray]],
        frozen_indices: Set[int],
        capture_indices: Set[int],
        steps: int,
    ) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
        """Dense-space dual-line loop. Returns step_results for capture_indices."""
        x0p_dense = self.x0_dense.clone()
        for gidx, ov in perturb_overrides.items():
            li = self.g2l.get(int(gidx))
            if li is None:
                continue
            vals = np.asarray(ov["raw_normed_values"], dtype=np.float32)
            vp = self.spot_valid_pos[li]
            dc = self.spot_dense_col[li]
            x0p_dense[li] = 0.0
            if len(dc) > 0:
                x0p_dense[li, dc] = torch.from_numpy(
                    vals[vp]).to(self.device)

        frozen_local = [self.g2l[int(g)] for g in frozen_indices
                        if int(g) in self.g2l]
        frozen_mask = torch.zeros(
            self.n_spots, dtype=torch.bool, device=self.device)
        if frozen_local:
            frozen_mask[torch.tensor(frozen_local, device=self.device)] = True

        xp_current = x0p_dense.clone()
        step_results: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

        for t in range(1, steps + 1):
            xp_k_raw = self.ridge.predict_batch(xp_current)
            delta = xp_k_raw - self.x_line_dense
            xp_k_true = x0p_dense + delta
            xp_k_true[frozen_mask] = x0p_dense[frozen_mask]

            step_results[t] = self._extract_spots(xp_k_true, capture_indices)
            xp_current = xp_k_true

        return step_results

    def _extract_spots(
        self,
        dense: torch.Tensor,
        indices: Set[int],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Convert dense GPU rows -> sparse dict for specific global indices."""
        result = {}
        dense_cpu = dense.cpu()
        for gidx in indices:
            li = self.g2l.get(int(gidx))
            if li is None:
                continue
            gids = self.spot_gids[li]
            vp = self.spot_valid_pos[li]
            dc = self.spot_dense_col[li]
            vals = np.zeros(len(gids), dtype=np.float32)
            if len(dc) > 0:
                vals[vp] = dense_cpu[li, dc].numpy()
            result[int(gidx)] = {
                "gene_ids": gids.copy(), "raw_normed_values": vals}
        return result



# -----------------------------------------------------------------------------
# FastKNNAverageEngine - GPU-accelerated dense-space KNN average inference
# -----------------------------------------------------------------------------

class FastKNNAverageEngine:
    """
    Pre-loads all spot data and neighbor indices into dense GPU tensors.
    This preserves KNN-average semantics while avoiding Python dict loops during
    every dual-line perturbation step.
    """

    def __init__(
        self,
        databank: SpatialDataBank,
        all_global_indices: List[int],
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_spots = len(all_global_indices)
        self.all_global_indices = all_global_indices
        self.g2l = {int(g): i for i, g in enumerate(all_global_indices)}
        self.l2g = {i: int(g) for i, g in enumerate(all_global_indices)}

        t0 = time.time()
        spot_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        all_gene_ids: Set[int] = set()
        for gidx in tqdm(all_global_indices, desc="[FastKNN] Collecting vocabulary"):
            gids, vals = get_spot_expression(databank, int(gidx))
            spot_cache[int(gidx)] = (gids, vals)
            all_gene_ids.update(int(g) for g in gids if int(g) >= 0)

        sorted_gids = sorted(all_gene_ids)
        self.gene_id_to_idx = {gid: i for i, gid in enumerate(sorted_gids)}
        self.n_genes = len(sorted_gids)

        self.spot_gids: List[np.ndarray] = []
        self.spot_valid_pos: List[np.ndarray] = []
        self.spot_dense_col: List[np.ndarray] = []
        self.x0_dense = torch.zeros(
            self.n_spots, self.n_genes, device=self.device)
        self.valid_mask = torch.zeros(
            self.n_spots, self.n_genes, dtype=torch.bool, device=self.device)

        for i, gidx in enumerate(tqdm(all_global_indices, desc="[FastKNN] Pre-loading")):
            gids, vals = spot_cache[int(gidx)]
            self.spot_gids.append(gids)

            valid_pos, dense_col = [], []
            for j, gid in enumerate(gids):
                col = self.gene_id_to_idx.get(int(gid), -1)
                if col >= 0:
                    valid_pos.append(j)
                    dense_col.append(col)
            vp = np.array(valid_pos, dtype=np.int64)
            dc = np.array(dense_col, dtype=np.int64)
            self.spot_valid_pos.append(vp)
            self.spot_dense_col.append(dc)

            if len(dc) > 0:
                self.x0_dense[i, dc] = torch.from_numpy(vals[vp]).to(self.device)
                self.valid_mask[i, dc] = True

        self._build_neighbor_table(databank)
        self.x_line_dense = self.predict_batch(self.x0_dense)

        elapsed = time.time() - t0
        mem_mb = (self.x0_dense.element_size() * self.x0_dense.nelement()
                  * 3 / 1024 / 1024)
        print(f"[FastKNN] Ready in {elapsed:.1f}s on {self.device}. "
              f"[{self.n_spots}, {self.n_genes}], ~{mem_mb:.0f} MB GPU")

    def _build_neighbor_table(self, databank: SpatialDataBank):
        neighbor_lists: List[List[int]] = []
        max_neighbors = 0
        for gidx in self.all_global_indices:
            try:
                neighbors = databank.get_neighbors_for_spot(int(gidx))
            except Exception:
                neighbors = []
            local_neighbors = [self.g2l[int(nb)] for nb in neighbors
                               if int(nb) in self.g2l]
            neighbor_lists.append(local_neighbors)
            max_neighbors = max(max_neighbors, len(local_neighbors))

        max_neighbors = max(max_neighbors, 1)
        nb_table = np.full((self.n_spots, max_neighbors), -1, dtype=np.int64)
        nb_mask = np.zeros((self.n_spots, max_neighbors), dtype=np.float32)
        for i, neighbors in enumerate(neighbor_lists):
            for j, li in enumerate(neighbors):
                nb_table[i, j] = li
                nb_mask[i, j] = 1.0

        self.neighbor_table = torch.from_numpy(nb_table).to(self.device)
        self.neighbor_mask = torch.from_numpy(nb_mask).to(self.device)

    def predict_batch(self, expr_matrix: torch.Tensor) -> torch.Tensor:
        nb_idx = self.neighbor_table.clamp(min=0)
        nb_expr = expr_matrix[nb_idx]
        nb_valid = self.valid_mask[nb_idx]
        nb_present = nb_valid & (self.neighbor_mask.unsqueeze(-1) > 0.5)

        counts = nb_present.sum(dim=1)
        sums = (nb_expr * nb_present.float()).sum(dim=1)
        avg = sums / counts.clamp(min=1).float()

        n_neighbors = self.neighbor_mask.sum(dim=1).long().unsqueeze(-1)
        common_mask = self.valid_mask & (n_neighbors > 0) & (counts == n_neighbors)
        return torch.where(common_mask, avg, expr_matrix)

    def run_dual_line(
        self,
        perturb_overrides: Dict[int, Dict[str, np.ndarray]],
        frozen_indices: Set[int],
        capture_indices: Set[int],
        steps: int,
    ) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
        x0p_dense = self.x0_dense.clone()
        for gidx, ov in perturb_overrides.items():
            li = self.g2l.get(int(gidx))
            if li is None:
                continue
            vals = np.asarray(ov["raw_normed_values"], dtype=np.float32)
            vp = self.spot_valid_pos[li]
            dc = self.spot_dense_col[li]
            x0p_dense[li] = 0.0
            if len(dc) > 0:
                x0p_dense[li, dc] = torch.from_numpy(vals[vp]).to(self.device)

        frozen_local = [self.g2l[int(g)] for g in frozen_indices
                        if int(g) in self.g2l]
        frozen_mask = torch.zeros(
            self.n_spots, dtype=torch.bool, device=self.device)
        if frozen_local:
            frozen_mask[torch.tensor(frozen_local, device=self.device)] = True

        xp_current = x0p_dense.clone()
        step_results: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        for t in range(1, steps + 1):
            xp_k_raw = self.predict_batch(xp_current)
            delta = xp_k_raw - self.x_line_dense
            xp_k_true = x0p_dense + delta
            xp_k_true[frozen_mask] = x0p_dense[frozen_mask]

            step_results[t] = self._extract_spots(xp_k_true, capture_indices)
            xp_current = xp_k_true

        return step_results

    def _extract_spots(
        self,
        dense: torch.Tensor,
        indices: Set[int],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        result = {}
        dense_cpu = dense.cpu()
        for gidx in indices:
            li = self.g2l.get(int(gidx))
            if li is None:
                continue
            gids = self.spot_gids[li]
            vp = self.spot_valid_pos[li]
            dc = self.spot_dense_col[li]
            vals = np.zeros(len(gids), dtype=np.float32)
            if len(dc) > 0:
                vals[vp] = dense_cpu[li, dc].numpy()
            result[int(gidx)] = {
                "gene_ids": gids.copy(), "raw_normed_values": vals}
        return result



# -----------------------------------------------------------------------------
# Single-pass inference (KNN average fallback only)
# -----------------------------------------------------------------------------

def run_single_pass_inference(
    databank: SpatialDataBank,
    all_global_indices: List[int],
    infer_mode: str,
    current_state: Dict[int, Dict[str, np.ndarray]],
    ridge_model: Optional[GPURidgeRegressionModel] = None,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in tqdm(all_global_indices, desc=desc):
        pred = knn_average_predict(databank, gidx, current_state)
        results[int(gidx)] = pred
    return results


# -----------------------------------------------------------------------------
# Dual-line iterative inference (KNN average fallback only)
# -----------------------------------------------------------------------------

def run_knn_dual_line_inference(
    databank: SpatialDataBank,
    all_global_indices: List[int],
    steps: int,
    frozen_indices: Set[int],
    x0_state: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    ridge_model: Optional[GPURidgeRegressionModel],
    capture_indices: Optional[Set[int]] = None,
    precomputed_x_line: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], Dict[int, Dict[str, np.ndarray]]]:
    """Dual-line iterative inference with KNN average."""

    x0p_base = {
        int(g): {
            "gene_ids": np.asarray(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in x0_prime_state.items()
    }
    xp_current = {
        int(g): {
            "gene_ids": np.asarray(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.asarray(v["raw_normed_values"], dtype=np.float32).copy(),
        }
        for g, v in x0_prime_state.items()
    }

    if precomputed_x_line is not None:
        x_k = precomputed_x_line
    else:
        x_k = run_single_pass_inference(
            databank, all_global_indices, "knn_avg", x0_state,
            desc="X line (once)")

    step_results: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    for t in range(1, steps + 1):
        xp_k_raw = run_single_pass_inference(
            databank, all_global_indices, "knn_avg", xp_current,
            desc=f"X' step {t}")

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
                x_k_vals = (x_k[gidx]["raw_normed_values"]
                            if gidx in x_k else np.zeros_like(xp_raw_vals))
                delta = xp_raw_vals - x_k_vals
                gid_to_delta = {int(g): delta[i]
                                for i, g in enumerate(gids_pred)}

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

        if capture_indices is not None:
            step_results[t] = {
                int(k): {"gene_ids": v["gene_ids"].copy(),
                          "raw_normed_values": v["raw_normed_values"].copy()}
                for k, v in xp_k_true.items() if int(k) in capture_indices
            }
        else:
            step_results[t] = {
                int(k): {"gene_ids": v["gene_ids"].copy(),
                          "raw_normed_values": v["raw_normed_values"].copy()}
                for k, v in xp_k_true.items()
            }

        xp_current = xp_k_true
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return step_results, xp_current




# -----------------------------------------------------------------------------
# Single perturbation experiment
# -----------------------------------------------------------------------------

def run_single_perturbation_ridge_fast(
    engine: FastRidgeEngine,
    databank: SpatialDataBank,
    config: Config,
    center_idx: int,
    target_center_idx: int,
    source_neighbors: List[int],
    target_neighbors: List[int],
    k: int,
    alpha: float,
    hop: int,
    steps: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """GPU-accelerated perturbation experiment via FastRidgeEngine."""
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_center = databank.get_spot_data(center_idx)
    raw_orig_gids = np.asarray(raw_center["gene_ids"], dtype=np.int64)
    raw_orig_vals = np.asarray(raw_center["raw_normed_values"], dtype=np.float32)
    raw_target = databank.get_spot_data(target_center_idx)
    raw_target_gids = np.asarray(raw_target["gene_ids"], dtype=np.int64)
    raw_target_vals = np.asarray(raw_target["raw_normed_values"], dtype=np.float32)


    perturb_overrides = create_interpolated_perturbation(
        databank, source_neighbors, target_neighbors, k, alpha, seed)
    frozen_indices = set(perturb_overrides.keys())

    capture_set = {center_idx, target_center_idx}
    capture_set.update(source_neighbors)

    step_results = engine.run_dual_line(
        perturb_overrides, frozen_indices, capture_set, steps)

    save_step_expression(step_results, center_idx, out_dir, config.vocab_file)
    convergence = compute_step_convergence(step_results, center_idx)

    step_metrics_vs_raw = []
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

    baseline_raw = compute_spot_similarity(
        raw_orig_vals, raw_orig_gids, raw_target_vals, raw_target_gids)
    baseline_raw["step"] = 0
    metrics_vs_raw = [baseline_raw] + step_metrics_vs_raw

    result = {
        "center_idx": center_idx,
        "target_center_idx": target_center_idx,
        "k": k, "alpha": alpha, "hop": hop, "steps": steps, "seed": seed,
        "n_perturbed_neighbors": len(frozen_indices),
        "n_source_neighbors": len(source_neighbors),
        "n_target_neighbors": len(target_neighbors),
        "infer_mode": "linear_reg_gpu",
        "metrics_vs_raw": metrics_vs_raw,
        "metrics": metrics_vs_raw,
        "convergence": convergence,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    conv_report = {
        "converged": (len(convergence) > 1
                      and convergence[-1]["mse"] < convergence[0]["mse"] * 0.5),
        "steps": convergence,
        "final_step_mse": convergence[-1]["mse"] if convergence else None,
    }
    with open(out_dir / "convergence_report.json", "w") as f:
        json.dump(conv_report, f, indent=2, default=str)

    return result


def run_single_perturbation_knn_fast(
    engine: FastKNNAverageEngine,
    databank: SpatialDataBank,
    config: Config,
    center_idx: int,
    target_center_idx: int,
    source_neighbors: List[int],
    target_neighbors: List[int],
    k: int,
    alpha: float,
    hop: int,
    steps: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """GPU-accelerated perturbation experiment via FastKNNAverageEngine."""
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_center = databank.get_spot_data(center_idx)
    raw_orig_gids = np.asarray(raw_center["gene_ids"], dtype=np.int64)
    raw_orig_vals = np.asarray(raw_center["raw_normed_values"], dtype=np.float32)
    raw_target = databank.get_spot_data(target_center_idx)
    raw_target_gids = np.asarray(raw_target["gene_ids"], dtype=np.int64)
    raw_target_vals = np.asarray(raw_target["raw_normed_values"], dtype=np.float32)


    perturb_overrides = create_interpolated_perturbation(
        databank, source_neighbors, target_neighbors, k, alpha, seed)
    frozen_indices = set(perturb_overrides.keys())

    capture_set = {center_idx, target_center_idx}
    capture_set.update(source_neighbors)

    step_results = engine.run_dual_line(
        perturb_overrides, frozen_indices, capture_set, steps)

    save_step_expression(step_results, center_idx, out_dir, config.vocab_file)
    convergence = compute_step_convergence(step_results, center_idx)

    step_metrics_vs_raw = []
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

    baseline_raw = compute_spot_similarity(
        raw_orig_vals, raw_orig_gids, raw_target_vals, raw_target_gids)
    baseline_raw["step"] = 0
    metrics_vs_raw = [baseline_raw] + step_metrics_vs_raw

    result = {
        "center_idx": center_idx,
        "target_center_idx": target_center_idx,
        "k": k, "alpha": alpha, "hop": hop, "steps": steps, "seed": seed,
        "n_perturbed_neighbors": len(frozen_indices),
        "n_source_neighbors": len(source_neighbors),
        "n_target_neighbors": len(target_neighbors),
        "infer_mode": "knn_avg",
        "metrics_vs_raw": metrics_vs_raw,
        "metrics": metrics_vs_raw,
        "convergence": convergence,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    conv_report = {
        "converged": (len(convergence) > 1
                      and convergence[-1]["mse"] < convergence[0]["mse"] * 0.5),
        "steps": convergence,
        "final_step_mse": convergence[-1]["mse"] if convergence else None,
    }
    with open(out_dir / "convergence_report.json", "w") as f:
        json.dump(conv_report, f, indent=2, default=str)

    return result


def run_single_perturbation_knn(
    databank: SpatialDataBank,
    config: Config,
    center_idx: int,
    target_center_idx: int,
    source_neighbors: List[int],
    target_neighbors: List[int],
    k: int,
    alpha: float,
    hop: int,
    steps: int,
    seed: int,
    out_dir: Path,
    all_global_indices: List[int],
    ridge_model: Optional[GPURidgeRegressionModel],
) -> Dict[str, Any]:
    """Run single perturbation experiment with KNN average baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_center = databank.get_spot_data(center_idx)
    raw_orig_gids = np.asarray(raw_center["gene_ids"], dtype=np.int64)
    raw_orig_vals = np.asarray(raw_center["raw_normed_values"], dtype=np.float32)
    raw_target = databank.get_spot_data(target_center_idx)
    raw_target_gids = np.asarray(raw_target["gene_ids"], dtype=np.int64)
    raw_target_vals = np.asarray(raw_target["raw_normed_values"], dtype=np.float32)


    perturb_overrides = create_interpolated_perturbation(
        databank, source_neighbors, target_neighbors, k, alpha, seed)
    frozen_indices = set(perturb_overrides.keys())

    x0_state: Dict[int, Dict[str, np.ndarray]] = {}
    x0_prime_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in all_global_indices:
        gids, vals = get_spot_expression(databank, gidx)
        entry = {"gene_ids": gids.copy(), "raw_normed_values": vals.copy()}
        x0_state[int(gidx)] = entry
        x0_prime_state[int(gidx)] = {
            "gene_ids": gids.copy(), "raw_normed_values": vals.copy()}
    for gidx, ov in perturb_overrides.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(ov["raw_normed_values"],
                                            dtype=np.float32),
        }

    capture_set = {center_idx, target_center_idx}
    capture_set.update(source_neighbors)

    step_results, _ = run_knn_dual_line_inference(
        databank, all_global_indices, steps, frozen_indices,
        x0_state, x0_prime_state, ridge_model,
        capture_indices=capture_set,
        precomputed_x_line=None,
    )

    save_step_expression(step_results, center_idx, out_dir, config.vocab_file)
    convergence = compute_step_convergence(step_results, center_idx)

    step_metrics_vs_raw = []
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

    baseline_raw = compute_spot_similarity(
        raw_orig_vals, raw_orig_gids, raw_target_vals, raw_target_gids)
    baseline_raw["step"] = 0
    metrics_vs_raw = [baseline_raw] + step_metrics_vs_raw

    result = {
        "center_idx": center_idx,
        "target_center_idx": target_center_idx,
        "k": k, "alpha": alpha, "hop": hop, "steps": steps, "seed": seed,
        "n_perturbed_neighbors": len(frozen_indices),
        "n_source_neighbors": len(source_neighbors),
        "n_target_neighbors": len(target_neighbors),
        "infer_mode": "knn_avg",
        "metrics_vs_raw": metrics_vs_raw,
        "metrics": metrics_vs_raw,
        "convergence": convergence,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    conv_report = {
        "converged": (len(convergence) > 1
                      and convergence[-1]["mse"] < convergence[0]["mse"] * 0.5),
        "steps": convergence,
        "final_step_mse": convergence[-1]["mse"] if convergence else None,
    }
    with open(out_dir / "convergence_report.json", "w") as f:
        json.dump(conv_report, f, indent=2, default=str)

    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="KNN/Ridge baseline for microenvironment perturbation")

    ap.add_argument("--dataset", required=True, choices=["1142243F", "1160920F", "HBRC", "PDAC"])
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--lmdb_path", default=None)
    ap.add_argument("--lmdb_manifest", default=None)
    ap.add_argument("--cache_mode", default="lmdb", choices=["h5", "lmdb"])
    ap.add_argument("--truth_csv", default=None)

    ap.add_argument("--infer_mode", required=True, choices=["knn_avg", "linear_reg_gpu"])
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--ridge_max_neighbors", type=int, default=8)
    ap.add_argument("--ridge_min_samples", type=int, default=10)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=5)
    ap.add_argument("--hops", type=str, default="1,2")
    ap.add_argument("--k_values_1hop", type=str, default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--k_values_2hop", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    ap.add_argument("--alpha_values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min_purity_healthy", type=float, default=0.5)
    ap.add_argument("--min_purity_cancer", type=float, default=0.5)
    ap.add_argument("--min_neighbors", type=int, default=4)
    ap.add_argument("--exclusion_hops", type=int, default=4)

    ap.add_argument("--directions", type=str, default="cancer_to_healthy,healthy_to_cancer")
    ap.add_argument("--pair_idx", type=int, default=None)

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

    print(f"[INFO] Dataset: {args.dataset}, Mode: {args.infer_mode}")
    print(f"[INFO] Pairs: {args.n_pairs}, Hops: {hops}, Steps: {args.steps}")

    databank = setup_databank(
        config, args.cache_dir, args.dataset,
        lmdb_path=args.lmdb_path, lmdb_manifest=args.lmdb_manifest,
        cache_mode=args.cache_mode)
    print(f"[OK] DataBank initialized")

    annotations = load_spot_annotations(args.dataset, args.cache_dir, truth_csv=args.truth_csv)
    n_cancer = sum(1 for v in annotations.values() if v == "cancer")
    n_healthy = sum(1 for v in annotations.values() if v == "healthy")
    print(f"[OK] Annotations: {len(annotations)} spots, {n_cancer} cancer, {n_healthy} healthy")

    pairs = select_spot_pairs(
        databank, annotations, args.n_pairs, args.seed,
        min_healthy_purity=args.min_purity_healthy,
        min_cancer_purity=args.min_purity_cancer,
        min_neighbors=args.min_neighbors,
        exclusion_hops=args.exclusion_hops)
    print(f"[OK] Selected {len(pairs)} spot pairs")

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "pairs_manifest.json", "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                    "n_pairs": len(pairs), "pairs": pairs}, f, indent=2)

    pair_range = [args.pair_idx] if args.pair_idx is not None else list(range(len(pairs)))

    all_global_indices = sorted(annotations.keys())
    print(f"[INFO] Total spots for inference: {len(all_global_indices)}")

    ridge_model = None
    fast_engine = None

    if args.infer_mode == "linear_reg_gpu":
        ridge_model = GPURidgeRegressionModel(
            alpha=args.ridge_alpha, max_neighbors=args.ridge_max_neighbors,
            device=str(device))
        ridge_model.train(databank, all_global_indices, current_state=None,
                          min_samples=args.ridge_min_samples)
        fast_engine = FastRidgeEngine(
            ridge_model, databank, all_global_indices)
        print(f"[OK] FastRidgeEngine ready - dual-line runs on GPU")
    elif args.infer_mode == "knn_avg":
        fast_engine = FastKNNAverageEngine(
            databank, all_global_indices, device=str(device))
        print(f"[OK] FastKNNAverageEngine ready - dual-line runs on GPU")


    total_exps = 0
    for h in hops:
        k_vals = k_values_1hop if h == 1 else k_values_2hop
        total_exps += (len(pair_range) * len(directions)
                       * len(k_vals) * len(alpha_values))
    print(f"[INFO] Total experiments: {total_exps}")

    all_results = []
    exp_count = 0

    for pi in pair_range:
        pair = pairs[pi]
        healthy_center = pair["healthy_center"]
        cancer_center = pair["cancer_center"]

        for direction in directions:
            center_idx = (cancer_center if direction == "cancer_to_healthy"
                          else healthy_center)
            target_idx = (healthy_center if direction == "cancer_to_healthy"
                          else cancer_center)

            for hop in hops:
                hop1, hop2 = get_hop_neighbors(databank, center_idx, hop)
                source_neighbors = hop1 if hop == 1 else hop1 + hop2
                t_hop1, t_hop2 = get_hop_neighbors(databank, target_idx, hop)
                target_neighbors = t_hop1 if hop == 1 else t_hop1 + t_hop2
                k_vals = k_values_1hop if hop == 1 else k_values_2hop

                for k in k_vals:
                    if k > len(source_neighbors):
                        continue
                    for alpha in alpha_values:
                        exp_count += 1
                        exp_tag = (f"pair{pi}/{direction}/"
                                   f"hop{hop}_k{k}_alpha{alpha:.1f}")
                        exp_out = (out_base / args.dataset / direction
                                   / f"pair{pi}"
                                   / f"hop{hop}_k{k}_alpha{alpha:.1f}")

                        if (exp_out / "metrics.json").exists():
                            print(f"[SKIP] {exp_tag}")
                            with open(exp_out / "metrics.json") as f:
                                all_results.append(json.load(f))
                            continue

                        t_exp = time.time()
                        print(f"\n[{exp_count}/{total_exps}] {exp_tag}")

                        if args.infer_mode == "linear_reg_gpu" and fast_engine is not None:
                            result = run_single_perturbation_ridge_fast(
                                engine=fast_engine,
                                databank=databank, config=config,
                                center_idx=center_idx,
                                target_center_idx=target_idx,
                                source_neighbors=source_neighbors,
                                target_neighbors=target_neighbors,
                                k=k, alpha=alpha, hop=hop,
                                steps=args.steps, seed=args.seed,
                                out_dir=exp_out)
                        elif args.infer_mode == "knn_avg" and fast_engine is not None:
                            result = run_single_perturbation_knn_fast(
                                engine=fast_engine,
                                databank=databank, config=config,
                                center_idx=center_idx,
                                target_center_idx=target_idx,
                                source_neighbors=source_neighbors,
                                target_neighbors=target_neighbors,
                                k=k, alpha=alpha, hop=hop,
                                steps=args.steps, seed=args.seed,
                                out_dir=exp_out)
                        else:
                            result = run_single_perturbation_knn(
                                databank=databank, config=config,
                                center_idx=center_idx,
                                target_center_idx=target_idx,
                                source_neighbors=source_neighbors,
                                target_neighbors=target_neighbors,
                                k=k, alpha=alpha, hop=hop,
                                steps=args.steps, seed=args.seed,
                                out_dir=exp_out,
                                all_global_indices=all_global_indices,
                                ridge_model=ridge_model,
                                )

                        elapsed_exp = time.time() - t_exp
                        print(f"  => {elapsed_exp:.2f}s")

                        result["pair_idx"] = pi
                        result["direction"] = direction
                        result["dataset"] = args.dataset
                        all_results.append(result)

                        with open(exp_out / "manifest.json", "w") as f:
                            json.dump({
                                "dataset": args.dataset,
                                "direction": direction,
                                "pair_idx": pi, "hop": hop,
                                "k": k, "alpha": alpha,
                                "center_idx": center_idx,
                                "target_center_idx": target_idx,
                                "infer_mode": args.infer_mode,
                                "seed": args.seed,
                                "steps": args.steps,
                            }, f, indent=2)

    # Save summary
    summary_path = out_base / f"summary_{args.dataset}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Saved summary to {summary_path}")

    # CSV
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
        for m in r.get("metrics_vs_raw", []) or []:
            row = dict(base_info)
            row["reference"] = "raw"
            row["step"] = m.get("step", -1)
            row["pearson"] = m.get("pearson")
            row["mse"] = m.get("mse")
            row["n_common_genes"] = m.get("n_common_genes")
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_base / f"dose_response_{args.dataset}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved dose-response CSV to {csv_path}")


if __name__ == "__main__":
    main()
