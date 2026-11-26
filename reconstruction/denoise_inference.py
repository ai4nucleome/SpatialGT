#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-pass SpatialGT denoising of observed section expression."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
import anndata as ad


import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRETRAIN_DIR = _PROJECT_ROOT / "pretrain"
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Config import Config
from spatial_databank import SpatialDataBank
from pretrain.checkpoint import load_model_checkpoint


from spatialgt_reconstruction_eval import (
    _setup_config_for_cache,
    _get_dataset_info,
    _build_all_loader,
    run_single_pass_inference,
    fix_seed,
)


def run_denoise(
    databank: SpatialDataBank,
    config: Config,
    model: nn.Module,
    device: torch.device,
    all_global_indices: List[int],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Reconstruct every center once from observed neighbor expression."""
    print(f"[INFO] Building frozen-state overrides for {len(all_global_indices)} spots ...")
    frozen_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in all_global_indices:
        sd = databank.get_spot_data(int(gidx))
        frozen_state[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
        }
    databank.set_runtime_spot_overrides(frozen_state)

    loader = _build_all_loader(
        databank, config, batch_size, num_workers,
        prefetch_factor, persistent_workers,
    )
    overrides = run_single_pass_inference(
        databank, loader, model, device,
        desc="Denoise (no mask)", show_progress=True,
    )
    print(f"[INFO] Inference complete: {len(overrides)} spots predicted "
          f"(expected {len(all_global_indices)}).")
    return overrides


def stack_to_matrix(
    overrides: Dict[int, Dict[str, np.ndarray]],
    spot_global_ids: List[int],
    n_genes: int,
    reference_gene_ids: np.ndarray,
) -> np.ndarray:
    """Align predicted token values to the section gene columns."""

    token_to_col: Dict[int, int] = {
        int(tid): col for col, tid in enumerate(reference_gene_ids)
    }
    if len(token_to_col) != n_genes:
        print(f"[WARN] reference gene_ids has {len(token_to_col)} unique tokens "
              f"but adata has {n_genes} genes. Some columns may stay zero.")

    M = np.zeros((len(spot_global_ids), n_genes), dtype=np.float32)
    missed_spots = 0
    unmapped_tokens = 0
    for row, gidx in enumerate(spot_global_ids):
        ov = overrides.get(int(gidx))
        if ov is None:
            missed_spots += 1
            continue
        gids = np.asarray(ov["gene_ids"], dtype=np.int64)
        vals = np.asarray(ov["raw_normed_values"], dtype=np.float32)

        keep = gids >= 0
        gids, vals = gids[keep], vals[keep]

        cols = np.fromiter(
            (token_to_col.get(int(t), -1) for t in gids),
            dtype=np.int64, count=len(gids),
        )
        valid = cols >= 0
        if (~valid).any():
            unmapped_tokens += int((~valid).sum())
        M[row, cols[valid]] = vals[valid]
    if missed_spots:
        print(f"[WARN] {missed_spots} spots had no prediction (kept as zero rows).")
    if unmapped_tokens:
        print(f"[WARN] {unmapped_tokens} (spot, gene) entries dropped - "
              f"token id not in reference gene_ids vocab.")
    return M


def main():
    ap = argparse.ArgumentParser(
        description="SpatialGT denoising inference.",
    )
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to SpatialGT checkpoint dir (with state_dict / safetensors).")
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="Path to preprocessed cache dir containing <dataset_name>/processed.h5ad.")
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--cache_mode", type=str, default="lmdb", choices=["h5", "lmdb"])
    ap.add_argument("--lmdb_path", type=str, default=None)
    ap.add_argument("--lmdb_manifest", type=str, default=None)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--variant_tag", type=str, default="finetune",
                    help="Free-form tag stored in manifest.json (e.g. finetune / pretrain / singleslice).")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_ids", type=str, default="0",
                    help="Comma-separated GPU ids; first is used.")
    args = ap.parse_args()

    fix_seed(args.seed)

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids and torch.cuda.is_available() \
             else torch.device("cpu")
    print(f"[INFO] device = {device}")

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = Path(args.ckpt)


    config = Config()
    config = _setup_config_for_cache(
        config, str(cache_dir), args.dataset_name, args.cache_mode,
        args.lmdb_path, args.lmdb_manifest,
    )
    print(f"[INFO] Initializing databank ...")
    dataset_paths = [str(cache_dir / args.dataset_name / "processed.h5ad")]
    databank = SpatialDataBank(
        dataset_paths=dataset_paths,
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False,
    )
    start_idx, n_spots = _get_dataset_info(
        cache_dir, args.dataset_name,
        lmdb_manifest_path=args.lmdb_manifest if args.cache_mode == "lmdb" else None,
    )
    spot_global_ids = list(range(start_idx, start_idx + n_spots))
    print(f"[INFO] dataset={args.dataset_name}  n_spots={n_spots}  "
          f"global_range=[{start_idx},{start_idx+n_spots-1}]")


    adata = ad.read_h5ad(str(cache_dir / args.dataset_name / "processed.h5ad"))
    n_genes = int(adata.n_vars)
    gene_names = list(adata.var_names)
    barcodes = list(adata.obs_names)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if "X_log1p" in adata.layers:
        L = adata.layers["X_log1p"]
        raw = (L.toarray() if issparse(L) else np.asarray(L)).astype(np.float32)
    else:
        raw = (adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)).astype(np.float32)
    print(f"[INFO] adata: {adata.n_obs} spots x {n_genes} genes  (raw matrix shape={raw.shape})")


    print(f"[INFO] Loading SpatialGT checkpoint: {ckpt_dir}")
    model = load_model_checkpoint(ckpt_dir, config, device=device)
    model.eval()


    overrides = run_denoise(
        databank=databank,
        config=config,
        model=model,
        device=device,
        all_global_indices=spot_global_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )


    ref_sd = databank.get_spot_data(int(spot_global_ids[0]))
    reference_gene_ids = np.asarray(ref_sd["gene_ids"], dtype=np.int64)
    print(f"[INFO] reference gene_ids array length = {len(reference_gene_ids)} "
          f"(expected {n_genes})")
    denoised = stack_to_matrix(overrides, spot_global_ids, n_genes, reference_gene_ids)
    print(f"[INFO] denoised matrix shape={denoised.shape}, "
          f"mean={denoised.mean():.4f}, std={denoised.std():.4f}, "
          f"min={denoised.min():.4f}, max={denoised.max():.4f}")


    valid_rows = np.linalg.norm(denoised, axis=1) > 1e-8
    if valid_rows.sum() > 0:
        from scipy.stats import pearsonr
        flat_raw = raw[valid_rows].ravel()
        flat_den = denoised[valid_rows].ravel()

        if flat_raw.size > 1_000_000:
            sel = np.random.RandomState(0).choice(flat_raw.size, 1_000_000, replace=False)
            flat_raw, flat_den = flat_raw[sel], flat_den[sel]
        global_pcc = pearsonr(flat_raw, flat_den)[0]
        print(f"[INFO] Global PCC(raw, denoised) over {valid_rows.sum()} valid spots = "
              f"{global_pcc:.4f}")


    npz_path = out_dir / "denoised_expression.npz"
    np.savez_compressed(
        npz_path,
        denoised        = denoised,
        raw             = raw,
        gene_names      = np.array(gene_names, dtype=object),
        gene_token_ids  = reference_gene_ids,
        spot_global_ids = np.asarray(spot_global_ids, dtype=np.int64),
        barcodes        = np.array(barcodes, dtype=object),
        coords          = coords,
    )
    print(f"[INFO] Saved: {npz_path}")

    manifest = {
        "task": "denoising",
        "method": "SpatialGT",
        "variant": args.variant_tag,
        "dataset": args.dataset_name,
        "n_spots": int(n_spots),
        "n_genes": int(n_genes),
        "device": str(device),
        "cache_mode": args.cache_mode,
        "global_pcc_raw_vs_denoised": float(global_pcc) if 'global_pcc' in locals() else None,
        "note": "Single forward pass over observed section expression; each center is "
                "reconstructed from its spatial neighbors.",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved: {out_dir / 'manifest.json'}")
    print(f"[DONE]")


if __name__ == "__main__":
    main()
