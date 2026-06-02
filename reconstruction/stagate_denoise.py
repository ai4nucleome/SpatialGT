#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGATE_pyG denoising baseline for DLPFC slices (per-slice train + reconstruct).

Reference:
  Dong & Zhang 2022 Nat Commun - graph attention auto-encoder that reconstructs
  spot expression from its spatial neighborhood, providing a denoising effect.
  Repo: https://github.com/QIFEIDKN/STAGATE_pyG

Output (under --out_dir, schema matches SpatialGT's denoise_inference.py):
  denoised_expression.npz
      denoised        : float32 (N_spots, N_genes)  STAGATE reconstructed X
      raw             : float32 (N_spots, N_genes)  raw log1p (adata.layers["X_log1p"])
      gene_names      : object  (N_genes,)
      spot_global_ids : int64   (N_spots,)          0..N-1 (per-slice training)
      barcodes        : object  (N_spots,)
      coords          : float32 (N_spots, 2)
  manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse, csr_matrix
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


def fix_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(
        description="STAGATE_pyG per-slice denoising baseline.",
    )
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="Path containing <dataset>/processed.h5ad (for example, <cache_dir>/<dataset_name>)")
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # STAGATE hyperparameters
    ap.add_argument("--graph_model", type=str, default="KNN",
                    choices=["KNN", "Radius"],
                    help="Spatial graph builder. Default KNN (k=8, matches SpatialGT).")
    ap.add_argument("--k_cutoff", type=int, default=8,
                    help="k for KNN spatial graph (used if graph_model=KNN).")
    ap.add_argument("--rad_cutoff", type=float, default=150.0,
                    help="Radius for Radius spatial graph (used if graph_model=Radius). "
                         "Default 150 px matches Visium tutorial.")
    ap.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 30],
                    help="STAGATE encoder hidden dims (default 512 30).")
    ap.add_argument("--n_epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_hvg", action="store_true",
                    help="If set, restrict training to genes flagged as 'highly_variable' "
                         "in adata.var. Default uses ALL genes (matches our 2669-HVG cache).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    fix_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # Lazy import after env+seed set
    try:
        from STAGATE_pyG import Cal_Spatial_Net, Stats_Spatial_Net, train_STAGATE
    except Exception as e:
        print(f"[ERROR] STAGATE_pyG import failed: {e}")
        sys.exit(2)

    h5ad_path = Path(args.cache_dir) / args.dataset_name / "processed.h5ad"
    print(f"[INFO] Reading {h5ad_path}")
    adata = ad.read_h5ad(str(h5ad_path))

    # STAGATE conventionally trains on log-normalized expression. Override
    # adata.X with X_log1p so the rest of the script (and STAGATE) use it.
    if "X_log1p" in adata.layers:
        L = adata.layers["X_log1p"]
        adata.X = L if issparse(L) else csr_matrix(np.asarray(L))
        print(f"[INFO] Using adata.layers['X_log1p'] as adata.X")
    else:
        # Already log-normalized in adata.X - keep as is
        if not issparse(adata.X):
            adata.X = csr_matrix(adata.X)
        print(f"[INFO] adata.X kept (no X_log1p layer found)")

    raw_dense = adata.X.toarray().astype(np.float32) if issparse(adata.X) \
                else np.asarray(adata.X, dtype=np.float32)

    barcodes   = list(adata.obs_names)
    gene_names = list(adata.var_names)
    coords     = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    n_spots, n_genes = raw_dense.shape
    print(f"[INFO] adata: {n_spots} spots x {n_genes} genes, "
          f"raw mean={raw_dense.mean():.3f}, std={raw_dense.std():.3f}")

    # 1) build spatial network
    print(f"[INFO] Cal_Spatial_Net (model={args.graph_model}, "
          f"k_cutoff={args.k_cutoff}, rad_cutoff={args.rad_cutoff})")
    if args.graph_model == "KNN":
        Cal_Spatial_Net(adata, k_cutoff=args.k_cutoff, model="KNN", verbose=True)
    else:
        Cal_Spatial_Net(adata, rad_cutoff=args.rad_cutoff, model="Radius", verbose=True)
    try:
        Stats_Spatial_Net(adata)
    except Exception as e:
        print(f"[WARN] Stats_Spatial_Net error (non-fatal): {e}")

    # Optionally restrict to HVG (default off to match SpatialGT setup)
    if args.use_hvg and "highly_variable" in adata.var.columns:
        n_hvg = int(adata.var["highly_variable"].sum())
        print(f"[INFO] Restricting STAGATE to {n_hvg} HVG (use_hvg=True)")
    elif args.use_hvg:
        print("[WARN] use_hvg=True but no highly_variable column - using ALL genes")

    # 2) train + reconstruct (in-place into adata.layers['STAGATE_ReX'])
    print(f"[INFO] train_STAGATE epochs={args.n_epochs}, lr={args.lr}, "
          f"hidden_dims={args.hidden_dims}")
    adata = train_STAGATE(
        adata,
        hidden_dims=list(args.hidden_dims),
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
        save_reconstrction=True,           # NB: typo is in upstream API
        device=device,
    )

    rex = adata.layers.get("STAGATE_ReX")
    if rex is None:
        print("[ERROR] STAGATE_ReX layer not found after training")
        sys.exit(3)
    if issparse(rex):
        rex = rex.toarray()
    rex = np.asarray(rex, dtype=np.float32)
    if rex.shape != raw_dense.shape:
        print(f"[WARN] ReX shape {rex.shape} != raw {raw_dense.shape} - "
              "likely because STAGATE trained on HVG subset. Padding zeros for "
              "non-HVG genes.")
        full = np.zeros_like(raw_dense)
        # Map back via gene names: STAGATE works on adata_Vars subset, but the
        # ReX layer is stored on the original adata, so the shape *should* match.
        # If we land here, raise instead of trying to guess column order.
        raise RuntimeError(
            f"Unexpected ReX shape {rex.shape}. Re-run with --use_hvg off."
        )

    # Quick sanity: global PCC between raw and STAGATE reconstruction
    valid_rows = np.linalg.norm(rex, axis=1) > 1e-8
    if valid_rows.sum() > 0:
        flat_raw = raw_dense[valid_rows].ravel()
        flat_den = rex[valid_rows].ravel()
        if flat_raw.size > 1_000_000:
            sel = np.random.RandomState(0).choice(flat_raw.size, 1_000_000, replace=False)
            flat_raw, flat_den = flat_raw[sel], flat_den[sel]
        global_pcc = float(pearsonr(flat_raw, flat_den)[0])
        print(f"[INFO] Global PCC(raw, denoised) = {global_pcc:.4f}")
    else:
        global_pcc = None

    print(f"[INFO] denoised matrix shape={rex.shape}, "
          f"mean={rex.mean():.4f}, std={rex.std():.4f}, "
          f"min={rex.min():.4f}, max={rex.max():.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "denoised_expression.npz"
    np.savez_compressed(
        npz_path,
        denoised        = rex,
        raw             = raw_dense,
        gene_names      = np.array(gene_names, dtype=object),
        spot_global_ids = np.arange(n_spots, dtype=np.int64),
        barcodes        = np.array(barcodes, dtype=object),
        coords          = coords,
    )
    print(f"[INFO] Saved: {npz_path}")

    manifest = {
        "task":            "denoising",
        "method":          "STAGATE_pyG",
        "variant":         "default",
        "dataset":         args.dataset_name,
        "graph_model":     args.graph_model,
        "k_cutoff":        args.k_cutoff,
        "rad_cutoff":      args.rad_cutoff,
        "hidden_dims":     list(args.hidden_dims),
        "n_epochs":        args.n_epochs,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "use_hvg":         bool(args.use_hvg),
        "n_spots":         int(n_spots),
        "n_genes":         int(n_genes),
        "device":          str(device),
        "global_pcc_raw_vs_denoised": global_pcc,
        "note":            "Per-slice STAGATE_pyG training; reconstruction stored as denoised matrix.",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved: {out_dir / 'manifest.json'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
