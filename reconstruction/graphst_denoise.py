#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphST denoising baseline for DLPFC slices (per-slice train + reconstruct).

Reference:
  Long Y. et al. 2023 Nat Commun - "Spatially informed clustering, integration,
  and deconvolution of spatial transcriptomics with GraphST". GCN auto-encoder
  with self-supervised contrastive learning that reconstructs the spot-level
  gene expression matrix.
  Repo: https://github.com/JinmiaoChenLab/GraphST

Output (under --out_dir, schema matches SpatialGT/STAGATE denoise scripts):
  denoised_expression.npz
      denoised        : float32 (N_spots, N_genes)  GraphST reconstructed X
      raw             : float32 (N_spots, N_genes)  raw log1p expression
      gene_names      : object  (N_genes,)
      spot_global_ids : int64   (N_spots,)          0..N-1 (per-slice)
      barcodes        : object  (N_spots,)
      coords          : float32 (N_spots, 2)
  manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, issparse
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
        description="GraphST per-slice denoising baseline.",
    )
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="Path containing <dataset>/processed.h5ad "
                         "(for example, <cache_dir>/<dataset_name>)")
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # GraphST hyperparameters (defaults from upstream tutorials)
    ap.add_argument("--epochs", type=int, default=600,
                    help="GraphST training epochs (default 600 - tutorial value).")
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dim_output", type=int, default=64,
                    help="GCN bottleneck dimension (default 64).")
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="Reconstruction loss weight (default 10).")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="Contrastive loss weight (default 1).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    fix_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    try:
        from GraphST import GraphST as GraphSTModule
    except Exception as e:
        print(f"[ERROR] GraphST import failed: {e}")
        sys.exit(2)

    # ------------------------------------------------------------------
    # Load and prepare adata
    # ------------------------------------------------------------------
    h5ad_path = Path(args.cache_dir) / args.dataset_name / "processed.h5ad"
    print(f"[INFO] Reading {h5ad_path}")
    adata = ad.read_h5ad(str(h5ad_path))

    # Use log1p-normalized expression as input (consistent with our other
    # denoise baselines). We bypass GraphST's internal Seurat-HVG selection
    # (which expects raw counts) by tagging *all* genes as highly_variable; the
    # underlying X is already HVG-filtered upstream (n_HVG = 2669 here).
    if "X_log1p" in adata.layers:
        L = adata.layers["X_log1p"]
        adata.X = L if issparse(L) else csr_matrix(np.asarray(L))
        print("[INFO] Using adata.layers['X_log1p'] as adata.X")
    else:
        if not issparse(adata.X):
            adata.X = csr_matrix(adata.X)
        print("[INFO] adata.X kept (no X_log1p layer found)")

    adata.var["highly_variable"] = True
    adata.uns.setdefault("log1p", {"base": None})  # placate scanpy

    raw_dense = adata.X.toarray().astype(np.float32) if issparse(adata.X) \
                else np.asarray(adata.X, dtype=np.float32)
    barcodes   = list(adata.obs_names)
    gene_names = list(adata.var_names)
    coords     = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    n_spots, n_genes = raw_dense.shape
    print(f"[INFO] adata: {n_spots} spots x {n_genes} genes "
          f"(treated as HVG=ALL since upstream cache is already HVG-3000-filtered)")

    # ------------------------------------------------------------------
    # Build + train GraphST
    # ------------------------------------------------------------------
    print(f"[INFO] Init GraphST (epochs={args.epochs}, dim_output={args.dim_output}, "
          f"alpha={args.alpha}, beta={args.beta}, lr={args.learning_rate})")
    model = GraphSTModule.GraphST(
        adata,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        dim_output=args.dim_output,
        alpha=args.alpha,
        beta=args.beta,
        random_seed=args.seed,
        deconvolution=False,
        datatype="10X",
    )
    print("[INFO] Training GraphST ...")
    out_adata = model.train()

    if "emb" not in out_adata.obsm.keys():
        print("[ERROR] adata.obsm['emb'] missing after training")
        sys.exit(3)

    emb_rec = np.asarray(out_adata.obsm["emb"], dtype=np.float32)
    if emb_rec.shape[1] != n_genes:
        # If GraphST kept its own HVG subset, embed back into the full gene
        # matrix using gene-name alignment. With our setup this branch should
        # not trigger (we forced highly_variable=True for all genes), but keep
        # it for safety.
        print(f"[WARN] emb shape {emb_rec.shape} != raw {raw_dense.shape}; "
              "attempting gene-name re-alignment.")
        sub_genes = list(out_adata.var_names[
            out_adata.var.get("highly_variable", np.ones(out_adata.n_vars, bool))
        ])
        col_idx = {g: i for i, g in enumerate(sub_genes)}
        full = np.zeros_like(raw_dense, dtype=np.float32)
        for j, g in enumerate(gene_names):
            if g in col_idx:
                full[:, j] = emb_rec[:, col_idx[g]]
        emb_rec = full

    # ------------------------------------------------------------------
    # Sanity check: global PCC(raw, denoised)
    # ------------------------------------------------------------------
    valid_rows = np.linalg.norm(emb_rec, axis=1) > 1e-8
    if valid_rows.sum() > 0:
        flat_raw = raw_dense[valid_rows].ravel()
        flat_den = emb_rec[valid_rows].ravel()
        if flat_raw.size > 1_000_000:
            sel = np.random.RandomState(0).choice(flat_raw.size, 1_000_000, replace=False)
            flat_raw, flat_den = flat_raw[sel], flat_den[sel]
        global_pcc = float(pearsonr(flat_raw, flat_den)[0])
        print(f"[INFO] Global PCC(raw, denoised) = {global_pcc:.4f}")
    else:
        global_pcc = None

    print(f"[INFO] denoised matrix shape={emb_rec.shape}, "
          f"mean={emb_rec.mean():.4f}, std={emb_rec.std():.4f}, "
          f"min={emb_rec.min():.4f}, max={emb_rec.max():.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "denoised_expression.npz"
    np.savez_compressed(
        npz_path,
        denoised        = emb_rec,
        raw             = raw_dense,
        gene_names      = np.array(gene_names, dtype=object),
        spot_global_ids = np.arange(n_spots, dtype=np.int64),
        barcodes        = np.array(barcodes, dtype=object),
        coords          = coords,
    )
    print(f"[INFO] Saved: {npz_path}")

    manifest = {
        "task":            "denoising",
        "method":          "GraphST",
        "variant":         "default",
        "dataset":         args.dataset_name,
        "epochs":          args.epochs,
        "learning_rate":   args.learning_rate,
        "weight_decay":    args.weight_decay,
        "dim_output":      args.dim_output,
        "alpha":           args.alpha,
        "beta":            args.beta,
        "n_spots":         int(n_spots),
        "n_genes":         int(n_genes),
        "device":          str(device),
        "global_pcc_raw_vs_denoised": global_pcc,
        "note":            "Per-slice GraphST training; reconstructed expression "
                           "stored as denoised matrix. HVG flag set to True for all "
                           "genes since upstream cache is already HVG-filtered.",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved: {out_dir / 'manifest.json'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
