#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEDR mouse stroke perturbation baseline.

This public script keeps only random-spot DEG perturbation and dual-line
iteration. The initial control state is the raw Sham expression matrix,
matching the original baseline implementation. Reference-slice output scoring
and exploratory selection branches have been removed.
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
import scipy.sparse as sp
from tqdm import tqdm
import anndata as ad

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_SEDR_ROOT = _REPO_ROOT / "baseline" / "SEDR"
for _path in (_REPO_ROOT, _SEDR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from SEDR.SEDR_model import Sedr
from SEDR.graph_func import graph_construction

def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load gene vocabulary and return lowercase mapping."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {k.lower(): int(v) for k, v in vocab.items()}


def _barcodes_to_indices(obs_names: List[str], barcodes: List[str]) -> List[int]:
    """Convert barcodes to indices in obs_names."""
    pos = {b: i for i, b in enumerate(obs_names)}
    return [pos[b] for b in barcodes if b in pos]


def _indices_to_barcodes(obs_names: List[str], indices: List[int]) -> List[str]:
    """Convert indices to barcodes."""
    return [obs_names[i] for i in indices if 0 <= i < len(obs_names)]


# ==============================================================================
# Data loading and preparation
# ==============================================================================

def load_adata_and_prepare(
    cache_dir: Path,
    dataset_name: str,
    vocab: Dict[str, int],
) -> Tuple[ad.AnnData, np.ndarray, np.ndarray, List[int]]:
    """
    Load processed.h5ad and prepare expression matrix with vocab gene_ids.
    
    Returns:
        adata: AnnData object
        X: expression matrix [n_spots, n_genes] as numpy array
        gene_ids: array of gene_ids corresponding to columns of X
        valid_gene_indices: indices of genes that are in vocab
    """
    h5ad_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(h5ad_path))
    
    # Map gene names to vocab gene_ids
    gene_names = list(adata.var_names)
    gene_ids = []
    valid_gene_indices = []
    
    for i, gene in enumerate(gene_names):
        gid = vocab.get(gene.lower(), None)
        if gid is not None:
            gene_ids.append(gid)
            valid_gene_indices.append(i)
    
    gene_ids = np.array(gene_ids, dtype=np.int64)
    
    # Use log1p-normalized expression (same as SpatialGT/KNN via SpatialDataBank)
    # to ensure all methods operate in the same data space.
    if "X_log1p" in adata.layers:
        X = adata.layers["X_log1p"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X[:, valid_gene_indices].astype(np.float32)
    
    data_source = "X_log1p" if "X_log1p" in adata.layers else "adata.X"
    print(f"[DATA] Loaded {dataset_name}: {adata.n_obs} spots, {len(valid_gene_indices)} genes (of {adata.n_vars}), source={data_source}")
    
    return adata, X, gene_ids, valid_gene_indices


def fix_seed(seed: int = 42):
    """Fix random seed for reproducibility."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_sedr(
    X: np.ndarray,
    adata: ad.AnnData,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    graph_knn_k: int = 6,
    rec_w: float = 10.0,
    gcn_w: float = 0.1,
    self_w: float = 1.0,
    device: str = "cuda:0",
    mode: str = "imputation",
    show_progress: bool = True,
    seed: int = 42,
) -> Sedr:
    """
    Train SEDR model on expression matrix.
    
    Args:
        X: expression matrix [n_spots, n_genes]
        adata: AnnData with spatial coordinates
        epochs: number of training epochs
        lr: learning rate
        weight_decay: weight decay
        graph_knn_k: number of neighbors for KNN graph
        rec_w: reconstruction loss weight
        gcn_w: GCN loss weight
        self_w: self-supervised loss weight
        device: torch device
        mode: "clustering" or "imputation"
        show_progress: whether to show progress bar
        seed: random seed for reproducibility
    
    Returns:
        Trained Sedr model
    """
    # Fix random seed for reproducibility
    fix_seed(seed)
    
    # Build graph
    graph_dict = graph_construction(adata, n=graph_knn_k, mode='KNN')
    
    # Create SEDR model
    sedr = Sedr(
        X=X,
        graph_dict=graph_dict,
        rec_w=rec_w,
        gcn_w=gcn_w,
        self_w=self_w,
        mode=mode,
        device=device,
    )
    
    # Train
    print(f"[SEDR] Training for {epochs} epochs...")
    sedr.train_without_dec(epochs=epochs, lr=lr, decay=weight_decay)
    
    return sedr


def sedr_reconstruct(
    sedr: Sedr,
    X: np.ndarray,
    adj_norm: torch.Tensor,
) -> np.ndarray:
    """
    Run SEDR reconstruction on expression matrix.
    
    Args:
        sedr: trained Sedr model
        X: input expression matrix [n_spots, n_genes]
        adj_norm: normalized adjacency matrix
    
    Returns:
        Reconstructed expression matrix [n_spots, n_genes]
    """
    sedr.model.eval()
    
    # Disable mask for deterministic inference
    sedr.model._mask_rate = 0.0
    
    X_tensor = torch.FloatTensor(X).to(sedr.device)
    
    with torch.no_grad():
        # Forward pass
        z, mu, logvar, de_feat, q, feat_x, gnn_z, loss = sedr.model(X_tensor, adj_norm)
    
    # Get reconstruction and clip to non-negative
    recon = de_feat.detach().cpu().numpy()
    recon = np.clip(recon, 0, None)
    
    return recon.astype(np.float32)


def sedr_reconstruct_with_roi_mask(
    sedr: Sedr,
    X: np.ndarray,
    adj_norm: torch.Tensor,
    mask_indices: List[int],
) -> np.ndarray:
    """
    Run SEDR reconstruction with specific ROI spots masked.
    
    This function masks the specified ROI spots (by adding the learned mask token)
    and lets SEDR reconstruct their expression based on the surrounding unmasked spots.
    This leverages SEDR's masked autoencoder capability for prediction.
    
    Args:
        sedr: trained Sedr model
        X: input expression matrix [n_spots, n_genes]
        adj_norm: normalized adjacency matrix
        mask_indices: list of spot indices to mask (the ROI spots to predict)
    
    Returns:
        Reconstructed expression matrix [n_spots, n_genes]
    """
    sedr.model.eval()
    
    # Disable the random mask - we'll apply our own mask
    sedr.model._mask_rate = 0.0
    
    X_tensor = torch.FloatTensor(X.copy()).to(sedr.device)
    
    # Manually apply mask token to the specified ROI spots
    # This is what SEDR does in encoding_mask_noise: out_x[token_nodes] += self.enc_mask_token
    mask_indices_tensor = torch.LongTensor(mask_indices).to(sedr.device)
    X_tensor[mask_indices_tensor] = X_tensor[mask_indices_tensor] + sedr.model.enc_mask_token
    
    with torch.no_grad():
        # Manual forward pass (bypassing the random mask in forward)
        # We need to replicate the forward logic but skip encoding_mask_noise
        
        # Encoder
        feat_x = sedr.model.encoder(X_tensor)
        
        # GCN layers
        hidden1 = sedr.model.gc1(feat_x, adj_norm)
        mu = sedr.model.gc2(hidden1, adj_norm)
        logvar = sedr.model.gc3(hidden1, adj_norm)
        
        # Reparameterize (in eval mode, just return mu)
        gnn_z = mu
        
        # Concatenate features
        z = torch.cat((feat_x, gnn_z), 1)
        
        # Decoder
        if hasattr(sedr.model, 'decoder'):
            if hasattr(sedr.model.decoder, 'forward'):
                # Check if decoder is GraphConvolution (SEDR_module) or Sequential (SEDR_impute_module)
                try:
                    # GraphConvolution decoder needs adj
                    de_feat = sedr.model.decoder(z, adj_norm)
                except TypeError:
                    # Sequential decoder doesn't need adj
                    de_feat = sedr.model.decoder(z)
            else:
                de_feat = sedr.model.decoder(z)
        else:
            de_feat = z
    
    # Get reconstruction and clip to non-negative
    recon = de_feat.detach().cpu().numpy()
    recon = np.clip(recon, 0, None)
    
    return recon.astype(np.float32)


def sedr_reconstruct_with_override(
    sedr: Sedr,
    X_current: np.ndarray,
    adj_norm: torch.Tensor,
    frozen_indices: Set[int],
    frozen_values: np.ndarray,
    ema_alpha: float = 0.0,
) -> np.ndarray:
    """
    Run SEDR reconstruction with frozen spots and optional EMA.
    
    Args:
        sedr: trained Sedr model
        X_current: current expression matrix [n_spots, n_genes]
        adj_norm: normalized adjacency matrix
        frozen_indices: set of spot indices to keep frozen
        frozen_values: frozen expression values [n_frozen_spots, n_genes]
        ema_alpha: EMA coefficient (0 = direct replacement)
    
    Returns:
        Updated expression matrix [n_spots, n_genes]
    """
    # Get reconstruction
    X_recon = sedr_reconstruct(sedr, X_current, adj_norm)
    
    # Apply EMA for non-frozen spots
    if ema_alpha > 0:
        X_next = ema_alpha * X_current + (1.0 - ema_alpha) * X_recon
    else:
        X_next = X_recon.copy()
    
    # Restore frozen spots
    frozen_list = sorted(frozen_indices)
    for i, spot_idx in enumerate(frozen_list):
        X_next[spot_idx] = frozen_values[i]
    
    return X_next.astype(np.float32)


def select_random_spots(roi_indices: List[int], n_spots: int, seed: int) -> List[int]:
    """Select scattered spots from the requested ROI."""
    rng = random.Random(seed)
    roi_list = sorted(set(int(x) for x in roi_indices))
    if n_spots <= 0 or len(roi_list) <= n_spots:
        return roi_list
    return rng.sample(roi_list, n_spots)


def compute_uniform_weights(selected_indices: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    weights = np.ones(len(selected_indices), dtype=np.float32)
    return weights, {"weighting": "uniform"}

def apply_deg_perturbation(
    X: np.ndarray,
    gene_ids: np.ndarray,
    selected_indices: List[int],
    weights: np.ndarray,
    deg_csv: Path,
    vocab: Dict[str, int],
    p_adj_thresh: float = 0.1,
    min_abs_logfc: float = 0.0,
    logfc_strength: float = 1.0,
    logfc_clip: float = 5.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply DEG-based perturbation to selected spots (log1p space).
    
    Formula: X_new = log1p(expm1(X_old) * 2^(logFC × strength × weight))
    Data must be log1p-normalized (from adata.layers['X_log1p']).
    """
    deg = pd.read_csv(deg_csv)
    if "avg_logFC" not in deg.columns:
        raise ValueError(f"DEG csv missing required column avg_logFC")
    
    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]
    
    p_col = "p_val_adj" if "p_val_adj" in deg.columns else ("p_val" if "p_val" in deg.columns else None)
    if p_col is not None:
        deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
        deg = deg[np.isfinite(deg[p_col])]
        deg = deg[deg[p_col].astype(float) < float(p_adj_thresh)]
    
    deg = deg[deg["avg_logFC"].abs() >= float(min_abs_logfc)]
    
    # Build gene_id -> logFC mapping
    gene_logfc: Dict[int, float] = {}
    for _, r in deg.iterrows():
        gid = vocab.get(str(r["gene"]).lower(), None)
        if gid is None:
            continue
        logfc_val = float(r["avg_logFC"])
        logfc_clipped = np.clip(logfc_val, -float(logfc_clip), float(logfc_clip))
        gene_logfc[int(gid)] = logfc_clipped
    
    # Create gene_id to column index mapping
    gid_to_col = {int(gid): i for i, gid in enumerate(gene_ids)}
    
    # Apply perturbation in log1p space (consistent with SpatialGT/KNN):
    #   X_new = log1p(expm1(X_old) * 2^(logFC * strength * weight))
    X_perturbed = X.copy()
    n_hits_per_spot = []
    
    for i, spot_idx in enumerate(selected_indices):
        if i >= len(weights):
            break
        wi = float(weights[i])
        n_hits = 0
        
        for gid, logfc_val in gene_logfc.items():
            if gid in gid_to_col:
                col_idx = gid_to_col[gid]
                fold_change = np.power(2.0, logfc_val * logfc_strength * wi)
                linear_val = np.expm1(X_perturbed[spot_idx, col_idx])
                X_perturbed[spot_idx, col_idx] = np.log1p(max(linear_val * fold_change, 0.0))
                n_hits += 1
        
        n_hits_per_spot.append(n_hits)
    
    # Ensure non-negative
    X_perturbed = np.maximum(X_perturbed, 0.0)
    
    meta = {
        "deg_csv": str(deg_csv),
        "p_adj_thresh": float(p_adj_thresh),
        "min_abs_logfc": float(min_abs_logfc),
        "logfc_strength": float(logfc_strength),
        "logfc_clip": float(logfc_clip),
        "n_deg_genes_mapped": len(gene_logfc),
        "n_hits_per_spot_mean": float(np.mean(n_hits_per_spot)) if n_hits_per_spot else 0,
    }
    
    return X_perturbed, meta

def save_step_npz_from_matrix(
    X: np.ndarray,
    gene_ids: np.ndarray,
    start_idx: int,
    path: Path,
) -> None:
    """Save dense expression matrix as per-spot NPZ (same format as SpatialGT).

    Each row of X becomes a spot with global_idx = start_idx + row_idx.
    All spots share the same gene_ids vector.
    """
    n_spots = X.shape[0]
    spot_indices = np.arange(start_idx, start_idx + n_spots, dtype=np.int64)
    save_dict = {"spot_indices": spot_indices}
    gene_ids_arr = np.asarray(gene_ids, dtype=np.int64)
    for i in range(n_spots):
        gidx = int(spot_indices[i])
        save_dict[f"gids_{gidx}"] = gene_ids_arr
        save_dict[f"vals_{gidx}"] = X[i].astype(np.float32)
    np.savez_compressed(str(path), **save_dict)


def save_full_expression_matrix_to_csv(
    X: np.ndarray,
    gene_ids: np.ndarray,
    obs_names: List[str],
    out_path: Path,
    vocab: Dict[str, int],
) -> None:
    """
    Save full expression matrix to CSV (SEDR version).
    
    Args:
        X: expression matrix [n_spots, n_genes]
        gene_ids: array of gene_ids for each column
        obs_names: list of barcode names for each spot
        out_path: output CSV file path
        vocab: vocabulary mapping gene_name -> gene_id
    """
    if X.size == 0:
        print(f"[WARN] Empty expression matrix, skipping save to {out_path}")
        return
    
    # Reverse vocab: gene_id -> gene_name
    id_to_gene = {int(v): k for k, v in vocab.items()}
    
    # Get gene names
    gene_names = [id_to_gene.get(int(gid), f"gene_{gid}") for gid in gene_ids]
    
    # Create DataFrame and save
    df = pd.DataFrame(X, index=obs_names, columns=gene_names)
    df.index.name = "barcode"
    df.to_csv(out_path)
    print(f"[OK] Saved full expression: {out_path} ({X.shape[0]} spots × {X.shape[1]} genes)")


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


def run_dual_line_sedr(
    model: Sedr,
    adj_norm,
    x0: np.ndarray,
    x0_prime: np.ndarray,
    frozen_indices: Set[int],
    steps: int,
) -> Dict[int, np.ndarray]:
    """Run dense dual-line propagation with a trained SEDR model."""
    x_line = sedr_reconstruct(model, x0, adj_norm)
    x_prime_current = x0_prime.copy()
    step_expressions: Dict[int, np.ndarray] = {}
    for step in range(1, steps + 1):
        print(f"[SEDR] Dual-line step {step}/{steps}")
        x_prime_raw = sedr_reconstruct(model, x_prime_current, adj_norm)
        delta = x_prime_raw - x_line
        x_prime_true = x0_prime + delta
        for idx in frozen_indices:
            x_prime_true[int(idx)] = x0_prime[int(idx)]
        x_prime_true = np.maximum(x_prime_true, 0.0).astype(np.float32)
        step_expressions[step] = x_prime_true.copy()
        x_prime_current = x_prime_true
        if len(frozen_indices) < x_prime_true.shape[0]:
            mask = np.ones(x_prime_true.shape[0], dtype=bool)
            mask[list(frozen_indices)] = False
            print(f"  delta mean={float(delta[mask].mean()):.6f}, abs_mean={float(np.abs(delta[mask]).mean()):.6f}")
    return step_expressions


def main():
    parser = argparse.ArgumentParser(description="SEDR mouse stroke perturbation baseline")
    parser.add_argument("--cache_sham", required=True)
    parser.add_argument("--sham_dataset_name", default="Sham1-1")
    parser.add_argument("--roi_manifest", required=True)
    parser.add_argument("--deg_csv", required=True)
    parser.add_argument("--vocab_file", default=str(_REPO_ROOT / "gene_embedding" / "vocab.json"))
    parser.add_argument("--perturb_target_roi", choices=["ICA", "PIA_P", "PIA_D"], default="ICA")
    parser.add_argument("--n_spots", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_adj_thresh", type=float, default=0.05)
    parser.add_argument("--min_abs_logfc", type=float, default=0.0)
    parser.add_argument("--logfc_strength", type=float, default=1.0)
    parser.add_argument("--logfc_clip", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no_freeze_perturbed", action="store_true")
    parser.add_argument("--sedr_epochs", type=int, default=200)
    parser.add_argument("--sedr_lr", type=float, default=0.01)
    parser.add_argument("--sedr_weight_decay", type=float, default=0.01)
    parser.add_argument("--sedr_graph_knn_k", type=int, default=6)
    parser.add_argument("--sedr_rec_w", type=float, default=10.0)
    parser.add_argument("--sedr_gcn_w", type=float, default=0.1)
    parser.add_argument("--sedr_self_w", type=float, default=1.0)
    parser.add_argument("--sedr_mode", choices=["clustering", "imputation"], default="imputation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_expr_steps", default="")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    fix_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expr_dir = out_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)

    vocab = _load_vocab(args.vocab_file)
    adata, X_raw, gene_ids, _ = load_adata_and_prepare(Path(args.cache_sham), args.sham_dataset_name, vocab)

    model = train_sedr(
        X_raw,
        adata,
        epochs=args.sedr_epochs,
        lr=args.sedr_lr,
        weight_decay=args.sedr_weight_decay,
        graph_knn_k=args.sedr_graph_knn_k,
        rec_w=args.sedr_rec_w,
        gcn_w=args.sedr_gcn_w,
        self_w=args.sedr_self_w,
        device=args.device,
        mode=args.sedr_mode,
        seed=args.seed,
    )

    print("[STEP] Using raw Sham expression as X_0")
    x0 = X_raw.astype(np.float32).copy()

    manifest = _load_json(Path(args.roi_manifest))
    sham_key = "sham" if "sham" in manifest else "Sham1_1"
    sham_obs_names = list(adata.obs_names)
    roi_barcodes = {roi: manifest[sham_key][roi]["barcodes"] for roi in ["ICA", "PIA_P", "PIA_D"]}
    roi_indices = {roi: _barcodes_to_indices(sham_obs_names, barcodes) for roi, barcodes in roi_barcodes.items()}

    selected_indices = select_random_spots(roi_indices[args.perturb_target_roi], args.n_spots, args.seed)
    weights, weight_meta = compute_uniform_weights(selected_indices)
    x0_prime, perturb_meta = apply_deg_perturbation(
        x0,
        gene_ids,
        selected_indices,
        weights,
        Path(args.deg_csv),
        vocab,
        args.p_adj_thresh,
        args.min_abs_logfc,
        args.logfc_strength,
        args.logfc_clip,
    )
    frozen_indices = set(selected_indices) if not args.no_freeze_perturbed else set()

    with open(out_dir / "perturb_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "SEDR",
            "algorithm": "dual_line_error_cancellation",
            "x0_source": "raw_sham_expression",
            "perturb_target_roi": args.perturb_target_roi,
            "spot_selection": "random",
            "n_spots_selected": len(selected_indices),
            "selected_indices": selected_indices,
            "selected_barcodes": _indices_to_barcodes(sham_obs_names, selected_indices),
            "seed": args.seed,
            "weights": weights.tolist(),
            "freeze_perturbed": not args.no_freeze_perturbed,
            **weight_meta,
            **perturb_meta,
        }, f, indent=2)

    save_step_npz_from_matrix(x0, gene_ids, 0, expr_dir / "ctrl.npz")
    save_step_npz_from_matrix(x0_prime, gene_ids, 0, expr_dir / "step0.npz")
    step_expressions = run_dual_line_sedr(model, model.adj_norm, x0, x0_prime, frozen_indices, args.steps)
    for step, X_step in step_expressions.items():
        save_step_npz_from_matrix(X_step, gene_ids, 0, expr_dir / f"step{step}.npz")

    save_steps = parse_save_expr_steps(args.save_expr_steps, args.steps)
    if save_steps:
        if 0 in save_steps:
            save_full_expression_matrix_to_csv(x0_prime, gene_ids, sham_obs_names, expr_dir / "expression_step0.csv", vocab)
        for step, X_step in step_expressions.items():
            if step in save_steps:
                save_full_expression_matrix_to_csv(X_step, gene_ids, sham_obs_names, expr_dir / f"expression_step{step}.csv", vocab)

    print("[OK] Finished SEDR perturbation baseline")


if __name__ == "__main__":
    main()
