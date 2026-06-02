#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEDR baseline for cancer/healthy microenvironment perturbation.

Same experimental design as SpatialGT/KNN:
  - Select N paired spots (healthy center + cancer center)
  - Linearly interpolate k neighbors toward target niche

SEDR uses graph autoencoder: one forward pass directly predicts center spot
expression from the full perturbed raw expression matrix.

Output format is identical to the SpatialGT version for unified comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm
import anndata as ad

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
_SEDR_ROOT = _REPO_ROOT / "baseline" / "SEDR"
for _path in (_REPO_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR, _SEDR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pretrain.Config import Config
from pretrain.spatial_databank import SpatialDataBank
from SEDR.SEDR_model import Sedr
from SEDR.graph_func import graph_construction

from microenv_perturbation import (
    load_spot_annotations,
    get_hop_neighbors,
    select_spot_pairs,
    create_interpolated_perturbation,
    compute_spot_similarity,
    setup_databank,
)


# -----------------------------------------------------------------------------
# SEDR training / inference
# -----------------------------------------------------------------------------

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    seed: int = 42,
) -> Sedr:
    fix_seed(seed)
    graph_dict = graph_construction(adata, n=graph_knn_k, mode='KNN')
    sedr = Sedr(X=X, graph_dict=graph_dict, rec_w=rec_w, gcn_w=gcn_w,
                self_w=self_w, mode=mode, device=device)
    print(f"[SEDR] Training for {epochs} epochs...")
    sedr.train_without_dec(epochs=epochs, lr=lr, decay=weight_decay)
    return sedr



def sedr_reconstruct_with_mask(
    sedr: Sedr,
    X: np.ndarray,
    adj_norm: torch.Tensor,
    mask_indices: List[int],
) -> np.ndarray:
    """
    Mask specified spots with learned mask token, then run SEDR forward pass.
    Masked spots are reconstructed purely from neighbor information via GCN.
    """
    sedr.model.eval()
    sedr.model._mask_rate = 0.0

    X_tensor = torch.FloatTensor(X.copy()).to(sedr.device)

    mask_idx_t = torch.LongTensor(mask_indices).to(sedr.device)
    X_tensor[mask_idx_t] = X_tensor[mask_idx_t] + sedr.model.enc_mask_token

    with torch.no_grad():
        feat_x = sedr.model.encoder(X_tensor)

        hidden1 = sedr.model.gc1(feat_x, adj_norm)
        mu = sedr.model.gc2(hidden1, adj_norm)

        gnn_z = mu
        z = torch.cat((feat_x, gnn_z), 1)

        if hasattr(sedr.model, 'decoder'):
            try:
                de_feat = sedr.model.decoder(z, adj_norm)
            except TypeError:
                de_feat = sedr.model.decoder(z)
        else:
            de_feat = z

    recon = de_feat.detach().cpu().numpy()
    return np.clip(recon, 0, None).astype(np.float32)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_adata_expression(
    cache_dir: Path,
    dataset_name: str,
    vocab: Dict[str, int],
) -> Tuple[ad.AnnData, np.ndarray, np.ndarray, List[int]]:
    """Load processed.h5ad, extract X_log1p expression and map genes to vocab."""
    h5ad_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(h5ad_path))

    gene_names = list(adata.var_names)
    gene_ids, valid_gene_indices = [], []
    for i, gene in enumerate(gene_names):
        gid = vocab.get(gene.lower())
        if gid is not None:
            gene_ids.append(gid)
            valid_gene_indices.append(i)

    gene_ids = np.array(gene_ids, dtype=np.int64)

    if "X_log1p" in adata.layers:
        X = adata.layers["X_log1p"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X[:, valid_gene_indices].astype(np.float32)

    print(f"[DATA] {dataset_name}: {adata.n_obs} spots, {len(valid_gene_indices)} genes")
    return adata, X, gene_ids, valid_gene_indices


# -----------------------------------------------------------------------------
# Mapping between global_idx and adata row index
# -----------------------------------------------------------------------------

def get_dataset_start_idx(cache_dir: Path, dataset_name: str) -> int:
    meta_path = cache_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    for info in meta.get("dataset_indices", []):
        ds_idx = info.get("dataset_idx", 0)
        ds_meta = meta.get("datasets", [{}])[ds_idx]
        if ds_meta.get("name") == dataset_name:
            return int(info.get("start_idx", 0))
    return 0


# -----------------------------------------------------------------------------
# Single perturbation experiment (SEDR - 1 step)
# -----------------------------------------------------------------------------

def run_single_perturbation_sedr(
    databank: SpatialDataBank,
    sedr_model: Sedr,
    adj_norm: torch.Tensor,
    X_raw: np.ndarray,
    gene_ids: np.ndarray,
    start_idx: int,
    center_idx: int,
    target_center_idx: int,
    source_neighbors: List[int],
    target_neighbors: List[int],
    k: int,
    alpha: float,
    hop: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Run single SEDR perturbation experiment.
    SEDR is one-shot: perturb neighbors -> SEDR reconstruct -> extract center.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    center_local = center_idx - start_idx
    target_local = target_center_idx - start_idx

    # Raw reference (ground truth in log-normalized space)
    raw_center_vals = X_raw[center_local]
    raw_target_vals = X_raw[target_local]

    # Create perturbation: modify neighbor rows in expression matrix
    perturb_overrides = create_interpolated_perturbation(
        databank, source_neighbors, target_neighbors, k, alpha, seed)

    X_perturbed = X_raw.copy()
    for gidx, ov in perturb_overrides.items():
        local = gidx - start_idx
        if 0 <= local < X_perturbed.shape[0]:
            ov_gids = np.asarray(ov["gene_ids"], dtype=np.int64)
            ov_vals = np.asarray(ov["raw_normed_values"], dtype=np.float32)
            gid_to_col = {int(gid): col for col, gid in enumerate(gene_ids)}
            for j, gid in enumerate(ov_gids):
                gid_int = int(gid)
                if gid_int in gid_to_col:
                    X_perturbed[local, gid_to_col[gid_int]] = ov_vals[j]

    # Mask center spot -> SEDR reconstructs it purely from perturbed neighbors
    X_pred = sedr_reconstruct_with_mask(
        sedr_model, X_perturbed, adj_norm,
        mask_indices=[center_local])

    pred_center_vals = X_pred[center_local]

    # --- Metrics ---
    # vs_raw
    baseline_raw = compute_spot_similarity(
        raw_center_vals, gene_ids, raw_target_vals, gene_ids)
    baseline_raw["step"] = 0
    step1_raw = compute_spot_similarity(
        pred_center_vals, gene_ids, raw_target_vals, gene_ids)
    step1_raw["step"] = 1
    metrics_vs_raw = [baseline_raw, step1_raw]


    result = {
        "center_idx": center_idx,
        "target_center_idx": target_center_idx,
        "k": k, "alpha": alpha, "hop": hop, "steps": 1, "seed": seed,
        "n_perturbed_neighbors": len(perturb_overrides),
        "n_source_neighbors": len(source_neighbors),
        "n_target_neighbors": len(target_neighbors),
        "infer_mode": "SEDR",
        "metrics_vs_raw": metrics_vs_raw,
        "metrics": metrics_vs_raw,
        "convergence": [],
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    with open(out_dir / "convergence_report.json", "w") as f:
        json.dump({
            "mode": "one_shot",
            "steps": [],
            "final_step_mse": None,
        }, f, indent=2, default=str)

    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SEDR baseline for microenvironment perturbation")

    ap.add_argument("--dataset", required=True, choices=["1142243F", "1160920F", "HBRC", "PDAC"])
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--lmdb_path", default=None)
    ap.add_argument("--lmdb_manifest", default=None)
    ap.add_argument("--cache_mode", default="lmdb", choices=["h5", "lmdb"])
    ap.add_argument("--truth_csv", default=None)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=5)
    ap.add_argument("--hops", type=str, default="1,2")
    ap.add_argument("--k_values_1hop", type=str, default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--k_values_2hop", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    ap.add_argument("--alpha_values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min_purity_healthy", type=float, default=0.5)
    ap.add_argument("--min_purity_cancer", type=float, default=0.5)
    ap.add_argument("--min_neighbors", type=int, default=4)
    ap.add_argument("--exclusion_hops", type=int, default=4)

    ap.add_argument("--directions", type=str, default="cancer_to_healthy,healthy_to_cancer")
    ap.add_argument("--pair_idx", type=int, default=None)

    # SEDR hyperparameters
    ap.add_argument("--sedr_epochs", type=int, default=200)
    ap.add_argument("--sedr_lr", type=float, default=0.01)
    ap.add_argument("--sedr_weight_decay", type=float, default=0.01)
    ap.add_argument("--sedr_graph_knn_k", type=int, default=6)
    ap.add_argument("--sedr_rec_w", type=float, default=10.0)
    ap.add_argument("--sedr_gcn_w", type=float, default=0.1)
    ap.add_argument("--sedr_self_w", type=float, default=1.0)
    ap.add_argument("--sedr_mode", default="imputation", choices=["clustering", "imputation"])
    ap.add_argument("--device", default="cuda:0")

    args = ap.parse_args()

    hops = [int(h) for h in args.hops.split(",")]
    k_values_1hop = [int(x) for x in args.k_values_1hop.split(",")]
    k_values_2hop = [int(x) for x in args.k_values_2hop.split(",")]
    alpha_values = [float(x) for x in args.alpha_values.split(",")]
    directions = [d.strip() for d in args.directions.split(",")]

    config = Config()
    config.device = args.device

    print(f"[INFO] Dataset: {args.dataset}, Method: SEDR")
    print(f"[INFO] Pairs: {args.n_pairs}, Hops: {hops}")

    # -- DataBank (for pair selection + neighbor topology + perturbation) --
    databank = setup_databank(
        config, args.cache_dir, args.dataset,
        lmdb_path=args.lmdb_path, lmdb_manifest=args.lmdb_manifest,
        cache_mode=args.cache_mode)

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

    # -- Load expression matrix for SEDR --
    vocab_path = config.vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {k.lower(): int(v) for k, v in vocab_json.items()}

    adata, X_raw, gene_ids, valid_gene_indices = load_adata_expression(
        Path(args.cache_dir), args.dataset, vocab)

    start_idx = get_dataset_start_idx(Path(args.cache_dir), args.dataset)
    print(f"[INFO] start_idx={start_idx}, X shape={X_raw.shape}")

    # -- Train SEDR --
    print("\n[SEDR] Training on raw expression...")
    sedr_model = train_sedr(
        X_raw, adata,
        epochs=args.sedr_epochs, lr=args.sedr_lr,
        weight_decay=args.sedr_weight_decay,
        graph_knn_k=args.sedr_graph_knn_k,
        rec_w=args.sedr_rec_w, gcn_w=args.sedr_gcn_w, self_w=args.sedr_self_w,
        device=args.device, mode=args.sedr_mode, seed=args.seed)


    # -- Run experiments --
    total_exps = 0
    for h in hops:
        k_vals = k_values_1hop if h == 1 else k_values_2hop
        total_exps += len(pair_range) * len(directions) * len(k_vals) * len(alpha_values)
    print(f"[INFO] Total experiments: {total_exps}")

    all_results = []
    exp_count = 0

    for pi in pair_range:
        pair = pairs[pi]
        healthy_center = pair["healthy_center"]
        cancer_center = pair["cancer_center"]

        for direction in directions:
            center_idx = cancer_center if direction == "cancer_to_healthy" else healthy_center
            target_idx = healthy_center if direction == "cancer_to_healthy" else cancer_center

            for hop in hops:
                hop1, hop2 = get_hop_neighbors(databank, center_idx, hop)
                source_neighbors = hop1 if hop == 1 else hop1 + hop2
                t_hop1, t_hop2 = get_hop_neighbors(databank, target_idx, hop)
                target_neighbors = t_hop1 if hop == 1 else t_hop1 + t_hop2
                k_vals = k_values_1hop if hop == 1 else k_values_2hop

                for k in k_vals:
                    if k > len(source_neighbors):
                        continue
                    for alpha_val in alpha_values:
                        exp_count += 1
                        exp_tag = f"pair{pi}/{direction}/hop{hop}_k{k}_alpha{alpha_val:.1f}"
                        exp_out = (out_base / args.dataset / direction /
                                   f"pair{pi}" / f"hop{hop}_k{k}_alpha{alpha_val:.1f}")

                        if (exp_out / "metrics.json").exists():
                            print(f"[SKIP] {exp_tag}")
                            with open(exp_out / "metrics.json") as f:
                                all_results.append(json.load(f))
                            continue

                        print(f"[{exp_count}/{total_exps}] {exp_tag}")
                        result = run_single_perturbation_sedr(
                            databank=databank,
                            sedr_model=sedr_model,
                            adj_norm=sedr_model.adj_norm,
                            X_raw=X_raw,
                            gene_ids=gene_ids,
                            start_idx=start_idx,
                            center_idx=center_idx,
                            target_center_idx=target_idx,
                            source_neighbors=source_neighbors,
                            target_neighbors=target_neighbors,
                            k=k, alpha=alpha_val, hop=hop,
                            seed=args.seed, out_dir=exp_out)

                        result["pair_idx"] = pi
                        result["direction"] = direction
                        result["dataset"] = args.dataset
                        all_results.append(result)

                        with open(exp_out / "manifest.json", "w") as f:
                            json.dump({
                                "dataset": args.dataset, "direction": direction,
                                "pair_idx": pi, "hop": hop, "k": k, "alpha": alpha_val,
                                "center_idx": center_idx, "target_center_idx": target_idx,
                                "infer_mode": "SEDR", "seed": args.seed,
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
