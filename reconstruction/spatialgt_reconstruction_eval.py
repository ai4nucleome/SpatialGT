#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-pass random-mask reconstruction on log-normalized expression."""

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
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr
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
from model_spatialpt import SpatialNeighborTransformer
from pretrain.checkpoint import load_model_checkpoint
from utils_train import process_batch, forward_pass


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_seed(seed: int = 42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_dataset_start_idx(cache_dir: Path, dataset_name: str, lmdb_manifest_path: Optional[str] = None) -> int:
    """Get dataset start index from metadata.json or LMDB manifest."""

    if lmdb_manifest_path:
        manifest_path = Path(lmdb_manifest_path)
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            meta = manifest.get("metadata", {})
            datasets = meta.get("datasets", manifest.get("datasets", []))
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


    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return 0
    meta = _load_json(meta_path)
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


def _get_dataset_info(cache_dir: Path, dataset_name: str, lmdb_manifest_path: Optional[str] = None) -> Tuple[int, int]:
    """Get (start_idx, n_spots) for a dataset from metadata or LMDB manifest."""

    if lmdb_manifest_path:
        manifest_path = Path(lmdb_manifest_path)
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            meta = manifest.get("metadata", {})
            datasets = meta.get("datasets", manifest.get("datasets", []))


            for ds_info in datasets:
                if ds_info.get("name") == dataset_name:
                    n_spots = ds_info.get("n_spots", 0)
                    break
            else:
                n_spots = manifest.get("total_spots", 0)


            dataset_idx = None
            for i, ds in enumerate(datasets):
                if ds.get("name") == dataset_name:
                    dataset_idx = i
                    break
            if dataset_idx is None:
                dataset_idx = 0
            start_idx = 0
            for info in meta.get("dataset_indices", []):
                if info.get("dataset_idx") == dataset_idx:
                    start_idx = int(info.get("start_idx", 0))
                    break

            return start_idx, n_spots


    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    meta = _load_json(meta_path)

    datasets = meta.get("datasets", [])
    n_spots = 0
    for ds_info in datasets:
        if ds_info.get("name") == dataset_name:
            n_spots = ds_info.get("n_spots", 0)
            break


    dataset_idx = None
    for i, ds in enumerate(datasets):
        if ds.get("name") == dataset_name:
            dataset_idx = i
            break
    if dataset_idx is None:
        dataset_idx = 0
    start_idx = 0
    for info in meta.get("dataset_indices", []):
        if info.get("dataset_idx") == dataset_idx:
            start_idx = int(info.get("start_idx", 0))
            break

    return start_idx, n_spots


def _setup_config_for_cache(
    config: Config,
    cache_dir: str,
    dataset_name: str,
    cache_mode: str = "h5",
    lmdb_path: Optional[str] = None,
    lmdb_manifest_path: Optional[str] = None,
) -> Config:
    """Configure Config object for the specified cache mode."""
    config.cache_dir = str(cache_dir)
    config.cache_mode = cache_mode.lower()
    config.strict_cache_only = True
    config.max_seq_len = getattr(config, "max_seq_len", 3000)
    config.mask_value = getattr(config, "mask_value", -1)
    config.pad_value = getattr(config, "pad_value", -2)
    config.pad_token = getattr(config, "pad_token", "[PAD]")
    config.cls_token = getattr(config, "cls_token", "[CLS]")
    config.use_metadata_vocabs = getattr(config, "use_metadata_vocabs", False)
    config.log_vocab_messages = getattr(config, "log_vocab_messages", False)
    config.data_is_raw = getattr(config, "data_is_raw", True)
    config.input_style = getattr(config, "input_style", "log1p")
    config.include_zero_gene = getattr(config, "include_zero_gene", True)
    config.filter_gene_by_counts = getattr(config, "filter_gene_by_counts", False)
    config.subset_hvg = getattr(config, "subset_hvg", 3000)

    if cache_mode.lower() == "lmdb":
        if lmdb_path:
            config.lmdb_path = str(lmdb_path)
            config.runtime_lmdb_path = str(lmdb_path)
        if lmdb_manifest_path:
            config.lmdb_manifest_path = str(lmdb_manifest_path)
            config.runtime_lmdb_manifest_path = str(lmdb_manifest_path)

    return config


def load_adata_and_expression(cache_dir: Path, dataset_name: str) -> Tuple[ad.AnnData, np.ndarray]:
    """Load processed.h5ad and extract expression matrix."""
    h5ad_path = cache_dir / dataset_name / "processed.h5ad"
    adata = ad.read_h5ad(str(h5ad_path))

    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    print(f"[DATA] Loaded {dataset_name}: {adata.n_obs} spots, {adata.n_vars} genes")

    return adata, X


def select_random_spots(
    all_global_indices: List[int],
    n_spots: int,
    seed: int,
) -> List[int]:
    """Randomly sample n_spots from all spots."""
    rng = random.Random(seed)
    if n_spots < 1 or n_spots > len(all_global_indices):
        raise ValueError("n_spots must be between one and the number of section spots")
    return rng.sample(all_global_indices, n_spots)


def _build_all_loader(
    databank: SpatialDataBank,
    config: Config,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    """Build data loader for full-slice inference."""
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


def run_single_pass_inference(
    databank: SpatialDataBank,
    loader,
    model: nn.Module,
    device: torch.device,
    desc: str = "Inference",
    show_progress: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run a single forward pass on the entire slice."""
    all_overrides: Dict[int, Dict[str, np.ndarray]] = {}

    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    iterator = tqdm(loader, desc=desc, disable=not show_progress)

    skipped_batches = 0
    processed_spots = 0

    for batch in iterator:
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            skipped_batches += 1
            continue


        structure = batch.get("structure", {})
        centers_global = structure.get("centers_global_indices", None)

        if centers_global is None:

            centers_global = structure.get("center_global_indices", None)

        if centers_global is None:

            centers_global = batch.get("centers_global_indices", None)

        if isinstance(centers_global, torch.Tensor):
            centers_global = centers_global.cpu().numpy().astype(int).tolist()
        elif centers_global is None:

            skipped_batches += 1
            continue

        batch_data = process_batch(batch, device, config=databank.config)

        with torch.no_grad():
            preds, _, _, _ = forward_pass(model, batch_data, config=databank.config)

        B = len(centers_global)
        center_gene_ids = batch_data["genes"][:B]
        center_pad_mask = batch_data["padding_attention_mask"][:B].to(torch.bool)

        gathered = preds.gather(1, center_gene_ids.clamp(min=0))
        gathered = gathered * center_pad_mask.to(gathered.dtype)

        gid_np = center_gene_ids.detach().cpu().numpy().astype(np.int64)
        val_np = gathered.detach().float().cpu().numpy().astype(np.float32)
        mask_np = center_pad_mask.detach().cpu().numpy()

        for i in range(B):
            gidx = int(centers_global[i])

            valid_mask = mask_np[i]
            valid_gids = gid_np[i][valid_mask]
            valid_vals = val_np[i][valid_mask]
            all_overrides[gidx] = {"gene_ids": valid_gids, "raw_normed_values": valid_vals}
            processed_spots += 1

    if skipped_batches > 0:
        print(f"[WARN] {desc}: Skipped {skipped_batches} batches")
    print(f"[INFO] {desc}: Processed {processed_spots} spots, collected {len(all_overrides)} unique overrides")

    return all_overrides


def run_masked_reconstruction(
    databank: SpatialDataBank,
    loader_factory: callable,
    model: nn.Module,
    device: torch.device,
    all_global_indices: List[int],
    masked_global_indices: Set[int],
    show_progress: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Zero all hidden rows before a single full-section forward pass."""
    all_indices = set(map(int, all_global_indices))
    masked = set(map(int, masked_global_indices))
    if not masked or not masked.issubset(all_indices):
        raise ValueError("The mask must be a nonempty subset of the section spots")
    databank.clear_runtime_spot_overrides()
    state = {}
    for gidx in all_global_indices:
        row = databank.get_spot_data(int(gidx))
        values = np.asarray(row["raw_normed_values"], dtype=np.float32)
        state[int(gidx)] = {
            "gene_ids": np.asarray(row["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.zeros_like(values) if int(gidx) in masked else values.copy(),
        }
    databank.set_runtime_spot_overrides(state)
    try:
        predictions = run_single_pass_inference(
            databank, loader_factory(), model, device,
            desc="Reconstruction (first forward)", show_progress=show_progress,
        )
    finally:
        databank.clear_runtime_spot_overrides()
    missing = masked.difference(predictions)
    if missing:
        raise RuntimeError(f"No prediction returned for {len(missing)} masked spots")
    return {gidx: predictions[gidx] for gidx in all_global_indices if gidx in masked}


def compute_spotwise_metrics(
    true_data: Dict[int, Dict[str, np.ndarray]],
    pred_data: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """Compute spot-wise PCC and RMSE for reconstructed masked spots."""
    pcc_list = []
    rmse_list = []

    for gidx in true_data:
        if gidx not in pred_data:
            continue

        true_gids = true_data[gidx]["gene_ids"]
        true_vals = true_data[gidx]["raw_normed_values"]
        pred_gids = pred_data[gidx]["gene_ids"]
        pred_vals = pred_data[gidx]["raw_normed_values"]

        if len(true_gids) == len(pred_gids) and np.array_equal(true_gids, pred_gids):
            true_vec = np.asarray(true_vals, dtype=np.float32)
            pred_vec = np.asarray(pred_vals, dtype=np.float32)
        else:
            true_gid_set = {int(g) for g in true_gids if g >= 0}
            pred_gid_set = {int(g) for g in pred_gids if g >= 0}
            common = sorted(true_gid_set & pred_gid_set)
            if not common:
                continue
            true_pos = {int(g): i for i, g in enumerate(true_gids)}
            pred_pos = {int(g): i for i, g in enumerate(pred_gids)}
            true_vec = np.array([true_vals[true_pos[g]] for g in common], dtype=np.float32)
            pred_vec = np.array([pred_vals[pred_pos[g]] for g in common], dtype=np.float32)

        if np.std(true_vec) > 1e-8 and np.std(pred_vec) > 1e-8:
            pcc, _ = pearsonr(true_vec, pred_vec)
        else:
            pcc = 0.0
        rmse = float(np.sqrt(np.mean((true_vec - pred_vec) ** 2)))
        pcc_list.append(float(pcc))
        rmse_list.append(rmse)

    return {
        "pcc": float(np.nanmean(pcc_list)) if pcc_list else float("nan"),
        "rmse": float(np.nanmean(rmse_list)) if rmse_list else float("nan"),
        "pcc_std": float(np.nanstd(pcc_list)) if pcc_list else float("nan"),
        "rmse_std": float(np.nanstd(rmse_list)) if rmse_list else float("nan"),
    }

def align_output_gene_names(gene_ids, vocabulary, section_gene_names):
    """Name exported columns by token ID, retaining section-specific gene casing."""
    lower_to_id = {}
    id_to_name = {}
    for name, token in vocabulary.items():
        lower_to_id.setdefault(str(name).lower(), int(token))
        id_to_name.setdefault(int(token), str(name))
    section_names = {}
    for name in section_gene_names:
        token = lower_to_id.get(str(name).lower())
        if token is not None:
            section_names.setdefault(token, str(name))
    names = []
    for token in gene_ids:
        token = int(token)
        if token not in id_to_name:
            raise ValueError(f"Output gene token {token} is absent from the checkpoint vocabulary")
        names.append(section_names.get(token, id_to_name[token]))
    return names


def main():
    ap = argparse.ArgumentParser(description="SpatialGT reconstruction evaluation")


    ap.add_argument("--ckpt", type=str, required=True, help="Path to SpatialGT checkpoint")


    ap.add_argument("--cache_dir", type=str, required=True, help="Path to preprocessed cache directory")
    ap.add_argument("--dataset_name", type=str, required=True, help="Dataset name (subdirectory in cache_dir)")
    ap.add_argument("--cache_mode", type=str, default="h5", choices=["h5", "lmdb"])
    ap.add_argument("--lmdb_path", type=str, default=None)
    ap.add_argument("--lmdb_manifest", type=str, default=None)


    ap.add_argument("--mask_mode", default="random", choices=["random"],
                    help="Random masking of complete spot expression profiles")
    ap.add_argument("--n_spots", type=int, default=50, help="Number of spots to mask")


    ap.add_argument("--steps", type=int, default=1, choices=[1],
                    help="Exactly one forward pass is used for reconstruction")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor")
    ap.add_argument("--persistent_workers", action="store_true", help="Use persistent workers")

    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs (comma-separated)")

    args = ap.parse_args()
    if args.n_spots < 1:
        ap.error("--n_spots must be positive")


    fix_seed(args.seed)


    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")


    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt)


    config = Config()
    config = _setup_config_for_cache(
        config, str(cache_dir), args.dataset_name, args.cache_mode,
        args.lmdb_path, args.lmdb_manifest
    )


    print(f"[INFO] Initializing databank...")


    dataset_paths = [str(cache_dir / args.dataset_name / "processed.h5ad")]
    databank = SpatialDataBank(
        dataset_paths=dataset_paths,
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False,
    )


    start_idx, n_spots = _get_dataset_info(
        cache_dir, args.dataset_name,
        lmdb_manifest_path=args.lmdb_manifest if args.cache_mode == "lmdb" else None
    )
    all_global_indices = list(range(start_idx, start_idx + n_spots))
    print(f"[INFO] Dataset has {n_spots} spots (global indices {start_idx} to {start_idx + n_spots - 1})")
    print(f"[INFO] Cache mode: {args.cache_mode}")

    if args.n_spots > n_spots:
        summary = {"status": "skipped", "reason": "requested mask exceeds section size",
                   "n_spots_requested": args.n_spots, "n_spots_available": n_spots}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[SKIP] Requested {args.n_spots} masked spots; section has {n_spots}.")
        return
    mask_indices = select_random_spots(all_global_indices, args.n_spots, args.seed)
    mask_set = set(mask_indices)


    true_data: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in mask_indices:
        sd = databank.get_spot_data(int(gidx))
        true_data[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
        }


    print(f"\n[INFO] Loading SpatialGT model from {ckpt_dir}...")
    model = load_model_checkpoint(ckpt_dir, config, device=device)
    model.eval()


    def loader_factory():
        return _build_all_loader(
            databank, config, args.batch_size, args.num_workers,
            args.prefetch_factor, args.persistent_workers
        )

    pred_data = run_masked_reconstruction(
        databank, loader_factory, model, device, all_global_indices, mask_set,
    )
    metrics = compute_spotwise_metrics(true_data, pred_data)
    results_rows = [{"step": 1, "metric": metric, "value": metrics[metric],
                     "std": metrics.get(f"{metric}_std", float("nan"))}
                    for metric in ("pcc", "rmse")]
    print(f"  First forward: PCC={metrics['pcc']:.4f}, RMSE={metrics['rmse']:.4f}")


    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(out_dir / "summary.csv", index=False)


    manifest = {
        "task": "reconstruction",
        "eval_script": "spatialgt_reconstruction_eval.py",
        "baseline_method": "SpatialGT",
        "dataset": args.dataset_name,
        "mask_mode": args.mask_mode,
        "n_spots": len(mask_indices),
        "n_spots_requested": args.n_spots,
        "steps": args.steps,
        "inference_passes": 1,
        "prediction_step": 1,
        "seed": args.seed,
        "device": str(device),
        "mask_indices": mask_indices,
        "masked_spot_value": "zero",
        "masking_note": "Masked spots are zero-initialized when acting as neighbors",
    }

    with open(out_dir / "reconstruction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


    final_metrics = metrics
    summary_dict = {
        "final_step": args.steps,
        "final_metrics": {
            m: final_metrics[m] for m in ["pcc", "rmse"]
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    ordered_mask = list(mask_indices)
    X_true_matrix = np.stack([true_data[gidx]["raw_normed_values"] for gidx in ordered_mask])
    gene_ids_arr = true_data[ordered_mask[0]]["gene_ids"]
    predicted_rows = []
    for gidx in ordered_mask:
        row = pred_data[gidx]
        if not np.array_equal(true_data[gidx]["gene_ids"], gene_ids_arr):
            raise ValueError("Masked rows have inconsistent gene panels")
        positions = {int(g): i for i, g in enumerate(row["gene_ids"])}
        if any(int(g) not in positions for g in gene_ids_arr):
            raise ValueError("Prediction is missing a gene in the evaluated panel")
        predicted_rows.append([row["raw_normed_values"][positions[int(g)]] for g in gene_ids_arr])
    adata_for_names = ad.read_h5ad(str(cache_dir / args.dataset_name / "processed.h5ad"), backed="r")
    try:
        section_gene_names = (adata_for_names.var["gene_name"].tolist()
                              if "gene_name" in adata_for_names.var else list(adata_for_names.var_names))
    finally:
        adata_for_names.file.close()
    with open(config.vocab_file, encoding="utf-8") as handle:
        gene_names = align_output_gene_names(gene_ids_arr, json.load(handle), section_gene_names)
    np.savez_compressed(
        out_dir / "reconstruction_expressions.npz",
        ground_truth=X_true_matrix, mask_indices=np.asarray(ordered_mask, dtype=np.int64),
        gene_ids=gene_ids_arr, gene_names=np.asarray(gene_names, dtype=str),
        step_1=np.asarray(predicted_rows, dtype=np.float32),
    )


    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)

    print(f"[INFO] Saved first-forward predictions: "
          f"{X_true_matrix.shape[0]} spots x {X_true_matrix.shape[1]} genes to {out_dir}")

    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()

