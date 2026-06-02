#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Endometrial cancer virtual-treatment perturbation with SpatialGT.

The script applies treatment-associated tumor DEG log-fold changes to selected
pre-treatment tumor spots and propagates the edited state through the spatial
graph with dual-line iterative inference. The initial control state is the
SpatialGT reconstruction of the unedited section, matching the perturbation
setting used for the EC analysis in the paper.

Output structure:
  {out_dir}/
    expression/ctrl.npz
    expression/step0.npz
    expression/step1.npz ...
    convergence/step_mse.csv
    convergence/convergence_report.json
    perturb_manifest.json
    summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import anndata as ad

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
for _path in (_REPO_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pretrain.Config import Config
from pretrain.spatial_databank import SpatialDataBank
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils_train import process_batch, forward_pass


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DATA_ROOT = _REPO_ROOT / "example_data" / "EC" / "GSE225691"
DEFAULT_FINETUNE_ROOT = _REPO_ROOT / "model" / "finetune"
DEG_DIR = _SCRIPT_DIR / "deg"

DATA_ROOT = str(DEFAULT_DATA_ROOT)
LMDB_ROOT = str(DEFAULT_DATA_ROOT / "lmdb")
PREPROCESSED_ROOT = str(DEFAULT_DATA_ROOT / "preprocessed_v3")
RAW_DATA_ROOT = str(DEFAULT_DATA_ROOT)

DEG_FILES = {
    "P034": "P034_perturbation_DEGs.csv",
    "P039": "P039_perturbation_DEGs.csv",
}

PATIENTS = {
    "P034": {
        "pre": "01_034_C1d1",
        "post": "01_034_C3d1",
    },
    "P039": {
        "pre": "01_039_C1d1",
        "post": "01_039_C3d1",
    },
}

# ============================================================================
# Utility functions
# ============================================================================

def load_vocab(vocab_path: str = None) -> Dict[str, int]:
    if vocab_path is None:
        vocab_path = Config().vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {k.lower(): int(v) for k, v in vocab.items()}


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
            print(f"[OK] Model loaded from {fpath}")
            return model

    raise FileNotFoundError(f"No model weights in {ckpt_dir}")


def setup_databank(
    config: Config,
    cache_dir: str,
    sample_name: str,
    lmdb_path: str,
    lmdb_manifest: str,
) -> SpatialDataBank:
    config.cache_dir = cache_dir
    config.cache_mode = "lmdb"
    config.strict_cache_only = True
    config.lmdb_path = lmdb_path
    config.runtime_lmdb_path = lmdb_path
    config.lmdb_manifest_path = lmdb_manifest
    config.runtime_lmdb_manifest_path = lmdb_manifest

    proc_path = str(Path(cache_dir) / sample_name / "processed.h5ad")
    bank = SpatialDataBank(
        dataset_paths=[proc_path],
        cache_dir=cache_dir,
        config=config,
        force_rebuild=False,
    )
    return bank


def build_loader(databank, config, batch_size=64, num_workers=4):
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


def run_full_inference(
    databank: SpatialDataBank,
    loader,
    model: nn.Module,
    device: torch.device,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    """Full-slice forward pass -> {global_idx: {gene_ids, raw_normed_values}}."""
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


# ============================================================================
# Annotation and DEG loading
# ============================================================================

def load_tumor_annotation(sample_name: str) -> pd.DataFrame:
    """
    Load spot-level celltype annotation from raw h5ad.
    Returns DataFrame with spot_idx, barcode, cell_type, is_tumor.
    """
    candidates = [
        os.path.join(RAW_DATA_ROOT, sample_name, sample_name, "outs", f"{sample_name}.h5ad"),
        os.path.join(RAW_DATA_ROOT, sample_name, f"{sample_name}.h5ad"),
    ]
    adata = None
    for p in candidates:
        if os.path.exists(p):
            adata = ad.read_h5ad(p)
            break
    if adata is None:
        raise FileNotFoundError(
            f"Raw h5ad not found for {sample_name}. Tried: {candidates}"
        )

    ct_col = None
    for col_name in ["celltype_relaxed", "celltype", "cell_type"]:
        if col_name in adata.obs.columns:
            ct_col = col_name
            break
    if ct_col is None:
        raise ValueError(
            f"No celltype column in {sample_name}. "
            f"Available: {list(adata.obs.columns)}"
        )

    df = pd.DataFrame({
        "spot_idx": np.arange(adata.n_obs),
        "barcode": adata.obs_names,
        "cell_type": adata.obs[ct_col].values,
    })
    df["is_tumor"] = df["cell_type"].str.lower().str.contains("tumor")

    n_tumor = df["is_tumor"].sum()
    print(f"[OK] Annotation for {sample_name}: {len(df)} spots, "
          f"{n_tumor} tumor ({ct_col})")
    return df


def load_deg(
    deg_file: str,
    p_thresh: float = 0.05,
    min_abs_logfc: float = 0.0,
) -> pd.DataFrame:
    """Load DEG file and filter by significance thresholds."""
    if not os.path.exists(deg_file):
        raise FileNotFoundError(f"DEG file not found: {deg_file}")

    deg = pd.read_csv(deg_file)

    if "avg_logFC" not in deg.columns and "logFC" in deg.columns:
        deg["avg_logFC"] = deg["logFC"]

    deg = deg[np.isfinite(deg["avg_logFC"].astype(float))]

    for p_col in ["p_val_adj", "padj_034", "padj_039", "p_val"]:
        if p_col in deg.columns:
            deg[p_col] = pd.to_numeric(deg[p_col], errors="coerce")
            deg = deg[np.isfinite(deg[p_col])]
            deg = deg[deg[p_col].astype(float) < p_thresh]
            break

    if min_abs_logfc > 0:
        deg = deg[deg["avg_logFC"].abs() >= min_abs_logfc]

    up = (deg["avg_logFC"] > 0).sum()
    down = (deg["avg_logFC"] < 0).sum()
    print(f"[DEG] {len(deg)} genes from {Path(deg_file).name} "
          f"(UP: {up}, DOWN: {down})")
    return deg


def build_perturbation_vector(
    deg: pd.DataFrame,
    vocab: Dict[str, int],
    logfc_clip: float = 5.0,
) -> Dict[int, float]:
    """Map DEG gene names -> vocab IDs with clipped logFC."""
    gene_logfc: Dict[int, float] = {}
    for _, row in deg.iterrows():
        gene_name = str(row["gene"]).lower()
        gid = vocab.get(gene_name)
        if gid is not None:
            logfc = float(row["avg_logFC"])
            gene_logfc[int(gid)] = float(np.clip(logfc, -logfc_clip, logfc_clip))
    print(f"[DEG] Mapped {len(gene_logfc)}/{len(deg)} genes to vocab")
    return gene_logfc


# ============================================================================
# Spot selection and perturbation
# ============================================================================

def select_tumor_spots(
    annotation: pd.DataFrame,
    n_spots: int,
    seed: int,
) -> List[int]:
    """
    Select tumor spots for perturbation.
    n_spots=0 selects all tumor spots.
    """
    tumor_indices = annotation[annotation["is_tumor"]]["spot_idx"].tolist()

    if n_spots <= 0 or n_spots >= len(tumor_indices):
        print(f"[INFO] Selecting ALL {len(tumor_indices)} tumor spots")
        return tumor_indices

    rng = random.Random(seed)
    selected = rng.sample(tumor_indices, n_spots)
    print(f"[INFO] Selected {len(selected)}/{len(tumor_indices)} tumor spots "
          f"(seed={seed})")
    return selected


def apply_perturbation(
    databank: SpatialDataBank,
    selected_indices: List[int],
    gene_logfc: Dict[int, float],
    dose_lambda: float = 1.0,
    base_state: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Apply DEG perturbation to selected spots.
    Formula: new_val = old_val x 2^(logFC x lambda)

    When `base_state` is provided, the edit is applied in the same
    reconstructed state space used for ctrl.npz and downstream dual-line
    propagation.
    """
    overrides: Dict[int, Dict[str, np.ndarray]] = {}
    n_hits_list = []
    base_source = "reconstructed x0_state" if base_state is not None else "raw databank"

    for gidx in selected_indices:
        gidx_i = int(gidx)
        if base_state is not None and gidx_i in base_state:
            sd = base_state[gidx_i]
        else:
            sd = databank.get_spot_data(gidx_i)
        gene_ids = np.asarray(sd["gene_ids"], dtype=np.int64)
        vals = np.asarray(sd["raw_normed_values"], dtype=np.float32).copy()

        n_hits = 0
        for j, gid in enumerate(gene_ids):
            if int(gid) in gene_logfc:
                logfc = gene_logfc[int(gid)]
                fold_change = np.power(2.0, logfc * dose_lambda)
                linear_val = np.expm1(vals[j])
                vals[j] = np.log1p(max(linear_val * fold_change, 0.0))
                n_hits += 1

        vals = np.maximum(vals, 0.0)
        n_hits_list.append(n_hits)
        overrides[int(gidx)] = {
            "gene_ids": gene_ids,
            "raw_normed_values": vals,
        }

    meta = {
        "dose_lambda": dose_lambda,
        "n_perturbed_spots": len(selected_indices),
        "n_deg_genes": len(gene_logfc),
        "mean_hits_per_spot": float(np.mean(n_hits_list)) if n_hits_list else 0,
        "max_hits_per_spot": int(np.max(n_hits_list)) if n_hits_list else 0,
        "base_source": base_source,
    }
    print(f"[OK] Perturbation applied: {meta['n_perturbed_spots']} spots, "
          f"mean {meta['mean_hits_per_spot']:.1f} gene hits/spot, "
          f"base={base_source}")
    return overrides, meta


# ============================================================================
# Convergence detection
# ============================================================================

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




# ============================================================================
# Expression saving
# ============================================================================

def save_step_npz(
    state: Dict[int, Dict[str, np.ndarray]],
    path: Path,
):
    """Save one step's expression to compressed npz."""
    spot_indices = sorted(state.keys())
    save_dict = {"spot_indices": np.array(spot_indices, dtype=np.int64)}

    for gidx in spot_indices:
        ov = state[gidx]
        save_dict[f"gids_{gidx}"] = ov["gene_ids"]
        save_dict[f"vals_{gidx}"] = ov["raw_normed_values"]

    np.savez_compressed(str(path), **save_dict)


def load_step_npz(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """Load expression from npz."""
    data = np.load(str(path), allow_pickle=True)
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("gids_"):
            gidx = int(key[5:])
            result[gidx] = {
                "gene_ids": data[f"gids_{gidx}"],
                "raw_normed_values": data[f"vals_{gidx}"],
            }
    return result


# ============================================================================
# Dual-line iterative inference with convergence
# ============================================================================

def run_dual_line_with_convergence(
    databank: SpatialDataBank,
    loader_factory,
    model: nn.Module,
    device: torch.device,
    max_steps: int,
    frozen_indices: Set[int],
    x0_prime_base: Dict[int, Dict[str, np.ndarray]],
    x0_prime_state: Dict[int, Dict[str, np.ndarray]],
    x_line: Dict[int, Dict[str, np.ndarray]],
    expr_dir: Path,
    conv_dir: Path,
    stopping_mode: str = "patience",
    patience: int = 3,
    emergency_div_factor: float = 3.0,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Dual-line iterative inference with MSE-minimum stopping.

    The step-wise MSE follows a U-shaped curve: initial descent as the
    perturbation is absorbed by the spatial graph, followed by a rise
    as model error accumulates. The optimal output is at the MSE trough.

    Stopping modes:
    - "patience": stop after `patience` steps without MSE improvement;
      recommend the step with minimum MSE (U-curve trough).
    - "retrospective": run all `max_steps`; recommend the step with
      minimum MSE.

    Both modes include emergency protection: immediate stop if
    step MSE > emergency_div_factor x best_mse.

    Returns (final_state, convergence_report).
    """
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

        # --- X' line: SpatialGT(X'_{t-1}_true) ---
        databank.clear_runtime_spot_overrides()
        databank.set_runtime_spot_overrides(xp_current)
        loader = loader_factory()
        xp_k_raw = run_full_inference(
            databank, loader, model, device, desc=f"X' step {t}"
        )

        # --- delta_k and X'_k_true ---
        xp_k_true: Dict[int, Dict[str, np.ndarray]] = {}
        for gidx in xp_k_raw:
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

        # --- Step MSE ---
        step_mse, n_compared = compute_step_mse(
            xp_k_true, prev_state, frozen_indices
        )
        mse_history.append(step_mse)

        # --- Track MSE minimum (U-curve trough) ---
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
            "BEST BEST" if steps_since_improvement == 0
            else f"patience {steps_since_improvement}/{patience}"
        )
        print(f"  MSE(step {t-1}->{t}) = {step_mse:.8f}  "
              f"[best={best_mse:.8f} @ step {best_step}]  {status_mark}")

        # --- Save step expression ---
        save_step_npz(xp_k_true, expr_dir / f"step{t}.npz")
        print(f"  Saved step{t}.npz ({len(xp_k_true)} spots)")

        prev_state = xp_k_true
        xp_current = xp_k_true
        actual_steps = t

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Emergency divergence protection ---
        if best_mse > 0 and step_mse > emergency_div_factor * best_mse:
            final_status = "emergency_stop"
            final_message = (
                f"Emergency stop: MSE {step_mse:.6f} exceeded "
                f"{emergency_div_factor}x best MSE {best_mse:.6f}"
            )
            print(f"\n[EMERGENCY STOP] {final_message}")
            break

        # --- Patience-based early stop ---
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
    print(f"         Use expression/step{best_step}.npz "
          f"for downstream analysis")
    print(f"{'='*50}")

    # --- Save convergence report ---
    mse_df = pd.DataFrame(mse_records)
    mse_df.to_csv(conv_dir / "step_mse.csv", index=False)

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
    with open(conv_dir / "convergence_report.json", "w") as f:
        json.dump(conv_report, f, indent=2)

    return xp_current, conv_report


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="EC Virtual Treatment Perturbation with SpatialGT"
    )

    # --- required ---
    ap.add_argument("--patient", required=True, choices=["P034", "P039"])
    ap.add_argument("--out_dir", required=True)

    # --- model ---
    ap.add_argument("--ckpt_dir", default=None,
                    help="Override model checkpoint (default: FINETUNE_ROOT/sample)")

    # --- perturbation control ---
    ap.add_argument("--n_spots", type=int, default=0,
                    help="Number of tumor spots to perturb (0 = all)")
    ap.add_argument("--dose_lambda", type=float, default=1.0,
                    help="Treatment intensity multiplier")
    ap.add_argument("--logfc_clip", type=float, default=5.0)

    # --- DEG source ---
    ap.add_argument("--deg_file", default=None,
                    help="Custom DEG file path; defaults to the patient-specific file in ./deg")

    # --- iteration & stopping ---
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--stopping_mode", default="patience",
                    choices=["patience", "retrospective"],
                    help="patience: early stop after N steps w/o MSE improvement; "
                         "retrospective: run all steps, pick best")
    ap.add_argument("--patience", type=int, default=3,
                    help="Steps without MSE improvement before stopping (patience mode)")
    ap.add_argument("--emergency_div_factor", type=float, default=3.0,
                    help="Emergency stop if step MSE > factor x best_mse")

    # --- data and model roots ---
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="Root directory containing preprocessed_v3, lmdb and raw h5ad files")
    ap.add_argument("--finetune_root", type=Path, default=DEFAULT_FINETUNE_ROOT,
                    help="Directory containing patient-specific finetuned checkpoints")

    # --- performance ---
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)

    # --- idempotency ---
    ap.add_argument("--skip_if_exists", action="store_true",
                    help="If --out_dir/summary.json already exists, exit 0 "
                         "immediately without re-running. Defense-in-depth "
                         "against duplicate launches; bash drivers also "
                         "perform the same check before invoking python.")

    args = ap.parse_args()

    global DATA_ROOT, LMDB_ROOT, PREPROCESSED_ROOT, RAW_DATA_ROOT
    DATA_ROOT = str(args.data_root)
    LMDB_ROOT = str(args.data_root / "lmdb")
    PREPROCESSED_ROOT = str(args.data_root / "preprocessed_v3")
    RAW_DATA_ROOT = str(args.data_root)

    # --- device ---
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "summary.json"
    if args.skip_if_exists and summary_path.exists():
        print(f"[SKIP] summary.json already exists at {summary_path}; "
              f"skipping (use without --skip_if_exists to force rerun).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    expr_dir = out_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)
    conv_dir = out_dir / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    patient = PATIENTS[args.patient]
    sample_name = patient["pre"]

    print("=" * 60)
    print("EC Virtual Treatment Perturbation (SpatialGT)")
    print("=" * 60)
    print(f"Patient       : {args.patient}")
    print(f"Sample        : {sample_name} (C1D1, pre-treatment)")
    print(f"n_spots       : {args.n_spots if args.n_spots > 0 else 'ALL tumor'}")
    print(f"Lambda        : {args.dose_lambda}")
    print(f"Max steps     : {args.max_steps}")
    print(f"Stopping mode : {args.stopping_mode}"
          f"{f' (patience={args.patience})' if args.stopping_mode == 'patience' else ''}")
    print(f"Emergency div : {args.emergency_div_factor}x")
    print(f"Seed          : {args.seed}")
    print(f"Device        : {device}")
    print(f"Output        : {out_dir}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load vocab, model, databank
    # ----------------------------------------------------------------
    vocab = load_vocab()
    id_to_gene = {v: k for k, v in vocab.items()}

    cache_dir = os.path.join(PREPROCESSED_ROOT, sample_name)
    lmdb_path = os.path.join(LMDB_ROOT, sample_name, "spatial_cache.lmdb")
    lmdb_manifest = os.path.join(
        LMDB_ROOT, sample_name, "spatial_cache.manifest.json"
    )
    ckpt_dir = args.ckpt_dir or str(args.finetune_root / sample_name)

    config = Config()
    config.device = str(device)
    model = load_model(config, ckpt_dir, device)
    databank = setup_databank(
        config, cache_dir, sample_name, lmdb_path, lmdb_manifest
    )
    print(f"[OK] DataBank initialized ({sample_name})")

    # ----------------------------------------------------------------
    # 2. Load annotations and DEGs
    # ----------------------------------------------------------------
    annotation = load_tumor_annotation(sample_name)

    if args.deg_file:
        deg_path = args.deg_file
    else:
        deg_path = DEG_DIR / DEG_FILES[args.patient]
    print(f"[DEG] path={deg_path}")

    deg = load_deg(deg_path)
    gene_logfc = build_perturbation_vector(deg, vocab, args.logfc_clip)

    # ----------------------------------------------------------------
    # 3. Select tumor spots
    # ----------------------------------------------------------------
    tumor_spots = select_tumor_spots(annotation, args.n_spots, args.seed)
    frozen_indices = set(int(x) for x in tumor_spots)

    # ----------------------------------------------------------------
    # 4. Build X_0 and compute X line (once)
    # ----------------------------------------------------------------
    print("[STEP] Computing X_0 = SpatialGT(raw data)...")
    databank.clear_runtime_spot_overrides()
    loader = build_loader(databank, config, args.batch_size, args.num_workers)
    x0_state = run_full_inference(
        databank, loader, model, device, desc="X_0 = SpatialGT(raw)"
    )
    print(f"[OK] X_0 (reconstructed): {len(x0_state)} spots")

    print("[STEP] Computing X line = SpatialGT(X_0) once...")
    databank.clear_runtime_spot_overrides()
    databank.set_runtime_spot_overrides(x0_state)
    loader = build_loader(databank, config, args.batch_size, args.num_workers)
    x_line = run_full_inference(
        databank, loader, model, device, desc="X line (precompute)"
    )
    print(f"[OK] X line: {len(x_line)} spots")

    # ----------------------------------------------------------------
    # 5. Build X'_0 (perturbed initial state)
    # ----------------------------------------------------------------
    print("[STEP] Building X'_0 (applying perturbation on tumor spots)...")
    x0_prime_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx, ov in x0_state.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": ov["gene_ids"].copy(),
            "raw_normed_values": ov["raw_normed_values"].copy(),
        }

    perturb_overrides, perturb_meta = apply_perturbation(
        databank,
        tumor_spots,
        gene_logfc,
        args.dose_lambda,
        base_state=x0_state,
    )
    for gidx, ov in perturb_overrides.items():
        x0_prime_state[int(gidx)] = {
            "gene_ids": np.asarray(ov["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(ov["raw_normed_values"], dtype=np.float32),
        }

    # Deep copy X'_0 base (immutable reference for delta correction)
    x0_prime_base: Dict[int, Dict[str, np.ndarray]] = {
        int(g): {
            "gene_ids": np.array(v["gene_ids"], dtype=np.int64).copy(),
            "raw_normed_values": np.array(
                v["raw_normed_values"], dtype=np.float32
            ).copy(),
        }
        for g, v in x0_prime_state.items()
    }

    # ----------------------------------------------------------------
    # 6. Save metadata and initial states
    # ----------------------------------------------------------------
    spot_meta = annotation.copy()
    spot_meta["is_perturbed"] = spot_meta["spot_idx"].isin(frozen_indices)
    spot_meta.to_csv(expr_dir / "spot_metadata.csv", index=False)

    save_step_npz(x0_state, expr_dir / "ctrl.npz")
    save_step_npz(x0_prime_state, expr_dir / "step0.npz")
    print("[OK] Saved ctrl.npz and step0.npz")

    manifest = {
        "patient": args.patient,
        "sample": sample_name,
        "post_treatment_sample": patient["post"],
        "deg_file": str(deg_path),
        "n_deg_genes_in_file": len(deg),
        "n_deg_genes_mapped": len(gene_logfc),
        "deg_genes": [
            id_to_gene.get(gid, f"id_{gid}") for gid in gene_logfc.keys()
        ],
        "deg_logfc": {
            id_to_gene.get(gid, f"id_{gid}"): lfc
            for gid, lfc in gene_logfc.items()
        },
        "n_tumor_spots_total": int(annotation["is_tumor"].sum()),
        "n_perturbed_spots": len(tumor_spots),
        "perturbed_spot_indices": [int(x) for x in tumor_spots],
        "dose_lambda": args.dose_lambda,
        "logfc_clip": args.logfc_clip,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "ckpt_dir": str(ckpt_dir),
        "x0_state": "SpatialGT reconstruction",
        "stopping_mode": args.stopping_mode,
        "patience": args.patience,
        "emergency_div_factor": args.emergency_div_factor,
        **perturb_meta,
    }
    with open(out_dir / "perturb_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ----------------------------------------------------------------
    # 7. Run dual-line inference with convergence detection
    # ----------------------------------------------------------------
    def loader_factory():
        return build_loader(
            databank, config, args.batch_size, args.num_workers
        )

    final_state, conv_report = run_dual_line_with_convergence(
        databank=databank,
        loader_factory=loader_factory,
        model=model,
        device=device,
        max_steps=args.max_steps,
        frozen_indices=frozen_indices,
        x0_prime_base=x0_prime_base,
        x0_prime_state=x0_prime_state,
        x_line=x_line,
        expr_dir=expr_dir,
        conv_dir=conv_dir,
        stopping_mode=args.stopping_mode,
        patience=args.patience,
        emergency_div_factor=args.emergency_div_factor,
    )

    # ----------------------------------------------------------------
    # 8. Save final summary
    # ----------------------------------------------------------------
    summary = {
        "patient": args.patient,
        "sample": sample_name,
        "n_perturbed_spots": len(tumor_spots),
        "n_total_spots": len(x0_state),
        "dose_lambda": args.dose_lambda,
        "best_step": conv_report["best_step"],
        "best_mse": conv_report["best_mse"],
        "convergence": conv_report,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Status        : {conv_report['status']}")
    print(f"Actual steps  : {conv_report['actual_steps']}")
    print(f"Best step     : {conv_report['best_step']} "
          f"(MSE = {conv_report['best_mse']:.8f})")
    print(f"Expression    : {expr_dir}/ "
          f"(ctrl, step0..step{conv_report['actual_steps']})")
    print(f"Recommended   : expression/step{conv_report['best_step']}.npz")
    print(f"Convergence   : {conv_dir}/")
    print(f"Manifest      : {out_dir / 'perturb_manifest.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
