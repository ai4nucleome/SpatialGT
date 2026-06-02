"""
Core perturbation engine: DEG application, dual-line iterative inference,
and convergence detection.

Wraps logic from niche_perturb_eval_v2.py and analyze_step_mse.py.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrain.Config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerturbationResult:
    """Container for perturbation results."""
    converged_step: int
    total_steps: int
    step_mse: List[float] = field(default_factory=list)
    best_step: Optional[int] = None
    best_mse: Optional[float] = None
    stopping_mode: str = "auto_best"
    stop_reason: str = "not_started"
    selected_step: Optional[int] = None
    output_dir: Optional[str] = None
    summary_path: Optional[str] = None
    step_output_paths: Dict[int, str] = field(default_factory=dict)
    update_mse: List[float] = field(default_factory=list)
    step_expressions: Dict[int, Dict[int, Dict]] = field(default_factory=dict)
    baseline_expressions: Dict[int, Dict] = field(default_factory=dict)
    perturbed_spots: List[int] = field(default_factory=list)
    gene_ids: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# DEG perturbation
# ─────────────────────────────────────────────────────────────────────────────

def load_vocab(vocab_path: str | Path) -> Dict[str, int]:
    with open(vocab_path, "r") as f:
        return json.load(f)


def apply_deg_perturbation(
    spot_data: Dict[str, np.ndarray],
    deg_df: pd.DataFrame,
    vocab: Dict[str, int],
    strength: float = 1.0,
    weight: float = 1.0,
    logfc_clip: float = 5.0,
    p_thresh: float = 0.1,
    min_abs_logfc: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Apply DEG-based perturbation to a single spot's expression.

    Formula: new_val = old_val * 2^(logFC * strength * weight)
    """
    gene_ids = spot_data["gene_ids"]
    values = spot_data["raw_normed_values"].copy().astype(np.float32)

    id_to_pos = {int(gid): i for i, gid in enumerate(gene_ids) if gid >= 0}

    if "p_val_adj" in deg_df.columns:
        filtered = deg_df[deg_df["p_val_adj"] < p_thresh]
    elif "p_val" in deg_df.columns:
        filtered = deg_df[deg_df["p_val"] < p_thresh]
    else:
        filtered = deg_df

    if min_abs_logfc > 0:
        filtered = filtered[filtered["avg_logFC"].abs() >= min_abs_logfc]

    for _, row in filtered.iterrows():
        gene = row["gene"]
        logfc = np.clip(row["avg_logFC"], -logfc_clip, logfc_clip)
        gid = vocab.get(gene)
        if gid is None or gid not in id_to_pos:
            continue
        pos = id_to_pos[gid]
        scale = 2.0 ** (logfc * strength * weight)
        values[pos] = max(0.0, values[pos] * scale)

    result = spot_data.copy()
    result["raw_normed_values"] = values
    return result


def apply_custom_gene_edits(
    spot_data: Dict[str, np.ndarray],
    edits: Dict[str, float],
    vocab: Dict[str, int],
    strength: float = 1.0,
    weight: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Apply manual per-gene logFC edits."""
    gene_ids = spot_data["gene_ids"]
    values = spot_data["raw_normed_values"].copy().astype(np.float32)
    id_to_pos = {int(gid): i for i, gid in enumerate(gene_ids) if gid >= 0}

    for gene_name, logfc in edits.items():
        gid = vocab.get(gene_name)
        if gid is None or gid not in id_to_pos:
            continue
        pos = id_to_pos[gid]
        scale = 2.0 ** (logfc * strength * weight)
        values[pos] = max(0.0, values[pos] * scale)

    result = spot_data.copy()
    result["raw_normed_values"] = values
    return result


def compute_gaussian_weights(
    spatial_coords: np.ndarray,
    perturb_indices: List[int],
) -> Dict[int, float]:
    """Gaussian spatial weights centered on the perturbation centroid."""
    centers = spatial_coords[perturb_indices]
    centroid = centers.mean(axis=0)
    dists = np.sqrt(((centers - centroid) ** 2).sum(axis=1))
    sigma = max(dists.std(), 1e-6)
    weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    return {idx: float(w) for idx, w in zip(perturb_indices, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_pass(
    databank,
    loader,
    model: nn.Module,
    device: torch.device,
    config: Config,
    desc: str = "Inference",
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run one forward pass over the full slice, return {global_idx: {gene_ids, raw_normed_values}}."""
    from pretrain.utils_train import process_batch, forward_pass

    overrides: Dict[int, Dict[str, np.ndarray]] = {}
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    for batch in tqdm(loader, desc=desc, leave=False):
        if isinstance(batch, dict) and batch.get("skip_batch", False):
            continue

        structure = batch.get("structure", {})
        centers_global = structure.get("centers_global_indices", None)
        if centers_global is None:
            centers_global = structure.get("center_global_indices", batch.get("centers_global_indices"))
        if centers_global is None:
            continue
        if isinstance(centers_global, torch.Tensor):
            centers_global = centers_global.cpu().numpy().astype(int).tolist()

        batch_data = process_batch(batch, device, config=config)
        with torch.no_grad():
            preds, _, _, _ = forward_pass(model, batch_data, config=config)

        B = len(centers_global)
        gids = batch_data["genes"][:B]
        pad_mask = batch_data["padding_attention_mask"][:B].bool()
        gathered = preds.gather(1, gids.clamp(min=0)) * pad_mask.to(preds.dtype)

        gid_np = gids.detach().cpu().numpy().astype(np.int64)
        val_np = gathered.detach().float().cpu().numpy().astype(np.float32)
        mask_np = pad_mask.detach().cpu().numpy()

        for i in range(B):
            gidx = int(centers_global[i])
            valid = mask_np[i]
            overrides[gidx] = {
                "gene_ids": gid_np[i][valid],
                "raw_normed_values": val_np[i][valid],
            }

    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Dual-line iterative inference with convergence detection
# ─────────────────────────────────────────────────────────────────────────────

def run_perturbation(
    model: nn.Module,
    databank,
    config: Config,
    perturb_spots: List[int],
    all_global_indices: List[int],
    deg_df: Optional[pd.DataFrame] = None,
    custom_gene_edits: Optional[Dict[str, float]] = None,
    strength: float = 1.0,
    weighting: str = "uniform",
    logfc_clip: float = 5.0,
    p_thresh: float = 0.1,
    min_abs_logfc: float = 0.0,
    max_steps: int = 20,
    stopping_mode: str = "auto_best",
    fixed_step: Optional[int] = None,
    convergence_threshold: float = 1e-5,
    patience: int = 3,
    emergency_div_factor: float = 10.0,
    freeze_perturbed: bool = True,
    device: str = "cuda:0",
    batch_size: int = 64,
    num_workers: int = 4,
    output_dir: Optional[str | Path] = None,
    step_callback: Optional[Callable] = None,
) -> PerturbationResult:
    """
    Full dual-line perturbation pipeline.

    1. Collect X_0 baseline (one-pass inference on original slice)
    2. Apply perturbation to selected spots -> X'_0
    3. Iterate dual-line inference until convergence
    """
    torch_device = torch.device(device)
    vocab = load_vocab(config.vocab_file)

    perturb_set = set(perturb_spots)
    frozen_set = set(all_global_indices) - perturb_set

    # ── Build loader factory ──
    def loader_factory():
        return databank.get_data_loader(
            split="train", batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers,
            stratify_by_batch=False, validation_split=0.0,
            use_distributed=False,
            persistent_workers=False,
            prefetch_factor=4 if num_workers > 0 else 2,
            is_training=False,
        )

    # ── Collect original spot data ──
    original_state: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in all_global_indices:
        sd = databank.get_spot_data(int(gidx))
        original_state[int(gidx)] = {
            "gene_ids": np.asarray(sd["gene_ids"], dtype=np.int64),
            "raw_normed_values": np.asarray(sd["raw_normed_values"], dtype=np.float32),
        }

    # ── Step 1: one-pass X_0 baseline ──
    if step_callback:
        step_callback(-1, 0.0, "Running baseline inference (X_0)...")
    databank.set_runtime_spot_overrides(original_state)
    loader = loader_factory()
    x0_overrides = _run_single_pass(databank, loader, model, torch_device, config, "X_0 baseline")

    # ── Step 2: apply perturbation to get X'_0 ──
    if step_callback:
        step_callback(-1, 0.0, "Applying perturbation...")

    spatial_coords = None
    if weighting == "gaussian":
        import anndata as ad
        adata_path = Path(config.cache_dir).glob("*/*.h5ad")
        for p in adata_path:
            adata = ad.read_h5ad(str(p))
            spatial_coords = np.array(adata.obsm["spatial"])
            break

    weights = {}
    if weighting == "gaussian" and spatial_coords is not None:
        local_indices = [gidx for gidx in perturb_spots]
        weights = compute_gaussian_weights(spatial_coords, local_indices)
    else:
        weights = {idx: 1.0 for idx in perturb_spots}

    perturbed_x0: Dict[int, Dict[str, np.ndarray]] = {}
    for gidx in all_global_indices:
        if gidx in perturb_set:
            sd = x0_overrides.get(gidx, original_state[gidx]).copy()
            w = weights.get(gidx, 1.0)
            if deg_df is not None:
                sd = apply_deg_perturbation(sd, deg_df, vocab, strength, w, logfc_clip, p_thresh, min_abs_logfc)
            if custom_gene_edits:
                sd = apply_custom_gene_edits(sd, custom_gene_edits, vocab, strength, w)
            perturbed_x0[gidx] = sd
        else:
            perturbed_x0[gidx] = x0_overrides.get(gidx, original_state[gidx]).copy()

    # ── Step 3: dual-line iterative inference ──
    result = PerturbationResult(
        converged_step=max_steps,
        total_steps=max_steps,
        perturbed_spots=list(perturb_spots),
        stopping_mode=stopping_mode,
    )
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = str(out_dir)
    else:
        out_dir = None

    current_perturbed = {k: v.copy() for k, v in perturbed_x0.items()}
    best_mse = float("inf")
    best_step = None
    stale_steps = 0

    # The X-line input is fixed at X_0 across all dual-line iterations, so its
    # prediction can be reused instead of recomputed every step.
    databank.set_runtime_spot_overrides(
        {k: x0_overrides.get(k, original_state[k]) for k in all_global_indices}
    )
    loader = loader_factory()
    x_pred_fixed = _run_single_pass(databank, loader, model, torch_device, config, "X fixed baseline")

    for step in range(max_steps):
        if step_callback:
            step_callback(step, step / max_steps, f"Step {step + 1}/{max_steps}")

        # X'-line: infer from current perturbed state
        databank.set_runtime_spot_overrides(current_perturbed)
        loader = loader_factory()
        xp_pred = _run_single_pass(databank, loader, model, torch_device, config, f"X' step {step}")

        # Compute delta and update: X'_new = X'_0 + (X'_pred - X_pred)
        new_perturbed: Dict[int, Dict[str, np.ndarray]] = {}
        for gidx in all_global_indices:
            if gidx in perturb_set and freeze_perturbed:
                new_perturbed[gidx] = perturbed_x0[gidx].copy()
                continue

            base = perturbed_x0.get(gidx, original_state[gidx])
            xp_val = xp_pred.get(gidx, current_perturbed.get(gidx, base))
            x_val = x_pred_fixed.get(gidx, x0_overrides.get(gidx, original_state[gidx]))

            delta = xp_val["raw_normed_values"] - x_val["raw_normed_values"]
            new_vals = base["raw_normed_values"] + delta
            new_vals = np.clip(new_vals, 0, None)

            new_perturbed[gidx] = {
                "gene_ids": base["gene_ids"].copy(),
                "raw_normed_values": new_vals.astype(np.float32),
            }

        # Update MSE drives convergence; effect MSE is user-facing and includes
        # direct perturbation effects relative to the baseline expression.
        update_mse_vals = []
        effect_mse_vals = []
        for gidx in all_global_indices:
            old_v = current_perturbed.get(gidx, {}).get("raw_normed_values")
            new_v = new_perturbed.get(gidx, {}).get("raw_normed_values")
            if old_v is not None and new_v is not None:
                if not (gidx in perturb_set and freeze_perturbed):
                    update_mse_vals.append(float(np.mean((old_v - new_v) ** 2)))
            base_v = x0_overrides.get(gidx, original_state[gidx]).get("raw_normed_values")
            if base_v is not None and new_v is not None:
                effect_mse_vals.append(float(np.mean((base_v - new_v) ** 2)))

        update_mse = float(np.mean(update_mse_vals)) if update_mse_vals else 0.0
        step_mse = float(np.mean(effect_mse_vals)) if effect_mse_vals else 0.0
        result.update_mse.append(update_mse)
        result.step_mse.append(step_mse)
        result.step_expressions[step] = {
            gidx: new_perturbed[gidx].copy() for gidx in all_global_indices
        }
        if out_dir is not None:
            step_path = out_dir / f"step_{step + 1:03d}_expressions.pkl"
            with open(step_path, "wb") as f:
                pickle.dump(result.step_expressions[step], f, protocol=pickle.HIGHEST_PROTOCOL)
            result.step_output_paths[step] = str(step_path)
            partial_summary = {
                "output_dir": result.output_dir,
                "step_output_paths": {str(k + 1): v for k, v in result.step_output_paths.items()},
                "step_mse": result.step_mse,
                "update_mse": result.update_mse,
                "current_step": step + 1,
                "stopping_mode": result.stopping_mode,
                "perturbed_spots": result.perturbed_spots,
            }
            with open(out_dir / "perturbation_summary.json", "w") as f:
                json.dump(partial_summary, f, indent=2)

        improved = step_mse < (best_mse - convergence_threshold)
        if improved or best_step is None:
            best_mse = step_mse
            best_step = step
            stale_steps = 0
        else:
            stale_steps += 1
        result.best_step = best_step
        result.best_mse = best_mse

        current_perturbed = new_perturbed

        if step_callback:
            step_callback(step, (step + 1) / max_steps, f"Step {step + 1} MSE: {step_mse:.2e}")

        # Check stopping policy.
        if stopping_mode == "fixed_steps":
            continue
        if stopping_mode != "auto_best":
            raise ValueError("stopping_mode must be 'auto_best' or 'fixed_steps'.")

        if best_mse > 0 and update_mse > best_mse * emergency_div_factor:
            result.converged_step = step + 1
            result.stop_reason = (
                f"emergency_stop: update MSE {update_mse:.3e} exceeded "
                f"{emergency_div_factor:g}x best MSE {best_mse:.3e}"
            )
            break
        if stale_steps >= patience:
            result.converged_step = step + 1
            result.stop_reason = f"auto_best_patience: no MSE improvement for {patience} steps"
            break
    else:
        result.stop_reason = "fixed_steps_complete" if stopping_mode == "fixed_steps" else "max_steps_reached"

    result.baseline_expressions = {
        gidx: x0_overrides.get(gidx, original_state[gidx])
        for gidx in all_global_indices
    }
    if out_dir is not None:
        baseline_path = out_dir / "baseline_expressions.pkl"
        with open(baseline_path, "wb") as f:
            pickle.dump(result.baseline_expressions, f, protocol=pickle.HIGHEST_PROTOCOL)
    if all_global_indices:
        first = result.baseline_expressions[all_global_indices[0]]
        result.gene_ids = first["gene_ids"]
    result.total_steps = len(result.step_mse)
    if result.best_step is None and result.step_mse:
        result.best_step = int(np.argmin(result.step_mse))
        result.best_mse = float(result.step_mse[result.best_step])
    if stopping_mode == "fixed_steps":
        selected = fixed_step if fixed_step is not None else max_steps
        selected = max(1, min(int(selected), result.total_steps))
        result.selected_step = selected - 1
        result.best_step = result.selected_step
        result.best_mse = float(result.step_mse[result.selected_step]) if result.step_mse else None
        result.converged_step = selected
    else:
        result.selected_step = result.best_step
        if result.stop_reason == "not_started":
            result.stop_reason = "auto_best_complete"
    if out_dir is not None:
        summary = {
            "output_dir": result.output_dir,
            "baseline_path": str(out_dir / "baseline_expressions.pkl"),
            "step_output_paths": {str(k + 1): v for k, v in result.step_output_paths.items()},
            "step_mse": result.step_mse,
            "update_mse": result.update_mse,
            "best_step": None if result.best_step is None else result.best_step + 1,
            "best_mse": result.best_mse,
            "selected_step": None if result.selected_step is None else result.selected_step + 1,
            "stopping_mode": result.stopping_mode,
            "stop_reason": result.stop_reason,
            "perturbed_spots": result.perturbed_spots,
        }
        summary_path = out_dir / "perturbation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        result.summary_path = str(summary_path)

    return result
