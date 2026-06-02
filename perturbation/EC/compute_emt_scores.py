#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-accelerated EMT scoring for EC virtual-treatment outputs.

The public workflow keeps the two EMT readouts used in the paper: the MSigDB
Hallmark EMT mean-expression score and a bidirectional EMT score defined as
mean(Hallmark EMT genes) minus mean(Mak epithelial genes). Both readouts are
reported in two-way and three-way common-gene universes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
for _path in (_REPO_ROOT, _PRETRAIN_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from pretrain.Config import Config

# -------------------------------------------------------------------------- #
# Paths & constants (identical to CPU script)                                #
# -------------------------------------------------------------------------- #

DEG_DIR = _SCRIPT_DIR / "deg"
MAK_SIGNATURE_PATH = DEG_DIR / "mak_pancancer_emt_signature.json"
MSIGDB_EMT_PATH = DEG_DIR / "msigdb_hallmark_emt.json"

DEFAULT_DATA_ROOT = _REPO_ROOT / "example_data" / "EC" / "GSE225691"
DATA_ROOT = DEFAULT_DATA_ROOT
PREPROC_ROOT = DATA_ROOT / "preprocessed_v3"

PATIENTS = {
    "P034": {"pre": "01_034_C1d1", "post": "01_034_C3d1"},
    "P039": {"pre": "01_039_C1d1", "post": "01_039_C3d1"},
}

GENE_ALIASES = {"gpr56": "adgrg1"}

SCORE_KEYS = [
    "hybrid_bidir",
    "hybrid_bidir_common",
    "msigdb_mean",
    "msigdb_mean_common",
]

SCORE_DESCRIPTIONS = {
    "hybrid_bidir":        "mean(MSigDB_EMT) - mean(Mak_E) (2-way)",
    "hybrid_bidir_common": "mean(MSigDB_EMT) - mean(Mak_E) (3-way)",
    "msigdb_mean":         "mean(MSigDB_EMT expression) (2-way)",
    "msigdb_mean_common":  "mean(MSigDB_EMT expression) (3-way)",
}

SCORE_FILENAMES = {k: f"emt_{k}.csv" for k in SCORE_KEYS}


# -------------------------------------------------------------------------- #
# Vocabulary, gene sets, gene universe                                       #
# -------------------------------------------------------------------------- #

def load_vocab(vocab_path: str = None) -> Dict[str, int]:
    if vocab_path is None:
        vocab_path = Config().vocab_file
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {k.lower(): int(v) for k, v in vocab.items()}


def _map_genes_to_ids(genes: List[str], vocab: Dict[str, int],
                      label: str = "") -> Set[int]:
    ids: Set[int] = set()
    for g in genes:
        name = GENE_ALIASES.get(g, g)
        if name in vocab:
            ids.add(vocab[name])
    if label:
        print(f"[EMT] {label}: {len(ids)}/{len(genes)} mapped to vocab")
    return ids


def load_gene_sets(vocab: Dict[str, int]) -> Dict[str, Set[int]]:
    with open(MSIGDB_EMT_PATH) as f:
        msigdb_genes = [g.lower() for g in json.load(f)["genes"]]
    with open(MAK_SIGNATURE_PATH) as f:
        mak_e_genes = [g.lower() for g in json.load(f)["epithelial_genes"]]

    msigdb_ids = _map_genes_to_ids(msigdb_genes, vocab, "MSigDB Hallmark EMT")
    mak_e_ids = _map_genes_to_ids(mak_e_genes, vocab, "Mak epithelial")

    return {
        "msigdb": msigdb_ids,
        "mak_e": mak_e_ids,
        "msigdb_total": len(msigdb_genes),
        "mak_e_total": len(mak_e_genes),
    }


def get_npz_gene_universe(patient_dir: Path) -> Set[int]:
    for d in sorted(patient_dir.iterdir()):
        ctrl = d / "expression" / "ctrl.npz"
        if ctrl.exists():
            data = np.load(str(ctrl))
            return set(int(g) for g in data["gids_0"])
    return set()


def compute_common_gene_ids(patient: str, vocab: Dict[str, int]) -> Set[int]:
    info = PATIENTS[patient]

    def _h5ad_ids(sample):
        path = PREPROC_ROOT / sample / sample / "processed.h5ad"
        a = ad.read_h5ad(str(path), backed="r")
        ids = set()
        for g in a.var_names:
            name = GENE_ALIASES.get(g.lower(), g.lower())
            vid = vocab.get(name)
            if vid is not None:
                ids.add(vid)
        return ids

    return _h5ad_ids(info["pre"]) & _h5ad_ids(info["post"])


def build_filtered_sets(full_sets, npz_gids, common_gids):
    sets_2way = {"msigdb": full_sets["msigdb"] & npz_gids,
                 "mak_e": full_sets["mak_e"] & npz_gids}
    sets_3way = {"msigdb": full_sets["msigdb"] & common_gids & npz_gids,
                 "mak_e": full_sets["mak_e"] & common_gids & npz_gids}
    return sets_2way, sets_3way


# -------------------------------------------------------------------------- #
# GPU vectorised scoring                                                     #
# -------------------------------------------------------------------------- #

def _build_target_mask(gene_ids: np.ndarray, target: Set[int],
                       device: torch.device) -> torch.Tensor:
    """bool tensor (n_genes,) - True if gene_ids[i] in target."""
    t = np.array([int(g) in target for g in gene_ids], dtype=bool)
    return torch.from_numpy(t).to(device)


def _mean_vec(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros(vals.shape[0], device=vals.device)
    return vals[:, mask].mean(dim=1)


def stack_npz(state: Dict[int, Tuple[np.ndarray, np.ndarray]]
              ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Convert {spot_idx: (gids, vals)} to (sorted_spot_idxs, gids, vals_mat).

    Asserts every spot shares the same gene_id vector - the case for SpatialGT
    HVG outputs.  If not, raises so the caller can fall back to per-spot CPU.
    """
    spot_idxs = sorted(state.keys())
    gids_first, _ = state[spot_idxs[0]]
    n_genes = len(gids_first)

    vals_mat = np.empty((len(spot_idxs), n_genes), dtype=np.float32)
    for i, s in enumerate(spot_idxs):
        gids_s, vals_s = state[s]
        if len(gids_s) != n_genes or not np.array_equal(gids_s, gids_first):
            raise ValueError(f"spot {s} has different gene_ids vector")
        vals_mat[i] = vals_s
    return spot_idxs, gids_first, vals_mat


def compute_all_scores_gpu(spot_idxs: List[int], gids: np.ndarray,
                           vals_mat: np.ndarray,
                           sets_2way: Dict[str, Set[int]],
                           sets_3way: Dict[str, Set[int]],
                           device: torch.device,
                           ) -> Dict[int, Dict[str, float]]:
    """Returns {spot_idx: {score_key: value}}, identical numbers to CPU code."""
    vals_t = torch.from_numpy(vals_mat).to(device)

    # masks (per gene set, computed once)
    mask_msigdb_2 = _build_target_mask(gids, sets_2way["msigdb"], device)
    mask_msigdb_3 = _build_target_mask(gids, sets_3way["msigdb"], device)
    mask_mak_e_2 = _build_target_mask(gids, sets_2way["mak_e"], device)
    mask_mak_e_3 = _build_target_mask(gids, sets_3way["mak_e"], device)

    msigdb_mean_2 = _mean_vec(vals_t, mask_msigdb_2)
    msigdb_mean_3 = _mean_vec(vals_t, mask_msigdb_3)
    mak_e_mean_2 = _mean_vec(vals_t, mask_mak_e_2)
    mak_e_mean_3 = _mean_vec(vals_t, mask_mak_e_3)

    bidir_2 = msigdb_mean_2 - mak_e_mean_2
    bidir_3 = msigdb_mean_3 - mak_e_mean_3

    scores = {
        "hybrid_bidir":        bidir_2.cpu().numpy(),
        "hybrid_bidir_common": bidir_3.cpu().numpy(),
        "msigdb_mean":         msigdb_mean_2.cpu().numpy(),
        "msigdb_mean_common":  msigdb_mean_3.cpu().numpy(),
    }

    out = {}
    for i, s in enumerate(spot_idxs):
        out[s] = {k: float(scores[k][i]) for k in SCORE_KEYS}
    return out


# -------------------------------------------------------------------------- #
# NPZ helpers                                                                #
# -------------------------------------------------------------------------- #

def load_npz(path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    data = np.load(str(path), allow_pickle=True)
    result = {}
    for key in data.files:
        if key.startswith("gids_"):
            gidx = int(key[5:])
            result[gidx] = (data[f"gids_{gidx}"], data[f"vals_{gidx}"])
    return result


def step_sort_key(name: str):
    if name == "ctrl":
        return (-1, 0)
    return (0, int(name.replace("step", "")))


# -------------------------------------------------------------------------- #
# Per-experiment processing (GPU)                                            #
# -------------------------------------------------------------------------- #

def process_experiment(exp_dir: Path, sets_2way, sets_3way, gene_info,
                       device: torch.device, force: bool = False):
    expr_dir = exp_dir / "expression"
    emt_dir = exp_dir / "emt"
    if not expr_dir.exists():
        return
    if (emt_dir / "emt_summary.json").exists() and not force:
        print(f"  [SKIP] already computed")
        return

    emt_dir.mkdir(parents=True, exist_ok=True)
    step_files = sorted(expr_dir.glob("step*.npz"))
    ctrl_file = expr_dir / "ctrl.npz"
    if not step_files:
        print(f"  [SKIP] no step files")
        return

    meta_path = expr_dir / "spot_metadata.csv"
    spot_meta = pd.read_csv(meta_path) if meta_path.exists() else None
    summary_path = exp_dir / "summary.json"
    best_step = None
    if summary_path.exists():
        with open(summary_path) as f:
            best_step = json.load(f).get("best_step")

    all_scores: Dict[str, Dict[int, Dict[str, float]]] = {}

    files_to_process = []
    if ctrl_file.exists():
        files_to_process.append(("ctrl", ctrl_file))
    for sf in step_files:
        files_to_process.append((sf.stem, sf))

    for step_name, fpath in files_to_process:
        state = load_npz(fpath)
        try:
            spot_idxs, gids, vals_mat = stack_npz(state)
            step_scores = compute_all_scores_gpu(
                spot_idxs, gids, vals_mat, sets_2way, sets_3way, device)
        except ValueError:
            # rare fallback: per-spot CPU
            step_scores = {}
            for s_idx, (g, v) in state.items():
                step_scores[s_idx] = _cpu_scores_one(g, v, sets_2way, sets_3way)
        all_scores[step_name] = step_scores

    all_spots = sorted(set().union(*[s.keys() for s in all_scores.values()]))
    step_cols = sorted(all_scores.keys(), key=step_sort_key)

    tumor_map = None
    if spot_meta is not None:
        tumor_map = spot_meta.set_index("spot_idx")["is_tumor"]

    score_dfs: Dict[str, pd.DataFrame] = {}
    for sk in SCORE_KEYS:
        data_dict = {"spot_idx": all_spots}
        for col in step_cols:
            scores = all_scores[col]
            data_dict[col] = [scores[s][sk] if s in scores else np.nan
                              for s in all_spots]
        df = pd.DataFrame(data_dict)
        if tumor_map is not None:
            df.insert(1, "is_tumor",
                      df["spot_idx"].map(tumor_map).fillna(False).astype(bool))
        df.to_csv(emt_dir / SCORE_FILENAMES[sk], index=False)
        score_dfs[sk] = df

    summary_out: Dict = {
        "scoring_methods": SCORE_DESCRIPTIONS,
        "gene_counts": gene_info,
        "n_spots": len(all_spots),
        "n_tumor_spots": int(score_dfs[SCORE_KEYS[0]]["is_tumor"].sum())
            if "is_tumor" in score_dfs[SCORE_KEYS[0]].columns else None,
        "steps": step_cols,
        "best_step": best_step,
    }
    for sk in SCORE_KEYS:
        df = score_dfs[sk]
        has_tumor = "is_tumor" in df.columns
        summary_out[f"{sk}_per_step"] = {}
        if has_tumor:
            summary_out[f"{sk}_tumor_per_step"] = {}
        for col in step_cols:
            vals = df[col].dropna()
            summary_out[f"{sk}_per_step"][col] = {
                "mean": round(float(vals.mean()), 6),
                "median": round(float(vals.median()), 6),
                "std": round(float(vals.std()), 6),
            }
            if has_tumor:
                t_vals = df.loc[df["is_tumor"], col].dropna()
                if len(t_vals) > 0:
                    summary_out[f"{sk}_tumor_per_step"][col] = {
                        "mean": round(float(t_vals.mean()), 6),
                        "median": round(float(t_vals.median()), 6),
                        "std": round(float(t_vals.std()), 6),
                    }

    with open(emt_dir / "emt_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    best_tag = f" (best_step={best_step})" if best_step else ""
    print(f"  [OK] {len(all_spots)} spots x {len(step_cols)} steps x "
          f"{len(SCORE_KEYS)} scores{best_tag}")


# -------------------------------------------------------------------------- #
# Per-spot CPU fallback (only used when gene-id stacking fails)              #
# -------------------------------------------------------------------------- #

def _cpu_scores_one(gene_ids, values, sets_2way, sets_3way):
    """Identical to compute_all_scores in CPU script - included as fallback."""
    def _mean(g, v, target):
        vals = [float(v[i]) for i in range(len(g)) if int(g[i]) in target]
        return float(np.mean(vals)) if vals else 0.0

    def _bidir(g, v, m_set, e_set):
        return _mean(g, v, m_set) - _mean(g, v, e_set)

    return {
        "hybrid_bidir":        _bidir(gene_ids, values, sets_2way["msigdb"], sets_2way["mak_e"]),
        "hybrid_bidir_common": _bidir(gene_ids, values, sets_3way["msigdb"], sets_3way["mak_e"]),
        "msigdb_mean":         _mean(gene_ids, values, sets_2way["msigdb"]),
        "msigdb_mean_common":  _mean(gene_ids, values, sets_3way["msigdb"]),
    }


# -------------------------------------------------------------------------- #
# Per-patient baselines (GPU vectorised)                                     #
# -------------------------------------------------------------------------- #

def _load_tumor_mask(sample, obs_names):
    raw_path = DATA_ROOT / sample / sample / "outs" / f"{sample}.h5ad"
    if not raw_path.exists():
        return np.zeros(len(obs_names), dtype=bool)
    raw = ad.read_h5ad(str(raw_path))
    for col in ["celltype_relaxed", "celltype"]:
        if col in raw.obs.columns:
            raw_ct = {bc: str(raw.obs.loc[bc, col]).lower() for bc in raw.obs_names}
            return np.array(["tumor" in raw_ct.get(bc, "") for bc in obs_names])
    return np.zeros(len(obs_names), dtype=bool)


def _stats(arr, mask=None):
    a = np.asarray(arr)
    result = {"all": {"mean": round(float(a.mean()), 6),
                      "std": round(float(a.std()), 6)}}
    if mask is not None and mask.any():
        t = a[mask]
        result["tumor"] = {"mean": round(float(t.mean()), 6),
                           "std": round(float(t.std()), 6)}
    return result


def compute_baselines(patient, patient_dir, vocab, full_sets,
                      npz_gids, common_gids, device, force: bool = False):
    bp = patient_dir / "baselines.json"
    if bp.exists() and not force:
        print(f"  [INFO] baselines exist -> {bp} (skip; use --force to redo)")
        return

    info = PATIENTS[patient]
    sets_2way, sets_3way = build_filtered_sets(full_sets, npz_gids, common_gids)
    # ---------- C1D1 ----------
    print(f"  C1D1: {info['pre']} h5ad (log-normalized) [GPU]")
    adata_pre = ad.read_h5ad(
        str(PREPROC_ROOT / info["pre"] / info["pre"] / "processed.h5ad"))
    tumor_mask = _load_tumor_mask(info["pre"], adata_pre.obs_names)

    var_lower = [g.lower() for g in adata_pre.var_names]
    h5ad_gids = np.array([vocab.get(GENE_ALIASES.get(g, g), -1)
                          for g in var_lower], dtype=np.int64)
    valid = h5ad_gids >= 0
    gene_ids_base = h5ad_gids[valid]

    X_pre = adata_pre.layers.get("X_log1p", adata_pre.X)
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()
    X_valid = np.asarray(X_pre[:, valid], dtype=np.float32)

    vals_mat = X_valid
    spot_idxs = list(range(vals_mat.shape[0]))
    print(f"    {vals_mat.shape[0]} spots, {int(tumor_mask.sum())} tumor, "
          f"{len(gene_ids_base)} genes")

    c1d1_scores = compute_all_scores_gpu(
        spot_idxs, gene_ids_base, vals_mat,
        sets_2way, sets_3way, device)
    c1d1_arrs = {k: np.array([c1d1_scores[s][k] for s in spot_idxs])
                 for k in SCORE_KEYS}

    c1d1 = {"source": f"{info['pre']} h5ad (log-normalized)",
            "n_spots": vals_mat.shape[0],
            "n_tumor": int(tumor_mask.sum()) if tumor_mask is not None else 0,
            "n_genes": len(gene_ids_base)}
    for k in SCORE_KEYS:
        c1d1[k] = _stats(c1d1_arrs[k], tumor_mask)

    # ---------- C3D1 ----------
    print(f"  C3D1: {info['post']} h5ad [GPU]")
    adata_post = ad.read_h5ad(
        str(PREPROC_ROOT / info["post"] / info["post"] / "processed.h5ad"))
    is_tumor_c3 = _load_tumor_mask(info["post"], adata_post.obs_names)

    var_lower = [g.lower() for g in adata_post.var_names]
    all_gids = np.array([vocab.get(GENE_ALIASES.get(g, g), -1)
                         for g in var_lower], dtype=np.int64)
    valid = all_gids >= 0
    gene_ids_c3 = all_gids[valid]
    X = adata_post.layers.get("X_log1p", adata_post.X)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_valid_c3 = np.asarray(X[:, valid], dtype=np.float32)
    spot_idxs_c3 = list(range(X_valid_c3.shape[0]))

    c3d1_keys = [k for k in SCORE_KEYS if k.endswith("_common")]
    # CPU script feeds (sets_3way, sets_3way) for both args -> identical here
    c3d1_scores = compute_all_scores_gpu(
        spot_idxs_c3, gene_ids_c3, X_valid_c3,
        sets_3way, sets_3way, device)
    c3d1_arrs = {k: np.array([c3d1_scores[s][k] for s in spot_idxs_c3])
                 for k in c3d1_keys}

    c3d1 = {"source": f"{info['post']} h5ad (native gene universe)",
            "n_spots": X_valid_c3.shape[0],
            "n_tumor": int(is_tumor_c3.sum()),
            "n_genes": int(valid.sum())}
    for k in c3d1_keys:
        c3d1[k] = _stats(c3d1_arrs[k], is_tumor_c3)

    # ---------- save per-spot CSVs ----------
    c1d1_df = pd.DataFrame({"spot_idx": spot_idxs, "is_tumor": tumor_mask})
    for k in SCORE_KEYS:
        c1d1_df[k] = c1d1_arrs[k]
    c1d1_df.to_csv(patient_dir / "c1d1_per_spot_scores.csv", index=False)

    c3d1_df = pd.DataFrame({"spot_idx": spot_idxs_c3, "is_tumor": is_tumor_c3})
    for k in c3d1_keys:
        c3d1_df[k] = c3d1_arrs[k]
    c3d1_df.to_csv(patient_dir / "c3d1_per_spot_scores.csv", index=False)

    baselines = {
        "patient": patient,
        "gene_info": {
            "npz_genes": len(npz_gids),
            "common_genes": len(common_gids),
            "msigdb_total": full_sets.get("msigdb_total", len(full_sets["msigdb"])),
            "msigdb_2way": len(sets_2way["msigdb"]),
            "msigdb_3way": len(sets_3way["msigdb"]),
            "mak_e_total": full_sets.get("mak_e_total", len(full_sets["mak_e"])),
            "mak_e_2way": len(sets_2way["mak_e"]),
            "mak_e_3way": len(sets_3way["mak_e"]),
        },
        "c1d1": c1d1,
        "c3d1": c3d1,
    }
    with open(patient_dir / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  [OK] baselines -> {patient_dir / 'baselines.json'}")


# -------------------------------------------------------------------------- #
# Main                                                                        #
# -------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="GPU-accelerated EMT scoring (drop-in for CPU script)")
    ap.add_argument("input_paths", nargs="+",
                    help="Experiment dir(s) or parent dir(s)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if results already exist")
    ap.add_argument("--device", default=None,
                    help="cuda:N or cpu (default: cuda:0 if available)")
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="Root directory containing preprocessed_v3 and raw h5ad files")
    args = ap.parse_args()

    global DATA_ROOT, PREPROC_ROOT
    DATA_ROOT = args.data_root
    PREPROC_ROOT = DATA_ROOT / "preprocessed_v3"

    device = torch.device(args.device) if args.device else \
             torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    vocab = load_vocab()
    full_sets = load_gene_sets(vocab)
    if not full_sets["msigdb"]:
        print("[ERROR] No MSigDB EMT genes mapped to vocab")
        sys.exit(1)

    def _detect_patient(exp_dir: Path) -> str:
        for cand in exp_dir.parents:
            if cand.name in PATIENTS:
                return cand.name
        return exp_dir.parent.name

    patient_exps: Dict[str, List[Path]] = {}
    for p_str in args.input_paths:
        p = Path(p_str)
        if (p / "expression").exists():
            patient_exps.setdefault(_detect_patient(p), []).append(p)
        else:
            for expr_sub in sorted(p.rglob("expression")):
                if expr_sub.is_dir():
                    exp_dir = expr_sub.parent
                    patient_exps.setdefault(_detect_patient(exp_dir), []).append(exp_dir)

    total = sum(len(v) for v in patient_exps.values())
    print(f"\n[INFO] {total} experiments for {len(patient_exps)} patients")

    done, failed = 0, 0
    for patient, exp_dirs in patient_exps.items():
        print(f"\n{'='*60}")
        print(f" {patient}: {len(exp_dirs)} experiments")
        print(f"{'='*60}")

        baseline_root = exp_dirs[0].parent
        while baseline_root.name not in PATIENTS and \
                baseline_root != baseline_root.parent:
            baseline_root = baseline_root.parent

        npz_gids = get_npz_gene_universe(exp_dirs[0].parent)
        if patient in PATIENTS:
            common_gids = compute_common_gene_ids(patient, vocab)
        else:
            print(f"  [WARN] Unknown patient {patient}, 3-way = 2-way")
            common_gids = npz_gids

        sets_2way, sets_3way = build_filtered_sets(
            full_sets, npz_gids, common_gids)
        gene_info = {
            "msigdb_2way": len(sets_2way["msigdb"]),
            "msigdb_3way": len(sets_3way["msigdb"]),
            "mak_e_2way": len(sets_2way["mak_e"]),
            "mak_e_3way": len(sets_3way["mak_e"]),
        }
        print(f"  npz genes:    {len(npz_gids)}")
        print(f"  common genes: {len(common_gids)}")
        print(f"  MSigDB 2-way: {gene_info['msigdb_2way']}, "
              f"3-way: {gene_info['msigdb_3way']}")
        print(f"  Mak_E  2-way: {gene_info['mak_e_2way']}, "
              f"3-way: {gene_info['mak_e_3way']}")

        for i, exp_dir in enumerate(exp_dirs):
            print(f"\n[{i+1}/{len(exp_dirs)}] {patient}/{exp_dir.name}")
            try:
                process_experiment(exp_dir, sets_2way, sets_3way, gene_info,
                                   device, args.force)
                done += 1
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        if patient in PATIENTS:
            print(f"\n[INFO] Computing baselines for {patient} -> {baseline_root}/")
            compute_baselines(patient, baseline_root, vocab, full_sets,
                              npz_gids, common_gids, device, args.force)

    print(f"\n{'='*50}")
    print(f"GPU EMT computation complete  ({len(SCORE_KEYS)} methods)")
    print(f"  Processed: {done}")
    print(f"  Failed:    {failed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
