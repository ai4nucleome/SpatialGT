#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microenvironment-stratified EC bystander analysis.

For every unedited tumor spot, the script computes delta_emt as predicted EMT
score minus the pre-treatment control score, local neighbor cell-type fractions,
the local fraction of edited tumor spots, and distance to the nearest edited
tumor spot. It then reports marginal Spearman, partial Spearman controlling for
edited-tumor fraction, partial Spearman controlling for distance, and OLS
standardized coefficients.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import KDTree
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
EC_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = EC_DIR.parent.parent / "example_data" / "EC" / "GSE225691"
DATA_ROOT = DEFAULT_DATA_ROOT
RESULT_ROOT = SCRIPT_DIR / "results"

PATIENTS_META = {"P034": "01_034_C1d1", "P039": "01_039_C1d1"}

# Merge sub-types into broader buckets so per-CT counts are usable
CT_MERGE = {"CD4_T": "T cell", "CD8_T": "T cell",
            "Mixed": "Other", "Undefined": "Other"}

# Order matters for plotting.
NEIGHBOR_TYPES = [
    ("perturbed_tumor",   "Perturbed Tumor"),
    ("unperturbed_tumor", "Unpert. Tumor"),
    ("CAFs",              "CAFs"),
    ("Immune",            "Immune (Macro+T)"),
    ("Stroma",            "Stroma"),
    ("Endothelial",       "Endothelial"),
]


# ------------------------------------------------------------------------- #
# Path helpers                                                              #
# ------------------------------------------------------------------------- #

def patient_run_root(output_root: Path, patient: str) -> Path:
    """Where lambda*_n*_seed* run directories live for this patient."""
    return output_root / patient


# ------------------------------------------------------------------------- #
# Data loading                                                              #
# ------------------------------------------------------------------------- #

def load_coords(patient: str) -> np.ndarray:
    sample = PATIENTS_META[patient]
    path = DATA_ROOT / sample / sample / "outs" / f"{sample}.h5ad"
    adata = ad.read_h5ad(str(path), backed="r")
    coords = np.array(adata.obsm["spatial"])
    adata.file.close()
    return coords


def compute_neighbor_fracs(meta: pd.DataFrame, coords: np.ndarray,
                           is_perturbed: np.ndarray, k: int = 15):
    ct_arr = meta["cell_type"].replace(CT_MERGE).values
    tumor_mask = meta["is_tumor"].values.astype(bool)
    n = len(meta)

    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)

    fracs = {key: np.zeros(n) for key, _ in NEIGHBOR_TYPES}
    n_actual = np.zeros(n, dtype=int)

    for i in range(n):
        nbs = [j for j in idx[i] if j != i][:k]
        if not nbs:
            continue
        n_actual[i] = len(nbs)
        ct_nb = ct_arr[nbs]
        tm_nb = tumor_mask[nbs]
        pt_nb = is_perturbed[nbs]
        total = len(nbs)
        fracs["perturbed_tumor"][i] = (tm_nb & pt_nb).sum() / total
        fracs["unperturbed_tumor"][i] = (tm_nb & ~pt_nb).sum() / total
        fracs["CAFs"][i] = (ct_nb == "CAFs").sum() / total
        fracs["Immune"][i] = np.isin(ct_nb, ["Macrophage", "T cell"]).sum() / total
        fracs["Stroma"][i] = (ct_nb == "Stroma").sum() / total
        fracs["Endothelial"][i] = (ct_nb == "Endothelial").sum() / total

    return fracs, n_actual


def get_spot_data(exp_dir: Path, metric: str, coords: np.ndarray,
                  k_neighbors: int) -> pd.DataFrame | None:
    emt_path = exp_dir / "emt" / f"emt_{metric}.csv"
    meta_path = exp_dir / "expression" / "spot_metadata.csv"
    if not emt_path.exists() or not meta_path.exists():
        return None

    emt = pd.read_csv(emt_path)
    meta = pd.read_csv(meta_path)

    tumor_mask = meta["is_tumor"].values.astype(bool)
    is_perturbed = (meta["is_perturbed"].values.astype(bool)
                    if "is_perturbed" in meta.columns
                    else np.ones(len(meta), dtype=bool))

    ctrl_emt = emt["ctrl"].values
    if "prediction" in emt.columns:
        delta_emt = emt["prediction"].values - ctrl_emt
    elif "best_step" in emt.columns:
        delta_emt = emt["best_step"].values - ctrl_emt
    else:
        step_cols = sorted([c for c in emt.columns if c.startswith("step")],
                           key=lambda x: int(x.replace("step", "")))
        if not step_cols:
            return None
        delta_emt = emt[step_cols[-1]].values - ctrl_emt

    unpert_tumor = tumor_mask & ~is_perturbed
    if unpert_tumor.sum() < 5:
        return None

    fracs, n_actual = compute_neighbor_fracs(meta, coords, is_perturbed,
                                             k=k_neighbors)

    perturbed_idx = np.where(tumor_mask & is_perturbed)[0]
    if len(perturbed_idx) == 0:
        return None
    pert_tree = KDTree(coords[perturbed_idx])
    dist_to_pert = pert_tree.query(coords, k=1)[0].ravel()

    idx = np.where(unpert_tumor)[0]
    rows = []
    for i in idx:
        rows.append({
            "delta_emt": delta_emt[i],
            "n_neighbors": int(n_actual[i]),
            "dist_to_perturbed": float(dist_to_pert[i]),
            **{f"frac_{k}": fracs[k][i] for k, _ in NEIGHBOR_TYPES},
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- #
# Statistics                                                                #
# ------------------------------------------------------------------------- #

def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Partial Spearman rho between x and y, controlling for z."""
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    rank_z = stats.rankdata(z)
    res_x = rank_x - np.polyval(np.polyfit(rank_z, rank_x, 1), rank_z)
    res_y = rank_y - np.polyval(np.polyfit(rank_z, rank_y, 1), rank_z)
    rho, p = stats.spearmanr(res_x, res_y)
    return float(rho), float(p)


def compute_vif(X: np.ndarray):
    n_cols = X.shape[1]
    vifs = []
    for j in range(n_cols):
        y_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)
        X_rest = np.column_stack([np.ones(len(y_j)), X_rest])
        beta, *_ = np.linalg.lstsq(X_rest, y_j, rcond=None)
        y_pred = X_rest @ beta
        ss_res = np.sum((y_j - y_pred) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs.append(1 / (1 - r2) if r2 < 1 else float("inf"))
    return vifs


def run_ols(df: pd.DataFrame, predictors: List[str], dep_var: str = "delta_emt"):
    y = df[dep_var].values
    X_raw = df[predictors].values
    n, p = X_raw.shape

    X_std = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-12)
    X = np.column_stack([np.ones(n), X_std])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    dof = n - p - 1
    if dof > 0:
        mse = ss_res / dof
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_vals = beta / (se + 1e-15)
        p_vals = 2 * stats.t.sf(np.abs(t_vals), dof)
    else:
        se = np.zeros(p + 1)
        t_vals = np.zeros(p + 1)
        p_vals = np.ones(p + 1)

    vifs = compute_vif(X_raw)

    out = {"r2": round(float(r2), 4), "n": int(n), "dof": int(dof)}
    for i, pred in enumerate(predictors):
        idx = i + 1
        out[pred] = {
            "beta_std": round(float(beta[idx]), 5),
            "se": round(float(se[idx]), 5),
            "t": round(float(t_vals[idx]), 3),
            "p": float(p_vals[idx]),
            "vif": round(float(vifs[i]), 2),
        }
    return out


def analyze_pooled(df: pd.DataFrame, label: str) -> dict:
    delta = df["delta_emt"].values
    n = len(df)

    out = {
        "label": label,
        "n_spots": int(n),
        "mean_delta": round(float(delta.mean()), 6),
        "median_delta": round(float(np.median(delta)), 6),
        "p_vs_zero": float(stats.ttest_1samp(delta, 0).pvalue),
    }

    # 1. marginal Spearman per neighbor type
    marginal = {}
    for key, _ in NEIGHBOR_TYPES:
        fv = df[f"frac_{key}"].values
        if np.std(fv) > 1e-8:
            rho, p = stats.spearmanr(fv, delta)
            rho, p = float(rho), float(p)
        else:
            rho, p = 0.0, 1.0
        marginal[key] = {"rho": round(rho, 4), "p": p}
    out["marginal_spearman"] = marginal

    # 2. partial Spearman ctrl perturbed_tumor
    frac_pt = df["frac_perturbed_tumor"].values
    partial_pt = {}
    for key, _ in NEIGHBOR_TYPES:
        if key == "perturbed_tumor":
            partial_pt[key] = marginal[key]
            continue
        fv = df[f"frac_{key}"].values
        if np.std(fv) > 1e-8 and np.std(frac_pt) > 1e-8:
            rho, p = partial_spearman(fv, delta, frac_pt)
        else:
            rho, p = 0.0, 1.0
        partial_pt[key] = {"rho": round(rho, 4), "p": p}
    out["partial_spearman_ctrl_pt"] = partial_pt

    # 2b. "partial-other" semantics for PT vs CAFs comparison:
    #     PT row    -> controlled for CAFs frac
    #     CAFs row  -> controlled for PT frac    (same as partial_pt above)
    #     other rows-> controlled for PT frac    (same as partial_pt above)
    #
    # This lets dumbbell plots show a meaningful partial value for the PT
    # panel (otherwise we would be "controlling PT for PT" which is undefined).
    frac_cafs = df["frac_CAFs"].values
    partial_other = {}
    for key, _ in NEIGHBOR_TYPES:
        if key == "perturbed_tumor":
            fv = df["frac_perturbed_tumor"].values
            if np.std(fv) > 1e-8 and np.std(frac_cafs) > 1e-8:
                rho, p = partial_spearman(fv, delta, frac_cafs)
            else:
                rho, p = 0.0, 1.0
        else:
            partial_other[key] = partial_pt[key]
            continue
        partial_other[key] = {"rho": round(rho, 4), "p": p}
    out["partial_spearman_ctrl_other"] = partial_other

    # 3. partial Spearman ctrl distance to nearest perturbed spot
    dist = df["dist_to_perturbed"].values
    partial_dist = {}
    for key, _ in NEIGHBOR_TYPES:
        fv = df[f"frac_{key}"].values
        if np.std(fv) > 1e-8 and np.std(dist) > 1e-8:
            rho, p = partial_spearman(fv, delta, dist)
        else:
            rho, p = 0.0, 1.0
        partial_dist[key] = {"rho": round(rho, 4), "p": p}
    out["partial_spearman_ctrl_dist"] = partial_dist

    # 4. multiple OLS: two informative model specifications
    preds_pt_cafs = ["frac_perturbed_tumor", "frac_CAFs"]
    preds_full = ["frac_perturbed_tumor", "frac_CAFs", "frac_Immune",
                  "frac_Stroma", "frac_Endothelial"]
    preds_full_geo = preds_full + ["dist_to_perturbed"]
    out["ols_pt_cafs"] = run_ols(df, preds_pt_cafs)
    out["ols_full"] = run_ols(df, preds_full)
    # OLS on standardized predictors: all neighbor fracs + distance to nearest
    # perturbed spot (continuous geometry confounder).
    out["ols_full_geo"] = run_ols(df, preds_full_geo)

    # Distance row (for long CSV + rigorous heatmap row 6): Spearman / partials
    # use the same naming convention as neighbor types.
    if np.std(dist) > 1e-8:
        rho_dm, p_dm = stats.spearmanr(dist, delta)
    else:
        rho_dm, p_dm = 0.0, 1.0
    if np.std(dist) > 1e-8 and np.std(frac_pt) > 1e-8:
        rho_dpt, p_dpt = partial_spearman(dist, delta, frac_pt)
    else:
        rho_dpt, p_dpt = 0.0, 1.0
    if np.std(dist) > 1e-8 and np.std(frac_cafs) > 1e-8:
        rho_dcf, p_dcf = partial_spearman(dist, delta, frac_cafs)
    else:
        rho_dcf, p_dcf = 0.0, 1.0
    out["distance_row"] = {
        "marg_rho": round(float(rho_dm), 4), "marg_p": float(p_dm),
        "part_pt_rho": round(float(rho_dpt), 4), "part_pt_p": float(p_dpt),
        "part_other_rho": round(float(rho_dcf), 4), "part_other_p": float(p_dcf),
    }

    return out


# ------------------------------------------------------------------------- #
# Main loop                                                                 #
# ------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_root", type=Path, default=EC_DIR / "output",
                    help="Root directory containing per-patient perturbation run folders")
    ap.add_argument("--patients", nargs="+", default=["P034", "P039"])
    ap.add_argument("--lambda_val", default="1.0")
    ap.add_argument("--metric", default="msigdb_mean_common")
    ap.add_argument("--k_neighbors", type=int, default=15)
    ap.add_argument("--n_spots", nargs="*", default=None,
                    help="optional filter, e.g. n40 n80")
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="Root directory containing EC raw h5ad files")
    args = ap.parse_args()

    global DATA_ROOT
    DATA_ROOT = args.data_root
    out_dir = RESULT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    pat_re = re.compile(rf"^lambda{re.escape(args.lambda_val)}_(n[A-Za-z0-9]+)_seed(\d+)$")

    for patient in args.patients:
        coords = load_coords(patient)
        run_root = patient_run_root(args.output_root, patient)
        if not run_root.exists():
            print(f"  [WARN] {run_root} missing -> skip {patient}")
            continue

        print(f"\n{'='*70}")
        print(f" {patient} | lambda={args.lambda_val} | k={args.k_neighbors}")
        print(f" {'='*70}")

        exp_map: dict[str, list[Path]] = {}
        for d in sorted(run_root.iterdir()):
            if not d.is_dir():
                continue
            m = pat_re.match(d.name)
            if not m:
                continue
            ns = m.group(1)
            if args.n_spots and ns not in args.n_spots:
                continue
            exp_map.setdefault(ns, []).append(d)

        if not exp_map:
            print(f"  [WARN] no matching runs for {patient}")
            continue

        ns_sorted = sorted(exp_map.keys(),
                           key=lambda x: (-1 if x == "nAll" else int(x[1:])))
        print(f"  settings: {ns_sorted} "
              f"(seeds: {[len(exp_map[ns]) for ns in ns_sorted]})")

        for ns in ns_sorted:
            dfs = []
            for ed in exp_map[ns]:
                sdf = get_spot_data(ed, args.metric, coords, args.k_neighbors)
                if sdf is not None:
                    dfs.append(sdf)
            if not dfs:
                print(f"    [WARN] no usable runs for {ns}")
                continue

            pooled = pd.concat(dfs, ignore_index=True)
            pooled.to_csv(out_dir / f"raw_{patient}_{ns}.csv", index=False)

            label = f"{patient}_{ns}"
            r = analyze_pooled(pooled, label)
            r["patient"] = patient
            r["n_spots_label"] = ns
            r["n_seeds"] = len(dfs)

            with open(out_dir / f"stats_{patient}_{ns}.json", "w") as f:
                json.dump(r, f, indent=2)

            print(f"\n  --- {ns} (n_pooled={r['n_spots']}, "
                  f"seeds={len(dfs)}) ---")
            print(f"    mean DeltaEMT={r['mean_delta']:+.5f}, "
                  f"p(!=0)={r['p_vs_zero']:.2e}")
            print(f"    {'CT':<22s} {'marg rho':>9s} {'part(ctl_PT)':>13s}"
                  f" {'part(ctl_d)':>13s} {'OLS beta':>9s}")
            print(f"    {'-'*68}")
            for key, lbl in NEIGHBOR_TYPES:
                m_r = r["marginal_spearman"][key]
                ppt = r["partial_spearman_ctrl_pt"][key]
                pdi = r["partial_spearman_ctrl_dist"][key]
                ols = r["ols_full"].get(f"frac_{key}", {"beta_std": float("nan"),
                                                         "p": 1.0})
                def s(p_):
                    return "***" if p_ < 1e-3 else "**" if p_ < 1e-2 \
                        else "*" if p_ < 5e-2 else "ns"
                print(f"    {lbl:<22s} "
                      f"{m_r['rho']:+.3f}{s(m_r['p']):>3s}  "
                      f"{ppt['rho']:+.3f}{s(ppt['p']):>3s}      "
                      f"{pdi['rho']:+.3f}{s(pdi['p']):>3s}      "
                      f"{ols['beta_std']:+.3f}{s(ols['p']):>3s}")

            all_results.append(r)

    if all_results:
        with open(out_dir / f"combined_lambda{args.lambda_val}.json", "w") as f:
            json.dump(all_results, f, indent=2)

        rows = []
        for r in all_results:
            ols_geo = r.get("ols_full_geo", {})
            for key, _ in NEIGHBOR_TYPES:
                m = r["marginal_spearman"][key]
                ppt = r["partial_spearman_ctrl_pt"][key]
                pot = r.get("partial_spearman_ctrl_other", {}).get(key, ppt)
                pdi = r["partial_spearman_ctrl_dist"][key]
                ols = r["ols_full"].get(f"frac_{key}", {})
                og = ols_geo.get(f"frac_{key}", {})
                rows.append({
                    "patient": r["patient"],
                    "n_spots": r["n_spots_label"],
                    "n_pooled": r["n_spots"],
                    "n_seeds": r["n_seeds"],
                    "neighbor_ct": key,
                    "marg_rho": m["rho"], "marg_p": m["p"],
                    "part_pt_rho": ppt["rho"], "part_pt_p": ppt["p"],
                    "part_other_rho": pot["rho"], "part_other_p": pot["p"],
                    "part_dist_rho": pdi["rho"], "part_dist_p": pdi["p"],
                    "ols_beta": ols.get("beta_std", float("nan")),
                    "ols_p": ols.get("p", float("nan")),
                    "ols_vif": ols.get("vif", float("nan")),
                    "ols_geo_beta": og.get("beta_std", float("nan")),
                    "ols_geo_p": og.get("p", float("nan")),
                    "ols_geo_vif": og.get("vif", float("nan")),
                })
            dr = r.get("distance_row")
            if dr and ols_geo:
                dcol = "dist_to_perturbed"
                ogd = ols_geo.get(dcol, {})
                rows.append({
                    "patient": r["patient"],
                    "n_spots": r["n_spots_label"],
                    "n_pooled": r["n_spots"],
                    "n_seeds": r["n_seeds"],
                    "neighbor_ct": dcol,
                    "marg_rho": dr["marg_rho"], "marg_p": dr["marg_p"],
                    "part_pt_rho": dr["part_pt_rho"], "part_pt_p": dr["part_pt_p"],
                    "part_other_rho": dr["part_other_rho"],
                    "part_other_p": dr["part_other_p"],
                    "part_dist_rho": float("nan"), "part_dist_p": float("nan"),
                    "ols_beta": float("nan"), "ols_p": float("nan"),
                    "ols_vif": float("nan"),
                    "ols_geo_beta": ogd.get("beta_std", float("nan")),
                    "ols_geo_p": ogd.get("p", float("nan")),
                    "ols_geo_vif": ogd.get("vif", float("nan")),
                })
        long_df = pd.DataFrame(rows)
        long_df.to_csv(out_dir / f"combined_lambda{args.lambda_val}.csv",
                       index=False)
        print(f"\n  Saved: {out_dir}/combined_lambda{args.lambda_val}.{{json,csv}}")

    print("\nDone.")


if __name__ == "__main__":
    main()
