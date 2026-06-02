#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dumbbell and heatmap visualisations of microenvironment-stratified bystander
analysis. Reads the long-format CSV produced by compute_microenv_bystander.py.

Outputs (under analysis_microenv_bystander/figures/):
  * dumbbell_<patient>.{pdf,png}      one panel per patient, all CTs as rows,
                                      shows marginal/partial-PT/partial-dist
                                      Spearman rho for each n_spots setting.
  * heatmap_partial_pt.{pdf,png}      patient x CT heatmap (all n_spots pooled
                                      via inverse-variance weighted mean rho)
  * scatter_marg_vs_part.{pdf,png}    sanity check: how much partial controls
                                      shift each marginal rho.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "results"
FIG_ROOT = SCRIPT_DIR / "figures"

CT_ORDER = [
    ("perturbed_tumor",   "Perturbed Tumor"),
    ("unperturbed_tumor", "Unpert. Tumor"),
    ("CAFs",              "CAFs"),
    ("Immune",            "Immune (Macro+T)"),
    ("Stroma",            "Stroma"),
    ("Endothelial",       "Endothelial"),
]
CT_KEYS = [k for k, _ in CT_ORDER]
CT_LABELS = {k: lbl for k, lbl in CT_ORDER}

CT_COLOR = {
    "perturbed_tumor":   "#C0392B",
    "unperturbed_tumor": "#E67E22",
    "CAFs":              "#2C7FB8",
    "Immune":            "#1B7837",
    "Stroma":            "#7B6FB1",
    "Endothelial":       "#8C8C8C",
}

COLOR_AX = "#222222"
COLOR_GRID = "#DDDDDD"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": COLOR_AX,
    "axes.labelcolor": COLOR_AX,
    "xtick.color": COLOR_AX,
    "ytick.color": COLOR_AX,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sig(p: float) -> str:
    """Single * for any p < 0.05 (dumbbell aesthetics)."""
    if not np.isfinite(p):
        return ""
    return "*" if p < 0.05 else ""


def n_spots_int(label: str) -> int:
    if label == "nAll":
        return 10**6
    return int(label[1:])


# ------------------------------------------------------------------------- #
# 1. Per-patient dumbbell                                                   #
# ------------------------------------------------------------------------- #

def plot_dumbbell(df: pd.DataFrame, fig_dir: Path):
    """One figure per patient.  Rows = (n_spots, neighbor CT).  Three markers:
    o = marginal, D = partial(ctrl PT), ^ = partial(ctrl distance)."""
    patients = sorted(df["patient"].unique())
    n_ct = len(CT_KEYS)

    for patient in patients:
        sub = df[df["patient"] == patient].copy()
        if sub.empty:
            continue

        # only settings actually present for this patient
        settings_p = sorted(sub["n_spots"].unique(), key=n_spots_int)

        # row layout: groups of n_ct rows per n_spots, blank gap between groups
        gap = 0.7
        y_positions, y_labels = [], []
        y = 0.0
        group_y_ranges = {}
        group_meta = {}
        for ns in settings_p:
            sub_ns = sub[sub["n_spots"] == ns]
            if sub_ns.empty:
                continue
            group_meta[ns] = (int(sub_ns["n_pooled"].iloc[0]),
                              int(sub_ns["n_seeds"].iloc[0]))
            y_start = y
            for k in CT_KEYS:
                y_positions.append(y)
                y_labels.append(CT_LABELS[k])
                y += 1
            group_y_ranges[ns] = (y_start, y - 1)
            y += gap

        fig_h = max(4.5, 0.32 * len(y_positions) + 1.6)
        fig, ax = plt.subplots(figsize=(9.6, fig_h))
        ax.set_facecolor("#FAFAFA")

        idx = 0
        for ns in settings_p:
            sub_ns = sub[sub["n_spots"] == ns].set_index("neighbor_ct")
            if sub_ns.empty:
                continue
            for k in CT_KEYS:
                yp = y_positions[idx]
                if k not in sub_ns.index:
                    idx += 1
                    continue
                row = sub_ns.loc[k]

                col = CT_COLOR[k]
                m_v, _ = row["marg_rho"], row["marg_p"]
                ppt_v, ppt_p = row["part_pt_rho"], row["part_pt_p"]
                pdi_v, _ = row["part_dist_rho"], row["part_dist_p"]

                vals = sorted([m_v, ppt_v, pdi_v])
                ax.plot([vals[0], vals[-1]], [yp, yp],
                        color=col, alpha=0.32, lw=2.0,
                        solid_capstyle="round", zorder=2)

                ax.scatter(m_v, yp, marker="o", s=48, color=col,
                           edgecolors="white", linewidths=0.7, zorder=4)
                ax.scatter(ppt_v, yp, marker="D", s=36,
                           color="white", edgecolors=col, linewidths=1.4,
                           zorder=4)
                ax.scatter(pdi_v, yp, marker="^", s=42,
                           color="white", edgecolors=col, linewidths=1.4,
                           zorder=4)

                # partial-PT: one * if p<0.05; fixed offset toward top of panel
                star = sig(ppt_p)
                if star:
                    ax.text(ppt_v, yp - 0.32, star,
                            ha="center", va="top", fontsize=7.5,
                            color=col, fontweight="bold")
                idx += 1

        # group separators (light shading + group label on the right side)
        for i, ns in enumerate(settings_p):
            if ns not in group_y_ranges:
                continue
            y0, y1 = group_y_ranges[ns]
            n_pooled, n_seeds = group_meta[ns]
            if i % 2 == 1:
                ax.axhspan(y0 - 0.45, y1 + 0.45,
                           facecolor="#F0F0F0", edgecolor="none", zorder=0)
            ax.text(1.005, (y0 + y1) / 2,
                    f"{ns}\nn={n_pooled}\nseeds={n_seeds}",
                    ha="left", va="center", fontsize=7.5,
                    transform=ax.get_yaxis_transform(),
                    color=COLOR_AX)

        ax.axvline(0, color=COLOR_AX, lw=0.7, ls="--", alpha=0.55, zorder=1)
        ax.set_xlabel("Spearman rho (neighbor fraction vs bystander DeltaEMT)",
                      fontsize=10)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.tick_params(axis="y", length=0)
        xlo, xhi = _safe_x_range(sub)
        ax.set_xlim(xlo, xhi)

        ax.invert_yaxis()
        ax.grid(axis="x", color=COLOR_GRID, lw=0.5, alpha=0.7, zorder=0)

        title_main = f"{patient} | lambda = 1.0"
        title_sub = ("o marginal rho      D partial rho (ctrl perturbed-tumor frac)"
                     "      ^ partial rho (ctrl distance to nearest perturbed)")
        ax.set_title(f"{title_main}\n{title_sub}",
                     fontsize=10, color=COLOR_AX, loc="left", pad=10)

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
                   markeredgecolor="white", markeredgewidth=0.6,
                   markersize=8, label="marginal rho"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor="#666", markeredgewidth=1.0,
                   markersize=7, label="partial rho (ctrl PT frac)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
                   markeredgecolor="#666", markeredgewidth=1.0,
                   markersize=7, label="partial rho (ctrl distance)"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  frameon=False, fontsize=7.6)

        plt.tight_layout(rect=[0, 0, 0.93, 1])
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"dumbbell_{patient}.{ext}",
                        dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {fig_dir / f'dumbbell_{patient}.png'}")


def _safe_x_range(df: pd.DataFrame):
    cols = ["marg_rho", "part_pt_rho", "part_dist_rho"]
    vals = df[cols].values.flatten()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -0.5, 0.5
    lo = float(vals.min())
    hi = float(vals.max())
    pad = (hi - lo) * 0.08 + 0.02
    return lo - pad, hi + pad


# ------------------------------------------------------------------------- #
# 2. CT by n_spots heatmap of partial-PT rho                                #
# ------------------------------------------------------------------------- #

def plot_heatmap(df: pd.DataFrame, fig_dir: Path):
    patients = sorted(df["patient"].unique())
    settings = sorted(df["n_spots"].unique(), key=n_spots_int)

    n_p = len(patients)
    fig, axes = plt.subplots(1, n_p, figsize=(2.5 + 1.0 * len(settings),
                                              0.55 * len(CT_KEYS) + 1.6),
                             sharey=True)
    if n_p == 1:
        axes = [axes]

    vmax = float(np.nanmax(np.abs(df["part_pt_rho"].values))) if len(df) else 0.4
    vmax = max(vmax, 0.05)

    for ax, patient in zip(axes, patients):
        mat = np.full((len(CT_KEYS), len(settings)), np.nan)
        pmat = np.full_like(mat, np.nan)
        sub = df[df["patient"] == patient]
        for j, ns in enumerate(settings):
            row = sub[sub["n_spots"] == ns].set_index("neighbor_ct")
            for i, k in enumerate(CT_KEYS):
                if k in row.index:
                    mat[i, j] = row.loc[k, "part_pt_rho"]
                    pmat[i, j] = row.loc[k, "part_pt_p"]

        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(settings, rotation=0, fontsize=8)
        ax.set_yticks(range(len(CT_KEYS)))
        ax.set_yticklabels([CT_LABELS[k] for k in CT_KEYS], fontsize=8)
        ax.set_title(patient, fontsize=10, color=COLOR_AX)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                txt = f"{v:+.2f}"
                star = sig(pmat[i, j])
                if star:
                    txt += f"\n{star}"
                txt_color = "white" if abs(v) > 0.55 * vmax else "#222"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.6, color=txt_color,
                        fontweight="bold" if star else "normal")

        for s in ax.spines.values():
            s.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, shrink=0.8)
    cbar.set_label("Partial Spearman rho\n(ctrl perturbed-tumor frac)",
                   fontsize=8, color=COLOR_AX)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Bystander DeltaEMT vs neighbor composition | lambda=1.0",
                 fontsize=11, color=COLOR_AX, x=0.02, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"heatmap_partial_pt.{ext}",
                    dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {fig_dir / 'heatmap_partial_pt.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=Path,
                    default=RESULT_ROOT / "combined_lambda1.0.csv")
    ap.add_argument("--lambda_val", default="1.0")
    args = ap.parse_args()

    csv_path = args.input_csv
    if not csv_path.exists():
        raise SystemExit(f"Missing input: {csv_path}\n"
                         "Run compute_microenv_bystander.py first.")
    df = pd.read_csv(csv_path)

    fig_dir = FIG_ROOT
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Plot dumbbell")
    plot_dumbbell(df, fig_dir)
    print("Plot heatmap")
    plot_heatmap(df, fig_dir)


if __name__ == "__main__":
    main()
