#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marker-based relaxed cell-type annotation for EC Visium sections.

The workflow follows the final annotation strategy used in the manuscript:
immune and endothelial compartments are assigned first from multi-marker rules,
then remaining spots are classified as tumor or CAF/stroma by comparing EPCAM
against ACTA2/COL1A1/FN1-associated fibroblast signal on log-normalized counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc

DEFAULT_SAMPLES = [
    ("01_034_C1d1", "P034 pre-treatment"),
    ("01_034_C3d1", "P034 post-treatment"),
    ("01_039_C1d1", "P039 pre-treatment"),
    ("01_039_C3d1", "P039 post-treatment"),
]

MARKER_GENES = [
    "EPCAM", "ACTA2", "COL1A1", "FN1", "PTPRC", "CD4", "CD8A",
    "CD163", "CD14", "CD68", "CD34", "PECAM1", "PROM1",
]


def _get_expression(adata: sc.AnnData, gene: str) -> np.ndarray:
    if gene in adata.var_names:
        x = adata[:, gene].X
        return x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    return np.zeros(adata.n_obs)


def annotate_relaxed(adata_raw: sc.AnnData, expr_thresh: float = 0.1) -> pd.Series:
    """Dominance-based annotation using log-normalized marker expression."""
    adata = adata_raw.copy()
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    get = lambda gene: _get_expression(adata, gene)
    epcam = get("EPCAM")
    acta2 = get("ACTA2")
    col1a1 = get("COL1A1")
    fn1 = get("FN1")
    ptprc = get("PTPRC")
    cd4 = get("CD4")
    cd8a = get("CD8A")
    cd163 = get("CD163")
    cd14 = get("CD14")
    cd68 = get("CD68")
    cd34 = get("CD34")
    pecam1 = get("PECAM1")
    prom1 = get("PROM1")

    labels = np.full(adata.n_obs, "Undefined", dtype=object)
    labels[(ptprc > 0) & (cd8a > 0) & (cd4 == 0)] = "CD8_T"
    labels[(ptprc > 0) & (cd4 > 0) & (cd8a == 0)] = "CD4_T"
    labels[(cd163 > 0) & (cd14 > 0) & (cd68 > 0)] = "Macrophage"
    labels[(cd34 > 0) & (pecam1 > 0) & (prom1 > 0)] = "Endothelial"

    for i in np.where(labels == "Undefined")[0]:
        has_tumor = epcam[i] > expr_thresh
        has_caf = (acta2[i] > expr_thresh) or (col1a1[i] > expr_thresh) or (fn1[i] > expr_thresh)
        if has_tumor and not has_caf:
            labels[i] = "Tumor"
        elif has_caf and not has_tumor:
            caf_all = (acta2[i] > expr_thresh) and (col1a1[i] > expr_thresh) and (fn1[i] > expr_thresh)
            labels[i] = "CAFs" if caf_all else "Stroma"
        elif has_tumor and has_caf:
            if epcam[i] > acta2[i]:
                labels[i] = "Tumor"
            elif acta2[i] > epcam[i]:
                labels[i] = "CAFs"
            else:
                labels[i] = "Mixed"

    return pd.Series(labels, index=adata_raw.obs_names, name="celltype_relaxed")


def compute_marker_scores(adata_raw: sc.AnnData) -> pd.DataFrame:
    adata = adata_raw.copy()
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    get = lambda gene: _get_expression(adata, gene)
    return pd.DataFrame({
        "epcam_expr": get("EPCAM"),
        "acta2_expr": get("ACTA2"),
        "caf_score": (get("ACTA2") + get("COL1A1") + get("FN1")) / 3,
        "tumor_score": get("EPCAM"),
        "immune_score": get("PTPRC"),
    }, index=adata_raw.obs_names)


def find_h5ad(data_dir: Path, sample_id: str) -> Path:
    candidates = [
        data_dir / sample_id / sample_id / "outs" / f"{sample_id}.h5ad",
        data_dir / sample_id / f"{sample_id}.h5ad",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find h5ad for {sample_id}: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root directory containing EC raw h5ad files")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expr_thresh", type=float, default=0.1)
    parser.add_argument("--save_h5ad", action="store_true",
                        help="Write celltype_relaxed back into each h5ad file")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Optional subset of sample IDs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.samples) if args.samples else None
    summary: Dict[str, Dict] = {}

    for sample_id, label in DEFAULT_SAMPLES:
        if selected and sample_id not in selected:
            continue
        h5ad_path = find_h5ad(args.data_dir, sample_id)
        print(f"[INFO] {label}: {h5ad_path}")
        adata = sc.read_h5ad(str(h5ad_path))
        labels = annotate_relaxed(adata, expr_thresh=args.expr_thresh)
        scores = compute_marker_scores(adata)

        out = pd.DataFrame({"barcode": labels.index, "celltype_relaxed": labels.values})
        out = pd.concat([out.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        out_path = args.output_dir / f"{sample_id}_annotations.csv"
        out.to_csv(out_path, index=False)

        counts = labels.value_counts().to_dict()
        summary[sample_id] = {"label": label, "counts": counts, "n_spots": int(adata.n_obs)}
        print(f"  saved {out_path.name}: {counts}")

        if args.save_h5ad:
            adata.obs["celltype_relaxed"] = labels.loc[adata.obs_names].values
            adata.write_h5ad(str(h5ad_path))
            print("  updated h5ad with celltype_relaxed")

    with open(args.output_dir / "annotation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
