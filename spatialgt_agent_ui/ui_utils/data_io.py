"""Data I/O helpers for h5ad files and DEG tables."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def load_h5ad(path: str | Path) -> ad.AnnData:
    adata = ad.read_h5ad(str(path))
    return adata


def get_expression_matrix(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    if issparse(X):
        X = X.toarray()
    return X.astype(np.float32)


def detect_celltype_columns(adata: ad.AnnData) -> List[str]:
    """Heuristic: find obs columns that look like cell-type annotations."""
    candidates = []
    for col in adata.obs.columns:
        if adata.obs[col].dtype == "category" or adata.obs[col].dtype == object:
            n_unique = adata.obs[col].nunique()
            if 2 <= n_unique <= 200:
                candidates.append(col)
    return candidates


def load_label_table(path: str | Path) -> pd.DataFrame:
    """Load a spot label table such as labels.csv."""
    df = pd.read_csv(path, sep=None, engine="python")
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "spot_id"})
    elif "spot_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "spot_id"})
    df["spot_id"] = df["spot_id"].astype(str)
    return df


def choose_label_column(labels: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Choose the most likely categorical annotation column."""
    if preferred and preferred in labels.columns:
        return preferred
    for candidate in ("Region", "region", "label", "Label", "cell_type", "celltype", "annotation"):
        if candidate in labels.columns:
            return candidate
    for col in labels.columns:
        if col in {"spot_id", "x", "y", "ID", "id"}:
            continue
        if labels[col].dtype == object or str(labels[col].dtype).startswith("category"):
            return col
    raise ValueError("No categorical label column found in label file.")


def attach_labels_to_adata(
    adata: ad.AnnData,
    label_path: str | Path,
    label_column: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Merge a spot label CSV into adata.obs.

    The preferred path is index matching by spot id. If the label ids do not
    overlap with adata.obs_names but the row counts match, labels are assigned
    by row order for small curated test fixtures.
    """
    labels = load_label_table(label_path)
    chosen_col = choose_label_column(labels, label_column)
    out_col = chosen_col if chosen_col not in adata.obs.columns else f"label_{chosen_col}"

    labels = labels.set_index("spot_id", drop=False)
    obs_index = adata.obs_names.astype(str)
    overlap = set(obs_index).intersection(set(labels.index.astype(str)))

    if overlap:
        aligned = labels.reindex(obs_index)
        adata.obs[out_col] = aligned[chosen_col].values
    elif len(labels) == adata.n_obs:
        adata.obs[out_col] = labels[chosen_col].values
    else:
        raise ValueError(
            f"Cannot align labels: {len(labels)} labels for {adata.n_obs} spots and no spot-id overlap."
        )

    adata.obs[out_col] = adata.obs[out_col].astype("category")
    counts = adata.obs[out_col].value_counts(dropna=False).to_dict()
    return out_col, {str(k): int(v) for k, v in counts.items()}


def parse_deg_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a DEG CSV and normalise column names.
    Supports common bulk formats:
    - gene + avg_logFC/log2FoldChange/logFC
    - gene + fold_change (converted to log2 fold change)
    - gene + direction/up_down with optional strength
    - a one-column gene list (defaults to avg_logFC=1.0)
    Optional: p_val_adj (or padj, p_val).
    """
    df = pd.read_csv(path, sep=None, engine="python")

    rename_map = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "_")
        if lc in ("gene", "gene_name", "symbol", "gene_symbol", "genes"):
            rename_map[col] = "gene"
        elif lc in ("avg_logfc", "log2foldchange", "logfc", "avg_log2fc", "log2fc", "log_fold_change"):
            rename_map[col] = "avg_logFC"
        elif lc in ("foldchange", "fold_change", "fc"):
            rename_map[col] = "fold_change"
        elif lc in ("direction", "regulation", "up_down", "perturbation"):
            rename_map[col] = "direction"
        elif lc in ("strength", "dose", "factor"):
            rename_map[col] = "strength"
        elif lc in ("p_val_adj", "padj", "p_value_adj", "fdr"):
            rename_map[col] = "p_val_adj"
        elif lc in ("p_val", "pvalue", "p_value"):
            rename_map[col] = "p_val"
    df = df.rename(columns=rename_map)

    if "gene" not in df.columns:
        if df.index.name and df.index.name.lower() in ("gene", "gene_name"):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "gene"})
        else:
            df.insert(0, "gene", df.iloc[:, 0])

    if "avg_logFC" not in df.columns:
        if "fold_change" in df.columns:
            fc = pd.to_numeric(df["fold_change"], errors="coerce").fillna(1.0).clip(lower=1e-12)
            df["avg_logFC"] = np.log2(fc)
        elif "direction" in df.columns:
            direction = df["direction"].astype(str).str.lower()
            sign = np.where(direction.str.contains("down|decrease|lower|inhibit|ko|kd"), -1.0, 1.0)
            strength = pd.to_numeric(df.get("strength", 1.0), errors="coerce").fillna(1.0)
            df["avg_logFC"] = sign * strength.astype(float)
        else:
            df["avg_logFC"] = 1.0

    if "p_val_adj" not in df.columns and "p_val" in df.columns:
        df["p_val_adj"] = df["p_val"]

    df["gene"] = df["gene"].astype(str)
    df["avg_logFC"] = pd.to_numeric(df["avg_logFC"], errors="coerce").fillna(0.0)

    return df


def get_spatial_coords(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        return np.array(adata.obsm["spatial"], dtype=np.float64)
    raise KeyError("adata.obsm['spatial'] not found.")
