"""Build EC tumor perturbation DEG files.

Source : {patient}_C3D1_vs_C1D1_tumor_DEG_raw.csv
Filter : p_val_adj < FDR_THRESH AND |avg_logFC| >= LOGFC_THRESH
Output : {patient}_perturbation_DEGs.csv

The default FDR threshold is 0.05 and the default absolute logFC threshold is
0.0, matching the perturbation DEG definition used in the EC analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = THIS_DIR

PATIENTS = ["P034", "P039"]


def load_emt_signatures(deg_dir: Path) -> tuple[set[str], set[str]]:
    """Return (msigdb_hallmark_emt set, mak_pancancer_emt set), both lower-case."""
    msigdb_path = deg_dir / "msigdb_hallmark_emt.json"
    mak_path = deg_dir / "mak_pancancer_emt_signature.json"

    msigdb: set[str] = set()
    mak: set[str] = set()

    if msigdb_path.exists():
        obj = json.loads(msigdb_path.read_text())
        for k in ("genes", "hallmark_emt", "EMT", "epithelial_mesenchymal_transition"):
            if k in obj and isinstance(obj[k], list):
                msigdb = {g.lower() for g in obj[k]}
                break
        if not msigdb:
            for v in obj.values():
                if isinstance(v, list):
                    msigdb = {g.lower() for g in v}
                    break
    else:
        print(f"[WARN] {msigdb_path} not found, EMT_hallmark annotation disabled")

    if mak_path.exists():
        obj = json.loads(mak_path.read_text())
        merged: set[str] = set()
        for v in obj.values():
            if isinstance(v, list):
                merged.update(g.lower() for g in v)
        mak = merged
    else:
        print(f"[WARN] {mak_path} not found, EMT_pancancer annotation disabled")

    print(f"[SIG] MSigDB Hallmark EMT: {len(msigdb)} genes")
    print(f"[SIG] Mak pan-cancer EMT : {len(mak)} genes")
    return msigdb, mak


def annotate_gene_set(gene: str, msigdb: set[str], mak: set[str]) -> str:
    g = gene.lower()
    in_m = g in msigdb
    in_p = g in mak
    if in_m and in_p:
        return "EMT_both"
    if in_m:
        return "EMT_hallmark"
    if in_p:
        return "EMT_pancancer"
    return "Other"


def build_for_patient(
    patient: str,
    src_dir: Path,
    out_dir: Path,
    fdr_thresh: float,
    logfc_thresh: float,
    msigdb_emt: set[str],
    mak_emt: set[str],
) -> None:
    src = src_dir / f"{patient}_C3D1_vs_C1D1_tumor_DEG_raw.csv"
    if not src.exists():
        raise FileNotFoundError(f"raw DEG not found: {src}")

    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]

    if "avg_logFC" not in df.columns and "avg_logfc" in df.columns:
        df = df.rename(columns={"avg_logfc": "avg_logFC"})
    if "p_val_adj" not in df.columns:
        if "padj" in df.columns:
            df = df.rename(columns={"padj": "p_val_adj"})
        else:
            raise ValueError(f"No p_val_adj column in {src}")

    df = df.dropna(subset=["avg_logFC", "p_val_adj", "gene"])
    df["avg_logFC"] = df["avg_logFC"].astype(float)
    df["p_val_adj"] = df["p_val_adj"].astype(float)

    n_total = len(df)
    sub = df[
        (df["p_val_adj"] < fdr_thresh)
        & (df["avg_logFC"].abs() >= logfc_thresh)
    ].copy()

    sub = sub.sort_values("avg_logFC", key=lambda s: -s.abs()).reset_index(drop=True)
    sub["gene_set"] = sub["gene"].apply(
        lambda g: annotate_gene_set(str(g), msigdb_emt, mak_emt)
    )

    cols = ["gene", "avg_logFC", "p_val_adj", "gene_set"]
    extra = [c for c in df.columns if c not in cols and c in sub.columns]
    out_df = sub[cols + extra]

    out_path = out_dir / f"{patient}_perturbation_DEGs.csv"
    out_df.to_csv(out_path, index=False)

    n_up = int((out_df["avg_logFC"] > 0).sum())
    n_dn = int((out_df["avg_logFC"] < 0).sum())
    set_counts = out_df["gene_set"].value_counts().to_dict()

    print(
        f"[OUT] {patient}: {len(out_df)}/{n_total} genes -> {out_path.name}\n"
        f"      up={n_up}  down={n_dn}\n"
        f"      gene_set breakdown: {set_counts}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir", type=Path, default=DEFAULT_SOURCE_DIR,
        help="directory containing *_tumor_DEG_raw.csv files",
    )
    ap.add_argument(
        "--out_dir", type=Path, default=THIS_DIR,
        help="output directory (default: this script's directory)",
    )
    ap.add_argument("--fdr_thresh", type=float, default=0.05,
                    help="match mouse_stroke p_adj_thresh=0.05")
    ap.add_argument("--logfc_thresh", type=float, default=0.0,
                    help="match mouse_stroke min_abs_logfc=0.0 (no |logFC| filter)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Build EC perturbation DEG files")
    print(f"  src         : {args.src_dir}")
    print(f"  out         : {args.out_dir}")
    print(f"  FDR < {args.fdr_thresh},  |logFC| >= {args.logfc_thresh}")
    print("=" * 60)

    msigdb_emt, mak_emt = load_emt_signatures(args.out_dir)

    for patient in PATIENTS:
        build_for_patient(
            patient=patient,
            src_dir=args.src_dir,
            out_dir=args.out_dir,
            fdr_thresh=args.fdr_thresh,
            logfc_thresh=args.logfc_thresh,
            msigdb_emt=msigdb_emt,
            mak_emt=mak_emt,
        )


if __name__ == "__main__":
    main()
