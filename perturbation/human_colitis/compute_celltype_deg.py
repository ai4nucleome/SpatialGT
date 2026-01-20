#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Differential Expression Genes (DEG) Between Cell Types or Conditions.

Supports two modes:
1. Cross-condition comparison: Same cell type across different conditions (UC vs HC, UC_VDZ vs UC)
2. Within-condition comparison: Different cell types within the same condition (MNP_activated vs MNP_normal)

Usage:
    # Mode 1: Cross-condition comparison
    python compute_celltype_deg.py --condition_a UC --condition_b HC --celltype MNP --output_dir ./output/deg
    python compute_celltype_deg.py --condition_a UC_VDZ --condition_b UC --celltype Fibroblast --output_dir ./output/deg

    # Mode 2: Within-condition comparison
    python compute_celltype_deg.py --condition UC --celltype_a MNP_activated --celltype_b MNP_normal --output_dir ./output/deg
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import warnings

warnings.filterwarnings('ignore')

# Script directory for relative paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Data path configuration
# Override with environment variable COLITIS_DATA_ROOT or pass via command line
DATA_ROOT = Path(os.environ.get("COLITIS_DATA_ROOT", _REPO_ROOT / "example_data" / "human_colitis"))
RAW_DATA_DIR = DATA_ROOT / "raw_data"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"

# Cell type mapping
CELLTYPE_MAPPING = {
    # MNP related
    "MNP": ["05-MNP", "05B-MNP_activated_inf"],
    "MNP_normal": ["05-MNP"],
    "MNP_activated": ["05B-MNP_activated_inf"],
    
    # Fibroblast related
    "Fibroblast": ["03A-Myofibroblast", "03B-Fibroblast", "03C-activated_fibroblast"],
    "Fibroblast_normal": ["03A-Myofibroblast", "03B-Fibroblast"],
    "Fibroblast_activated": ["03C-activated_fibroblast"],
    
    # IEC (epithelial cells) related
    "IEC": ["01A-IEC_crypt_base", "01B-IEC_transit_amplifying", 
            "01C-IEC_crypt_top_colonocyte", "01D-IEC_base_2"],
    "IEC_crypt_base": ["01A-IEC_crypt_base"],
    "IEC_colonocyte": ["01C-IEC_crypt_top_colonocyte"],
    
    # Other immune cells
    "T_cell": ["04A-CD4_T", "04B-CD8_T", "04C-Treg"],
    "B_cell": ["06-B"],
    "Plasma": ["02A-IgA_plasma", "02B-IgG_plasma"],
    "Endothelial": ["07-Endothelial"],
    "Mast": ["08-Mast"],
}

# Condition to sample mapping
CONDITION_SAMPLES = {
    "UC": ["HS5_UC_R_0", "HS7_UC_L_0", "HS7_UC_R_1", "HS8_UC_L_1"],
    "UC_VDZ": ["HS10_VDZ_L_1", "HS10_VDZ_R_0", "HS11_VDZ_L_3", "HS11_VDZ_R_0", "HS12_VDZ_L_1"],
    "HC": ["HS1_HC_L_0_fov18", "HS1_HC_L_0_fov19", "HS1_HC_R_0", "HS2_HC_L_0", 
           "HS2_HC_R_0", "HS3_HC_R_0", "HS4_HC_L_0", "HS4_HC_R_0"],
}


def load_condition_data(condition: str, use_preprocessed: bool = True) -> ad.AnnData:
    """Load and merge all sample data for a given condition."""
    samples = CONDITION_SAMPLES.get(condition, [])
    if not samples:
        raise ValueError(f"Unknown condition: {condition}")
    
    adatas = []
    for sample in samples:
        if use_preprocessed:
            processed_path = PREPROCESSED_DIR / condition / sample / "processed.h5ad"
            if not processed_path.exists():
                processed_path = PREPROCESSED_DIR / sample / sample / "processed.h5ad"
            if not processed_path.exists():
                processed_path = RAW_DATA_DIR / sample / f"{sample}.h5ad"
        else:
            processed_path = RAW_DATA_DIR / sample / f"{sample}.h5ad"
        
        if processed_path.exists():
            adata = sc.read_h5ad(processed_path)
            adata.obs['sample'] = sample
            adata.obs['condition'] = condition
            adatas.append(adata)
            print(f"  Loaded {sample}: {adata.n_obs} cells")
        else:
            print(f"  [WARN] File not found: {processed_path}")
    
    if not adatas:
        raise FileNotFoundError(f"No data files found for condition: {condition}")
    
    # Merge data
    combined = ad.concat(adatas, join='inner', merge='same')
    print(f"  Combined {condition}: {combined.n_obs} cells, {combined.n_vars} genes")
    
    return combined


def subset_by_celltype(
    adata: ad.AnnData, 
    celltype: str, 
    annotation_col: str = "fine_annotation"
) -> ad.AnnData:
    """Subset data by cell type."""
    if annotation_col not in adata.obs.columns:
        raise ValueError(f"Annotation column '{annotation_col}' not found in obs")
    
    # Get target cell types
    target_types = CELLTYPE_MAPPING.get(celltype, [celltype])
    
    mask = adata.obs[annotation_col].isin(target_types)
    subset = adata[mask].copy()
    
    print(f"  Subset {celltype}: {subset.n_obs} cells (from {len(target_types)} types)")
    
    return subset


def ensure_log1p_layer(adata: ad.AnnData, layer_key: str = "X_log1p") -> None:
    """Ensure log1p layer exists."""
    if layer_key in adata.layers:
        return
    
    # Save original counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    
    # Normalize + log1p
    orig_X = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.layers["X_normed"] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers[layer_key] = adata.X.copy()
    adata.X = orig_X


def compute_deg(
    adata_a: ad.AnnData,
    adata_b: ad.AnnData,
    group_a_name: str = "A",
    group_b_name: str = "B",
    use_layer: str = "X_log1p",
    method: str = "wilcoxon",
    n_top: int = 3000,
) -> pd.DataFrame:
    """
    Compute differential expression genes between two groups.
    
    Args:
        adata_a: Group A AnnData
        adata_b: Group B AnnData (reference)
        group_a_name: Group A name
        group_b_name: Group B name
        use_layer: Layer to use
        method: DEG computation method
        n_top: Number of top genes to return
    
    Returns:
        DEG results DataFrame
    """
    # Align genes
    common_genes = adata_a.var_names.intersection(adata_b.var_names)
    if len(common_genes) == 0:
        raise ValueError("No common genes between the two datasets")
    
    sub_a = adata_a[:, common_genes].copy()
    sub_b = adata_b[:, common_genes].copy()
    
    # Ensure log1p layer exists
    ensure_log1p_layer(sub_a, layer_key=use_layer)
    ensure_log1p_layer(sub_b, layer_key=use_layer)
    
    # Make obs_names unique
    sub_a.obs_names = [f"{group_a_name}_{i}" for i in range(sub_a.n_obs)]
    sub_b.obs_names = [f"{group_b_name}_{i}" for i in range(sub_b.n_obs)]
    
    # Set condition labels
    sub_a.obs["condition"] = group_a_name
    sub_b.obs["condition"] = group_b_name
    
    # Merge
    sub_a.var_names_make_unique()
    sub_b.var_names_make_unique()
    combo = ad.concat([sub_a, sub_b], join="inner", merge="same")
    combo.obs["condition"] = combo.obs["condition"].astype("category")
    
    # Ensure layer exists
    if use_layer not in combo.layers:
        ensure_log1p_layer(combo, layer_key=use_layer)
    
    print(f"  DEG: {group_a_name} ({sub_a.n_obs} cells) vs {group_b_name} ({sub_b.n_obs} cells), {combo.n_vars} genes")
    
    # Run DEG analysis
    sc.tl.rank_genes_groups(
        combo,
        groupby="condition",
        groups=[group_a_name],
        reference=group_b_name,
        method=method,
        n_genes=min(n_top, combo.n_vars),
        layer=use_layer,
    )
    
    # Get results
    res = sc.get.rank_genes_groups_df(combo, group=group_a_name)
    
    # Standardize column names
    res = res.rename(columns={
        "names": "gene",
        "logfoldchanges": "avg_logFC",
        "pvals_adj": "p_val_adj",
        "pvals": "p_val",
    })
    
    return res


def main():
    parser = argparse.ArgumentParser(description="Compute DEG between cell types across conditions or within condition")
    
    # Condition arguments
    parser.add_argument("--condition", type=str, default=None,
                       help="Single condition for within-condition comparison (UC, UC_VDZ, HC)")
    parser.add_argument("--condition_a", type=str, default=None,
                       help="Condition A (UC, UC_VDZ, HC)")
    parser.add_argument("--condition_b", type=str, default=None,
                       help="Condition B as reference (UC, UC_VDZ, HC)")
    
    # Cell type arguments
    parser.add_argument("--celltype", type=str, default=None,
                       help="Cell type for cross-condition comparison (MNP, Fibroblast, IEC, etc.)")
    parser.add_argument("--celltype_a", type=str, default=None,
                       help="Cell type A for within-condition comparison")
    parser.add_argument("--celltype_b", type=str, default=None,
                       help="Cell type B as reference for within-condition comparison")
    
    # Other arguments
    parser.add_argument("--output_dir", type=str, default="./output/deg",
                       help="Output directory")
    parser.add_argument("--annotation_col", type=str, default="fine_annotation",
                       help="Cell type annotation column name")
    parser.add_argument("--use_layer", type=str, default="X_log1p",
                       help="Layer to use for DEG computation")
    parser.add_argument("--method", type=str, default="wilcoxon",
                       help="DEG method (wilcoxon, t-test, etc.)")
    parser.add_argument("--n_top", type=int, default=3000,
                       help="Number of top genes to return")
    parser.add_argument("--use_raw", action="store_true",
                       help="Use raw data instead of preprocessed")
    
    args = parser.parse_args()
    
    # Determine comparison mode
    if args.celltype_a and args.celltype_b:
        # Mode 2: Compare different cell types within same condition
        mode = "celltype_comparison"
        
        if args.condition:
            condition = args.condition
        elif args.condition_a:
            condition = args.condition_a
        else:
            raise ValueError("Must specify --condition or --condition_a for celltype comparison")
        
        celltype_a = args.celltype_a
        celltype_b = args.celltype_b
        
        print("=" * 60)
        print("Compute Cell Type DEG (Within-Condition Comparison)")
        print("=" * 60)
        print(f"Condition: {condition}")
        print(f"Cell Type A: {celltype_a}")
        print(f"Cell Type B: {celltype_b} (reference)")
        print(f"Output Dir: {args.output_dir}")
        print("=" * 60)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"\n[1/3] Loading {condition} data...")
        adata = load_condition_data(condition, use_preprocessed=not args.use_raw)
        
        # Subset cell types
        print(f"\n[2/3] Subsetting cells...")
        print(f"  Subsetting to {celltype_a}...")
        sub_a = subset_by_celltype(adata, celltype_a, args.annotation_col)
        print(f"  Subsetting to {celltype_b}...")
        sub_b = subset_by_celltype(adata, celltype_b, args.annotation_col)
        
        if sub_a.n_obs == 0 or sub_b.n_obs == 0:
            print(f"[ERROR] No cells found for one or both cell types")
            return
        
        # Compute DEG
        print(f"\n[3/3] Computing DEG...")
        deg_df = compute_deg(
            adata_a=sub_a,
            adata_b=sub_b,
            group_a_name=celltype_a,
            group_b_name=celltype_b,
            use_layer=args.use_layer,
            method=args.method,
            n_top=args.n_top,
        )
        
        # Save results
        output_file = output_dir / f"{condition}_{celltype_a}_vs_{celltype_b}_DEG.csv"
        deg_df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved DEG results to: {output_file}")
        
        # Print statistics
        n_up = len(deg_df[deg_df["avg_logFC"] > 0])
        n_down = len(deg_df[deg_df["avg_logFC"] < 0])
        n_sig = len(deg_df[deg_df["p_val_adj"] < 0.05])
        
        print(f"\n[Summary]")
        print(f"  Total genes: {len(deg_df)}")
        print(f"  Up-regulated in {celltype_a}: {n_up}")
        print(f"  Down-regulated in {celltype_a}: {n_down}")
        print(f"  Significant (adj.p < 0.05): {n_sig}")
        
        # Show top genes
        print(f"\nTop 10 up-regulated genes in {celltype_a}:")
        print(deg_df.head(10)[['gene', 'avg_logFC', 'p_val_adj']].to_string(index=False))
        
        print(f"\nTop 10 down-regulated genes in {celltype_a}:")
        print(deg_df.tail(10)[['gene', 'avg_logFC', 'p_val_adj']].to_string(index=False))
        
        # Save metadata
        meta = {
            "mode": "celltype_comparison",
            "condition": condition,
            "celltype_a": celltype_a,
            "celltype_b": celltype_b,
            "n_cells_a": int(sub_a.n_obs),
            "n_cells_b": int(sub_b.n_obs),
            "n_genes": int(len(deg_df)),
            "n_up": int(n_up),
            "n_down": int(n_down),
            "n_significant": int(n_sig),
            "method": args.method,
            "use_layer": args.use_layer,
        }
        
        meta_file = output_dir / f"{condition}_{celltype_a}_vs_{celltype_b}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"\n[OK] Saved metadata to: {meta_file}")
        
    else:
        # Mode 1: Compare same cell type across different conditions
        mode = "condition_comparison"
        
        if not args.condition_a or not args.condition_b:
            raise ValueError("Must specify --condition_a and --condition_b for condition comparison")
        if not args.celltype:
            raise ValueError("Must specify --celltype for condition comparison")
        
        print("=" * 60)
        print("Compute Cell Type DEG (Cross-Condition Comparison)")
        print("=" * 60)
        print(f"Condition A: {args.condition_a}")
        print(f"Condition B: {args.condition_b}")
        print(f"Cell Type: {args.celltype}")
        print(f"Output Dir: {args.output_dir}")
        print("=" * 60)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"\n[1/4] Loading {args.condition_a} data...")
        adata_a = load_condition_data(args.condition_a, use_preprocessed=not args.use_raw)
        
        print(f"\n[2/4] Loading {args.condition_b} data...")
        adata_b = load_condition_data(args.condition_b, use_preprocessed=not args.use_raw)
        
        # Subset cell types
        print(f"\n[3/4] Subsetting to {args.celltype} cells...")
        sub_a = subset_by_celltype(adata_a, args.celltype, args.annotation_col)
        sub_b = subset_by_celltype(adata_b, args.celltype, args.annotation_col)
        
        if sub_a.n_obs == 0 or sub_b.n_obs == 0:
            print(f"[ERROR] No cells found for {args.celltype} in one or both conditions")
            return
        
        # Compute DEG
        print(f"\n[4/4] Computing DEG...")
        deg_df = compute_deg(
            adata_a=sub_a,
            adata_b=sub_b,
            group_a_name=args.condition_a,
            group_b_name=args.condition_b,
            use_layer=args.use_layer,
            method=args.method,
            n_top=args.n_top,
        )
        
        # Save results
        output_file = output_dir / f"{args.condition_a}_vs_{args.condition_b}_{args.celltype}_DEG.csv"
        deg_df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved DEG results to: {output_file}")
        
        # Print statistics
        n_up = len(deg_df[deg_df["avg_logFC"] > 0])
        n_down = len(deg_df[deg_df["avg_logFC"] < 0])
        n_sig = len(deg_df[deg_df["p_val_adj"] < 0.05])
        
        print(f"\n[Summary]")
        print(f"  Total genes: {len(deg_df)}")
        print(f"  Up-regulated in {args.condition_a}: {n_up}")
        print(f"  Down-regulated in {args.condition_a}: {n_down}")
        print(f"  Significant (adj.p < 0.05): {n_sig}")
        
        # Show top genes
        print(f"\nTop 10 up-regulated genes in {args.condition_a}:")
        print(deg_df.head(10)[['gene', 'avg_logFC', 'p_val_adj']].to_string(index=False))
        
        print(f"\nTop 10 down-regulated genes in {args.condition_a}:")
        print(deg_df.tail(10)[['gene', 'avg_logFC', 'p_val_adj']].to_string(index=False))
        
        # Save metadata
        meta = {
            "mode": "condition_comparison",
            "condition_a": args.condition_a,
            "condition_b": args.condition_b,
            "celltype": args.celltype,
            "n_cells_a": int(sub_a.n_obs),
            "n_cells_b": int(sub_b.n_obs),
            "n_genes": int(len(deg_df)),
            "n_up": int(n_up),
            "n_down": int(n_down),
            "n_significant": int(n_sig),
            "method": args.method,
            "use_layer": args.use_layer,
        }
        
        meta_file = output_dir / f"{args.condition_a}_vs_{args.condition_b}_{args.celltype}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"\n[OK] Saved metadata to: {meta_file}")


if __name__ == "__main__":
    main()
