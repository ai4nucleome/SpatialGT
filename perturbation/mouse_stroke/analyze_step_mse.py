#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Step-wise MSE for Non-Perturbed Spots

This script calculates the MSE (Mean Squared Error) and MAE (Mean Absolute Error)
between adjacent steps for NON-PERTURBED spots only.

This helps observe how perturbation propagates through the tissue during iteration,
by measuring expression changes in spots that were NOT initially perturbed.

Usage:
    python analyze_step_mse.py --expr_dirs <expr_dir1> [expr_dir2 ...]
    python analyze_step_mse.py --expr_dirs output/expression --output_dir output/analysis

Note:
    The expression CSV files must contain an "is_perturbed" column (0 or 1).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List, Optional


def load_expression_data(expr_dir: Path, step: int) -> pd.DataFrame:
    """Load expression data for a specific step."""
    file_path = expr_dir / f"pert{step}_expression.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Expression file not found: {file_path}")
    
    df = pd.read_csv(file_path, index_col=0)
    return df


def get_non_perturbed_mask(df: pd.DataFrame) -> pd.Series:
    """Get boolean mask for non-perturbed spots."""
    if "is_perturbed" in df.columns:
        return df["is_perturbed"] == 0
    else:
        # If is_perturbed column doesn't exist, use all spots
        print("  [WARN] 'is_perturbed' column not found, using all spots")
        return pd.Series([True] * len(df), index=df.index)


def calculate_step_mse_non_perturbed(
    expr_dir: Path, 
    max_step: int = 10
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate MSE between adjacent steps for non-perturbed spots only.
    
    Args:
        expr_dir: Directory containing expression CSV files
        max_step: Maximum step to analyze
    
    Returns:
        results_df: DataFrame with step-wise MSE/MAE statistics
        summary: Dictionary with summary statistics
    """
    results = []
    
    # Load step 0
    prev_df = load_expression_data(expr_dir, 0)
    
    # Get numeric columns (exclude metadata columns)
    non_numeric_cols = ["is_perturbed", "cell_type", "cluster", "region"]
    numeric_cols = [c for c in prev_df.columns if c not in non_numeric_cols and prev_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    # Get non-perturbed mask from step 0 (should be consistent across steps)
    non_perturbed_mask = get_non_perturbed_mask(prev_df)
    n_non_perturbed = non_perturbed_mask.sum()
    n_perturbed = len(prev_df) - n_non_perturbed
    
    print(f"[INFO] Total spots: {len(prev_df)}")
    print(f"[INFO] Non-perturbed spots: {n_non_perturbed}")
    print(f"[INFO] Perturbed spots: {n_perturbed}")
    print(f"[INFO] Numeric columns: {len(numeric_cols)}")
    
    for step in range(1, max_step + 1):
        try:
            curr_df = load_expression_data(expr_dir, step)
        except FileNotFoundError:
            print(f"  [WARN] Step {step} not found, stopping")
            break
        
        # Ensure index alignment
        common_idx = prev_df.index.intersection(curr_df.index)
        
        # Filter to non-perturbed spots only
        non_perturbed_idx = [idx for idx in common_idx if non_perturbed_mask.loc[idx]]
        
        if len(non_perturbed_idx) == 0:
            print(f"  [WARN] No non-perturbed spots found in step {step}")
            continue
        
        # Get numeric values for non-perturbed spots
        prev_vals = prev_df.loc[non_perturbed_idx, numeric_cols].values
        curr_vals = curr_df.loc[non_perturbed_idx, numeric_cols].values
        
        # Calculate MSE and MAE per spot
        diff = curr_vals - prev_vals
        mse_per_spot = np.mean(diff ** 2, axis=1)
        mae_per_spot = np.mean(np.abs(diff), axis=1)
        
        # Also calculate per-gene statistics
        mse_per_gene = np.mean(diff ** 2, axis=0)
        mae_per_gene = np.mean(np.abs(diff), axis=0)
        
        results.append({
            "step_from": step - 1,
            "step_to": step,
            "step_pair": f"{step-1}→{step}",
            # Spot-level statistics
            "mean_mse": np.mean(mse_per_spot),
            "std_mse": np.std(mse_per_spot),
            "median_mse": np.median(mse_per_spot),
            "max_mse": np.max(mse_per_spot),
            "mean_mae": np.mean(mae_per_spot),
            "std_mae": np.std(mae_per_spot),
            "median_mae": np.median(mae_per_spot),
            "max_mae": np.max(mae_per_spot),
            # Gene-level statistics
            "mean_mse_gene": np.mean(mse_per_gene),
            "std_mse_gene": np.std(mse_per_gene),
            "mean_mae_gene": np.mean(mae_per_gene),
            "std_mae_gene": np.std(mae_per_gene),
            # Counts
            "n_spots": len(non_perturbed_idx),
            "n_genes": len(numeric_cols),
        })
        
        prev_df = curr_df
    
    results_df = pd.DataFrame(results)
    
    summary = {
        "total_spots": int(len(prev_df)),
        "n_non_perturbed": int(n_non_perturbed),
        "n_perturbed": int(n_perturbed),
        "n_genes": int(len(numeric_cols)),
        "n_steps_analyzed": int(len(results)),
    }
    
    return results_df, summary


def plot_mse_trend(
    results_df: pd.DataFrame, 
    output_path: Path, 
    title: str = "Non-Perturbed Spots MSE Analysis"
):
    """Plot MSE/MAE trend between adjacent steps."""
    if len(results_df) == 0:
        print("[WARN] No results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = results_df["step_to"].values
    
    # MSE plot
    ax1 = axes[0]
    ax1.errorbar(steps, results_df["mean_mse"], yerr=results_df["std_mse"], 
                 marker='o', capsize=3, capthick=1, color='#2E86AB', 
                 linewidth=2, markersize=8, label='Mean ± Std')
    ax1.plot(steps, results_df["median_mse"], 
             marker='s', color='#E94F37', linewidth=1.5, 
             markersize=6, linestyle='--', alpha=0.7, label='Median')
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("MSE (per non-perturbed spot)", fontsize=12)
    ax1.set_title("MSE Between Adjacent Steps\n(Non-Perturbed Spots Only)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(steps)
    
    # MAE plot
    ax2 = axes[1]
    ax2.errorbar(steps, results_df["mean_mae"], yerr=results_df["std_mae"], 
                 marker='o', capsize=3, capthick=1, color='#2E86AB', 
                 linewidth=2, markersize=8, label='Mean ± Std')
    ax2.plot(steps, results_df["median_mae"], 
             marker='s', color='#E94F37', linewidth=1.5, 
             markersize=6, linestyle='--', alpha=0.7, label='Median')
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("MAE (per non-perturbed spot)", fontsize=12)
    ax2.set_title("MAE Between Adjacent Steps\n(Non-Perturbed Spots Only)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(steps)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved plot: {output_path}")


def analyze_single_experiment(
    expr_dir: Path, 
    output_dir: Optional[Path] = None, 
    exp_name: str = "",
    max_step: int = 10,
) -> pd.DataFrame:
    """Analyze a single experiment."""
    if output_dir is None:
        output_dir = expr_dir.parent
    
    print(f"\n[INFO] Analyzing: {expr_dir}")
    
    # Calculate MSE for non-perturbed spots
    results_df, summary = calculate_step_mse_non_perturbed(expr_dir, max_step)
    
    if len(results_df) == 0:
        print("[WARN] No results generated")
        return results_df
    
    # Save CSV
    csv_path = output_dir / "step_mse_non_perturbed.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved: {csv_path}")
    
    # Save summary
    summary_path = output_dir / "step_mse_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved: {summary_path}")
    
    # Plot
    plot_path = output_dir / "step_mse_non_perturbed.png"
    plot_mse_trend(results_df, plot_path, title=exp_name or "Non-Perturbed Spots MSE Analysis")
    
    # Print results
    print("\n[Results - Non-Perturbed Spots MSE]")
    print(results_df[["step_pair", "mean_mse", "std_mse", "mean_mae", "n_spots"]].to_string(index=False))
    
    return results_df


def analyze_multiple_experiments(
    base_dirs: List[str], 
    output_dir: Path,
    max_step: int = 10,
) -> pd.DataFrame:
    """Analyze multiple experiments and compare."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        
        # Find expression directory
        if (base_path / "expression").exists():
            expr_dir = base_path / "expression"
            exp_name = base_path.name
        elif base_path.name == "expression":
            expr_dir = base_path
            exp_name = base_path.parent.name
        else:
            print(f"[WARN] Cannot find expression dir in: {base_path}")
            continue
        
        try:
            results_df, _ = calculate_step_mse_non_perturbed(expr_dir, max_step)
            if len(results_df) > 0:
                results_df["experiment"] = exp_name
                all_results.append(results_df)
                print(f"[OK] Processed: {exp_name}")
        except Exception as e:
            print(f"[ERROR] Failed to process {expr_dir}: {e}")
    
    if not all_results:
        print("[ERROR] No valid experiments found")
        return pd.DataFrame()
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / "combined_step_mse_non_perturbed.csv", index=False)
    print(f"\n[OK] Saved combined results: {output_dir / 'combined_step_mse_non_perturbed.csv'}")
    
    # Plot comparison
    plot_comparison(combined_df, output_dir)
    
    # Print summary
    print("\n[Summary by Step]")
    summary = combined_df.groupby("step_to").agg({
        "mean_mse": ["mean", "std"],
        "mean_mae": ["mean", "std"],
    }).round(6)
    print(summary)
    
    return combined_df


def plot_comparison(combined_df: pd.DataFrame, output_dir: Path):
    """Plot comparison across multiple experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    experiments = combined_df["experiment"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # MSE comparison
    ax1 = axes[0]
    for i, exp in enumerate(experiments):
        subset = combined_df[combined_df["experiment"] == exp]
        ax1.plot(subset["step_to"], subset["mean_mse"], 
                 marker='o', label=exp, color=colors[i], linewidth=2, markersize=6)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Mean MSE", fontsize=12)
    ax1.set_title("MSE Comparison (Non-Perturbed Spots)", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # MAE comparison
    ax2 = axes[1]
    for i, exp in enumerate(experiments):
        subset = combined_df[combined_df["experiment"] == exp]
        ax2.plot(subset["step_to"], subset["mean_mae"], 
                 marker='s', label=exp, color=colors[i], linewidth=2, markersize=6)
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Mean MAE", fontsize=12)
    ax2.set_title("MAE Comparison (Non-Perturbed Spots)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "step_mse_comparison_non_perturbed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved comparison plot: {output_dir / 'step_mse_comparison_non_perturbed.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze step-wise MSE for non-perturbed spots"
    )
    parser.add_argument(
        "--expr_dirs", type=str, nargs="+", required=True,
        help="Expression directories to analyze (contain pert0_expression.csv, etc.)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_step", type=int, default=10,
        help="Maximum step to analyze (default: 10)"
    )
    
    args = parser.parse_args()
    
    if len(args.expr_dirs) == 1:
        # Single experiment
        expr_dir = Path(args.expr_dirs[0])
        if expr_dir.name != "expression":
            if (expr_dir / "expression").exists():
                expr_dir = expr_dir / "expression"
        output_dir = Path(args.output_dir) if args.output_dir else expr_dir.parent
        analyze_single_experiment(expr_dir, output_dir, max_step=args.max_step)
    else:
        # Multiple experiments
        output_dir = Path(args.output_dir) if args.output_dir else Path("./step_mse_analysis")
        analyze_multiple_experiments(args.expr_dirs, output_dir, max_step=args.max_step)


if __name__ == "__main__":
    main()
