"""Page 5: Visualize and export perturbation results."""

import sys
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from core._patch_torchtext import patch as _ptt; _ptt()  # noqa

import streamlit as st
from core.session import init_session_state; init_session_state()

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ui_utils.data_io import get_spatial_coords
from ui_utils.spatial_plot import gene_expression_map

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("5. Results")

result = st.session_state.get("inference_results")
if result is None:
    st.warning("No inference results. Please run inference first (Page 4).")
    st.stop()

adata = st.session_state.adata
coords = get_spatial_coords(adata)

# ── Step selector ─────────────────────────────────────────────────────────────
available_steps = sorted(result.step_expressions.keys())
if not available_steps:
    st.error("No step expression data available.")
    st.stop()

best_step = (
    result.selected_step
    if getattr(result, "selected_step", None) is not None
    else (result.best_step if getattr(result, "best_step", None) is not None else result.converged_step - 1)
)
selected_step = st.select_slider(
    "Select inference step to view",
    options=available_steps,
    value=min(best_step, max(available_steps)),
)

# ── Build expression matrices ─────────────────────────────────────────────────

def _build_expr_vector(expr_dict, gene_ids_ref, n_spots):
    """Convert {global_idx: {gene_ids, raw_normed_values}} to matrix [n_spots, n_genes]."""
    n_genes = len(gene_ids_ref)
    mat = np.zeros((n_spots, n_genes), dtype=np.float32)
    for gidx, data in expr_dict.items():
        if int(gidx) < n_spots:
            mat[int(gidx)] = data["raw_normed_values"][:n_genes]
    return mat


n_spots = adata.n_obs
gene_ids_ref = result.gene_ids
if gene_ids_ref is None:
    st.error("Gene ID reference not available.")
    st.stop()

n_genes = len(gene_ids_ref)

baseline_mat = _build_expr_vector(result.baseline_expressions, gene_ids_ref, n_spots)
perturbed_mat = _build_expr_vector(result.step_expressions[selected_step], gene_ids_ref, n_spots)
delta_mat = perturbed_mat - baseline_mat

# ── Gene name mapping ─────────────────────────────────────────────────────────
import json
from core.model_manager import make_config
config = make_config(device="cpu")
with open(config.vocab_file) as f:
    vocab = json.load(f)
id_to_gene = {v: k for k, v in vocab.items()}
gene_names = [id_to_gene.get(int(gid), f"gene_{gid}") for gid in gene_ids_ref]

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Selected Step", (best_step + 1) if best_step is not None else result.converged_step)
with col2:
    st.metric("Total Steps", result.total_steps)
with col3:
    mean_delta = float(np.mean(np.abs(delta_mat)))
    st.metric("Mean |Δ Expression|", f"{mean_delta:.4f}")
with col4:
    n_affected = int(np.sum(np.any(np.abs(delta_mat) > 0.01, axis=1)))
    st.metric("Affected Spots", n_affected)

if getattr(result, "best_mse", None) is not None:
    st.caption(
        f"Stopping mode: {getattr(result, 'stopping_mode', 'auto_best')} | "
        f"Best MSE: {result.best_mse:.3e} | Reason: {getattr(result, 'stop_reason', 'N/A')}"
    )
if getattr(result, "output_dir", None):
    st.info(f"Result files saved in `{result.output_dir}`")
    if getattr(result, "summary_path", None):
        st.markdown(f"**Summary JSON**: `{result.summary_path}`")
    if getattr(result, "step_output_paths", None):
        paths_df = pd.DataFrame({
            "Step": [k + 1 for k in sorted(result.step_output_paths)],
            "Expression pickle path": [result.step_output_paths[k] for k in sorted(result.step_output_paths)],
        })
        with st.expander("Saved step expression files", expanded=True):
            st.dataframe(paths_df, use_container_width=True, hide_index=True)

# ── Convergence curve ─────────────────────────────────────────────────────────
with st.expander("Convergence Curve", expanded=True):
    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(
        x=list(range(1, len(result.step_mse) + 1)),
        y=result.step_mse,
        mode="lines+markers",
    ))
    fig_mse.update_layout(xaxis_title="Step", yaxis_title="MSE", yaxis_type="log", height=350)
    st.plotly_chart(fig_mse, use_container_width=True)
    if getattr(result, "update_mse", None):
        fig_update = go.Figure()
        fig_update.add_trace(go.Scatter(
            x=list(range(1, len(result.update_mse) + 1)),
            y=result.update_mse,
            mode="lines+markers",
        ))
        fig_update.update_layout(xaxis_title="Step", yaxis_title="Update MSE", yaxis_type="log", height=300)
        st.plotly_chart(fig_update, use_container_width=True)

# ── Gene expression viewer ───────────────────────────────────────────────────
st.divider()
st.subheader("Gene Expression Comparison")

gene_search = st.selectbox("Select a gene to visualize", options=gene_names)
if gene_search:
    gene_idx = gene_names.index(gene_search)

    col_before, col_after, col_diff = st.columns(3)
    with col_before:
        st.markdown("**Baseline (X₀)**")
        fig_b = gene_expression_map(coords, baseline_mat[:, gene_idx], f"{gene_search} (baseline)", point_size=3, height=450)
        st.plotly_chart(fig_b, use_container_width=True)
    with col_after:
        st.markdown(f"**Perturbed (Step {selected_step})**")
        fig_p = gene_expression_map(coords, perturbed_mat[:, gene_idx], f"{gene_search} (perturbed)", point_size=3, height=450)
        st.plotly_chart(fig_p, use_container_width=True)
    with col_diff:
        st.markdown("**Δ Expression**")
        fig_d = gene_expression_map(coords, delta_mat[:, gene_idx], f"{gene_search} (Δ)", point_size=3, height=450, colorscale="RdBu_r")
        st.plotly_chart(fig_d, use_container_width=True)

# ── Top changed genes ────────────────────────────────────────────────────────
st.divider()
st.subheader("Top Changed Genes")

mean_abs_delta = np.mean(np.abs(delta_mat), axis=0)
top_k = st.slider("Number of top genes to show", 10, 100, 30)
top_indices = np.argsort(mean_abs_delta)[::-1][:top_k]

top_df = pd.DataFrame({
    "Gene": [gene_names[i] for i in top_indices],
    "Mean |Δ|": [mean_abs_delta[i] for i in top_indices],
    "Mean Baseline": [np.mean(baseline_mat[:, i]) for i in top_indices],
    "Mean Perturbed": [np.mean(perturbed_mat[:, i]) for i in top_indices],
})
st.dataframe(top_df, use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Export Results")

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    if st.button("Download Expression CSV"):
        export_df = pd.DataFrame(perturbed_mat, columns=gene_names)
        export_df.insert(0, "spot_index", range(n_spots))
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV",
            csv_buf.getvalue(),
            file_name=f"perturbed_expression_step{selected_step}.csv",
            mime="text/csv",
        )

with col_ex2:
    if st.button("Download Delta NPZ"):
        buf = io.BytesIO()
        np.savez(buf,
                 baseline=baseline_mat,
                 perturbed=perturbed_mat,
                 delta=delta_mat,
                 gene_names=np.array(gene_names),
                 perturbed_spots=np.array(result.perturbed_spots))
        st.download_button(
            "Download NPZ",
            buf.getvalue(),
            file_name=f"perturbation_results_step{selected_step}.npz",
            mime="application/octet-stream",
        )
