"""Page 3: Configure perturbation parameters."""

import sys
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from core._patch_torchtext import patch as _ptt; _ptt()  # noqa

import streamlit as st
from core.session import init_session_state; init_session_state()
import pandas as pd
_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ui_utils.data_io import parse_deg_csv

st.set_page_config(page_title="Perturbation Config", page_icon="⚙️", layout="wide")
st.title("3. Perturbation Configuration")

if st.session_state.get("adata") is None:
    st.warning("Please upload data first (Page 1).")
    st.stop()

n_sel = len(st.session_state.get("selected_spots", []))
if n_sel == 0:
    st.warning("No spots selected. Please select perturbation targets in the Spatial Viewer (Page 2).")

st.info(f"**{n_sel}** spots selected for perturbation.")

# ── Initialize persistent config defaults in session_state ────────────────────
_DEFAULTS = {
    "pc_mode": "Upload DEG File",
    "pc_deg_df": None,
    "pc_deg_path": "",
    "pc_custom_edits": {},
    "pc_selected_genes": [],
    "pc_p_thresh": 0.1,
    "pc_min_logfc": 0.0,
    "pc_strength": 1.0,
    "pc_weighting": "uniform",
    "pc_max_steps": 20,
    "pc_stopping_mode": "auto_best",
    "pc_fixed_step": 10,
    "pc_convergence_threshold": 1e-5,
    "pc_patience": 3,
    "pc_emergency_div_factor": 10.0,
    "pc_freeze_perturbed": True,
    "pc_logfc_clip": 5.0,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

for cfg_key, widget_key in {
    "strength": "pc_strength",
    "weighting": "pc_weighting",
    "max_steps": "pc_max_steps",
    "stopping_mode": "pc_stopping_mode",
    "fixed_step": "pc_fixed_step",
    "convergence_threshold": "pc_convergence_threshold",
    "patience": "pc_patience",
    "emergency_div_factor": "pc_emergency_div_factor",
    "freeze_perturbed": "pc_freeze_perturbed",
    "logfc_clip": "pc_logfc_clip",
}.items():
    cfg = st.session_state.get("perturbation_config", {})
    if cfg.get(cfg_key) is not None:
        st.session_state[widget_key] = cfg[cfg_key]

# ── Perturbation mode ─────────────────────────────────────────────────────────
st.subheader("Perturbation Direction")
mode = st.radio(
    "How to define the perturbation?",
    ["Upload DEG File", "Manual Gene Editing", "Both (DEG + Manual Override)"],
    horizontal=True,
    key="pc_mode",
)

deg_df = st.session_state.pc_deg_df
custom_edits = st.session_state.pc_custom_edits

# ── DEG upload ────────────────────────────────────────────────────────────────
if mode in ("Upload DEG File", "Both (DEG + Manual Override)"):
    st.markdown("#### DEG File")
    st.caption("CSV with columns: gene, avg_logFC, (optional) p_val_adj")

    tab_deg_upload, tab_deg_path = st.tabs(["Upload CSV", "Server Path"])

    with tab_deg_upload:
        deg_file = st.file_uploader("Upload DEG CSV", type=["csv", "tsv", "txt"], key="pc_deg_uploader")
        if deg_file is not None:
            try:
                deg_df = parse_deg_csv(deg_file)
                st.session_state.pc_deg_df = deg_df
                st.success(f"Loaded {len(deg_df)} DEGs")
            except Exception as e:
                st.error(f"Failed to parse DEG file: {e}")

    with tab_deg_path:
        deg_path = st.text_input("DEG CSV path on server", key="pc_deg_path")
        if deg_path and Path(deg_path).exists():
            try:
                new_df = parse_deg_csv(deg_path)
                deg_df = new_df
                st.session_state.pc_deg_df = deg_df
                st.success(f"Loaded {len(deg_df)} DEGs from {deg_path}")
            except Exception as e:
                st.error(f"Failed to parse: {e}")

    if deg_df is not None:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            p_thresh = st.number_input(
                "p-value threshold", value=st.session_state.pc_p_thresh,
                min_value=0.001, max_value=1.0, step=0.01, key="pc_p_thresh",
            )
        with col_f2:
            min_logfc = st.number_input(
                "Min |logFC|", value=st.session_state.pc_min_logfc,
                min_value=0.0, step=0.1, key="pc_min_logfc",
            )

        if "p_val_adj" in deg_df.columns:
            filtered = deg_df[deg_df["p_val_adj"] < p_thresh]
        else:
            filtered = deg_df
        if min_logfc > 0:
            filtered = filtered[filtered["avg_logFC"].abs() >= min_logfc]

        st.markdown(f"**{len(filtered)}** genes pass filters (from {len(deg_df)} total)")

        with st.expander("Preview filtered DEGs", expanded=False):
            st.dataframe(filtered.head(50), use_container_width=True)

# ── Manual gene editing ───────────────────────────────────────────────────────
if mode in ("Manual Gene Editing", "Both (DEG + Manual Override)"):
    st.markdown("#### Manual Gene Editing")
    adata = st.session_state.adata
    gene_names = list(adata.var_names)

    selected_genes = st.multiselect(
        "Search and select genes to edit",
        options=gene_names,
        default=st.session_state.pc_selected_genes,
        max_selections=50,
        key="pc_selected_genes",
    )

    if selected_genes:
        st.markdown("Set logFC for each gene (positive = upregulate, negative = downregulate):")
        prev_edits = st.session_state.pc_custom_edits
        edit_df = pd.DataFrame({
            "gene": selected_genes,
            "logFC": [prev_edits.get(g, 0.0) for g in selected_genes],
        })
        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            num_rows="fixed",
            key="pc_gene_editor",
            column_config={
                "gene": st.column_config.TextColumn("Gene", disabled=True),
                "logFC": st.column_config.NumberColumn("logFC", min_value=-5.0, max_value=5.0, step=0.1),
            },
        )
        custom_edits = {row["gene"]: row["logFC"] for _, row in edited.iterrows() if row["logFC"] != 0.0}
        st.session_state.pc_custom_edits = custom_edits
        if custom_edits:
            st.markdown(f"**{len(custom_edits)}** genes with manual edits")

# ── Parameters ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Inference Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    strength = st.slider("Perturbation strength (α)", 0.1, 3.0,
                          st.session_state.pc_strength, 0.1, key="pc_strength")
    weighting = st.selectbox("Spatial weighting", ["uniform", "gaussian"],
                              index=["uniform", "gaussian"].index(st.session_state.pc_weighting),
                              key="pc_weighting")
with col2:
    stopping_mode = st.selectbox(
        "Stopping strategy",
        ["auto_best", "fixed_steps"],
        index=["auto_best", "fixed_steps"].index(st.session_state.pc_stopping_mode),
        help="auto_best selects the lowest-MSE step with patience; fixed_steps runs the specified number of steps.",
        key="pc_stopping_mode",
    )
    max_steps = st.slider("Max iteration steps", 5, 50,
                           st.session_state.pc_max_steps, key="pc_max_steps")
    if stopping_mode == "fixed_steps":
        fixed_step = st.slider("Use step", 1, max_steps, min(st.session_state.pc_fixed_step, max_steps), key="pc_fixed_step")
        convergence_thresh = st.session_state.pc_convergence_threshold
        patience = st.session_state.pc_patience
        emergency_div_factor = st.session_state.pc_emergency_div_factor
    else:
        fixed_step = None
        convergence_thresh = st.number_input("Min MSE improvement",
                                             value=st.session_state.pc_convergence_threshold,
                                             format="%.1e", key="pc_convergence_threshold")
        patience = st.number_input("Patience", value=st.session_state.pc_patience,
                                   min_value=1, max_value=20, step=1, key="pc_patience")
        emergency_div_factor = st.number_input("Emergency divergence factor",
                                               value=st.session_state.pc_emergency_div_factor,
                                               min_value=1.0, max_value=1000.0, step=1.0,
                                               key="pc_emergency_div_factor")
with col3:
    freeze_perturbed = st.checkbox("Freeze perturbed spots",
                                    value=st.session_state.pc_freeze_perturbed,
                                    key="pc_freeze_perturbed")
    logfc_clip = st.number_input("logFC clip", value=st.session_state.pc_logfc_clip,
                                  min_value=1.0, max_value=10.0, key="pc_logfc_clip")

# ── Save config to perturbation_config for inference page ─────────────────────
st.session_state.perturbation_config = {
    "deg_df": deg_df,
    "custom_edits": custom_edits,
    "strength": strength,
    "weighting": weighting,
    "max_steps": max_steps,
    "stopping_mode": stopping_mode,
    "fixed_step": fixed_step,
    "convergence_threshold": convergence_thresh,
    "patience": patience,
    "emergency_div_factor": emergency_div_factor,
    "freeze_perturbed": freeze_perturbed,
    "logfc_clip": logfc_clip,
    "p_thresh": st.session_state.pc_p_thresh,
    "min_logfc": st.session_state.pc_min_logfc,
}

st.divider()
ready = (deg_df is not None or len(custom_edits) > 0) and n_sel > 0
if ready:
    st.success("Configuration complete. Proceed to **Run Inference** (Page 4).")
else:
    missing = []
    if n_sel == 0:
        missing.append("select perturbation spots")
    if deg_df is None and len(custom_edits) == 0:
        missing.append("define perturbation direction (DEG or manual edits)")
    st.warning(f"Still need to: {', '.join(missing)}")
