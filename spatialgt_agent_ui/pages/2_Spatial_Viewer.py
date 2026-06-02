"""Page 2: Interactive spatial visualization with spot selection."""

import sys
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from core._patch_torchtext import patch as _ptt; _ptt()  # noqa

import streamlit as st
from core.session import init_session_state; init_session_state()
_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ui_utils.data_io import detect_celltype_columns, get_spatial_coords
from ui_utils.spatial_plot import spatial_scatter

st.set_page_config(page_title="Spatial Viewer", page_icon="🗺️", layout="wide")
st.title("2. Spatial Viewer & Spot Selection")

if st.session_state.get("adata") is None:
    st.warning("Please upload data first (Page 1).")
    st.stop()

adata = st.session_state.adata
coords = get_spatial_coords(adata)

# ── Color options ─────────────────────────────────────────────────────────────
ct_cols = detect_celltype_columns(adata)
color_col = None
if ct_cols:
    default_col = st.session_state.get("label_column")
    options = ["None"] + ct_cols
    default_idx = options.index(default_col) if default_col in options else (1 if ct_cols else 0)
    color_col = st.selectbox("Color by", options, index=default_idx)
    if color_col == "None":
        color_col = None

# ── Selection mode ────────────────────────────────────────────────────────────
st.markdown(
    """
    **How to select spots:**
    - Use the **lasso** or **box select** tool in the Plotly toolbar (top-right of the chart)
    - Or select by **cell type** using the panel below
    """
)

col_chart, col_panel = st.columns([3, 1])

with col_chart:
    fig = spatial_scatter(
        coords,
        color_values=adata.obs[color_col].values if color_col else None,
        color_label=color_col or "Spot",
        selected_indices=st.session_state.selected_spots,
        point_size=4,
        height=700,
        title="Select spots for perturbation",
    )
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="spatial_select",
    )

    if event and event.selection and event.selection.point_indices:
        # Plotly returns point indices across all traces; take from trace 0
        raw_indices = []
        for pts in event.selection.points:
            if hasattr(pts, "point_index"):
                raw_indices.append(pts.point_index)
        if not raw_indices:
            raw_indices = event.selection.point_indices
        st.session_state.selected_spots = sorted(set(raw_indices))

with col_panel:
    st.subheader("Selection Panel")
    n_sel = len(st.session_state.selected_spots)
    st.metric("Selected Spots", n_sel)

    if color_col and ct_cols:
        st.markdown("**Label counts:**")
        counts = adata.obs[color_col].astype(str).value_counts()
        st.dataframe(
            [{"label": k, "count": int(v)} for k, v in counts.items()],
            use_container_width=True,
            hide_index=True,
            height=220,
        )
        st.markdown("**Select by cell type:**")
        unique_types = sorted(adata.obs[color_col].unique())
        for ct in unique_types:
            if st.button(f"Select all {ct}", key=f"sel_{ct}"):
                mask = adata.obs[color_col] == ct
                indices = list(mask[mask].index)
                int_indices = [adata.obs.index.get_loc(i) for i in indices]
                st.session_state.selected_spots = sorted(set(int_indices))
                st.rerun()

        st.markdown("**Random select by label:**")
        rand_label = st.selectbox("Label", unique_types, key="sv_rand_label")
        rand_n = st.number_input("N", min_value=1, max_value=int(counts.max()), value=min(20, int(counts.max())), key="sv_rand_n")
        if st.button("Random select", key="sv_rand_select"):
            import random
            mask = adata.obs[color_col].astype(str) == str(rand_label)
            candidates = [adata.obs.index.get_loc(i) for i in mask[mask].index]
            chosen = sorted(random.sample(candidates, min(int(rand_n), len(candidates))))
            st.session_state.selected_spots = chosen
            st.rerun()

    st.divider()
    if st.button("Clear Selection"):
        st.session_state.selected_spots = []
        st.rerun()

    if n_sel > 0:
        st.markdown(f"First 10 indices: `{st.session_state.selected_spots[:10]}`")
