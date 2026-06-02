"""Page 1: Upload and preview spatial transcriptomics data."""

import sys
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from core._patch_torchtext import patch as _ptt; _ptt()  # noqa

import streamlit as st
from core.session import init_session_state; init_session_state()
import tempfile
_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ui_utils.data_io import attach_labels_to_adata, load_h5ad, detect_celltype_columns, get_spatial_coords
from ui_utils.spatial_plot import spatial_scatter

st.set_page_config(page_title="Upload Data", page_icon="📂", layout="wide")
st.title("1. Upload Data")

# ── Data source selection ─────────────────────────────────────────────────────
tab_upload, tab_local = st.tabs(["Upload .h5ad File", "Server Local Path"])

with tab_upload:
    uploaded = st.file_uploader("Choose an .h5ad file", type=["h5ad", "h5"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        st.session_state.adata_path = tmp_path

with tab_local:
    local_path = st.text_input(
        "Path to .h5ad on server",
        placeholder="/path/to/your/data.h5ad",
    )
    if local_path and Path(local_path).exists():
        st.session_state.adata_path = local_path
    elif local_path:
        st.warning("File not found on server.")

# ── Load and preview ──────────────────────────────────────────────────────────
if st.session_state.get("adata_path"):
    path = st.session_state.adata_path

    if st.session_state.adata is None or st.session_state.get("_last_loaded") != path:
        with st.spinner("Loading data..."):
            adata = load_h5ad(path)
            st.session_state.adata = adata
            st.session_state._last_loaded = path

    adata = st.session_state.adata

    st.success(f"Loaded: **{adata.n_obs}** spots, **{adata.n_vars}** genes")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Overview")
        st.markdown(f"- **Spots**: {adata.n_obs}")
        st.markdown(f"- **Genes**: {adata.n_vars}")
        ct_cols = detect_celltype_columns(adata)
        if ct_cols:
            st.markdown(f"- **Cell type columns**: {', '.join(ct_cols)}")
        else:
            st.markdown("- **Cell type columns**: None detected")
        st.markdown(f"- **obs columns**: {', '.join(adata.obs.columns[:20].tolist())}")

    with col2:
        st.subheader("Spatial Preview")
        try:
            coords = get_spatial_coords(adata)
            color_col = None
            if ct_cols:
                color_col = ct_cols[0]
            fig = spatial_scatter(
                coords,
                color_values=adata.obs[color_col].values if color_col else None,
                color_label=color_col or "Cell Type",
                point_size=3,
                height=500,
                title="Spatial Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Cannot plot spatial: {e}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Preprocessing")

    cache_dir = st.text_input(
        "Cache directory",
        value=st.session_state.get("cache_dir") or str(_REPO_ROOT / "spatialgt_agent_ui" / "workspace" / "cache"),
        key="p1_cache_dir",
    )

    col_prep1, col_prep2, col_prep3 = st.columns(3)
    with col_prep1:
        max_neighbors = st.number_input("Max neighbors (k)", value=8, min_value=4, max_value=20)
    with col_prep2:
        force_rebuild = st.checkbox("Force rebuild", value=False)
    with col_prep3:
        build_lmdb_flag = st.checkbox(
            "Build LMDB cache (default)",
            value=True,
            help="Builds an LMDB database after h5 preprocessing. This is the default runtime cache mode.",
        )

    if st.button("Run Preprocessing", type="primary"):
        from core.preprocessor import preprocess_h5ad, build_lmdb
        from core.model_manager import make_config

        config = make_config(cache_dir=cache_dir, device=st.session_state.device, cache_mode="h5")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)

        with st.spinner("Preprocessing (h5)..."):
            preprocess_h5ad(
                h5ad_path=path,
                cache_dir=cache_dir,
                config=config,
                max_neighbors=max_neighbors,
                force_rebuild=force_rebuild,
                progress_callback=_progress,
            )

        st.session_state.cache_dir = cache_dir
        st.session_state.cache_mode = "h5"
        st.session_state.lmdb_path = None
        st.session_state.lmdb_manifest_path = None
        st.session_state.preprocessing_done = True
        st.success("H5 preprocessing complete!")

        if build_lmdb_flag:
            lmdb_progress = st.progress(0.0)
            lmdb_status = st.empty()

            def _lmdb_progress(pct, msg):
                lmdb_progress.progress(pct)
                lmdb_status.text(msg)

            with st.spinner("Building LMDB cache..."):
                lmdb_path, manifest_path = build_lmdb(
                    cache_dir=cache_dir,
                    config=config,
                    progress_callback=_lmdb_progress,
                )

            st.session_state.cache_mode = "lmdb"
            st.session_state.lmdb_path = lmdb_path
            st.session_state.lmdb_manifest_path = manifest_path
            st.success(f"LMDB cache built at `{lmdb_path}`")

    # ── Optional labels ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Labels / Annotations")
    label_path = st.text_input(
        "Label CSV path on server",
        value=st.session_state.get("label_path") or "",
        placeholder="/path/to/labels.csv",
        key="p1_label_path",
    )
    label_col = st.text_input(
        "Label column (optional)",
        value=st.session_state.get("label_column") or "",
        placeholder="Region",
        key="p1_label_column",
    )
    if st.button("Load Labels"):
        try:
            out_col, counts = attach_labels_to_adata(adata, label_path, label_col or None)
            st.session_state.adata = adata
            st.session_state.label_path = label_path
            st.session_state.label_column = out_col
            st.session_state.label_counts = counts
            st.success(f"Labels loaded into `{out_col}`")
            st.dataframe(
                [{"label": k, "count": v} for k, v in counts.items()],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Failed to load labels: {e}")

    # Show current cache status
    if st.session_state.preprocessing_done:
        mode = st.session_state.cache_mode
        st.info(f"Current cache mode: **{mode.upper()}**"
                + (f" | LMDB: `{st.session_state.lmdb_path}`" if mode == "lmdb" else ""))

    # ── Model selection ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Selection")

    model_source = st.radio(
        "Model source",
        ["HuggingFace Pretrained", "Local Checkpoint"],
        horizontal=True,
    )

    def _current_config():
        from core.model_manager import make_config
        return make_config(
            cache_dir=cache_dir,
            device=st.session_state.device,
            cache_mode=st.session_state.get("cache_mode", "h5"),
            lmdb_path=st.session_state.get("lmdb_path"),
            lmdb_manifest_path=st.session_state.get("lmdb_manifest_path"),
        )

    if model_source == "HuggingFace Pretrained":
        if st.button("Download & Load Pretrained Model"):
            from core.model_manager import download_pretrained, load_model
            with st.spinner("Downloading pretrained model from HuggingFace..."):
                ckpt_dir = download_pretrained()
            config = _current_config()
            with st.spinner("Loading model..."):
                model = load_model(ckpt_dir, config, st.session_state.device)
            st.session_state.model = model
            st.session_state.model_path = str(ckpt_dir)
            st.success(f"Model loaded from {ckpt_dir}")
    else:
        ckpt_input = st.text_input(
            "Checkpoint path",
            value=str(_REPO_ROOT / "output" / "model_2" / "checkpoint-94620"),
        )
        if st.button("Load Local Model"):
            from core.model_manager import load_model
            config = _current_config()
            with st.spinner("Loading model..."):
                model = load_model(ckpt_input, config, st.session_state.device)
            st.session_state.model = model
            st.session_state.model_path = ckpt_input
            st.success(f"Model loaded from {ckpt_input}")
