"""
SpatialGT Agent UI — Main entry point.

Launch with:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parent
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from core._patch_torchtext import patch as _patch_tt  # noqa: E402
_patch_tt()

import os

import streamlit as st


def _check_runtime_env() -> tuple[bool, str]:
    """Check that critical runtime packages import cleanly."""
    msgs = []
    ok = True
    for mod in ("torch", "tensorboard", "streamlit",
                "plotly", "anndata", "scanpy"):
        try:
            __import__(mod)
        except Exception as e:
            ok = False
            msgs.append(f"`import {mod}` failed: {e}")
    return ok, "\n".join(msgs)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

st.set_page_config(
    page_title="SpatialGT",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


from core.session import init_session_state
init_session_state()

# Default GPU id 0 unless explicitly overridden.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# -- Sidebar: Navigation + status ---------------------------------------------
with st.sidebar:
    st.title("SpatialGT")
    st.caption("Spatial Perturbation Response Prediction")
    st.divider()
    _ok, _msg = _check_runtime_env()
    if not _ok:
        st.error("**Runtime environment problem detected:**\n\n" + _msg)
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            st.caption(f"Visible GPUs: `{visible}` | selected device: `{st.session_state.get('device', 'cuda:0')}`")
            for i in range(_torch.cuda.device_count()):
                try:
                    _gpu_name = _torch.cuda.get_device_name(i)
                except Exception:
                    _gpu_name = "?"
                st.caption(f"cuda:{i} — {_gpu_name}")
        else:
            st.warning("CUDA not available — running on CPU.")
    except Exception:
        pass
    st.info("Use the **Agent** page for AI-powered natural language control.")

# -- Main area: landing page ---------------------------------------------------
st.title("SpatialGT Virtual Perturbation Platform")
st.markdown(
    """
    Welcome to the **SpatialGT** interactive platform for spatial transcriptomics
    virtual perturbation analysis.

    ### Workflow

    1. **Upload Data** — Load your `.h5ad` spatial transcriptomics slice
    2. **Spatial Viewer** — Visualize spots and select perturbation targets
    3. **Perturbation Config** — Define perturbation (DEG / manual / template)
    4. **Run Inference** — Execute dual-line iterative inference with convergence detection
    5. **Results** — Explore and export perturbation effects

    Use the **pages** in the left sidebar to navigate through each step.
    The **AI Assistant** is available on every page to answer questions.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data Loaded", "Yes" if st.session_state.adata is not None else "No")
with col2:
    st.metric("Model Loaded", "Yes" if st.session_state.model is not None else "No")
with col3:
    n_sel = len(st.session_state.selected_spots)
    st.metric("Selected Spots", n_sel if n_sel > 0 else "—")
