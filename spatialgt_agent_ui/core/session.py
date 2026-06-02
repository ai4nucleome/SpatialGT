"""Shared session state initialization for all pages."""

from copy import deepcopy

import streamlit as st


PERTURBATION_DEFAULTS = {
    "deg_df": None,
    "custom_edits": {},
    "strength": 1.0,
    "weighting": "uniform",
    "max_steps": 20,
    "batch_size": 64,
    "num_workers": 4,
    "stopping_mode": "auto_best",
    "fixed_step": None,
    "convergence_threshold": 1e-5,
    "patience": 3,
    "emergency_div_factor": 10.0,
    "freeze_perturbed": True,
    "logfc_clip": 5.0,
    "p_thresh": 0.1,
    "min_logfc": 0.0,
}


def init_session_state():
    defaults = {
        "adata": None,
        "adata_path": None,
        "cache_dir": None,
        "cache_mode": "lmdb",
        "lmdb_path": None,
        "lmdb_manifest_path": None,
        "databank": None,
        "model": None,
        "model_path": None,
        "device": "cuda:0",
        "visible_gpus": "",
        "gpu_info": [],
        "selected_spots": [],
        "label_path": None,
        "label_column": None,
        "label_counts": {},
        "perturbation_config": deepcopy(PERTURBATION_DEFAULTS),
        "inference_results": None,
        "task_status": "idle",
        "task_name": None,
        "task_message": "",
        "task_pid": None,
        "task_output_dir": None,
        "task_process": None,
        "task_thread": None,
        "task_stop_requested": False,
        "task_cleanup_paths": [],
        "last_error": None,
        "chat_history": [],
        "preprocessing_done": False,
        "finetuning_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    cfg = deepcopy(PERTURBATION_DEFAULTS)
    cfg.update(st.session_state.get("perturbation_config") or {})
    st.session_state.perturbation_config = cfg
