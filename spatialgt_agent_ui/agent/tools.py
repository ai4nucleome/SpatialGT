"""
Agent tool definitions: functions the LLM agent can invoke.
Each tool returns a dict with 'success' and 'message' keys.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import numpy as np

_UI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _find_h5ad(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        matches = sorted(p.glob("*.h5ad"))
        if not matches:
            raise FileNotFoundError(f"No .h5ad file found in directory: {p}")
        return matches[0]
    return p


def _sync_config_widgets(cfg: Dict[str, Any]) -> None:
    """Keep manual Streamlit controls aligned with agent-written config."""
    mapping = {
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
        "p_thresh": "pc_p_thresh",
        "min_logfc": "pc_min_logfc",
    }
    for cfg_key, widget_key in mapping.items():
        if cfg_key in cfg and cfg[cfg_key] is not None:
            st.session_state[widget_key] = cfg[cfg_key]
    if cfg.get("custom_edits") is not None:
        edits = dict(cfg.get("custom_edits") or {})
        st.session_state.pc_custom_edits = edits
        st.session_state.pc_selected_genes = list(edits.keys())
    if cfg.get("deg_df") is not None:
        st.session_state.pc_deg_df = cfg["deg_df"]


def _set_task(status: str, name: str | None = None, message: str = "") -> None:
    st.session_state.task_status = status
    st.session_state.task_name = name
    st.session_state.task_message = message
    st.session_state.last_error = None if status != "error" else message


TOOL_DEFINITIONS = [
    {
        "name": "load_data",
        "description": "Load a .h5ad spatial transcriptomics file, or a directory containing one .h5ad file",
        "parameters": {
            "path": {"type": "string", "description": "Path to .h5ad file or dataset directory on server", "required": True},
        },
    },
    {
        "name": "detect_gpus",
        "description": "Detect available GPUs and current device selection",
        "parameters": {},
    },
    {
        "name": "set_device",
        "description": "Set the device or visible GPU list for model loading, finetuning, and inference",
        "parameters": {
            "device": {"type": "string", "description": "Device such as 'cuda:0' or 'cpu'"},
            "visible_gpus": {"type": "string", "description": "Comma-separated physical GPU ids, e.g. '0' or '0,1'"},
        },
    },
    {
        "name": "preprocess",
        "description": "Preprocess the loaded data (build neighbor graph, spot cache). Must load_data first.",
        "parameters": {
            "cache_dir": {"type": "string", "description": "Cache directory path (optional, auto-generated if omitted)"},
            "max_neighbors": {"type": "integer", "description": "K for KNN graph (default 8)"},
            "build_lmdb": {"type": "boolean", "description": "Also build LMDB cache for faster inference (default true)"},
        },
    },
    {
        "name": "load_labels",
        "description": "Load a spot label CSV and attach it to the loaded AnnData obs",
        "parameters": {
            "path": {"type": "string", "description": "Path to label CSV, e.g. labels.csv", "required": True},
            "label_column": {"type": "string", "description": "Optional label column, e.g. Region"},
        },
    },
    {
        "name": "load_model",
        "description": "Load SpatialGT model weights",
        "parameters": {
            "source": {"type": "string", "description": "'huggingface' or 'local'", "required": True},
            "path": {"type": "string", "description": "Local checkpoint path (only needed if source='local')"},
        },
    },
    {
        "name": "select_spots_by_type",
        "description": "Select perturbation spots by cell type annotation",
        "parameters": {
            "cell_type": {"type": "string", "description": "Cell type name to select", "required": True},
            "column": {"type": "string", "description": "obs column name with cell type labels (auto-detected if omitted)"},
        },
    },
    {
        "name": "select_spots_random",
        "description": "Randomly select N spots for perturbation",
        "parameters": {
            "n": {"type": "integer", "description": "Number of spots to select", "required": True},
        },
    },
    {
        "name": "select_spots_by_label_random",
        "description": "Randomly select N spots from a specific label value",
        "parameters": {
            "label": {"type": "string", "description": "Label value, e.g. cancer", "required": True},
            "n": {"type": "integer", "description": "Number of spots to select", "required": True},
            "column": {"type": "string", "description": "Label column, defaults to loaded label column or auto-detected"},
            "seed": {"type": "integer", "description": "Random seed (default 42)"},
        },
    },
    {
        "name": "select_spots_by_indices",
        "description": "Select specific spot indices for perturbation",
        "parameters": {
            "indices": {"type": "array", "description": "List of spot indices", "required": True},
        },
    },
    {
        "name": "set_deg_file",
        "description": "Load a DEG CSV file to define perturbation direction",
        "parameters": {
            "path": {"type": "string", "description": "Path to DEG CSV file", "required": True},
            "p_thresh": {"type": "number", "description": "p-value threshold (default 0.1)"},
            "min_logfc": {"type": "number", "description": "Minimum absolute logFC (default 0)"},
        },
    },
    {
        "name": "set_gene_edits",
        "description": "Manually set logFC for specific genes",
        "parameters": {
            "edits": {"type": "object", "description": "Dict of {gene_name: logFC_value}", "required": True},
        },
    },
    {
        "name": "set_perturbation_params",
        "description": "Set perturbation inference parameters",
        "parameters": {
            "strength": {"type": "number", "description": "Perturbation strength alpha (default 1.0)"},
            "weighting": {"type": "string", "description": "'uniform' or 'gaussian' (default 'uniform')"},
            "max_steps": {"type": "integer", "description": "Max iteration steps (default 20)"},
            "stopping_mode": {"type": "string", "description": "'auto_best' to stop by best-step patience or 'fixed_steps' to run a specified number of steps"},
            "fixed_step": {"type": "integer", "description": "Step to use in fixed_steps mode (optional; defaults to max_steps)"},
            "convergence_threshold": {"type": "number", "description": "MSE convergence threshold (default 1e-5)"},
            "patience": {"type": "integer", "description": "Auto-best patience for non-improving MSE steps"},
            "emergency_div_factor": {"type": "number", "description": "Emergency stop if MSE exceeds this factor times best MSE"},
            "freeze_perturbed": {"type": "boolean", "description": "Freeze perturbed spots (default true)"},
            "logfc_clip": {"type": "number", "description": "Max absolute logFC (default 5.0)"},
        },
    },
    {
        "name": "finetune_model",
        "description": "Finetune the currently loaded model on the current slice and reload the latest checkpoint",
        "parameters": {
            "epochs": {"type": "integer", "description": "Number of finetuning epochs (default 5)"},
            "batch_size": {"type": "integer", "description": "Batch size (default 32)"},
            "device": {"type": "string", "description": "Device override, e.g. cuda:0"},
        },
    },
    {
        "name": "run_inference",
        "description": "Execute the perturbation inference pipeline. Requires: data loaded, preprocessed, model loaded, spots selected, perturbation configured.",
        "parameters": {},
    },
    {
        "name": "stop_task",
        "description": "Stop the current finetuning or inference task and clean partial outputs",
        "parameters": {},
    },
    {
        "name": "get_status",
        "description": "Get current pipeline status: what's loaded, configured, and ready",
        "parameters": {},
    },
    {
        "name": "get_cell_types",
        "description": "List available cell type annotations in the loaded data",
        "parameters": {},
    },
    {
        "name": "get_gene_list",
        "description": "Search for genes in the loaded data",
        "parameters": {
            "query": {"type": "string", "description": "Gene name search query (partial match)"},
            "limit": {"type": "integer", "description": "Max results to return (default 20)"},
        },
    },
]


def get_tools_description() -> str:
    """Format tool definitions for the system prompt."""
    lines = []
    for t in TOOL_DEFINITIONS:
        params = t.get("parameters", {})
        param_strs = []
        for pname, pdef in params.items():
            req = " (REQUIRED)" if pdef.get("required") else ""
            param_strs.append(f"    - {pname}: {pdef['type']} — {pdef['description']}{req}")
        param_block = "\n".join(param_strs) if param_strs else "    (no parameters)"
        lines.append(f"- **{t['name']}**: {t['description']}\n{param_block}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch and execute a tool by name."""
    dispatch = {
        "load_data": _tool_load_data,
        "detect_gpus": _tool_detect_gpus,
        "set_device": _tool_set_device,
        "preprocess": _tool_preprocess,
        "load_labels": _tool_load_labels,
        "load_model": _tool_load_model,
        "select_spots_by_type": _tool_select_by_type,
        "select_spots_random": _tool_select_random,
        "select_spots_by_label_random": _tool_select_by_label_random,
        "select_spots_by_indices": _tool_select_by_indices,
        "set_deg_file": _tool_set_deg,
        "set_gene_edits": _tool_set_gene_edits,
        "set_perturbation_params": _tool_set_params,
        "finetune_model": _tool_finetune_model,
        "run_inference": _tool_run_inference,
        "stop_task": _tool_stop_task,
        "get_status": _tool_get_status,
        "get_cell_types": _tool_get_cell_types,
        "get_gene_list": _tool_get_gene_list,
    }
    fn = dispatch.get(name)
    if fn is None:
        return {"success": False, "message": f"Unknown tool: {name}"}
    try:
        return fn(**params)
    except Exception as e:
        return {"success": False, "message": f"Tool error ({name}): {e}\n{traceback.format_exc()[:500]}"}


def _tool_load_data(path: str) -> Dict:
    from ui_utils.data_io import load_h5ad, detect_celltype_columns
    p = _find_h5ad(path)
    if not p.exists():
        return {"success": False, "message": f"File not found: {p}"}
    adata = load_h5ad(p)
    st.session_state.adata = adata
    st.session_state.adata_path = str(p)
    st.session_state.inference_results = None
    st.session_state.selected_spots = []
    ct_cols = detect_celltype_columns(adata)
    return {
        "success": True,
        "message": f"Loaded {adata.n_obs} spots, {adata.n_vars} genes. "
                   f"Cell type columns: {ct_cols if ct_cols else 'none detected'}",
    }


def _tool_detect_gpus() -> Dict:
    try:
        import torch
    except Exception as e:
        st.session_state.gpu_info = []
        return {
            "success": True,
            "message": f"PyTorch is not importable in this process ({e}). Use the run.sh SpatialGT environment for GPU detection.",
            "gpus": [],
        }

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            })
    st.session_state.gpu_info = gpus
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    st.session_state.visible_gpus = visible
    if not gpus:
        return {"success": True, "message": "No CUDA GPU detected. Current device: cpu", "gpus": []}
    lines = [f"Current device: {st.session_state.get('device', 'cuda:0')} | visible={visible or 'all'}"]
    lines += [f"GPU {g['index']}: {g['name']} ({g['total_memory_gb']} GB)" for g in gpus]
    return {"success": True, "message": "\n".join(lines), "gpus": gpus}


def _tool_set_device(device: str = None, visible_gpus: str = None) -> Dict:
    if visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        st.session_state.visible_gpus = visible_gpus
        if device is None:
            device = "cuda:0"
    if device:
        st.session_state.device = device
        st.session_state.model = None
        st.session_state.model_path = None
        msg = f"Device set to {device}. Model was cleared and must be reloaded on the new device."
    else:
        msg = f"Device unchanged: {st.session_state.get('device', 'cuda:0')}"
    return {"success": True, "message": msg}


def _tool_load_labels(path: str, label_column: str = None) -> Dict:
    adata = st.session_state.get("adata")
    if adata is None:
        return {"success": False, "message": "No data loaded. Load .h5ad before labels."}
    p = Path(path)
    if not p.exists():
        return {"success": False, "message": f"Label file not found: {path}"}
    from ui_utils.data_io import attach_labels_to_adata
    col, counts = attach_labels_to_adata(adata, p, label_column)
    st.session_state.adata = adata
    st.session_state.label_path = str(p)
    st.session_state.label_column = col
    st.session_state.label_counts = counts
    return {
        "success": True,
        "message": f"Labels loaded into obs['{col}']. Counts: {counts}",
        "label_column": col,
        "counts": counts,
    }


def _tool_preprocess(
    cache_dir: str = None,
    max_neighbors: int = 8,
    build_lmdb: bool = True,
) -> Dict:
    if st.session_state.adata is None:
        return {"success": False, "message": "No data loaded. Call load_data first."}

    if st.session_state.get("preprocessing_done") and (
        not build_lmdb or st.session_state.get("cache_mode") == "lmdb"
    ):
        mode = st.session_state.get("cache_mode", "h5")
        return {
            "success": True,
            "message": f"Already preprocessed ({mode.upper()} mode). Skipping.",
        }

    from core.preprocessor import preprocess_h5ad, build_lmdb as _build_lmdb
    from core.model_manager import make_config

    if not cache_dir:
        cache_dir = str(_UI_ROOT / "workspace" / "cache")

    config = make_config(cache_dir=cache_dir, device=st.session_state.device, cache_mode="h5")
    preprocess_h5ad(
        h5ad_path=st.session_state.adata_path,
        cache_dir=cache_dir,
        config=config,
        max_neighbors=max_neighbors,
    )
    st.session_state.cache_dir = cache_dir
    st.session_state.cache_mode = "h5"
    st.session_state.lmdb_path = None
    st.session_state.lmdb_manifest_path = None
    st.session_state.preprocessing_done = True
    msg = f"Preprocessing complete (h5 mode, k={max_neighbors})."

    if build_lmdb:
        lmdb_path, manifest_path = _build_lmdb(cache_dir=cache_dir, config=config)
        st.session_state.cache_mode = "lmdb"
        st.session_state.lmdb_path = lmdb_path
        st.session_state.lmdb_manifest_path = manifest_path
        msg += f" LMDB cache also built."

    return {"success": True, "message": msg}


def _tool_load_model(source: str, path: str = None) -> Dict:
    # Skip if model is already loaded
    if st.session_state.model is not None:
        existing = st.session_state.get("model_path", "unknown")
        return {
            "success": True,
            "message": f"Model already loaded from {existing}. Skipping reload. "
                       f"Use get_status to check current state.",
        }

    from core.model_manager import load_model, make_config, download_pretrained

    cache_dir = st.session_state.get("cache_dir")
    config = make_config(
        cache_dir=cache_dir,
        device=st.session_state.device,
        cache_mode=st.session_state.get("cache_mode", "h5"),
        lmdb_path=st.session_state.get("lmdb_path"),
        lmdb_manifest_path=st.session_state.get("lmdb_manifest_path"),
    )

    if source == "huggingface":
        ckpt_dir = download_pretrained()
        model = load_model(ckpt_dir, config, st.session_state.device)
        st.session_state.model = model
        st.session_state.model_path = str(ckpt_dir)
        return {"success": True, "message": f"Pretrained model loaded from HuggingFace ({ckpt_dir})"}
    elif source == "local":
        if not path:
            path = str(_REPO_ROOT / "output" / "model_2" / "checkpoint-94620")
        if not Path(path).exists():
            return {"success": False, "message": f"Checkpoint not found: {path}"}
        model = load_model(path, config, st.session_state.device)
        st.session_state.model = model
        st.session_state.model_path = path
        return {"success": True, "message": f"Model loaded from {path}"}
    else:
        return {"success": False, "message": f"Unknown source: {source}. Use 'huggingface' or 'local'."}


def _tool_select_by_type(cell_type: str, column: str = None) -> Dict:
    adata = st.session_state.adata
    if adata is None:
        return {"success": False, "message": "No data loaded."}

    from ui_utils.data_io import detect_celltype_columns
    if not column:
        cols = detect_celltype_columns(adata)
        if not cols:
            return {"success": False, "message": "No cell type columns detected. Specify 'column' parameter."}
        column = cols[0]

    if column not in adata.obs.columns:
        return {"success": False, "message": f"Column '{column}' not found. Available: {list(adata.obs.columns)}"}

    mask = adata.obs[column] == cell_type
    if mask.sum() == 0:
        available = sorted(adata.obs[column].unique())
        return {"success": False, "message": f"No spots with cell_type='{cell_type}'. Available types: {available}"}

    indices = [int(adata.obs.index.get_loc(i)) for i in mask[mask].index]
    st.session_state.selected_spots = indices
    return {"success": True, "message": f"Selected {len(indices)} spots of type '{cell_type}' (column: {column})"}


def _tool_select_random(n: int) -> Dict:
    adata = st.session_state.adata
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    import random
    n = min(n, adata.n_obs)
    indices = sorted(random.sample(range(adata.n_obs), n))
    st.session_state.selected_spots = indices
    return {"success": True, "message": f"Randomly selected {len(indices)} spots."}


def _tool_select_by_label_random(label: str, n: int, column: str = None, seed: int = 42) -> Dict:
    adata = st.session_state.adata
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    if not column:
        column = st.session_state.get("label_column")
    if not column:
        from ui_utils.data_io import detect_celltype_columns
        cols = detect_celltype_columns(adata)
        column = cols[0] if cols else None
    if not column or column not in adata.obs.columns:
        return {"success": False, "message": f"Label column not found. Available obs columns: {list(adata.obs.columns)}"}

    values = adata.obs[column].astype(str)
    matches = np.where(values.str.lower().values == str(label).lower())[0].tolist()
    if not matches:
        available = sorted(values.unique().tolist())
        return {"success": False, "message": f"No spots with {column}='{label}'. Available labels: {available}"}
    rng = random.Random(seed)
    chosen = sorted(rng.sample(matches, min(int(n), len(matches))))
    st.session_state.selected_spots = chosen
    return {
        "success": True,
        "message": f"Selected {len(chosen)} random spots from {column}='{label}' (seed={seed}).",
        "indices": chosen,
    }


def _tool_select_by_indices(indices: List[int]) -> Dict:
    adata = st.session_state.adata
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    valid = [i for i in indices if 0 <= i < adata.n_obs]
    st.session_state.selected_spots = valid
    return {"success": True, "message": f"Selected {len(valid)} spots by index."}


def _tool_set_deg(path: str, p_thresh: float = 0.1, min_logfc: float = 0.0) -> Dict:
    if not Path(path).exists():
        return {"success": False, "message": f"DEG file not found: {path}"}
    from ui_utils.data_io import parse_deg_csv
    deg_df = parse_deg_csv(path)

    cfg = st.session_state.perturbation_config or {}
    cfg["deg_df"] = deg_df
    cfg["p_thresh"] = p_thresh
    cfg["min_logfc"] = min_logfc
    st.session_state.perturbation_config = cfg
    st.session_state.pc_deg_df = deg_df
    st.session_state.pc_deg_path = path
    st.session_state.pc_p_thresh = p_thresh
    st.session_state.pc_min_logfc = min_logfc
    _sync_config_widgets(cfg)

    n_pass = len(deg_df)
    if "p_val_adj" in deg_df.columns:
        n_pass = len(deg_df[deg_df["p_val_adj"] < p_thresh])
    if min_logfc > 0:
        n_pass = len(deg_df[deg_df["avg_logFC"].abs() >= min_logfc])
    return {"success": True, "message": f"DEG file loaded: {len(deg_df)} genes total, {n_pass} pass filters (p<{p_thresh}, |logFC|>={min_logfc})"}


def _tool_set_gene_edits(edits: Dict[str, float]) -> Dict:
    cfg = st.session_state.perturbation_config or {}
    cfg["custom_edits"] = {str(k): float(v) for k, v in edits.items()}
    st.session_state.perturbation_config = cfg
    _sync_config_widgets(cfg)
    return {"success": True, "message": f"Set manual logFC edits for {len(edits)} genes: {cfg['custom_edits']}"}


def _tool_set_params(
    strength: float = None,
    weighting: str = None,
    max_steps: int = None,
    stopping_mode: str = None,
    fixed_step: int = None,
    convergence_threshold: float = None,
    patience: int = None,
    emergency_div_factor: float = None,
    freeze_perturbed: bool = None,
    logfc_clip: float = None,
) -> Dict:
    cfg = st.session_state.perturbation_config or {}
    updated = []
    if strength is not None:
        cfg["strength"] = strength; updated.append(f"strength={strength}")
    if weighting is not None:
        cfg["weighting"] = weighting; updated.append(f"weighting={weighting}")
    if max_steps is not None:
        cfg["max_steps"] = max_steps; updated.append(f"max_steps={max_steps}")
    if stopping_mode is not None:
        if stopping_mode not in ("auto_best", "fixed_steps"):
            return {"success": False, "message": "stopping_mode must be 'auto_best' or 'fixed_steps'."}
        cfg["stopping_mode"] = stopping_mode; updated.append(f"stopping_mode={stopping_mode}")
    if fixed_step is not None:
        cfg["fixed_step"] = fixed_step; updated.append(f"fixed_step={fixed_step}")
    if convergence_threshold is not None:
        cfg["convergence_threshold"] = convergence_threshold; updated.append(f"threshold={convergence_threshold}")
    if patience is not None:
        cfg["patience"] = patience; updated.append(f"patience={patience}")
    if emergency_div_factor is not None:
        cfg["emergency_div_factor"] = emergency_div_factor; updated.append(f"emergency_div_factor={emergency_div_factor}")
    if freeze_perturbed is not None:
        cfg["freeze_perturbed"] = freeze_perturbed; updated.append(f"freeze={freeze_perturbed}")
    if logfc_clip is not None:
        cfg["logfc_clip"] = logfc_clip; updated.append(f"logfc_clip={logfc_clip}")
    st.session_state.perturbation_config = cfg
    _sync_config_widgets(cfg)
    return {"success": True, "message": f"Updated params: {', '.join(updated)}"}


def _tool_run_inference() -> Dict:
    adata = st.session_state.adata
    model = st.session_state.model
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    if not st.session_state.get("preprocessing_done"):
        return {"success": False, "message": "Data not preprocessed."}
    if model is None:
        return {"success": False, "message": "No model loaded."}
    if not st.session_state.selected_spots:
        return {"success": False, "message": "No spots selected."}

    cfg = st.session_state.perturbation_config or {}
    if cfg.get("deg_df") is None and not cfg.get("custom_edits"):
        return {"success": False, "message": "No perturbation direction defined (need DEG file or gene edits)."}

    from core.perturber import run_perturbation
    from core.model_manager import make_config
    from pretrain.spatial_databank import SpatialDataBank

    cache_dir = Path(st.session_state.cache_dir)
    lmdb_path = st.session_state.get("lmdb_path")
    lmdb_manifest_path = st.session_state.get("lmdb_manifest_path")
    if not lmdb_path:
        candidate = cache_dir / "lmdb" / "spatial_cache.lmdb"
        if candidate.exists():
            lmdb_path = str(candidate)
            st.session_state.lmdb_path = lmdb_path
    if not lmdb_manifest_path:
        candidate = cache_dir / "lmdb" / "spatial_cache.manifest.json"
        if candidate.exists():
            lmdb_manifest_path = str(candidate)
            st.session_state.lmdb_manifest_path = lmdb_manifest_path
    cache_mode = "lmdb" if lmdb_path and lmdb_manifest_path else st.session_state.get("cache_mode", "h5")
    st.session_state.cache_mode = cache_mode

    config = make_config(
        cache_dir=cache_dir,
        device=st.session_state.device,
        cache_mode=cache_mode,
        lmdb_path=lmdb_path,
        lmdb_manifest_path=lmdb_manifest_path,
    )

    dataset_name = Path(st.session_state.adata_path).stem
    processed_path = cache_dir / dataset_name / "processed.h5ad"
    if not processed_path.exists():
        processed_path = cache_dir / dataset_name / f"{dataset_name}.h5ad"
    config.dataset_paths = [str(processed_path)]

    databank = SpatialDataBank(
        dataset_paths=[str(processed_path)],
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False,
    )

    all_indices = list(range(adata.n_obs))
    result_dir = Path(st.session_state.cache_dir) / "perturbation_results"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.task_output_dir = str(result_dir)
    st.session_state.task_cleanup_paths = [str(result_dir)]

    st.session_state.task_stop_requested = False
    _set_task("running", "inference", "Running perturbation inference...")

    def _step_callback(step, progress, message):
        st.session_state.task_message = message
        if st.session_state.get("task_stop_requested"):
            raise KeyboardInterrupt("Inference stopped by user.")

    try:
        result = run_perturbation(
            model=model,
            databank=databank,
            config=config,
            perturb_spots=st.session_state.selected_spots,
            all_global_indices=all_indices,
            deg_df=cfg.get("deg_df"),
            custom_gene_edits=cfg.get("custom_edits"),
            strength=cfg.get("strength", 1.0),
            weighting=cfg.get("weighting", "uniform"),
            logfc_clip=cfg.get("logfc_clip", 5.0),
            p_thresh=cfg.get("p_thresh", 0.1),
            min_abs_logfc=cfg.get("min_logfc", 0.0),
            max_steps=cfg.get("max_steps", 20),
            stopping_mode=cfg.get("stopping_mode", "auto_best"),
            fixed_step=cfg.get("fixed_step"),
            convergence_threshold=cfg.get("convergence_threshold", 1e-5),
            patience=cfg.get("patience", 3),
            emergency_div_factor=cfg.get("emergency_div_factor", 10.0),
            freeze_perturbed=cfg.get("freeze_perturbed", True),
            device=st.session_state.device,
            batch_size=int(cfg.get("batch_size", 64)),
            num_workers=int(cfg.get("num_workers", 4)),
            output_dir=result_dir,
            step_callback=_step_callback,
        )
    except KeyboardInterrupt as e:
        st.session_state.inference_results = None
        _set_task("stopped", "inference", str(e))
        return {"success": False, "message": str(e)}

    st.session_state.inference_results = result
    _set_task("completed", "inference", "Inference complete.")

    converge_msg = (
        f"Selected best step {result.best_step + 1} (MSE={result.best_mse:.2e})"
        if getattr(result, "best_step", None) is not None and result.best_mse is not None
        else f"Ran {result.total_steps} steps"
    )
    mse_info = f", final MSE: {result.step_mse[-1]:.2e}" if result.step_mse else ""
    return {
        "success": True,
        "message": f"Inference complete. {converge_msg}{mse_info}. "
                   f"Go to Results page to explore the output.",
    }


def _tool_finetune_model(epochs: int = 5, batch_size: int = 32, device: str = None) -> Dict:
    if st.session_state.get("model_path") is None:
        return {"success": False, "message": "No base model loaded. Load a model before finetuning."}
    if not st.session_state.get("preprocessing_done"):
        return {"success": False, "message": "Data not preprocessed. Preprocess before finetuning."}

    from core.finetuner import run_finetune

    use_device = device or st.session_state.get("device", "cuda:0")
    ft_output = Path(st.session_state.cache_dir) / "finetune_output"
    log_path = ft_output / "finetune.log"
    st.session_state.task_output_dir = str(ft_output)
    st.session_state.task_cleanup_paths = [str(ft_output)]
    st.session_state.task_stop_requested = False
    _set_task("running", "finetune", f"Finetuning for {epochs} epochs on {use_device}...")

    proc = run_finetune(
        base_ckpt=st.session_state.model_path,
        cache_dir=st.session_state.cache_dir,
        output_dir=str(ft_output),
        dataset_name=st.session_state.adata_path,
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=use_device,
        visible_gpus=st.session_state.get("visible_gpus") or None,
        cache_mode=st.session_state.get("cache_mode", "h5"),
        lmdb_path=st.session_state.get("lmdb_path"),
        lmdb_manifest_path=st.session_state.get("lmdb_manifest_path"),
        log_file=str(log_path),
    )
    st.session_state.task_process = proc
    st.session_state.task_pid = proc.pid
    return {
        "success": True,
        "message": (
            f"Finetuning started in background (pid={proc.pid}). "
            f"Monitor live loss in the Run/Status panel. Log: {log_path}"
        ),
        "pid": proc.pid,
        "log_path": str(log_path),
        "output_dir": str(ft_output),
    }


def _cleanup_task_outputs() -> None:
    for raw in st.session_state.get("task_cleanup_paths", []) or []:
        p = Path(raw)
        try:
            if p.exists():
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        except Exception:
            pass
    st.session_state.inference_results = None
    st.session_state.task_pid = None
    st.session_state.task_process = None


def _tool_stop_task() -> Dict:
    st.session_state.task_stop_requested = True
    proc = st.session_state.get("task_process")
    if proc is not None and getattr(proc, "poll", lambda: None)() is None:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
    _cleanup_task_outputs()
    _set_task("stopped", st.session_state.get("task_name"), "Task stopped and partial outputs cleaned.")
    return {"success": True, "message": "Stop requested. Running subprocess was killed and partial outputs were cleaned."}


def _tool_get_status() -> Dict:
    adata = st.session_state.adata
    cfg = st.session_state.perturbation_config or {}
    parts = []
    parts.append(f"Device: {st.session_state.get('device', 'cuda:0')} "
                 f"(visible GPUs: {st.session_state.get('visible_gpus') or os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    parts.append(f"Data: {'✅ ' + str(adata.n_obs) + ' spots, ' + str(adata.n_vars) + ' genes' if adata else '❌ not loaded'}")
    parts.append(f"Preprocessed: {'✅' if st.session_state.get('preprocessing_done') else '❌'}"
                 f" (mode: {st.session_state.get('cache_mode', 'N/A')})")
    if st.session_state.get("label_column"):
        parts.append(f"Labels: ✅ {st.session_state.label_column} counts={st.session_state.get('label_counts', {})}")
    else:
        parts.append("Labels: ❌ not loaded")
    parts.append(f"Model: {'✅ ' + str(st.session_state.get('model_path', '')) if st.session_state.model else '❌ not loaded'}")
    n_sel = len(st.session_state.get("selected_spots", []))
    parts.append(f"Selected spots: {n_sel if n_sel > 0 else '❌ none'}")
    has_deg = cfg.get("deg_df") is not None
    has_edits = bool(cfg.get("custom_edits"))
    parts.append(f"Perturbation: {'✅ DEG' if has_deg else ''}{'✅ manual edits' if has_edits else ''}"
                 f"{'❌ not configured' if not has_deg and not has_edits else ''}")
    parts.append(f"Params: α={cfg.get('strength', 1.0)}, weighting={cfg.get('weighting', 'uniform')}, "
                 f"mode={cfg.get('stopping_mode', 'auto_best')}, max_steps={cfg.get('max_steps', 20)}, "
                 f"patience={cfg.get('patience', 3)}")
    parts.append(f"Task: {st.session_state.get('task_status', 'idle')} "
                 f"{st.session_state.get('task_name') or ''} {st.session_state.get('task_message') or ''}")
    parts.append(f"Results: {'✅ available' if st.session_state.get('inference_results') else '❌ not run yet'}")
    return {"success": True, "message": "\n".join(parts)}


def _tool_get_cell_types(column: str = None) -> Dict:
    adata = st.session_state.adata
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    from ui_utils.data_io import detect_celltype_columns
    if not column:
        cols = detect_celltype_columns(adata)
        if not cols:
            return {"success": True, "message": "No cell type annotation columns detected."}
        column = cols[0]
    if column not in adata.obs.columns:
        return {"success": False, "message": f"Column '{column}' not found."}
    vc = adata.obs[column].value_counts()
    lines = [f"Column: {column}"] + [f"  {ct}: {count} spots" for ct, count in vc.items()]
    return {"success": True, "message": "\n".join(lines)}


def _tool_get_gene_list(query: str = "", limit: int = 20) -> Dict:
    adata = st.session_state.adata
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    genes = list(adata.var_names)
    if query:
        q = query.lower()
        matches = [g for g in genes if q in g.lower()]
    else:
        matches = genes
    n_total = len(matches)
    matches = matches[:limit]
    return {"success": True, "message": f"Found {n_total} genes (showing {len(matches)}): {matches}"}
