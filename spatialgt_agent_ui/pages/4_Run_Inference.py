"""Page 4: Execute perturbation inference with real-time convergence monitoring."""

import sys
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from core._patch_torchtext import patch as _ptt; _ptt()  # noqa

import streamlit as st
from core.session import init_session_state; init_session_state()
import time

import numpy as np
import plotly.graph_objects as go

_UI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

st.set_page_config(page_title="Run Inference", page_icon="🚀", layout="wide")
st.title("4. Run Inference")

# ── Pre-flight checks ────────────────────────────────────────────────────────
checks = {
    "Data loaded": st.session_state.get("adata") is not None,
    "Preprocessing done": st.session_state.get("preprocessing_done", False),
    "Model loaded": st.session_state.get("model") is not None,
    "Spots selected": len(st.session_state.get("selected_spots", [])) > 0,
    "Perturbation configured": bool(
        st.session_state.get("perturbation_config", {}).get("deg_df") is not None
        or st.session_state.get("perturbation_config", {}).get("custom_edits")
    ),
}

st.subheader("Pre-flight Checklist")
all_ok = True
for label, ok in checks.items():
    icon = "✅" if ok else "❌"
    st.markdown(f"{icon} {label}")
    if not ok:
        all_ok = False

if not all_ok:
    st.warning("Please complete all steps before running inference.")
    st.stop()

st.success("All checks passed. Ready to run.")
st.divider()

if st.session_state.get("task_status") == "running":
    st.warning(f"Running task: {st.session_state.get('task_name')} — {st.session_state.get('task_message', '')}")
    if st.button("Stop Current Task", type="secondary"):
        from agent.tools import execute_tool
        result = execute_tool("stop_task", {})
        st.info(result.get("message"))
        st.rerun()

# ── Optional finetuning ──────────────────────────────────────────────────────
with st.expander("Optional: Finetune model on this slice", expanded=False):
    ft_epochs = st.number_input("Finetune epochs", value=10, min_value=1, max_value=200)
    running_ft = (
        st.session_state.get("task_name") == "finetune"
        and st.session_state.get("task_process") is not None
        and st.session_state.task_process.poll() is None
    )
    if running_ft:
        st.info(f"Finetuning is running (pid={st.session_state.get('task_pid')}).")
        c_ft1, c_ft2 = st.columns(2)
        with c_ft1:
            if st.button("Stop Finetuning"):
                from agent.tools import execute_tool
                result = execute_tool("stop_task", {})
                st.warning(result.get("message"))
                st.rerun()
        with c_ft2:
            st.caption("Refresh the page to poll completion.")
        st.stop()

    if st.button("Start Finetuning"):
        from core.finetuner import run_finetune, find_latest_checkpoint
        from core.model_manager import load_model, make_config

        ft_output = Path(st.session_state.cache_dir) / "finetune_output"
        log_path = ft_output / "finetune.log"

        st.info("Finetuning started in background.")
        proc = run_finetune(
            base_ckpt=st.session_state.model_path,
            cache_dir=st.session_state.cache_dir,
            output_dir=str(ft_output),
            dataset_name=st.session_state.adata_path,
            epochs=ft_epochs,
            device=st.session_state.device,
            visible_gpus=st.session_state.get("visible_gpus") or None,
            cache_mode=st.session_state.get("cache_mode", "h5"),
            lmdb_path=st.session_state.get("lmdb_path"),
            lmdb_manifest_path=st.session_state.get("lmdb_manifest_path"),
            log_file=str(log_path),
        )
        st.session_state.task_name = "finetune"
        st.session_state.task_status = "running"
        st.session_state.task_message = f"Finetuning for {ft_epochs} epochs"
        st.session_state.task_process = proc
        st.session_state.task_pid = proc.pid
        st.session_state.task_output_dir = str(ft_output)
        st.session_state.task_cleanup_paths = [str(ft_output)]
        st.rerun()

    proc = st.session_state.get("task_process")
    if st.session_state.get("task_name") == "finetune" and proc is not None and proc.poll() is not None:
        from core.finetuner import find_latest_checkpoint
        from core.model_manager import load_model, make_config
        ft_output = Path(st.session_state.cache_dir) / "finetune_output"
        if proc.returncode == 0:
            ckpt = find_latest_checkpoint(ft_output)
            if ckpt:
                config = make_config(
                    cache_dir=st.session_state.cache_dir,
                    device=st.session_state.device,
                    cache_mode=st.session_state.get("cache_mode", "h5"),
                    lmdb_path=st.session_state.get("lmdb_path"),
                    lmdb_manifest_path=st.session_state.get("lmdb_manifest_path"),
                )
                model = load_model(ckpt, config, st.session_state.device)
                st.session_state.model = model
                st.session_state.model_path = str(ckpt)
                st.session_state.finetuning_done = True
                st.session_state.task_status = "completed"
                st.session_state.task_message = f"Finetuning complete: {ckpt}"
                st.success(f"Finetuning complete! Model updated from {ckpt}")
            else:
                st.session_state.task_status = "error"
                st.warning("Finetuning finished but no checkpoint found.")
        else:
            st.session_state.task_status = "error"
            st.error(f"Finetuning failed (exit code {proc.returncode})")
        st.session_state.task_process = None
        st.session_state.task_pid = None

# ── Run perturbation ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Perturbation Inference")

cfg = st.session_state.perturbation_config

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Perturbed Spots", len(st.session_state.selected_spots))
with col_info2:
    st.metric("Strength (α)", cfg.get("strength", 1.0))
with col_info3:
    st.metric("Max Steps", cfg.get("max_steps", 20))
st.caption(
    f"Stopping: {cfg.get('stopping_mode', 'auto_best')} | "
    f"patience={cfg.get('patience', 3)} | fixed_step={cfg.get('fixed_step')}"
)

if st.button("Start Perturbation Inference", type="primary"):
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

    adata = st.session_state.adata
    dataset_name = Path(st.session_state.adata_path).stem
    processed_path = cache_dir / dataset_name / "processed.h5ad"
    if not processed_path.exists():
        processed_path = cache_dir / dataset_name / f"{dataset_name}.h5ad"
    config.dataset_paths = [str(processed_path)]

    with st.spinner(f"Initializing databank ({cache_mode} mode)..."):
        databank = SpatialDataBank(
            dataset_paths=[str(processed_path)],
            cache_dir=str(cache_dir),
            config=config,
            force_rebuild=False,
        )

    all_indices = list(range(adata.n_obs))
    result_dir = Path(st.session_state.cache_dir) / "perturbation_results"
    if result_dir.exists():
        import shutil
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.task_output_dir = str(result_dir)
    st.session_state.task_cleanup_paths = [str(result_dir)]

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    mse_chart_placeholder = st.empty()
    mse_values = []

    def _step_callback(step, progress, message):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(message)
        st.session_state.task_message = message
        if st.session_state.get("task_stop_requested"):
            raise KeyboardInterrupt("Inference stopped by user.")

    st.session_state.task_status = "running"
    st.session_state.task_name = "inference"
    st.session_state.task_stop_requested = False
    try:
        with st.spinner("Running dual-line iterative inference..."):
            result = run_perturbation(
                model=st.session_state.model,
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
        st.session_state.task_status = "stopped"
        st.session_state.inference_results = None
        st.error(str(e))
        st.stop()

    st.session_state.inference_results = result
    st.session_state.task_status = "completed"
    progress_bar.progress(1.0)

    # ── Display convergence curve ─────────────────────────────────────────
    st.subheader("Convergence")
    if result.step_mse:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(result.step_mse) + 1)),
            y=result.step_mse,
            mode="lines+markers",
            name="Step-to-step MSE",
        ))
        fig.update_layout(
            xaxis_title="Step",
            yaxis_title="MSE",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    if getattr(result, "best_step", None) is not None:
        st.success(
            f"Selected step **{result.best_step + 1}** "
            f"(best MSE {result.best_mse:.2e}; reason: {result.stop_reason})"
        )
    elif result.converged_step < result.total_steps:
        st.success(f"Converged at step **{result.converged_step}** (total {result.total_steps} steps)")
    else:
        st.info(f"Completed {result.total_steps} steps (did not converge early)")

    st.success("Inference complete! Proceed to **Results** (Page 5).")
