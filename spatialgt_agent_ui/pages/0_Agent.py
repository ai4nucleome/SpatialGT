"""
Page 0: Agent — Natural language control of the full SpatialGT pipeline.
Left panel: chat interface. Right panel: live status dashboard.
"""

import sys
import re
import time
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from core._patch_torchtext import patch as _ptt; _ptt()  # noqa

import streamlit as st
from core.session import init_session_state; init_session_state()

_REPO_ROOT = _UI_ROOT.parent
_PRETRAIN_DIR = _REPO_ROOT / "pretrain"
for _path in (_REPO_ROOT, _PRETRAIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

st.set_page_config(page_title="SpatialGT Agent", page_icon="🤖", layout="wide")

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


def _handle_agent_prompt(user_input: str) -> None:
    """Run the LLM agent and append streamed events to chat history."""
    if not user_input.strip():
        return
    st.session_state.agent_messages.append({"type": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    from agent.client import agent_chat

    all_text_parts = []
    for event in agent_chat(user_input):
        msg_type = "assistant" if event["type"] == "text" else event["type"]
        msg_entry = dict(event)
        msg_entry["type"] = msg_type
        st.session_state.agent_messages.append(msg_entry)

        if event["type"] == "text":
            all_text_parts.append(event["content"])
        elif event["type"] == "tool_result":
            all_text_parts.append(
                f"[{event['name']}]: {event['result'].get('message', '')}"
            )

    full_reply = "\n".join(all_text_parts)
    if full_reply:
        st.session_state.chat_history.append({"role": "assistant", "content": full_reply})

    st.rerun()


def _parse_latest_finetune_metrics(log_path: Path) -> dict:
    """Extract recent train loss/step/epoch from the finetune log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    metrics = {}
    train_loss = re.findall(r"'train_loss':\s*([0-9.eE+-]+)", text)
    if train_loss:
        metrics["train_loss"] = float(train_loss[-1])
    loss = re.findall(r"'loss':\s*([0-9.eE+-]+)", text)
    if loss:
        metrics["loss"] = float(loss[-1])
    steps = re.findall(r"'step':\s*([0-9]+)", text)
    if steps:
        metrics["step"] = int(steps[-1])
    epoch = re.findall(r"'epoch':\s*([0-9.eE+-]+)", text)
    if epoch:
        metrics["epoch"] = float(epoch[-1])
    if "Starting finetuning" in text:
        metrics.setdefault("phase", "training")
    if "Saved finetuned_state_dict.pth" in text or "'train_runtime':" in text:
        metrics["phase"] = "finished"
    return metrics


def _render_task_monitor() -> None:
    """Show live task state, finetune loss, and inference MSE outputs."""
    task_name = st.session_state.get("task_name")
    task_status = st.session_state.get("task_status", "idle")
    task_msg = st.session_state.get("task_message", "")
    output_dir = st.session_state.get("task_output_dir")

    if not task_name or task_status == "idle":
        st.caption("No active task.")
        return

    st.info(f"Task: `{task_name}` | Status: `{task_status}` | {task_msg}")

    if task_name == "finetune" and output_dir:
        ft_output = Path(output_dir)
        log_path = ft_output / "finetune.log"
        metrics = _parse_latest_finetune_metrics(log_path)
        m1, m2, m3 = st.columns(3)
        m1.metric("Phase", metrics.get("phase", "starting"))
        m2.metric("Loss", f"{metrics.get('loss', metrics.get('train_loss', 0.0)):.4g}" if metrics else "—")
        m3.metric("Epoch", f"{metrics.get('epoch', 0.0):.2f}" if metrics else "—")
        if log_path.exists():
            with st.expander("Finetune log tail", expanded=False):
                lines = log_path.read_text(errors="ignore").splitlines()
                st.code("\n".join(lines[-20:]) or "(empty)", language="text")

        proc = st.session_state.get("task_process")
        if proc is not None and proc.poll() is not None:
            from core.finetuner import find_latest_checkpoint
            from core.model_manager import load_model, make_config
            if proc.returncode == 0:
                ckpt = find_latest_checkpoint(ft_output)
                if ckpt:
                    cfg = make_config(
                        cache_dir=st.session_state.cache_dir,
                        device=st.session_state.device,
                        cache_mode=st.session_state.get("cache_mode", "lmdb"),
                        lmdb_path=st.session_state.get("lmdb_path"),
                        lmdb_manifest_path=st.session_state.get("lmdb_manifest_path"),
                    )
                    st.session_state.model = load_model(ckpt, cfg, st.session_state.device)
                    st.session_state.model_path = str(ckpt)
                    st.session_state.finetuning_done = True
                    st.session_state.task_status = "completed"
                    st.session_state.task_message = f"Finetuning complete: {ckpt}"
            else:
                st.session_state.task_status = "error"
                st.session_state.task_message = f"Finetuning failed with exit code {proc.returncode}"
            st.session_state.task_process = None
            st.session_state.task_pid = None

    result_dir = None
    if output_dir:
        p = Path(output_dir)
        if (p / "perturbation_summary.json").exists():
            result_dir = p
    if result_dir is not None:
        import json
        summary = json.loads((result_dir / "perturbation_summary.json").read_text())
        step_mse = summary.get("step_mse") or []
        update_mse = summary.get("update_mse") or []
        if step_mse:
            c1, c2, c3 = st.columns(3)
            c1.metric("Completed steps", len(step_mse))
            c2.metric("Latest step MSE", f"{step_mse[-1]:.3e}")
            c3.metric("Latest update MSE", f"{update_mse[-1]:.3e}" if update_mse else "—")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(step_mse) + 1)), y=step_mse, mode="lines+markers", name="step_mse"))
            if update_mse:
                fig.add_trace(go.Scatter(x=list(range(1, len(update_mse) + 1)), y=update_mse, mode="lines+markers", name="update_mse"))
            fig.update_layout(height=260, yaxis_type="log", xaxis_title="Step", yaxis_title="MSE")
            st.plotly_chart(fig, use_container_width=True)


def _run_inference_with_live_ui(progress_bar, status_text, mse_box) -> dict:
    """Run perturbation from the Agent page with live Streamlit progress."""
    import shutil
    from core.perturber import run_perturbation
    from core.model_manager import make_config
    from pretrain.spatial_databank import SpatialDataBank

    adata = st.session_state.adata
    model = st.session_state.model
    cfg = st.session_state.perturbation_config or {}
    if adata is None:
        return {"success": False, "message": "No data loaded."}
    if model is None:
        return {"success": False, "message": "No model loaded."}
    if not st.session_state.get("preprocessing_done"):
        return {"success": False, "message": "Data not preprocessed."}
    if not st.session_state.get("selected_spots"):
        return {"success": False, "message": "No spots selected."}
    if cfg.get("deg_df") is None and not cfg.get("custom_edits"):
        return {"success": False, "message": "No perturbation direction configured."}

    cache_dir = Path(st.session_state.cache_dir)
    lmdb_path = st.session_state.get("lmdb_path") or str(cache_dir / "lmdb" / "spatial_cache.lmdb")
    lmdb_manifest_path = st.session_state.get("lmdb_manifest_path") or str(cache_dir / "lmdb" / "spatial_cache.manifest.json")
    cache_mode = "lmdb" if Path(lmdb_path).exists() and Path(lmdb_manifest_path).exists() else st.session_state.get("cache_mode", "h5")
    st.session_state.cache_mode = cache_mode
    if cache_mode == "lmdb":
        st.session_state.lmdb_path = lmdb_path
        st.session_state.lmdb_manifest_path = lmdb_manifest_path

    config = make_config(
        cache_dir=cache_dir,
        device=st.session_state.device,
        cache_mode=cache_mode,
        lmdb_path=lmdb_path if cache_mode == "lmdb" else None,
        lmdb_manifest_path=lmdb_manifest_path if cache_mode == "lmdb" else None,
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

    result_dir = cache_dir / "perturbation_results"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.task_name = "inference"
    st.session_state.task_status = "running"
    st.session_state.task_output_dir = str(result_dir)
    st.session_state.task_cleanup_paths = [str(result_dir)]
    st.session_state.task_stop_requested = False

    live_step_mse = []
    live_update_mse = []

    def _step_callback(step, progress, message):
        st.session_state.task_message = message
        progress_bar.progress(min(max(float(progress), 0.0), 1.0))
        status_text.info(message)
        if st.session_state.get("task_stop_requested"):
            raise KeyboardInterrupt("Inference stopped by user.")
        summary_path = result_dir / "perturbation_summary.json"
        if summary_path.exists():
            import json
            summary = json.loads(summary_path.read_text())
            live_step_mse[:] = summary.get("step_mse", [])
            live_update_mse[:] = summary.get("update_mse", [])
        if live_step_mse:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(live_step_mse) + 1)), y=live_step_mse, mode="lines+markers", name="step_mse"))
            if live_update_mse:
                fig.add_trace(go.Scatter(x=list(range(1, len(live_update_mse) + 1)), y=live_update_mse, mode="lines+markers", name="update_mse"))
            fig.update_layout(height=260, yaxis_type="log", xaxis_title="Step", yaxis_title="MSE")
            mse_box.plotly_chart(fig, use_container_width=True)

    try:
        result = run_perturbation(
            model=model,
            databank=databank,
            config=config,
            perturb_spots=st.session_state.selected_spots,
            all_global_indices=list(range(adata.n_obs)),
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
        st.session_state.task_message = str(e)
        return {"success": False, "message": str(e)}

    st.session_state.inference_results = result
    st.session_state.task_status = "completed"
    st.session_state.task_message = f"Inference complete: selected step {(result.selected_step or 0) + 1}"
    return {
        "success": True,
        "message": f"Inference complete. Summary: {result.summary_path}",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Layout: left chat (45%) | right dashboard (55%)
# ═══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([5, 6])

# ── LEFT: Chat History ────────────────────────────────────────────────────────
with left_col:
    st.header("Agent")
    st.caption("Natural language control of the SpatialGT pipeline.")

    chat_container = st.container(height=600)
    with chat_container:
        if not st.session_state.agent_messages:
            st.markdown(
                "**Try saying:**\n"
                "- Load `/path/to/data.h5ad`\n"
                "- Preprocess the loaded data\n"
                "- Load the pretrained model\n"
                "- Randomly select 50 spots\n"
                "- Show the current status\n"
                "- What cell types are in the data?"
            )
        for msg in st.session_state.agent_messages:
            if msg["type"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["type"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
            elif msg["type"] == "tool_call":
                with st.chat_message("assistant"):
                    st.code(f"Calling: {msg['name']}({msg.get('params', {})})", language="json")
            elif msg["type"] == "tool_result":
                with st.chat_message("assistant"):
                    result = msg["result"]
                    icon = "✅" if result.get("success") else "❌"
                    st.markdown(f"{icon} **{msg['name']}**: {result.get('message', '')}")
            elif msg["type"] == "error":
                with st.chat_message("assistant"):
                    st.error(msg["content"])

    with st.form("agent_inline_input_form", clear_on_submit=True):
        prompt = st.text_area(
            "Agent input",
            placeholder="Example: load my h5ad file, preprocess it, then randomly select 20 spots with label cancer",
            height=90,
            key="agent_inline_prompt",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send to Agent", type="primary", use_container_width=True)
    if submitted:
        _handle_agent_prompt(prompt)

    # Quick action buttons
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("📋 Status", use_container_width=True):
            st.session_state.agent_messages.append({"type": "user", "content": "Show status"})
            from agent.tools import execute_tool
            result = execute_tool("get_status", {})
            st.session_state.agent_messages.append({
                "type": "tool_result", "name": "get_status", "result": result,
            })
            st.rerun()
    with qc2:
        if st.button("🧬 Cell Types", use_container_width=True):
            st.session_state.agent_messages.append({"type": "user", "content": "Show cell types"})
            from agent.tools import execute_tool
            result = execute_tool("get_cell_types", {})
            st.session_state.agent_messages.append({
                "type": "tool_result", "name": "get_cell_types", "result": result,
            })
            st.rerun()
    with qc3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.agent_messages = []
            st.rerun()

# ── RIGHT: Dashboard ─────────────────────────────────────────────────────────
with right_col:
    st.header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        loaded = st.session_state.adata is not None
        st.metric("Data", f"{st.session_state.adata.n_obs} spots" if loaded else "—")
    with c2:
        st.metric("Model", "Loaded" if st.session_state.model else "—")
    with c3:
        n_sel = len(st.session_state.selected_spots)
        st.metric("Selected", str(n_sel) if n_sel > 0 else "—")
    with c4:
        has_results = st.session_state.inference_results is not None
        st.metric("Results", "Ready" if has_results else "—")

    tab_setup, tab_spatial, tab_params, tab_run, tab_status, tab_results = st.tabs(
        ["Setup", "Spatial / Labels", "Perturbation", "Run", "Status", "Results"]
    )

    with tab_setup:
        from agent.tools import execute_tool

        st.markdown("#### GPU / Device")
        c_gpu1, c_gpu2, c_gpu3 = st.columns([1, 1, 2])
        with c_gpu1:
            if st.button("Detect GPUs", use_container_width=True):
                st.session_state.agent_messages.append({
                    "type": "tool_result", "name": "detect_gpus",
                    "result": execute_tool("detect_gpus", {}),
                })
                st.rerun()
        with c_gpu2:
            device_choice = st.text_input("Device", value=st.session_state.get("device", "cuda:0"), key="agent_device_choice")
        with c_gpu3:
            visible_choice = st.text_input("Visible GPU ids", value=st.session_state.get("visible_gpus", ""), placeholder="0 or 0,1", key="agent_visible_choice")
        if st.button("Apply Device", use_container_width=True):
            result = execute_tool("set_device", {"device": device_choice, "visible_gpus": visible_choice or None})
            st.session_state.agent_messages.append({"type": "tool_result", "name": "set_device", "result": result})
            st.rerun()

        st.markdown("#### Data")
        data_path = st.text_input(
            "Data .h5ad or folder",
            value=st.session_state.get("adata_path") or "",
            placeholder="/path/to/your/data.h5ad",
            key="agent_data_path",
        )
        if st.button("Load Data", type="primary", use_container_width=True):
            result = execute_tool("load_data", {"path": data_path})
            st.session_state.agent_messages.append({"type": "tool_result", "name": "load_data", "result": result})
            st.rerun()

        label_path = st.text_input(
            "Label CSV",
            value=st.session_state.get("label_path") or "",
            placeholder="/path/to/labels.csv",
            key="agent_label_path",
        )
        label_col = st.text_input("Label column", value=st.session_state.get("label_column") or "Region", key="agent_label_col")
        if st.button("Load Labels", use_container_width=True):
            result = execute_tool("load_labels", {"path": label_path, "label_column": label_col or None})
            st.session_state.agent_messages.append({"type": "tool_result", "name": "load_labels", "result": result})
            st.rerun()
        if st.session_state.get("label_counts"):
            st.dataframe(
                [{"label": k, "count": v} for k, v in st.session_state.label_counts.items()],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Preprocessing")
        cache_dir = st.text_input(
            "Cache directory",
            value=st.session_state.get("cache_dir") or str(_UI_ROOT / "workspace" / "cache"),
            key="agent_cache_dir",
        )
        prep_k = st.number_input("Max neighbors", min_value=4, max_value=20, value=8, key="agent_prep_k")
        if st.button("Preprocess + Build LMDB", use_container_width=True):
            result = execute_tool("preprocess", {"cache_dir": cache_dir, "max_neighbors": int(prep_k), "build_lmdb": True})
            st.session_state.agent_messages.append({"type": "tool_result", "name": "preprocess", "result": result})
            st.rerun()

    with tab_status:
        adata = st.session_state.adata
        cfg = st.session_state.perturbation_config or {}
        st.markdown("#### Live Task Monitor")
        _render_task_monitor()
        st.divider()
        status_items = [
            ("Data", f"✅ {adata.n_obs} spots, {adata.n_vars} genes" if adata else "❌ Not loaded"),
            ("Preprocessed", f"✅ {st.session_state.get('cache_mode', 'h5').upper()}" if st.session_state.get("preprocessing_done") else "❌"),
            ("Labels", f"✅ {st.session_state.get('label_column')}" if st.session_state.get("label_column") else "❌"),
            ("Model", f"✅ {Path(st.session_state.get('model_path', '')).name}" if st.session_state.model else "❌"),
            ("Selected", f"✅ {len(st.session_state.selected_spots)} spots" if st.session_state.selected_spots else "❌"),
            ("Perturbation", "✅ Configured" if cfg.get("deg_df") is not None or cfg.get("custom_edits") else "❌"),
            ("Task", f"{st.session_state.get('task_status', 'idle')} {st.session_state.get('task_message', '')}"),
            ("Results", "✅ Available" if st.session_state.inference_results else "❌"),
        ]
        for label, value in status_items:
            st.markdown(f"**{label}**: {value}")

    with tab_spatial:
        if st.session_state.adata is not None:
            from ui_utils.data_io import get_spatial_coords, detect_celltype_columns
            from ui_utils.spatial_plot import spatial_scatter
            from agent.tools import execute_tool
            try:
                coords = get_spatial_coords(st.session_state.adata)
                ct_cols = detect_celltype_columns(st.session_state.adata)
                color_col = st.session_state.get("label_column") if st.session_state.get("label_column") in ct_cols else (ct_cols[0] if ct_cols else None)
                if ct_cols:
                    color_col = st.selectbox(
                        "Color by",
                        ct_cols,
                        index=ct_cols.index(color_col) if color_col in ct_cols else 0,
                        key="agent_spatial_color",
                    )
                color_vals = st.session_state.adata.obs[color_col].values if color_col else None
                fig = spatial_scatter(
                    coords, color_values=color_vals,
                    color_label=color_col if color_col else "Spot",
                    selected_indices=st.session_state.selected_spots,
                    point_size=3, height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
                if color_col:
                    counts = st.session_state.adata.obs[color_col].astype(str).value_counts()
                    c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1])
                    with c_sel1:
                        label = st.selectbox("Label for random selection", counts.index.tolist(), key="agent_label_select")
                    with c_sel2:
                        n_rand = st.number_input("N", min_value=1, max_value=int(counts.max()), value=min(20, int(counts.max())), key="agent_label_n")
                    with c_sel3:
                        st.metric("Selected", len(st.session_state.selected_spots))
                    if st.button("Random Select by Label", use_container_width=True):
                        result = execute_tool(
                            "select_spots_by_label_random",
                            {"label": label, "n": int(n_rand), "column": color_col, "seed": 42},
                        )
                        st.session_state.agent_messages.append({
                            "type": "tool_result", "name": "select_spots_by_label_random", "result": result,
                        })
                        st.rerun()
            except Exception as e:
                st.caption(f"Cannot render: {e}")
        else:
            st.caption("Load data to see spatial view.")

    with tab_params:
        from agent.tools import execute_tool
        cfg = st.session_state.perturbation_config or {}
        st.markdown("#### Manual Gene Edits")
        if st.session_state.adata is not None:
            genes = list(st.session_state.adata.var_names)
            default_edits = cfg.get("custom_edits") or {}
            gene_pick = st.multiselect(
                "Genes",
                options=genes,
                default=[g for g in default_edits.keys() if g in genes],
                max_selections=30,
                key="agent_gene_pick",
            )
            edit_values = {}
            if gene_pick:
                cols = st.columns(min(3, len(gene_pick)))
                for i, gene in enumerate(gene_pick):
                    with cols[i % len(cols)]:
                        edit_values[gene] = st.number_input(
                            gene,
                            value=float(default_edits.get(gene, 0.0)),
                            min_value=-10.0,
                            max_value=10.0,
                            step=0.1,
                            key=f"agent_edit_{gene}",
                        )
            if st.button("Apply Gene Edits", use_container_width=True):
                edits = {g: v for g, v in edit_values.items() if v != 0.0}
                result = execute_tool("set_gene_edits", {"edits": edits})
                st.session_state.agent_messages.append({"type": "tool_result", "name": "set_gene_edits", "result": result})
                st.rerun()
        else:
            st.caption("Load data to configure genes.")

        st.markdown("#### Bulk DEG List")
        st.caption("Accepted formats: gene+logFC/log2FoldChange, gene+fold_change, gene+direction(+strength), or one-column gene list.")
        deg_upload = st.file_uploader("Upload DEG list CSV/TSV/TXT", type=["csv", "tsv", "txt"], key="agent_deg_upload")
        deg_path = st.text_input("Or DEG file path on server", value="", placeholder="/path/to/deg.csv", key="agent_deg_path")
        col_deg1, col_deg2 = st.columns(2)
        with col_deg1:
            if st.button("Apply Uploaded DEG", use_container_width=True, disabled=deg_upload is None):
                upload_dir = _UI_ROOT / "workspace" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                saved_path = upload_dir / deg_upload.name
                saved_path.write_bytes(deg_upload.getvalue())
                result = execute_tool("set_deg_file", {"path": str(saved_path)})
                st.session_state.agent_messages.append({"type": "tool_result", "name": "set_deg_file", "result": result})
                st.rerun()
        with col_deg2:
            if st.button("Apply Server DEG Path", use_container_width=True, disabled=not bool(deg_path)):
                result = execute_tool("set_deg_file", {"path": deg_path})
                st.session_state.agent_messages.append({"type": "tool_result", "name": "set_deg_file", "result": result})
                st.rerun()

        st.markdown("#### Inference Parameters")
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            strength = st.number_input("Strength α", value=float(cfg.get("strength", 1.0)), min_value=0.1, max_value=10.0, step=0.1, key="agent_strength")
            weighting = st.selectbox("Weighting", ["uniform", "gaussian"], index=["uniform", "gaussian"].index(cfg.get("weighting", "uniform")), key="agent_weighting")
        with c_p2:
            stopping_mode = st.selectbox("Stopping", ["auto_best", "fixed_steps"], index=["auto_best", "fixed_steps"].index(cfg.get("stopping_mode", "auto_best")), key="agent_stopping")
            max_steps = st.number_input("Max steps", value=int(cfg.get("max_steps", 20)), min_value=1, max_value=100, step=1, key="agent_max_steps")
            fixed_step = st.number_input("Fixed/use step", value=int(cfg.get("fixed_step") or min(10, int(max_steps))), min_value=1, max_value=int(max_steps), step=1, key="agent_fixed_step")
        with c_p3:
            patience = st.number_input("Patience", value=int(cfg.get("patience", 3)), min_value=1, max_value=20, step=1, key="agent_patience")
            threshold = st.number_input("Min MSE improvement", value=float(cfg.get("convergence_threshold", 1e-5)), format="%.1e", key="agent_threshold")
            freeze = st.checkbox("Freeze perturbed spots", value=bool(cfg.get("freeze_perturbed", True)), key="agent_freeze")
        if st.button("Apply Perturbation Parameters", use_container_width=True):
            params = {
                "strength": float(strength),
                "weighting": weighting,
                "max_steps": int(max_steps),
                "stopping_mode": stopping_mode,
                "fixed_step": int(fixed_step) if stopping_mode == "fixed_steps" else None,
                "convergence_threshold": float(threshold),
                "patience": int(patience),
                "freeze_perturbed": bool(freeze),
            }
            result = execute_tool("set_perturbation_params", params)
            st.session_state.agent_messages.append({"type": "tool_result", "name": "set_perturbation_params", "result": result})
            st.rerun()

    with tab_run:
        from agent.tools import execute_tool

        st.markdown("#### Model")
        model_col1, model_col2 = st.columns([1, 2])
        with model_col1:
            source = st.selectbox("Source", ["huggingface", "local"], key="agent_model_source")
        with model_col2:
            ckpt_path = st.text_input(
                "Checkpoint path",
                value=st.session_state.get("model_path") or "",
                key="agent_ckpt_path",
            )
        if st.button("Load Model", use_container_width=True):
            params = {"source": source}
            if source == "local":
                params["path"] = ckpt_path
            result = execute_tool("load_model", params)
            st.session_state.agent_messages.append({"type": "tool_result", "name": "load_model", "result": result})
            st.rerun()

        st.markdown("#### Finetune")
        ft_epochs = st.number_input("Epochs", min_value=1, max_value=200, value=5, key="agent_ft_epochs")
        if st.button("Finetune Model", use_container_width=True):
            result = execute_tool("finetune_model", {"epochs": int(ft_epochs), "device": st.session_state.get("device", "cuda:0")})
            st.session_state.agent_messages.append({"type": "tool_result", "name": "finetune_model", "result": result})
            st.rerun()
        _render_task_monitor()

        st.markdown("#### Perturbation Inference")
        cfg = st.session_state.perturbation_config or {}
        st.caption(
            f"Selected spots={len(st.session_state.get('selected_spots', []))}; "
            f"genes={len(cfg.get('custom_edits') or {})}; "
            f"mode={cfg.get('stopping_mode', 'auto_best')}; max_steps={cfg.get('max_steps', 20)}"
        )
        run_col1, run_col2 = st.columns(2)
        with run_col1:
            if st.button("Run Inference", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                mse_box = st.empty()
                result = _run_inference_with_live_ui(progress_bar, status_text, mse_box)
                st.session_state.agent_messages.append({"type": "tool_result", "name": "run_inference", "result": result})
                st.rerun()
        with run_col2:
            if st.button("Stop Task", use_container_width=True):
                result = execute_tool("stop_task", {})
                st.session_state.agent_messages.append({"type": "tool_result", "name": "stop_task", "result": result})
                st.rerun()

    with tab_results:
        result = st.session_state.inference_results
        if result:
            import plotly.graph_objects as go
            if result.step_mse:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(result.step_mse) + 1)),
                    y=result.step_mse, mode="lines+markers",
                ))
                fig.update_layout(xaxis_title="Step", yaxis_title="MSE", yaxis_type="log", height=300)
                st.plotly_chart(fig, use_container_width=True)
            if getattr(result, "best_step", None) is not None:
                st.info(
                    f"Selected step {result.best_step + 1}; "
                    f"best MSE={result.best_mse:.2e}; reason={getattr(result, 'stop_reason', 'N/A')}"
                )
                if getattr(result, "output_dir", None):
                    st.markdown(f"**Saved results**: `{result.output_dir}`")
                if getattr(result, "summary_path", None):
                    st.markdown(f"**Summary**: `{result.summary_path}`")
                if getattr(result, "step_output_paths", None):
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame({
                            "Step": [k + 1 for k in sorted(result.step_output_paths)],
                            "Path": [result.step_output_paths[k] for k in sorted(result.step_output_paths)],
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                converge_msg = (f"Converged at step {result.converged_step}" if result.converged_step < result.total_steps
                                else f"Ran {result.total_steps} steps")
                st.info(converge_msg)
        else:
            st.caption("Run inference to see results.")

if (
    st.session_state.get("task_status") == "running"
    and st.session_state.get("task_name") == "finetune"
    and st.session_state.get("task_process") is not None
):
    time.sleep(5)
    st.rerun()
