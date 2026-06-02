"""
LLM Agent client with tool execution loop.
Uses SiliconFlow API (OpenAI-compatible) with Qwen3-8B.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv

_UI_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_UI_ROOT / ".env", override=False)

import streamlit as st
from openai import OpenAI

from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_QA_ONLY
from .tools import execute_tool

_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-8B"
_MAX_TOOL_ROUNDS = 5


def _get_status_summary() -> str:
    """Build a compact status string for the system prompt."""
    parts = []
    parts.append(f"Device: {st.session_state.get('device', 'cuda:0')} (visible GPUs: {st.session_state.get('visible_gpus') or 'default'})")
    adata = st.session_state.get("adata")
    parts.append(f"Data: {'LOADED (' + str(adata.n_obs) + ' spots)' if adata else 'NOT loaded'}")
    parts.append(f"Preprocessed: {'YES (' + st.session_state.get('cache_mode', 'h5') + ')' if st.session_state.get('preprocessing_done') else 'NO'}")
    parts.append(f"Labels: {st.session_state.get('label_column') or 'NOT loaded'} counts={st.session_state.get('label_counts', {})}")
    parts.append(f"Model: {'LOADED (' + str(st.session_state.get('model_path', '')) + ')' if st.session_state.get('model') else 'NOT loaded'}")
    n_sel = len(st.session_state.get("selected_spots", []))
    parts.append(f"Selected spots: {n_sel if n_sel > 0 else 'NONE'}")
    cfg = st.session_state.get("perturbation_config") or {}
    has_pert = cfg.get("deg_df") is not None or bool(cfg.get("custom_edits"))
    parts.append(
        f"Perturbation configured: {'YES' if has_pert else 'NO'}; "
        f"mode={cfg.get('stopping_mode', 'auto_best')}; max_steps={cfg.get('max_steps', 20)}"
    )
    parts.append(f"Task: {st.session_state.get('task_status', 'idle')} {st.session_state.get('task_message', '')}")
    parts.append(f"Results: {'AVAILABLE' if st.session_state.get('inference_results') else 'NOT run'}")
    return "\n".join(parts)


def _get_client() -> OpenAI | None:
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("SILICONFLOW_BASE_URL", _DEFAULT_BASE_URL),
    )


def _extract_actions(text: str) -> list[dict]:
    """Extract JSON action blocks from LLM response text."""
    actions = []
    pattern = r'```json\s*(\{[^`]*?\})\s*```'
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            if "action" in obj:
                actions.append(obj)
        except json.JSONDecodeError:
            continue
    if not actions:
        pattern2 = r'\{["\s]*action["\s]*:'
        for match in re.finditer(pattern2, text):
            start = match.start()
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    obj = json.loads(text[start:end])
                    if "action" in obj:
                        actions.append(obj)
                except json.JSONDecodeError:
                    continue
    return actions


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 responses."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def agent_chat(user_message: str) -> Generator[dict, None, None]:
    """
    Run the agent loop. Yields event dicts:
      {"type": "text", "content": "..."}
      {"type": "tool_call", "name": "...", "params": {...}}
      {"type": "tool_result", "name": "...", "result": {...}}
      {"type": "error", "content": "..."}
    """
    client = _get_client()
    if client is None:
        yield {"type": "error", "content": "SILICONFLOW_API_KEY not set. Configure it in .env file."}
        return

    history = st.session_state.get("chat_history", [])

    # Inject current pipeline status so the LLM knows what's already done
    status_summary = _get_status_summary()
    system_with_status = SYSTEM_PROMPT + f"\n\n## Current Pipeline Status\n{status_summary}"

    messages = [{"role": "system", "content": system_with_status}]
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    model = os.getenv("SILICONFLOW_MODEL", _DEFAULT_MODEL)

    for round_idx in range(_MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
            )
            reply = resp.choices[0].message.content or ""
        except Exception as e:
            yield {"type": "error", "content": f"API error: {e}"}
            return

        reply_clean = _strip_think_tags(reply)

        # If stripping <think> tags removed everything, fall back to raw reply
        if not reply_clean.strip():
            reply_clean = re.sub(r'</?think>', '', reply).strip()

        actions = _extract_actions(reply_clean)
        text_before = re.sub(r'```json\s*\{[^`]*?\}\s*```', '', reply_clean).strip()

        if not actions and not text_before:
            # Last resort: yield whatever the model said
            yield {"type": "text", "content": reply_clean if reply_clean else "(No response)"}
            break

        if text_before:
            yield {"type": "text", "content": text_before}

        if not actions:
            break

        for action in actions:
            tool_name = action["action"]
            tool_params = action.get("params", {})
            yield {"type": "tool_call", "name": tool_name, "params": tool_params}

            result = execute_tool(tool_name, tool_params)
            yield {"type": "tool_result", "name": tool_name, "result": result}

            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": f"[Tool Result for {tool_name}]: {json.dumps(result, ensure_ascii=False)}",
            })

    return


def render_chat_panel():
    """Legacy sidebar chat (Q&A only, no tool execution)."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history[-10:]:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"{role_icon} {msg['content'][:200]}")

    user_input = st.chat_input("Ask about SpatialGT...", key="agent_chat_input")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        client = _get_client()
        if client:
            model = os.getenv("SILICONFLOW_MODEL", _DEFAULT_MODEL)
            messages = [{"role": "system", "content": SYSTEM_PROMPT_QA_ONLY}]
            messages.extend(st.session_state.chat_history[-6:])
            try:
                resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512, temperature=0.7)
                raw = resp.choices[0].message.content or ""
                reply = _strip_think_tags(raw)
                if not reply.strip():
                    reply = re.sub(r'</?think>', '', raw).strip() or "(No response)"
            except Exception as e:
                reply = f"Error: {e}"
        else:
            reply = "API key not set."
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
