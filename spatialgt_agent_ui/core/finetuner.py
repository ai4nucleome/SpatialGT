"""
Finetune wrapper: runs finetuning as a subprocess for non-blocking UI.
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FINETUNE_SCRIPT = _REPO_ROOT / "finetune" / "finetune.py"


def run_finetune(
    base_ckpt: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    epochs: int = 10,
    unfreeze_last_n: int = 8,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    device: str = "cuda:0",
    visible_gpus: str | None = None,
    cache_mode: str = "h5",
    lmdb_path: str | Path | None = None,
    lmdb_manifest_path: str | Path | None = None,
    num_workers: int = 8,
    validation_interval: int = 50,
    checkpoint_interval: int = 500,
    save_total_limit: int = 50,
    log_file: Optional[str | Path] = None,
) -> subprocess.Popen:
    """
    Launch finetuning as a background subprocess.

    Returns the Popen handle so the UI can poll for completion.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_FINETUNE_SCRIPT),
        "--base_ckpt", str(base_ckpt),
        "--cache_dir", str(cache_dir),
        "--output_dir", str(output_dir),
        "--unfreeze_last_n", str(unfreeze_last_n),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size),
        "--num_epochs", str(epochs),
        "--validation_interval", str(validation_interval),
        "--checkpoint_interval", str(checkpoint_interval),
        "--save_total_limit", str(save_total_limit),
        "--validation_split", "0.00",
        "--device", device,
    ]
    if cache_mode == "lmdb":
        if not lmdb_path or not lmdb_manifest_path:
            raise ValueError("lmdb_path and lmdb_manifest_path are required for LMDB finetuning.")
        cmd.extend([
            "--cache_mode", "lmdb",
            "--lmdb_path", str(lmdb_path),
            "--lmdb_manifest_path", str(lmdb_manifest_path),
            "--num_workers", str(num_workers),
        ])

    log_handle = None
    if log_file:
        log_handle = open(log_file, "w")

    env = os.environ.copy()
    if visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle or subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc


def find_latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """After successful training, prefer its final save over interval checkpoints."""
    output_dir = Path(output_dir)
    weight_names = ("model.safetensors", "pytorch_model.bin", "finetuned_state_dict.pth")
    if any((output_dir / name).is_file() for name in weight_names):
        return output_dir
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=_ckpt_step, reverse=True)
    for ckpt in ckpts:
        if any((ckpt / name).is_file() for name in weight_names):
            return ckpt
    return None


def _ckpt_step(p: Path) -> int:
    try:
        return int(p.name.split("-")[-1])
    except (ValueError, IndexError):
        return 0


def is_finetuning_complete(proc: subprocess.Popen) -> bool:
    return proc.poll() is not None
