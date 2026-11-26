"""
Model management: download from HuggingFace, load checkpoints, configure runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrain.Config import Config
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.checkpoint import (
    load_checkpoint_state_dict, load_model_checkpoint, resolve_checkpoint_file,
)

HF_PRETRAINED_REPO = "Bgoood/SpatialGT-Pretrained"
_DEFAULT_VOCAB_FILE = _REPO_ROOT / "vocab.json"


def download_from_hf(repo_id: str, local_dir: str | Path, allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns)
    return local_dir


def ensure_gene_vocabulary(vocab_file: str | Path | None = None) -> Path:
    """Return the local gene-ID vocabulary file."""
    path = Path(vocab_file) if vocab_file is not None else _DEFAULT_VOCAB_FILE
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Gene vocabulary not found: {path}. Clone the complete SpatialGT "
            "repository, which includes vocab.json, or provide vocab_file."
        )
    return path


def make_config(
    cache_dir: str | Path | None = None,
    vocab_file: str | Path | None = None,
    device: str = "cuda:0",
    max_neighbors: int = 8,
    subset_hvg: int = 3000,
    cache_mode: str = "h5",
    lmdb_path: str | None = None,
    lmdb_manifest_path: str | None = None,
) -> Config:
    """Build a Config with paths overridden for the UI environment."""
    config = Config()

    config.vocab_file = str(ensure_gene_vocabulary(vocab_file))

    if cache_dir:
        config.cache_dir = str(cache_dir)
    config.device = device
    config.max_neighbors = max_neighbors
    config.subset_hvg = subset_hvg
    config.max_seq_len = subset_hvg

    config.cache_mode = cache_mode
    config.strict_cache_only = True

    if cache_mode == "lmdb" and lmdb_path:
        config.lmdb_path = lmdb_path
        config.runtime_lmdb_path = lmdb_path
        if lmdb_manifest_path:
            config.lmdb_manifest_path = lmdb_manifest_path
            config.runtime_lmdb_manifest_path = lmdb_manifest_path

    return config


def load_model(
    ckpt_path: str | Path,
    config: Optional[Config] = None,
    device: str = "cuda:0",
) -> SpatialNeighborTransformer:
    """
    Load SpatialNeighborTransformer from a checkpoint directory.
    Supports: model.safetensors, pytorch_model.bin, finetuned_state_dict.pth
    """
    if config is None:
        config = Config()
    model = load_model_checkpoint(ckpt_path, config, device=device)
    model.eval()
    return model


def _load_state_dict(ckpt_dir: Path) -> dict:
    """Read model tensor weights."""
    return load_checkpoint_state_dict(ckpt_dir)


def download_pretrained(local_dir: str | Path | None = None) -> Path:
    """Download inference weights and return the directory that contains them."""
    if local_dir is None:
        local_dir = _REPO_ROOT / "model" / "pretrain_ckpt"
    root = download_from_hf(
        HF_PRETRAINED_REPO,
        local_dir,
        allow_patterns=["checkpoint/model.safetensors", "model.safetensors"],
    )
    checkpoint_dir = root / "checkpoint"
    if checkpoint_dir.is_dir():
        return resolve_checkpoint_file(checkpoint_dir).parent
    return resolve_checkpoint_file(root).parent
