"""
Model management: download from HuggingFace, load checkpoints, configure runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretrain.Config import Config
from pretrain.model_spatialpt import SpatialNeighborTransformer

HF_PRETRAINED_REPO = "Bgoood/SpatialGT-Pretrained"
HF_GENE_EMBEDDING_REPO = "Bgoood/SpatialGT-GeneEmbedding"

_DEFAULT_GENE_EMB_DIR = _REPO_ROOT / "gene_embedding"


def download_from_hf(repo_id: str, local_dir: str | Path) -> Path:
    from huggingface_hub import snapshot_download
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def ensure_gene_embeddings(gene_emb_dir: str | Path | None = None) -> Path:
    """Make sure gene embedding files exist; download from HF if needed."""
    if gene_emb_dir:
        d = Path(gene_emb_dir)
    else:
        d = _DEFAULT_GENE_EMB_DIR

    vocab = d / "vocab.json"
    emb = d / "pretrained_gene_embeddings.pt"
    if vocab.exists() and emb.exists():
        return d

    print(f"[ModelManager] Downloading gene embeddings to {d} ...")
    download_from_hf(HF_GENE_EMBEDDING_REPO, d)
    return d


def make_config(
    cache_dir: str | Path | None = None,
    gene_emb_dir: str | Path | None = None,
    device: str = "cuda:0",
    max_neighbors: int = 8,
    subset_hvg: int = 3000,
    cache_mode: str = "h5",
    lmdb_path: str | None = None,
    lmdb_manifest_path: str | None = None,
) -> Config:
    """Build a Config with paths overridden for the UI environment."""
    config = Config()

    gene_dir = ensure_gene_embeddings(gene_emb_dir)
    config.vocab_file = str(gene_dir / "vocab.json")
    config.pretrained_gene_embeddings_path = str(gene_dir / "pretrained_gene_embeddings.pt")

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
        config = make_config(device=device)

    model = SpatialNeighborTransformer(config)

    ckpt_path = Path(ckpt_path)
    sd = _load_state_dict(ckpt_path)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_state_dict(ckpt_dir: Path) -> dict:
    for name, loader in [
        ("finetuned_state_dict.pth", lambda p: torch.load(str(p), map_location="cpu")),
        ("model.safetensors", _load_safetensors),
        ("pytorch_model.bin", lambda p: _load_bin(p)),
    ]:
        f = ckpt_dir / name
        if f.exists():
            print(f"[ModelManager] Loading weights from {f}")
            return loader(f)
    raise FileNotFoundError(f"No model weights found in {ckpt_dir}")


def _load_safetensors(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _load_bin(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def download_pretrained(local_dir: str | Path | None = None) -> Path:
    if local_dir is None:
        local_dir = _REPO_ROOT / "model" / "pretrain_ckpt"
    return download_from_hf(HF_PRETRAINED_REPO, local_dir)
