"""Load SpatialGT model checkpoints."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from .model_spatialpt import SpatialNeighborTransformer
except ImportError:  # Direct execution from pretrain/.
    from model_spatialpt import SpatialNeighborTransformer


WEIGHT_NAMES = ("finetuned_state_dict.pth", "model.safetensors", "pytorch_model.bin")


def resolve_checkpoint_file(path: str | Path) -> Path:
    """Accept a weights file, checkpoint directory, or downloaded HF model root."""
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix not in (".pth", ".pt", ".bin", ".safetensors"):
            raise ValueError(f"Unsupported checkpoint file: {path}")
        return path
    for directory in (path, path / "checkpoint"):
        for filename in WEIGHT_NAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"No SpatialGT weights found in {path} or its checkpoint/ subdirectory; "
        f"expected one of {', '.join(WEIGHT_NAMES)}. "
        "For a training run, select the specific checkpoint directory."
    )


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a complete tensor state dictionary on CPU, including common wrappers."""
    weight_file = resolve_checkpoint_file(path)
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(weight_file), device="cpu")
    else:
        state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {weight_file}")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"Expected tensor-only model weights in {weight_file}")
    state = dict(state)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def configure_from_state_dict(config, state: Mapping[str, torch.Tensor]):
    """Set shape-defined architecture fields; attention-head count remains configured."""
    required = ("gene_embedding_layer.weight", "mask_token", "reconstruction_head.0.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Incomplete SpatialGT checkpoint: missing {', '.join(missing)}")
    embedding = state["gene_embedding_layer.weight"]
    head = state["reconstruction_head.0.weight"]
    mask_token = state["mask_token"]
    if embedding.ndim != 2 or head.ndim != 2 or mask_token.ndim != 1:
        raise ValueError("Invalid embedding, reconstruction-head or mask-token tensor shape")
    n_genes, embedding_dim = embedding.shape
    hidden_dim = mask_token.shape[0]
    if tuple(head.shape) != (n_genes, hidden_dim):
        raise ValueError("Checkpoint vocabulary/hidden dimensions disagree between embedding and head")
    projection_key = "gene_pretrained_projection.weight"
    use_projection = projection_key in state
    if use_projection and tuple(state[projection_key].shape) != (hidden_dim, embedding_dim):
        raise ValueError("Checkpoint gene-projection dimensions disagree with the embedding")
    if not use_projection and embedding_dim != hidden_dim:
        raise ValueError("Checkpoint lacks the gene projection required by its embedding dimension")
    layer_ids = {
        int(match.group(1))
        for key in state
        for match in [re.match(r"^transformer_encoder\.layers\.(\d+)\.", key)]
        if match
    }
    if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError("Checkpoint has missing or non-contiguous transformer layers")
    n_heads = int(config.num_heads)
    if n_heads < 1 or hidden_dim % n_heads:
        raise ValueError("Configured attention-head count must divide checkpoint hidden dimension")
    config.num_genes = int(n_genes)
    config.gene_embedding_dim = int(embedding_dim)
    config.use_gene_projection = use_projection
    config.d_model = int(hidden_dim)
    config.num_layers = len(layer_ids)
    return config


def load_model_checkpoint(path: str | Path, config, device="cpu") -> SpatialNeighborTransformer:
    """Instantiate the checkpoint architecture and require every model tensor to load."""
    state = load_checkpoint_state_dict(path)
    model_config = configure_from_state_dict(copy.copy(config), state)
    model = SpatialNeighborTransformer(model_config)
    model.load_state_dict(state, strict=True)
    return model.to(device)
