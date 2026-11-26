#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialGT Fine-tuning Script

This script handles fine-tuning of a pretrained SpatialGT model on new datasets.
Default configuration:
- Full fine-tuning with all 8 transformer layers unfrozen
- 100 epochs of training
- Reconstruction head always unfrozen

Usage:
    # Single GPU
    python finetune.py --base_ckpt <checkpoint_path> --cache_dir <cache> --output_dir <output>
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 finetune.py --base_ckpt <checkpoint> ...
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def _configure_device_environment(argv):
    """Select a single requested logical GPU before importing the CUDA stack."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=None)
    args, _ = parser.parse_known_args(argv)
    device = args.device
    if not device or not device.startswith("cuda:"):
        return device
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return "cuda"  # torchrun selects each rank's device.
    index = int(device.split(":", 1)[1])
    if index < 0:
        raise ValueError("CUDA device index must be non-negative")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if index >= len(devices):
            raise ValueError("Requested CUDA index is outside CUDA_VISIBLE_DEVICES")
        selected = devices[index]
    else:
        selected = str(index)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return "cuda:0"


_ENTRY_DEVICE = _configure_device_environment(sys.argv[1:]) if __name__ == "__main__" else None

import torch


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))


from pretrain.spatial_databank import SpatialDataBank
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.utils import setup_logger, seed_everything
from pretrain.utils_train import process_batch, forward_pass
from pretrain.loss import compute_reconstruction_loss as loss_fn

from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

from finetune.Config import FinetuneConfig
from pretrain.checkpoint import load_checkpoint_state_dict, load_model_checkpoint


def freeze_all(model: torch.nn.Module) -> None:
    """Freeze all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_encoder_layers(model: SpatialNeighborTransformer, n: int) -> None:
    """
    Unfreeze the last N encoder layers.
    
    Args:
        model: The SpatialGT model
        n: Number of layers to unfreeze from the end
    """
    n = int(max(n, 0))
    if n <= 0:
        return
    
    layers = getattr(model.transformer_encoder, "layers", None)
    if layers is None:
        raise AttributeError("model.transformer_encoder.layers not found")
    
    # Unfreeze last n layers
    for layer in list(layers)[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def unfreeze_head(model: SpatialNeighborTransformer) -> None:
    """Unfreeze the reconstruction head."""
    for p in model.reconstruction_head.parameters():
        p.requires_grad = True


def unfreeze_embeddings(model: SpatialNeighborTransformer) -> None:
    """Unfreeze gene embedding layers."""
    for p in model.gene_embedding_layer.parameters():
        p.requires_grad = True
    if hasattr(model, 'gene_pretrained_projection') and model.gene_pretrained_projection is not None:
        for p in model.gene_pretrained_projection.parameters():
            p.requires_grad = True


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def is_cache_ready(cache_dir: Path) -> bool:
    """
    Check if cache directory has required files for SpatialDataBank.
    
    Required:
    - metadata.json
    - neighbors/neighbors_index.json
    - At least one dataset subdirectory with spots.h5
    """
    if not (cache_dir / "metadata.json").exists():
        return False
    if not (cache_dir / "neighbors" / "neighbors_index.json").exists():
        return False

    for child in cache_dir.iterdir():
        if child.is_dir() and (child / "spots.h5").exists():
            return True
    return False


def ensure_neighbors_index(cache_dir: Path, dataset_name: str) -> None:
    """
    Ensure neighbors/neighbors_index.json exists and is valid.
    
    Creates/rebuilds the index file if necessary.
    """
    neighbors_dir = cache_dir / "neighbors"
    neighbors_dir.mkdir(parents=True, exist_ok=True)
    index_file = neighbors_dir / "neighbors_index.json"

    def try_load() -> bool:
        if not index_file.exists():
            return False
        try:
            with open(index_file, "r") as f:
                json.load(f)
            return True
        except Exception:
            return False

    if try_load():
        return

    # Rebuild minimal mapping for single-dataset fine-tuning.
    pattern = str(neighbors_dir / f"neighbors_{dataset_name}_n*.h5")
    matches = sorted(glob.glob(pattern))
    if not matches:
        matches = sorted(glob.glob(str(neighbors_dir / "neighbors_*_n*.h5")))
    if not matches:
        raise FileNotFoundError(
            f"No neighbor h5 files found under {neighbors_dir}; "
            "cannot rebuild neighbors_index.json"
        )

    rebuilt = {"0": matches[0]}
    with open(index_file, "w") as f:
        json.dump(rebuilt, f, indent=2)
    print(f"[INFO] Rebuilt neighbors_index.json -> {index_file}")


class FinetuneDataCollator:
    """Custom data collator for fine-tuning."""
    
    def __init__(self, config):
        self.config = config
        
    def __call__(self, features):
        """Return batch as-is (already collated by DataLoader)."""
        return features


class FinetuneTrainer(Trainer):
    """Custom Trainer for SpatialGT fine-tuning."""
    
    def __init__(self, config, total_spots, train_loader, val_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.total_spots = total_spots
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self._step_log_interval = getattr(config, 'print_interval', 10)
    
    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        return self.val_dataloader

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Skip placeholder batches
        if isinstance(inputs, dict) and inputs.get("skip_batch", False):
            try:
                any_param = next(model.parameters())
                loss = any_param.sum() * 0.0
            except StopIteration:
                loss = torch.tensor(0.0, requires_grad=True)
            return (loss, None) if return_outputs else loss

        device = next(model.parameters()).device
        batch_data = process_batch(inputs, device, config=self.config)

        # Use AMP if enabled
        use_amp = bool(self.args.fp16 or self.args.bf16)
        if use_amp:
            autocast_cm = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            autocast_cm = nullcontext()

        with autocast_cm:
            predictions, targets, valid_mask, _ = forward_pass(model, batch_data, config=self.config)
            loss = loss_fn(predictions, targets, valid_mask)

        outputs = {"predict": predictions, "targets": targets}
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        if isinstance(inputs, dict) and inputs.get("skip_batch", False):
            zero = next(model.parameters()).sum() * 0.0
            return (zero, None, None)

        device = next(model.parameters()).device
        batch_data = process_batch(inputs, device, config=self.config)
        
        use_amp = bool(self.args.fp16 or self.args.bf16)
        if use_amp:
            autocast_cm = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            autocast_cm = nullcontext()

        with torch.no_grad():
            with autocast_cm:
                predictions, targets, valid_mask, _ = forward_pass(model, batch_data, config=self.config)
                loss = loss_fn(predictions, targets, valid_mask)
                loss = loss.detach()

        return (loss, None, None)


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(description="SpatialGT Fine-tuning")
    
    # Required arguments
    parser.add_argument("--base_ckpt", required=True, 
                        help="Path to pretrained checkpoint directory")
    parser.add_argument("--cache_dir", required=True,
                        help="Cache directory with preprocessed data")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--device", default=None,
                        help="Runtime device (default: CUDA when available, otherwise CPU)")
    parser.add_argument("--vocab_file", default=None,
                        help="Vocabulary JSON associated with the checkpoint")
    parser.add_argument("--use_existing_cache", action="store_true",
                        help="Use the preprocessed cache")
    
    # Optional: dataset source
    parser.add_argument("--dataset_folder", default=None,
                        help="Folder containing .h5ad file(s) (not needed if cache exists)")
    
    # Fine-tuning strategy
    parser.add_argument("--unfreeze_last_n", type=int, default=8,
                        help="Number of encoder layers to unfreeze (default: 8 = full fine-tuning)")
    parser.add_argument("--unfreeze_embeddings", action="store_true",
                        help="Also unfreeze gene embeddings")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--cache_mode", choices=["h5", "lmdb"], default="h5",
                        help="Cache backend")
    parser.add_argument("--lmdb_path", type=str, default=None,
                        help="LMDB file path (required if cache_mode=lmdb)")
    parser.add_argument("--lmdb_manifest_path", type=str, default=None,
                        help="LMDB manifest path (required if cache_mode=lmdb)")
    
    # Checkpointing
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=10,
                        help="Maximum checkpoints to keep")
    
    # Validation (optional)
    parser.add_argument("--validation_split", type=float, default=0.0,
                        help="Validation split ratio (0 = no validation)")
    parser.add_argument("--validation_interval", type=int, default=500,
                        help="Validation interval in steps")
    
    args = parser.parse_args()
    

    logger = setup_logger()
    
    # Paths
    base_ckpt = Path(args.base_ckpt)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = _ENTRY_DEVICE or args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    has_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not has_cuda:
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    logger.info(f"CUDA available: {has_cuda}")
    
    # Check cache readiness
    if not is_cache_ready(cache_dir):
        raise RuntimeError(
            f"Cache not ready at {cache_dir}. "
            "Please run preprocessing first using pretrain/preprocess.py"
        )
    

    with open(cache_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    
    dataset_paths = []
    for ds in meta.get("datasets", []):
        proc_path = Path(ds.get("cache_dir", "")) / "processed.h5ad"
        if proc_path.exists():
            dataset_paths.append(str(proc_path))
        

    if not dataset_paths:
        logger.warning(f"Scanning {cache_dir} for processed.h5ad files...")
        for subdir in cache_dir.iterdir():
            if subdir.is_dir() and subdir.name not in ("neighbors", "__pycache__"):
                proc_path = subdir / "processed.h5ad"
                if proc_path.exists():
                    dataset_paths.append(str(proc_path))
        
    if not dataset_paths:
        raise FileNotFoundError(f"No processed.h5ad found in {cache_dir}")
    
    logger.info(f"Found {len(dataset_paths)} dataset(s)")
    
    # Initialize config
    config = FinetuneConfig()
    config.output_dir = str(output_dir)
    config.cache_dir = str(cache_dir)
    config.cache_mode = args.cache_mode
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.max_steps = args.max_steps
    config.eval_batch_size = args.eval_batch_size
    config.validation_split = args.validation_split
    config.validation_interval = args.validation_interval
    config.checkpoint_interval = args.checkpoint_interval
    config.device = device
    if args.vocab_file:
        config.vocab_file = args.vocab_file
    
    # H5 mode: force single worker for stability
    if config.cache_mode == "h5" and config.num_workers > 1:
        logger.info(f"H5 cache mode: setting num_workers=1 for stability")
        config.num_workers = 1
    
    # LMDB configuration
    if config.cache_mode == "lmdb":
        if not args.lmdb_path or not args.lmdb_manifest_path:
            raise ValueError("--lmdb_path and --lmdb_manifest_path required for lmdb mode")
        config.lmdb_path = args.lmdb_path
        config.lmdb_manifest_path = args.lmdb_manifest_path

    seed_everything(config.random_seed)


    first_dataset_name = Path(dataset_paths[0]).parent.name
    ensure_neighbors_index(cache_dir, first_dataset_name)

    # Distributed training detection
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if has_cuda and not is_distributed and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "For multiple GPUs, launch with torchrun --nproc_per_node=N; "
            "otherwise select one GPU with --device cuda:N or CUDA_VISIBLE_DEVICES."
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Distributed: WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={local_rank}")


    logger.info("Loading SpatialDataBank...")
    databank = SpatialDataBank(
        dataset_paths=dataset_paths,
        cache_dir=str(cache_dir),
        config=config,
        force_rebuild=False
    )
    
    train_loader = databank.get_data_loader(
        split="train",
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        stratify_by_batch=False,
        validation_split=config.validation_split,
        use_distributed=is_distributed,
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        rank=int(os.environ.get("RANK", 0)),
        persistent_workers=(config.num_workers > 0),
    )
    
    val_loader = None
    if config.validation_split > 0:
        val_loader = databank.get_data_loader(
            split="val",
            batch_size=config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            stratify_by_batch=False,
            validation_split=config.validation_split,
            use_distributed=is_distributed,
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            persistent_workers=(config.num_workers > 0),
        )

    logger.info(f"Train batches: {len(train_loader)}")
    if len(train_loader) == 0:
        raise RuntimeError(
            "Fine-tuning train_loader has zero batches. Check cache metadata, "
            "neighbor files, and batch_size/drop_last settings."
        )
    

    logger.info("Creating model...")
    logger.info(f"Loading checkpoint from: {base_ckpt}")
    model = load_model_checkpoint(base_ckpt, config, device="cpu")
    config = model.config
    
    # Apply fine-tuning strategy
    logger.info(f"Fine-tuning strategy: unfreeze last {args.unfreeze_last_n} layers + head")
    freeze_all(model)
    unfreeze_last_encoder_layers(model, args.unfreeze_last_n)
    unfreeze_head(model)
    
    if args.unfreeze_embeddings:
        logger.info("Also unfreezing embeddings")
        unfreeze_embeddings(model)
    
    params = count_params(model)
    logger.info(f"Parameters: total={params['total']:,}, trainable={params['trainable']:,}")
    
    # Training arguments
    has_validation = val_loader is not None
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.accumulation_steps,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        num_train_epochs=float(config.num_epochs),
        eval_strategy="steps" if has_validation else "no",
        eval_steps=config.validation_interval if has_validation else None,
        save_strategy="steps",
        save_steps=config.checkpoint_interval,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        logging_strategy="steps",
        logging_steps=config.print_interval,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.num_warmup_steps,
        lr_scheduler_type="cosine",
        fp16=has_cuda,
        use_cpu=not has_cuda,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=config.pin_memory,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,
        seed=config.random_seed,
    )


    data_collator = FinetuneDataCollator(config)
    trainer = FinetuneTrainer(
        config=config,
        total_spots=databank.metadata.get("total_spots", 0),
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=(val_loader.dataset if val_loader else None),
        compute_metrics=None,
        data_collator=data_collator,
        callbacks=[],
    )


    logger.info(f"Starting fine-tuning for {config.num_epochs} epochs...")
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        config.save(str(output_dir / "config.json"))

        state_dict_path = output_dir / "finetuned_model.pth"
        torch.save(model.state_dict(), str(state_dict_path))
        logger.info(f"Saved model state dict to: {state_dict_path}")
    
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
