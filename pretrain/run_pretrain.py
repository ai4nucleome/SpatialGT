"""
SpatialGT Pretraining Script

This script handles the pretraining of the SpatialGT model using the 
Hugging Face Trainer for distributed training and efficient checkpointing.

Usage:
    # Single GPU
    python run_pretrain.py
    
    # Multi-GPU with torchrun
    torchrun --standalone --nproc_per_node=4 run_pretrain.py
"""

import os
import sys
import time
import glob
import shutil
import logging
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback

warnings.filterwarnings("ignore")

# Local imports
from utils import setup_logger, seed_everything
from Config import Config
from spatial_databank import SpatialDataBank
from model_spatialpt import SpatialNeighborTransformer
from loss import compute_reconstruction_loss as loss_fn
from utils_train import process_batch, forward_pass

# Initialize logger
logger = setup_logger()


class SpatialDataCollator:
    """Custom data collator for Trainer."""
    
    def __init__(self, config):
        self.config = config
        
    def __call__(self, features):
        """Collate a batch of data."""
        return features


class SpatialTrainer(Trainer):
    """Custom Trainer subclass for spatial transcriptomics data."""
    
    def __init__(self, config, total_spots, train_loader, val_loader, predict_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.total_spots = total_spots
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.predict_loader = predict_loader
        self._eval_total_steps = len(val_loader) if val_loader is not None else 0
        self._eval_progress_steps = 0
        self._log_eval_progress = False
        self._step_log_interval = getattr(config, 'print_interval', 10)
    
    def get_train_dataloader(self) -> DataLoader:
        """Return training dataloader (avoid re-preparing with accelerator)."""
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Return evaluation dataloader."""
        return self.val_dataloader

    def log(self, logs, start_time=None):
        """Override logging to add custom metrics."""
        if self.state.global_step % self.args.logging_steps == 0 and torch.cuda.is_available():
            dev = torch.cuda.current_device()
            logs["memory_allocated_GB"] = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            logs["memory_reserved_GB"] = torch.cuda.memory_reserved(dev) / (1024 ** 3)
        
        dataset = getattr(getattr(self, "train_dataloader", None), "dataset", None)
        if hasattr(dataset, "cache_hits"):
            total = max(dataset.cache_hits + dataset.cache_misses, 1)
            logs["data/cache_hit_ratio"] = float(dataset.cache_hits) / float(total)
        
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute training loss."""
        # Skip placeholder batches
        if isinstance(inputs, dict) and inputs.get("skip_batch", False):
            try:
                any_param = next(model.parameters())
                loss = any_param.sum() * 0.0
            except StopIteration:
                device = self.args.device if hasattr(self, "args") else None
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            return (loss, None) if return_outputs else loss

        device = next(model.parameters()).device
        batch_data = process_batch(inputs, device, config=self.config)

        # Let Trainer manage AMP
        use_amp = bool(self.args.fp16 or self.args.bf16)
        if use_amp:
            autocast_cm = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            autocast_cm = nullcontext()

        with autocast_cm:
            predictions, targets, valid_mask, _ = forward_pass(model, batch_data, config=self.config)
            loss = loss_fn(predictions, targets, valid_mask)

        # Record loss value for logging
        try:
            self._last_loss_value = loss.detach().float().item() if torch.is_tensor(loss) else float(loss)
        except Exception:
            self._last_loss_value = None

        # Log step progress (main process only)
        try:
            is_main = self._is_main_process()
            
            if is_main:
                interval = max(1, int(self._step_log_interval))
                current_step = int(self.state.global_step)
                next_step = current_step + 1
                total_steps = int(self.state.max_steps) if self.state.max_steps is not None else None
                should_log = (next_step % interval == 0)
                
                if should_log:
                    try:
                        current_lr = float(self.optimizer.param_groups[0]['lr']) if hasattr(self, 'optimizer') and self.optimizer is not None else float('nan')
                    except Exception:
                        current_lr = float('nan')
                    
                    total_txt = f"/{total_steps}" if total_steps is not None else ""
                    lr_txt = f", LR: {current_lr:.7f}" if current_lr == current_lr else ""
                    loss_val = self._last_loss_value
                    loss_txt = f", Loss: {loss_val:.4f}" if isinstance(loss_val, (int, float)) else ""
                    
                    line = f"[StepLogger] Step {next_step}{total_txt}{loss_txt}{lr_txt}"
                    logger.info(line)
                    print(line, flush=True)
        except Exception:
            pass

        outputs = {"predict": predictions, "targets": targets}
        return (loss, outputs) if return_outputs else loss
    
    def _is_main_process(self):
        """Check if this is the main process."""
        try:
            return bool(self.is_world_process_zero())
        except Exception:
            pass
        
        try:
            return os.environ.get('LOCAL_RANK', '0') in ['0', '-1']
        except Exception:
            pass
        
        try:
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            pass
        
        return False
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Callback at the beginning of each epoch."""
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(state.epoch)
        return control

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        """Run one prediction step."""
        # Skip placeholder batches
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

        if prediction_loss_only:
            if self._log_eval_progress:
                self._eval_progress_steps += 1
                interval = max(1, getattr(self.config, "print_interval", 100))
                if (self._eval_progress_steps % interval == 0) or (self._eval_total_steps and self._eval_progress_steps >= self._eval_total_steps):
                    logger.info(f"[Eval Progress] step {self._eval_progress_steps}/{self._eval_total_steps or '?'}")
            return (loss, None, None)

        preds_detached = predictions[valid_mask].detach()
        targs_detached = targets[valid_mask].detach()

        return (loss, preds_detached, targs_detached)

    def evaluate(self, *args, **kwargs):
        """Run evaluation."""
        if self.val_dataloader is not None:
            self._eval_total_steps = len(self.val_dataloader)
            self._eval_progress_steps = 0
            self._log_eval_progress = True
            logger.info(f"Starting validation, num_batches={self._eval_total_steps}")
        
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            if self._log_eval_progress:
                logger.info("Validation completed")
                self._log_eval_progress = False


def prepare_runtime_cache(config):
    """Copy LMDB cache/manifest to runtime location if requested."""
    if getattr(config, "cache_mode", "h5").lower() != "lmdb":
        return
    
    lmdb_src = Path(config.lmdb_path)
    manifest_src = Path(config.lmdb_manifest_path)
    
    if not lmdb_src.exists():
        raise FileNotFoundError(f"LMDB file not found: {lmdb_src}")
    if not manifest_src.exists():
        raise FileNotFoundError(f"LMDB manifest not found: {manifest_src}")
    
    runtime_lmdb = lmdb_src
    runtime_manifest = manifest_src
    
    if getattr(config, "copy_lmdb_to_shm", False):
        target_dir = Path(config.shared_cache_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        runtime_lmdb = target_dir / lmdb_src.name
        runtime_manifest = target_dir / manifest_src.name
        
        if (not runtime_lmdb.exists()) or (lmdb_src.stat().st_mtime > runtime_lmdb.stat().st_mtime):
            logger.info(f"Copying LMDB cache to {runtime_lmdb}")
            shutil.copy2(lmdb_src, runtime_lmdb)
        if (not runtime_manifest.exists()) or (manifest_src.stat().st_mtime > runtime_manifest.stat().st_mtime):
            shutil.copy2(manifest_src, runtime_manifest)
    
    config.runtime_lmdb_path = str(runtime_lmdb)
    config.runtime_lmdb_manifest_path = str(runtime_manifest)
    logger.info(f"Using LMDB cache: {config.runtime_lmdb_path}")


def train_with_trainer(config, args):
    """
    Train using Hugging Face Trainer.
    
    Args:
        config: Configuration object
        args: Command line arguments dict
    """
    global logger
    logger = setup_logger()
    
    # Set random seed for reproducibility
    seed_everything(config.random_seed)
    
    # Prepare runtime cache (LMDB)
    prepare_runtime_cache(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize TensorBoard
    experiment_name = f"{config.model_name}_{args.get('experiment_id', 'default')}"
    log_dir = Path(config.output_dir) / 'tensorboard' / experiment_name
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Distributed training setup
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = env_world_size if env_world_size > 0 else 1
    env_rank = int(os.environ.get("RANK", "0"))
    rank = env_rank if env_rank >= 0 else 0
    if world_size <= 1 and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
    use_distributed = world_size > 1
    
    # Load data
    logger.info("Initializing SpatialDataBank...")
    try:
        databank = SpatialDataBank(
            dataset_paths=args['dataset_paths'],
            cache_dir=config.cache_dir,
            config=config,
            force_rebuild=False
        )
        
        # Prepare vocabulary
        vocab = databank.prepare_vocabulary()
        logger.info(f"Vocabulary size: {len(vocab)}")
        
        # Get data loaders
        train_loader = databank.get_data_loader(
            split="train",
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            stratify_by_batch=getattr(config, 'stratify_by_batch', False),
            validation_split=config.validation_split,
            use_distributed=use_distributed,
            world_size=world_size,
            rank=rank,
            persistent_workers=False
        )

        val_loader = databank.get_data_loader(
            split="val",
            batch_size=getattr(config, 'eval_batch_size', config.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            stratify_by_batch=getattr(config, 'stratify_by_batch', False),
            validation_split=config.validation_split,
            use_distributed=use_distributed,
            world_size=world_size,
            rank=rank,
            persistent_workers=False
        )
        
        logger.info(f"Data loaders created: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        
    except Exception as e:
        logger.error(f"Failed to initialize databank: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Create model
    logger.info("Creating model...")
    model = SpatialNeighborTransformer(config)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    params_info = get_parameter_number(model)
    logger.info(f"Model parameters: {params_info}")

    def get_model_size_mb(model):
        param_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        buffer_size = sum(b.numel() * 4 for b in model.buffers()) / (1024 * 1024)
        total_size = param_size + buffer_size
        return {
            'Parameters Size (MB)': param_size,
            'Buffers Size (MB)': buffer_size,
            'Total Size (MB)': total_size,
            'Total Size (GB)': total_size / 1024
        }

    model_size_info = get_model_size_mb(model)
    logger.info(f"Model memory footprint: {model_size_info}")

    # Configure training arguments
    training_args = TrainingArguments(
        ignore_data_skip=True,
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=getattr(config, 'eval_batch_size', config.batch_size),
        gradient_accumulation_steps=config.accumulation_steps,
        eval_accumulation_steps=getattr(config, "eval_accumulation_steps", None),
        max_steps=config.max_steps if getattr(config, "max_steps", None) is not None else -1,
        eval_strategy="steps",
        eval_steps=config.validation_interval,
        save_strategy="steps",
        save_steps=config.checkpoint_interval,
        save_total_limit=1000,
        logging_dir=str(log_dir),
        logging_strategy="steps",
        logging_steps=config.print_interval,
        log_on_each_node=False,
        prediction_loss_only=True,
        learning_rate=config.learning_rate,
        warmup_steps=getattr(config, 'num_warmup_steps', 500),
        lr_scheduler_type="cosine",
        weight_decay=config.weight_decay,
        fp16=True,
        dataloader_drop_last=True,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=getattr(config, "pin_memory", True),
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.random_seed,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )
    
    # Create data collator and trainer
    data_collator = SpatialDataCollator(config)
    
    trainer = SpatialTrainer(
        config=config,
        total_spots=databank.metadata["total_spots"],
        train_loader=train_loader,
        val_loader=val_loader,
        predict_loader=None,
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset if val_loader else None,
        compute_metrics=None,
        data_collator=data_collator,
        callbacks=[]
    )
    
    # Start training
    logger.info("Starting training...")
    
    if args.get('resume_from_checkpoint'):
        train_result = trainer.train(resume_from_checkpoint=args['resume_from_checkpoint'])
    else:
        train_result = trainer.train()

    # Save final model
    logger.info("Training complete, saving model...")
    trainer.save_model()
    
    # Save model state without DDP wrapper
    final_model_path = os.path.join(config.output_dir, 'final_spatial_neighbor_model.pth')
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model state saved to: {final_model_path}")
    
    # Output training results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    if val_loader:
        logger.info("Running final evaluation...")
        metrics = trainer.evaluate(prediction_loss_only=True)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training pipeline complete!")
    return trainer


def main():
    """Main function - parse command line arguments and start training."""
    parser = argparse.ArgumentParser(description='SpatialGT Pretraining')
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    logger = setup_logger()

    # Set local_rank for distributed training
    if 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        config.local_rank = args.local_rank

    if torch.cuda.is_available() and config.local_rank != -1:
        torch.cuda.set_device(config.local_rank)
        logger.info(f"Set current GPU to local_rank={config.local_rank}")
    
    if config.local_rank not in [0, -1]:
        logger.disabled = True

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(config.output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Build dataset paths
    dataset_paths = []
    if getattr(config, "dataset_list", None) and os.path.exists(config.dataset_list):
        with open(config.dataset_list, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        
        for folder in folder_paths:
            h5ad_files = glob.glob(os.path.join(folder, "*.h5ad"))
            if h5ad_files:
                dataset_paths.extend(h5ad_files)
            else:
                logger.warning(f"No .h5ad files found in: {folder}")
        
        logger.info(f"Found {len(dataset_paths)} preprocessed h5ad files")
    elif getattr(config, "dataset_paths", None):
        dataset_paths = list(config.dataset_paths)
        logger.info(f"Using config.dataset_paths, count={len(dataset_paths)}")
    else:
        raise ValueError("No dataset_list or dataset_paths provided, cannot load datasets")

    # Build arguments dict
    args_dict = {
        'dataset_paths': dataset_paths,
        'cache_dir': config.cache_dir,
        'output_dir': config.output_dir,
        'tensorboard_dir': tensorboard_dir,
        'resume_from_checkpoint': getattr(config, 'resume_from_checkpoint', None),
        'print_interval': getattr(config, 'print_interval', 1),
        'experiment_id': getattr(config, 'experiment_id', time.strftime("%Y%m%d-%H%M%S")),
    }

    # Log CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA available, device count: {torch.cuda.device_count()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        try:
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {device_name}, Memory: {device_mem:.2f} GB")
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
    else:
        logger.error("CUDA not available, please check environment setup")
        sys.exit(1)

    torch.cuda.empty_cache()

    logger.info("Starting training...")
    try:
        if torch.cuda.device_count() > 1:
            logger.info(f"Detected {torch.cuda.device_count()} GPUs, using Accelerate/DDP (handled by Trainer)")
        
        trainer = train_with_trainer(config, args_dict)
        logger.info("Training complete!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    try:
        print("Starting script...", flush=True)
        sys.stdout.flush()
        main()
    except Exception as e:
        if 'logger' in locals():
            logger.exception(f"Runtime exception: {e}")
        else:
            print(f"Exception before logger initialization: {e}")
        raise
