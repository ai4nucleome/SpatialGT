"""
Training Utilities for SpatialGT

This module provides utility functions for the training pipeline including:
- Batch processing and data movement
- Learning rate scheduling
- Logging and checkpointing
- TensorBoard integration
"""

import os
import sys
import time
import json
import math
import glob
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Non-GUI backend for server environments
warnings.filterwarnings("ignore")

from utils import setup_logger
from Config import Config
from spatial_databank import SpatialDataBank

logger = setup_logger()


def debug_batch_structure(batch, rank=0):
    """Debug function: print batch data structure."""
    if rank == 0:
        logger.info("=== Batch Data Structure Debug ===")
        if isinstance(batch, list):
            logger.info(f"Batch type: list, length: {len(batch)}")
            if len(batch) > 0:
                logger.info(f"First element keys: {list(batch[0].keys()) if isinstance(batch[0], dict) else 'not dict'}")
                if isinstance(batch[0], dict) and 'flat' in batch[0]:
                    logger.info(f"Keys in 'flat': {list(batch[0]['flat'].keys())}")
                if isinstance(batch[0], dict) and 'structure' in batch[0]:
                    logger.info(f"Keys in 'structure': {list(batch[0]['structure'].keys())}")
        else:
            logger.info(f"Batch type: dict")
            logger.info(f"Top-level keys: {list(batch.keys()) if isinstance(batch, dict) else 'not dict'}")
            if isinstance(batch, dict) and 'flat' in batch:
                logger.info(f"Keys in 'flat': {list(batch['flat'].keys())}")
            if isinstance(batch, dict) and 'structure' in batch:
                logger.info(f"Keys in 'structure': {list(batch['structure'].keys())}")
        logger.info("=== End Structure Debug ===")


def process_batch(batch, device, config=None):
    """
    Process batch data and move to device.
    
    Args:
        batch: Input batch dictionary
        device: Target device (cuda/cpu)
        config: Configuration object
    
    Returns:
        Processed batch dictionary with tensors on the target device
    """
    # Early return: skip placeholder batches marked by collate_fn
    if isinstance(batch, dict) and batch.get("skip_batch", False):
        L = 2  # Minimum length
        out = {
            "skip_batch": True,
            "genes": torch.zeros((1, L), dtype=torch.long, device=device),
            "raw_normed_values": torch.zeros((1, L), dtype=torch.float32, device=device),
            "binned_values": torch.zeros((1, L), dtype=torch.float32, device=device),
            "padding_attention_mask": torch.ones((1, L), dtype=torch.bool, device=device),
            "nonzero_genes": torch.zeros((1, 1), dtype=torch.long, device=device),
            "nonzero_expr": torch.zeros((1, 1), dtype=torch.float32, device=device),
            "nonzero_raw_normed": torch.zeros((1, 1), dtype=torch.float32, device=device),
            "nonzero_attention_mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
            "batch_to_spots_map": [(0, 0)],
            "center_indices": torch.tensor([0], dtype=torch.long, device=device),
            "platform_id": torch.zeros((1,), dtype=torch.long, device=device),
            "organ_id": torch.zeros((1,), dtype=torch.long, device=device),
            "disease_id": torch.zeros((1,), dtype=torch.long, device=device),
        }
        return out

    # Print structure once for debugging
    if not hasattr(process_batch, '_debug_printed'):
        debug_batch_structure(batch, rank=0)
        process_batch._debug_printed = True

    if not isinstance(batch, dict):
        raise ValueError(f"Expected batch to be a dict, got: {type(batch)}")
    
    if 'flat' in batch and 'structure' in batch:
        return process_standard_format_batch(batch, device, config=config)
    else:
        raise ValueError("Data must be in standard format with 'flat' and 'structure' keys")


def process_standard_format_batch(batch, device, config=None):
    """
    Process standard format batch with 'flat' and 'structure' keys.
    
    Args:
        batch: Standard format batch dictionary
        device: Target device
        config: Configuration object
    
    Returns:
        Processed batch dictionary
    """
    # Early return for placeholder batches
    if isinstance(batch, dict) and batch.get("skip_batch", False):
        L = 2
        out = {
            "skip_batch": True,
            "genes": torch.zeros((1, L), dtype=torch.long, device=device),
            "raw_normed_values": torch.zeros((1, L), dtype=torch.float32, device=device),
            "binned_values": torch.zeros((1, L), dtype=torch.float32, device=device),
            "padding_attention_mask": torch.ones((1, L), dtype=torch.bool, device=device),
            "nonzero_genes": torch.zeros((1, 1), dtype=torch.long, device=device),
            "nonzero_expr": torch.zeros((1, 1), dtype=torch.float32, device=device),
            "nonzero_raw_normed": torch.zeros((1, 1), dtype=torch.float32, device=device),
            "nonzero_attention_mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
            "batch_to_spots_map": [(0, 0)],
            "center_indices": torch.tensor([0], dtype=torch.long, device=device),
            "platform_id": torch.zeros((1,), dtype=torch.long, device=device),
            "organ_id": torch.zeros((1,), dtype=torch.long, device=device),
            "disease_id": torch.zeros((1,), dtype=torch.long, device=device),
        }
        return out

    flat = batch["flat"]
    structure = batch["structure"]
    out = {}

    # Expression data
    out["genes"] = flat["genes"].to(device) if flat.get("genes") is not None else None
    out["raw_normed_values"] = flat["raw_normed_values"].to(device) if flat.get("raw_normed_values") is not None else None
    out["binned_values"] = flat["binned_values"].to(device) if flat.get("binned_values") is not None else None
    out["padding_attention_mask"] = flat["padding_attention_mask"].to(device) if flat.get("padding_attention_mask") is not None else None

    # Non-zero expression data
    out["nonzero_genes"] = flat["nonzero_genes"].to(device) if flat.get("nonzero_genes") is not None else None
    out["nonzero_expr"] = flat["nonzero_expr"].to(device) if flat.get("nonzero_expr") is not None else None
    out["nonzero_raw_normed"] = flat["nonzero_raw_normed"].to(device) if flat.get("nonzero_raw_normed") is not None else None
    out["nonzero_attention_mask"] = flat["nonzero_attention_mask"].to(device) if flat.get("nonzero_attention_mask") is not None else None

    # Structure information
    batch_to_spots_map = structure.get("batch_to_spots_map", None)
    center_indices = structure.get("center_indices", None)
    platform_ids = structure.get("platform_ids", None)
    organ_ids = structure.get("organ_ids", None)
    disease_ids = structure.get("disease_ids", None)

    out["batch_to_spots_map"] = batch_to_spots_map
    out["center_indices"] = center_indices.to(device) if isinstance(center_indices, torch.Tensor) else center_indices
    out["platform_id"] = platform_ids.to(device) if isinstance(platform_ids, torch.Tensor) else platform_ids
    out["organ_id"] = organ_ids.to(device) if isinstance(organ_ids, torch.Tensor) else organ_ids
    out["disease_id"] = disease_ids.to(device) if isinstance(disease_ids, torch.Tensor) else disease_ids
    out["batch_local_to_global"] = structure.get("batch_local_to_global", None)
    out["centers_global_indices"] = structure.get("centers_global_indices", None)
    out["neighbors_local_rows_per_center"] = structure.get("neighbors_local_rows_per_center", None)

    return out


def forward_pass(model, batch_data, config):
    """
    Model forward pass wrapper.
    
    Args:
        model: The SpatialGT model
        batch_data: Processed batch data
        config: Configuration object
    
    Returns:
        Model outputs (predictions, targets, valid_masks, hidden_states)
    """
    # Use log1p normalized values as input (standard preprocessing)
    input_values = batch_data["raw_normed_values"]

    outputs = model(
        genes=batch_data["genes"],
        input_values=input_values,
        padding_attention_mask=batch_data["padding_attention_mask"],
        batch_to_global_map=batch_data.get("batch_to_spots_map"),
        center_indices=batch_data.get("center_indices"),
        nonzero_genes=batch_data.get("nonzero_genes"),
        nonzero_expr=batch_data.get("nonzero_expr"),
        nonzero_attention_mask=batch_data.get("nonzero_attention_mask"),
        platform_ids=batch_data.get("platform_id"),
        organ_ids=batch_data.get("organ_id"),
        disease_ids=batch_data.get("disease_id"),
        batch_local_to_global=batch_data.get("batch_local_to_global"),
        centers_global_indices=batch_data.get("centers_global_indices"),
        neighbors_local_rows_per_center=batch_data.get("neighbors_local_rows_per_center"),
        is_training=model.training,
    )
    return outputs


def get_embeddings(model, batch_data, config):
    """
    Get spot embeddings from model.
    
    Args:
        model: The SpatialGT model
        batch_data: Processed batch data
        config: Configuration object
    
    Returns:
        Spot embeddings tensor
    """
    input_values = batch_data["raw_normed_values"]

    outputs = model.get_embeddings(
        genes=batch_data["genes"],
        input_values=input_values,
        padding_attention_mask=batch_data["padding_attention_mask"],
        center_indices=batch_data.get("center_indices")
    )
    return outputs


def validate_center_indices(center_indices, batch_size):
    """
    Validate center indices for correctness.
    
    Args:
        center_indices: List of center spot indices
        batch_size: Total batch size
    
    Returns:
        List of valid center indices
    """
    if not isinstance(center_indices, (list, tuple)):
        logger.warning(f"center_indices should be list or tuple, got: {type(center_indices)}")
        return list(range(batch_size))
    
    valid_indices = []
    for idx in center_indices:
        if isinstance(idx, int) and 0 <= idx < batch_size:
            valid_indices.append(idx)
        else:
            logger.warning(f"Invalid center index: {idx}, batch_size: {batch_size}")
    
    if not valid_indices:
        logger.warning("No valid center indices, using defaults")
        return list(range(batch_size))
    
    return valid_indices


def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5, 
    min_lr_ratio=0.1,
    init_lr_ratio=0.1,
    last_epoch=-1
):
    """
    Create warmup + cosine annealing learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (0.5 = half cycle)
        min_lr_ratio: Minimum learning rate ratio to initial LR
        init_lr_ratio: Starting learning rate ratio during warmup
        last_epoch: Last epoch number (for resuming)
    
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        # Ensure current_step doesn't exceed total steps
        current_step = min(current_step, num_training_steps - 1)
        
        # Warmup phase: linear increase to initial learning rate
        if current_step < num_warmup_steps:
            return init_lr_ratio + (1.0 - init_lr_ratio) * (current_step / max(1, num_warmup_steps))
        
        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, progress)
        cos_value = math.cos(math.pi * num_cycles * progress)
        decay_factor = 0.5 * (1.0 + cos_value)
        
        # Ensure learning rate doesn't go below min_lr_ratio * initial_lr
        adjusted_decay = min_lr_ratio + (1 - min_lr_ratio) * decay_factor
        
        return adjusted_decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def log_training_status(
    epoch, batch_idx, acc_step_count, curr_loss, 
    optimizer, train_loader, rank=0, writer=None,
    is_accumulation_step=False, additional_info=None, log_to_tensorboard=True,
    config=None, logger=None
):
    """
    Unified training status logging function.
    
    Args:
        epoch: Current epoch
        batch_idx: Current batch index
        acc_step_count: Accumulated step count
        curr_loss: Current loss value
        optimizer: Optimizer (for learning rate)
        train_loader: Training data loader (for progress calculation)
        rank: Current process rank
        writer: TensorBoard SummaryWriter
        is_accumulation_step: Whether this is the last step of gradient accumulation
        additional_info: Additional information dict to log
        log_to_tensorboard: Whether to log to TensorBoard
        config: Training configuration object
        logger: Logger instance
    """
    if rank != 0:
        return
    
    # Basic information
    current_lr = optimizer.param_groups[0]['lr']
    curr_mem = torch.cuda.memory_allocated(rank) / (1024 ** 3)
    peak_mem = torch.cuda.max_memory_allocated(rank) / (1024 ** 3)
    
    # Calculate progress
    progress_batch = (batch_idx + 1) / len(train_loader) * 100
    if config and hasattr(config, 'num_epochs'):
        progress_epoch = (epoch * len(train_loader) + batch_idx + 1) / (config.num_epochs * len(train_loader)) * 100
    else:
        progress_epoch = progress_batch

    # Log output
    if is_accumulation_step:
        # Detailed output for accumulation steps
        logger.info(f"\n{'='*20} Training Status [E{epoch+1} S{acc_step_count}] {'='*20}")
        if config and hasattr(config, 'num_epochs'):
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx+1}/{len(train_loader)}")
        else:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}")
            
        logger.info(f"Accumulated steps: {acc_step_count}, Learning rate: {current_lr:.7f}")
        logger.info(f"Current loss: {curr_loss:.4f}")
        
        if additional_info and 'recent_avg_loss' in additional_info:
            logger.info(f"Recent average loss: {additional_info['recent_avg_loss']:.4f}")
            
        logger.info(f"GPU memory: allocated={curr_mem:.2f}GB, peak={peak_mem:.2f}GB")
        logger.info(f"Progress: current epoch {progress_batch:.1f}%, total {progress_epoch:.1f}%")
        
        # Log additional info
        if additional_info:
            for key, value in additional_info.items():
                if key != 'recent_avg_loss' and not key.startswith('_'):
                    if isinstance(value, float):
                        logger.info(f"{key}: {value:.6f}")
                    else:
                        logger.info(f"{key}: {value}")
                        
        logger.info(f"{'='*58}\n")
    else:
        # Simple output for regular batches
        logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} ({progress_batch:.1f}%), "
                   f"Loss: {curr_loss:.4f}, LR: {current_lr:.7f}")
    
    # TensorBoard logging
    if log_to_tensorboard and writer is not None:
        if torch.isnan(torch.tensor(curr_loss)) or torch.isinf(torch.tensor(curr_loss)):
            logger.warning(f"Skipping TensorBoard write, loss is NaN/Inf: {curr_loss}")
            return
        
        global_step = acc_step_count
        writer.add_scalar('Loss/Train', curr_loss, global_step)
        writer.add_scalar('LearningRate', current_lr, global_step)
        writer.add_scalar('Memory/Allocated_GB', curr_mem, global_step)
        writer.add_scalar('Memory/Peak_GB', peak_mem, global_step)
        writer.add_scalar('Progress/Epoch', epoch + 1, global_step)
        writer.add_scalar('Progress/Epoch_Percent', progress_epoch, global_step)
        
        # Additional metrics
        if additional_info:
            for key, value in additional_info.items():
                if key.startswith('_') or not isinstance(value, (int, float, torch.Tensor)) or isinstance(value, bool):
                    continue
                
                if torch.is_tensor(value):
                    value = value.item()

                if 'loss' in key.lower():
                    clean_key = ''.join(word.capitalize() for word in key.split('_'))
                    writer.add_scalar(f'Loss_Components/Train/{clean_key}', value, global_step)
                elif 'acc' in key.lower():
                    clean_key = ''.join(word.capitalize() for word in key.split('_'))
                    writer.add_scalar(f'Accuracy/Train/{clean_key}', value, global_step)
                elif key not in ['learning_rate', 'lr_change']:
                    writer.add_scalar(f'Metrics/{key}', value, global_step)

        writer.flush()


def save_checkpoint(
    epoch, acc_step_count, model, optimizer, scheduler, curr_loss,
    args, rank=0, is_best=False, is_temp=True, extra_data=None
):
    """
    Unified checkpoint saving function.
    
    Args:
        epoch: Current epoch
        acc_step_count: Accumulated step count
        model: Model to save
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        curr_loss: Current loss value
        args: Arguments dict, must contain 'output_dir'
        rank: Current process rank
        is_best: Whether this is the best model so far
        is_temp: Whether this is a temporary checkpoint
        extra_data: Additional data to save
    """
    if rank != 0:
        return
        
    # Ensure output directory exists
    output_dir = args.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
        
    # Determine save path
    if is_best:
        checkpoint_path = os.path.join(output_dir, f'best_model_e{epoch+1}_s{acc_step_count}.pth')
    elif is_temp:
        checkpoint_path = os.path.join(output_dir, f'temp_checkpoint_e{epoch+1}_s{acc_step_count}.pth')
    else:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'acc_step': acc_step_count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': curr_loss,
        'batch_idx': extra_data.get('batch_idx', 0) if extra_data else 0,
        'completed_steps': extra_data.get('completed_steps', 0) if extra_data else 0,
    }
    
    # Add scheduler state if exists
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
    # Add extra data
    if extra_data:
        checkpoint_data.update(extra_data)
        
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"{'Best' if is_best else 'Temp' if is_temp else 'Epoch'} checkpoint saved: {checkpoint_path}")
    
    # Clean up old temporary checkpoints
    if is_temp:
        try:
            checkpoint_files = sorted(glob.glob(os.path.join(output_dir, f'temp_checkpoint_e{epoch+1}_*.pth')))
            if len(checkpoint_files) > 3:  # Keep latest 3
                for old_ckpt in checkpoint_files[:-3]:
                    os.remove(old_ckpt)
                    logger.info(f"Deleted old checkpoint: {old_ckpt}")
        except Exception as e:
            logger.warning(f"Error cleaning up old checkpoints: {e}")


def setup_tensorboard(log_dir="runs"):
    """
    Set up TensorBoard logger.
    
    Args:
        log_dir: TensorBoard log directory, defaults to "runs"
    
    Returns:
        SummaryWriter object or None if initialization fails
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    abs_log_dir = str(log_path.absolute())
    logger.info(f"Initializing TensorBoard SummaryWriter, log directory: {abs_log_dir}")
    
    try:
        # Test write access
        test_file = log_path / "test_write_access.txt"
        with open(test_file, 'w') as f:
            f.write("Test file to verify write access")
        test_file.unlink()
        
        writer = SummaryWriter(log_dir=abs_log_dir)
        
        # Test write a simple scalar
        writer.add_scalar("System/initialization_test", 1.0, 0)
        writer.flush()
        
        logger.info(f"TensorBoard SummaryWriter initialized successfully")
        return writer
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoard: {e}")
        logger.error(f"Please check directory permissions: {abs_log_dir}")
        return None
