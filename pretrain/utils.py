"""
Utility Functions for SpatialGT

This module provides utility functions for logging, reproducibility,
and distributed training setup.
"""

import os
import sys
import time
import json
import random
import logging
from datetime import timedelta
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from Config import Config


def setup_logger():
    """
    Set up the main logger for SpatialGT.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('stGPT')
    logger.setLevel(logging.INFO)
    
    # Check if handlers already added
    if not logger.handlers:
        # Console handler with stdout
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Create module-level logger
logger = setup_logger()


def seed_everything(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def cleanup():
    """Clean up distributed environment."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def process_id_tensor(id_data, device):
    """
    Process ID data and ensure correct format.
    
    Args:
        id_data: ID data (can be int, list, or tensor)
        device: Target device
    
    Returns:
        Tensor on the target device
    """
    if id_data is None:
        return torch.tensor([0], device=device)
    
    if isinstance(id_data, (int, float)):
        return torch.tensor([int(id_data)], device=device)
    
    if isinstance(id_data, (list, tuple)):
        try:
            return torch.tensor([int(x) for x in id_data], device=device)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert ID data: {id_data}, using default")
            return torch.tensor([0], device=device)
    
    if isinstance(id_data, torch.Tensor):
        return id_data.to(device)
    
    logger.warning(f"Unknown ID data type: {type(id_data)}, using default")
    return torch.tensor([0], device=device)


def move_to_device(data, device):
    """
    Move data to specified device.
    
    Args:
        data: Data to move (can be tensor, list, or None)
        device: Target device
    
    Returns:
        Data on the target device
    """
    if data is None:
        return None
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    
    if isinstance(data, (list, tuple)):
        try:
            return torch.tensor(data, device=device)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert data to tensor: {type(data)}")
            return data
    
    return data


def log_memory(name, rank=0):  
    """
    Log current memory usage.
    
    Args:
        name: Name/label for the log entry
        rank: GPU rank
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"GPU {rank} Memory [{name}]: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={peak:.2f}GB")


def get_memory_logger():
    """Create or get dedicated memory tracking logger."""
    mem_logger = logging.getLogger('memory_tracker')
    
    if mem_logger.handlers:
        return mem_logger
        
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    mem_logger.addHandler(handler)
    mem_logger.setLevel(logging.INFO)
    
    return mem_logger


def log_model_size(model):
    """
    Analyze and log model size.
    
    Args:
        model: PyTorch model
    """
    mem_logger = get_memory_logger()
    
    # Calculate total parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model memory size
    model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    mem_logger.info(f"Total trainable parameters: {param_count:,}")
    mem_logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    # Analyze size by module
    module_sizes = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in module_sizes:
            module_sizes[module_name] = 0
        module_sizes[module_name] += param.nelement() * param.element_size()
    
    # Output major module sizes
    for module_name, size in sorted(module_sizes.items(), key=lambda x: x[1], reverse=True):
        mem_logger.info(f"  - {module_name}: {size/(1024*1024):.2f} MB ({size/model_size_bytes*100:.1f}%)")


def monitor_resources(rank=0):
    """
    Monitor system resource usage.
    
    Args:
        rank: Current process rank
    
    Returns:
        dict: Resource usage information
    """
    if rank != 0:
        return {}
        
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed, cannot monitor CPU and memory")
        return {}
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    
    # GPU usage
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            
            gpu_util = "unknown"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except ImportError:
                try:
                    import GPUtil
                    gpu_list = GPUtil.getGPUs()
                    if i < len(gpu_list):
                        gpu_util = gpu_list[i].load * 100
                except ImportError:
                    pass
            
            gpu_info[f"gpu_{i}"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "utilization": gpu_util
            }
    
    logger.info(f"System resources: CPU={cpu_percent}%, Memory={memory_percent}% ({memory_used_gb:.1f}/{memory_total_gb:.1f}GB)")
    for gpu_id, gpu_data in gpu_info.items():
        logger.info(f"GPU {gpu_id}: {gpu_data['allocated_gb']:.1f}GB allocated, "
                   f"{gpu_data['reserved_gb']:.1f}GB reserved, peak={gpu_data['max_allocated_gb']:.1f}GB, "
                   f"utilization={gpu_data['utilization']}%")
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_used_gb": memory_used_gb,
        "memory_total_gb": memory_total_gb,
        "gpu_info": gpu_info
    }


def setup(rank, world_size, args_dict):
    """
    Initialize distributed training environment.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        args_dict: Arguments dict containing master_addr and master_port
    """
    master_addr = args_dict.get('master_addr', 'localhost')
    master_port = args_dict.get('master_port', '12355')
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    logger.info(f"Process {rank} initializing distributed environment: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    
    # Try NCCL first (better performance), fall back to Gloo
    backend = "nccl"
    
    try:
        logger.info(f"Process {rank} trying NCCL backend...")
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=2))
        logger.info(f"Process {rank} successfully initialized NCCL backend")
    except Exception as e:
        logger.warning(f"Process {rank} NCCL initialization failed: {e}")
        logger.info(f"Process {rank} falling back to Gloo backend...")
        backend = "gloo"
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=5))
            logger.info(f"Process {rank} successfully initialized Gloo backend")
        except Exception as e2:
            logger.error(f"Process {rank} Gloo backend also failed: {e2}")
            raise RuntimeError(f"All distributed backends failed: NCCL({e}), Gloo({e2})")
    
    # Set current device
    torch.cuda.set_device(rank)
    
    # Print device info
    device_name = torch.cuda.get_device_name(rank)
    logger.info(f"Process {rank} using GPU (logical ID: {rank}): {device_name}")


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    """
    Get sequence length from tensor.
    
    Args:
        src: Source tensor
        batch_first: Whether batch dimension is first
    
    Returns:
        Sequence length or None for nested tensors
    """
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # Unbatched: S, E
            return src_size[0]
        else:
            # Batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]
