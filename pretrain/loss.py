"""
Loss Functions for SpatialGT

This module provides loss functions for training the SpatialGT model,
including reconstruction loss with various options.
"""

import os
import torch
import torch.nn.functional as F


def compute_reconstruction_loss(predictions, targets, valid_masks, loss_type='mse'):
    """
    Compute reconstruction loss (only at valid gene positions).
    
    Args:
        predictions: [num_centers, num_genes] - Predicted values
        targets: [num_centers, num_genes] - Ground truth values
        valid_masks: [num_centers, num_genes] - True indicates positions to compute loss
        loss_type: 'mse', 'huber', 'weighted_mse', or 'combined'
    
    Returns:
        loss: Scalar loss value
    """
    # Only compute loss at valid positions
    pred_valid = predictions[valid_masks]  # [total_valid_genes]
    target_valid = targets[valid_masks]  # [total_valid_genes]
    
    # Optional debug output (only prints once when DEBUG_LOSS=1)
    if os.getenv('DEBUG_LOSS', '0') == '1':
        if not hasattr(compute_reconstruction_loss, "_printed_once"):
            print("[DEBUG_LOSS] pred_valid shape:", tuple(pred_valid.shape))
            print("[DEBUG_LOSS] target_valid shape:", tuple(target_valid.shape))
            compute_reconstruction_loss._printed_once = True

    if loss_type == 'mse':
        return F.mse_loss(pred_valid, target_valid)
    
    elif loss_type == 'huber':
        return F.huber_loss(pred_valid, target_valid, delta=1.0)
    
    elif loss_type == 'weighted_mse':
        # Give higher weight to highly expressed genes
        weights = torch.log1p(target_valid)
        return (weights * (pred_valid - target_valid) ** 2).mean()
    
    elif loss_type == 'combined':
        # MSE + Pearson correlation
        mse = F.mse_loss(pred_valid, target_valid)
        
        # Pearson correlation (computed per center spot, then averaged)
        num_centers = predictions.shape[0]
        pearson_losses = []
        
        for i in range(num_centers):
            mask_i = valid_masks[i]
            pred_i = predictions[i][mask_i]
            target_i = targets[i][mask_i]
            
            # Skip if too few genes
            if len(pred_i) < 2:
                continue
            
            # Center the values
            pred_centered = pred_i - pred_i.mean()
            target_centered = target_i - target_i.mean()
            
            # Pearson correlation
            numerator = (pred_centered * target_centered).sum()
            pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
            target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
            pearson = numerator / (pred_std * target_std)
            
            pearson_losses.append(1 - pearson)
        
        if pearson_losses:
            pearson_loss = torch.stack(pearson_losses).mean()
        else:
            pearson_loss = torch.tensor(0.0, device=predictions.device)
        
        return mse + 0.1 * pearson_loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
