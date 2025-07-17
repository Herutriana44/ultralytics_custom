"""
Utility functions for tensor operations in YOLO-SWIN integration
File: utils/tensor_utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition tensor into windows for SWIN Transformer
    
    Args:
        x: Input tensor (B, H, W, C)
        window_size: Size of window
        
    Returns:
        Windows tensor (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # Check if dimensions are divisible by window_size
    if H % window_size != 0 or W % window_size != 0:
        # Pad the tensor to make it divisible
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H, W = x.shape[1], x.shape[2]
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, original_H: int = None, original_W: int = None) -> torch.Tensor:
    """
    Reverse window partition
    
    Args:
        windows: Windows tensor (num_windows*B, window_size, window_size, C)
        window_size: Size of window
        H: Height of image (potentially padded)
        W: Width of image (potentially padded)
        original_H: Original height before padding
        original_W: Original width before padding
        
    Returns:
        Tensor (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    
    # Remove padding if original dimensions were provided
    if original_H is not None and original_W is not None:
        x = x[:, :original_H, :original_W, :]
    
    return x

def yolo_to_swin_format(x: torch.Tensor) -> torch.Tensor:
    """
    Convert YOLO tensor format (B, C, H, W) to SWIN format (B, H, W, C)
    
    Args:
        x: Input tensor in YOLO format (B, C, H, W)
        
    Returns:
        Tensor in SWIN format (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1).contiguous()

def swin_to_yolo_format(x: torch.Tensor) -> torch.Tensor:
    """
    Convert SWIN tensor format (B, H, W, C) to YOLO format (B, C, H, W)
    
    Args:
        x: Input tensor in SWIN format (B, H, W, C)
        
    Returns:
        Tensor in YOLO format (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2).contiguous()

def make_divisible(v: int, divisor: int = 8) -> int:
    """
    Make channel number divisible by divisor (for efficient computation)
    
    Args:
        v: Value to make divisible
        divisor: Divisor value
        
    Returns:
        New value divisible by divisor
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    
    Args:
        x: Input tensor
        drop_prob: Drop probability
        training: Whether in training mode
        
    Returns:
        Output tensor with drop path applied
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., 
                  a: float = -2., b: float = 2.) -> torch.Tensor:
    """
    Truncated normal initialization
    
    Args:
        tensor: Tensor to initialize
        mean: Mean of the distribution
        std: Standard deviation
        a: Lower bound
        b: Upper bound
        
    Returns:
        Initialized tensor
    """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def calculate_feature_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate feature statistics for quality monitoring
    
    Args:
        x: Input feature tensor (B, C, H, W) or (B, H, W, C)
        
    Returns:
        Tuple of (mean, variance) tensors
    """
    # Flatten spatial dimensions
    if x.dim() == 4:
        if x.shape[1] < x.shape[-1]:  # YOLO format (B, C, H, W)
            x_flat = x.view(x.shape[0], x.shape[1], -1)  # (B, C, H*W)
        else:  # SWIN format (B, H, W, C)
            x_flat = x.view(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)  # (B, C, H*W)
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    
    mean = torch.mean(x_flat, dim=-1)  # (B, C)
    var = torch.var(x_flat, dim=-1)   # (B, C)
    
    return mean, var

def ensure_tensor_compatibility(x: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Ensure tensor compatibility with target shape through interpolation or padding
    
    Args:
        x: Input tensor
        target_shape: Target spatial shape (H, W)
        
    Returns:
        Resized tensor
    """
    if len(target_shape) != 2:
        raise ValueError("target_shape must be (H, W)")
    
    target_h, target_w = target_shape
    
    if x.dim() == 4:  # (B, C, H, W) or (B, H, W, C)
        if x.shape[1] < x.shape[-1]:  # YOLO format (B, C, H, W)
            current_h, current_w = x.shape[2], x.shape[3]
            if (current_h, current_w) != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        else:  # SWIN format (B, H, W, C)
            current_h, current_w = x.shape[1], x.shape[2]
            if (current_h, current_w) != (target_h, target_w):
                # Convert to YOLO format, interpolate, then convert back
                x = swin_to_yolo_format(x)
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
                x = yolo_to_swin_format(x)
    
    return x

def create_attention_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Create attention mask for shifted window attention
    
    Args:
        H: Height
        W: Width  
        window_size: Window size
        shift_size: Shift size
        device: Device to create mask on
        
    Returns:
        Attention mask or None if no masking needed
    """
    if shift_size > 0:
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -window_size),
                   slice(-window_size, -shift_size),
                   slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                   slice(-window_size, -shift_size),  
                   slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    else:
        return None