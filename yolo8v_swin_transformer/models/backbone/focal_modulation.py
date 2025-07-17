"""
Focal Modulation Layer for enhanced attention mechanism
Paper: "Focal Modulation Networks" https://arxiv.org/abs/2203.11009
File: models/backbone/focal_modulation.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class FocalModulation(nn.Module):
    """
    Focal Modulation Layer - Enhanced attention mechanism that replaces traditional self-attention
    
    The key idea is to use a hierarchical contextualization approach where we:
    1. Extract query and context from input
    2. Apply hierarchical focal convolutions with different kernel sizes
    3. Aggregate contexts with learnable gating mechanism
    4. Apply focal modulation to generate final output
    
    Args:
        dim: Number of input channels
        focal_window: Base window size for focal attention  
        focal_level: Number of focal levels (hierarchy depth)
        focal_factor: Factor to increase kernel size at each level
        bias: Whether to use bias in linear layers
        proj_drop: Projection dropout rate
        use_postln_in_modulation: Whether to use post layer norm
        normalize_modulator: Whether to normalize modulator values
    """
    
    def __init__(
        self, 
        dim: int,
        focal_window: int = 7,
        focal_level: int = 2, 
        focal_factor: int = 2,
        bias: bool = True,
        proj_drop: float = 0.,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        
        # Pre-processing projection: input -> query, context, gates
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        
        # Modulation projection
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Focal convolution layers for hierarchical context aggregation
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim, dim, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=kernel_size // 2, 
                        groups=dim,  # Depthwise convolution for efficiency
                        bias=False
                    ),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
            
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Modulation
        
        Args:
            x: Input tensor in format (B, H, W, C) - SWIN format
        
        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Pre-processing projection
        # Split into query, context, and gates
        x_proj = self.f(x)  # (B, H, W, 2*C + focal_level + 1)
        q, ctx, gates = torch.split(
            x_proj, 
            (C, C, self.focal_level + 1), 
            dim=-1
        )
        
        # Apply softmax to gates for proper weighting
        gates = F.softmax(gates, dim=-1)
        
        # Hierarchical context aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            # Convert to YOLO format (B, C, H, W) for convolution
            ctx_conv = ctx.permute(0, 3, 1, 2).contiguous()
            
            # Apply focal convolution at level l
            ctx_conv = self.focal_layers[l](ctx_conv)
            
            # Convert back to SWIN format (B, H, W, C)
            ctx_level = ctx_conv.permute(0, 2, 3, 1).contiguous()
            
            # Ensure spatial dimensions match before adding
            if ctx_level.shape[1:3] != (H, W):
                ctx_level = F.interpolate(
                    ctx_level.permute(0, 3, 1, 2), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Apply gating for this level
            ctx_all = ctx_all + ctx_level * gates[:, :, :, l:l+1]
        
        # Global context (average pooling across spatial dimensions)
        ctx_global = ctx.mean(dim=[1, 2], keepdim=True)  # (B, 1, 1, C)
        ctx_global = self.act(ctx_global)
        
        # Add global context with gating
        ctx_all = ctx_all + ctx_global * gates[:, :, :, self.focal_level:]
        
        # Normalize modulator if requested
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        
        # Generate modulator through 1x1 convolution
        # Convert to YOLO format for convolution
        modulator = ctx_all.permute(0, 3, 1, 2).contiguous()
        modulator = self.h(modulator)
        # Convert back to SWIN format
        modulator = modulator.permute(0, 2, 3, 1).contiguous()
        
        # Focal modulation: element-wise multiplication
        x_out = q * modulator
        
        # Post layer norm if requested
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # Output projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out

    def extra_repr(self) -> str:
        """String representation of module parameters"""
        return (f'dim={self.dim}, focal_window={self.focal_window}, '
                f'focal_level={self.focal_level}, focal_factor={self.focal_factor}')

class FocalModulationYOLO(nn.Module):
    """
    Focal Modulation adapted for YOLO tensor format (B, C, H, W)
    This version works directly with YOLO format without conversion
    """
    
    def __init__(
        self, 
        dim: int,
        focal_window: int = 7,
        focal_level: int = 2, 
        focal_factor: int = 2,
        bias: bool = True,
        proj_drop: float = 0.,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        
        # Pre-processing convolution for YOLO format
        self.f = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        
        # Modulation projection
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        
        self.act = nn.GELU()
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_drop = nn.Dropout2d(proj_drop)
        
        # Focal convolution layers
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim, dim, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=kernel_size // 2, 
                        groups=dim,
                        bias=False
                    ),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
            
        if self.use_postln_in_modulation:
            # For YOLO format, use GroupNorm instead of LayerNorm
            self.ln = nn.GroupNorm(1, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for YOLO format input
        
        Args:
            x: Input tensor in format (B, C, H, W) - YOLO format
        
        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Pre-processing projection
        x_proj = self.f(x)  # (B, 2*C + focal_level + 1, H, W)
        q, ctx, gates = torch.split(
            x_proj, 
            (C, C, self.focal_level + 1), 
            dim=1
        )
        
        # Apply softmax to gates
        gates = F.softmax(gates, dim=1)
        
        # Hierarchical context aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            # Apply focal convolution at level l
            ctx_level = self.focal_layers[l](ctx)
            
            # Apply gating for this level
            ctx_all = ctx_all + ctx_level * gates[:, l:l+1, :, :]
        
        # Global context (global average pooling)
        ctx_global = F.adaptive_avg_pool2d(ctx, 1)  # (B, C, 1, 1)
        ctx_global = self.act(ctx_global)
        
        # Add global context with gating
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:, :, :]
        
        # Normalize modulator if requested
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        
        # Generate modulator
        modulator = self.h(ctx_all)
        
        # Focal modulation
        x_out = q * modulator
        
        # Post normalization if requested
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # Output projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out

    def extra_repr(self) -> str:
        """String representation of module parameters"""
        return (f'dim={self.dim}, focal_window={self.focal_window}, '
                f'focal_level={self.focal_level}, focal_factor={self.focal_factor}')


# Factory function to create appropriate focal modulation layer
def create_focal_modulation(
    dim: int,
    input_format: str = "swin",  # "swin" or "yolo"
    focal_window: int = 7,
    focal_level: int = 2,
    focal_factor: int = 2,
    bias: bool = True,
    proj_drop: float = 0.,
    use_postln_in_modulation: bool = False,
    normalize_modulator: bool = False
) -> nn.Module:
    """
    Factory function to create focal modulation layer based on input format
    
    Args:
        dim: Number of input channels
        input_format: "swin" for (B, H, W, C) or "yolo" for (B, C, H, W)
        Other args: Same as FocalModulation
    
    Returns:
        Appropriate focal modulation layer
    """
    if input_format.lower() == "swin":
        return FocalModulation(
            dim=dim,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            bias=bias,
            proj_drop=proj_drop,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator
        )
    elif input_format.lower() == "yolo":
        return FocalModulationYOLO(
            dim=dim,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            bias=bias,
            proj_drop=proj_drop,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator
        )
    else:
        raise ValueError(f"Unsupported input_format: {input_format}. Use 'swin' or 'yolo'")


# Testing utilities
if __name__ == "__main__":
    # Test focal modulation layers
    
    print("Testing FocalModulation (SWIN format)...")
    focal_swin = FocalModulation(dim=96, focal_level=2)
    x_swin = torch.randn(2, 56, 56, 96)  # (B, H, W, C)
    out_swin = focal_swin(x_swin)
    print(f"Input shape: {x_swin.shape}, Output shape: {out_swin.shape}")
    
    print("\nTesting FocalModulationYOLO (YOLO format)...")
    focal_yolo = FocalModulationYOLO(dim=96, focal_level=2)
    x_yolo = torch.randn(2, 96, 56, 56)  # (B, C, H, W)
    out_yolo = focal_yolo(x_yolo)
    print(f"Input shape: {x_yolo.shape}, Output shape: {out_yolo.shape}")
    
    print("\nTesting factory function...")
    focal_factory = create_focal_modulation(dim=96, input_format="swin")
    print(f"Created focal modulation: {type(focal_factory).__name__}")
    
    print("\nAll tests passed!")