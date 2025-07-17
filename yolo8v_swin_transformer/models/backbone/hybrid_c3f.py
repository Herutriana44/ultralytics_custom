"""
Hybrid C3F Module - Integration of C3F with SWIN Transformer and Focal Modulation
This module replaces the original C3F in YOLO v8 with enhanced SWIN-based processing
File: models/backbone/hybrid_c3f.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Import from our custom modules
try:
    from ...utils.tensor_utils import (
        yolo_to_swin_format, swin_to_yolo_format, 
        make_divisible, ensure_tensor_compatibility
    )
    from .focal_modulation import FocalModulation, create_focal_modulation
    from .swin_transformer import SwinTransformerBlock
except ImportError:
    # Fallback imports for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.tensor_utils import (
        yolo_to_swin_format, swin_to_yolo_format, 
        make_divisible, ensure_tensor_compatibility
    )
    from models.backbone.focal_modulation import FocalModulation, create_focal_modulation
    from models.backbone.swin_transformer import SwinTransformerBlock

class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def autopad(k, p=None, d=1):
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

class Bottleneck(nn.Module):
    """Standard bottleneck for C3F"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3F(nn.Module):
    """
    Original C3F module from YOLO v8
    Cross Stage Partial Network with 3 convolutions and Focus
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SWINAdapter(nn.Module):
    """
    Adapter layer to bridge YOLO and SWIN formats
    Handles tensor format conversion and dimension matching
    """
    def __init__(self, c1, c2, img_size, patch_size=1, use_conv_proj=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.img_size = img_size if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        self.patch_size = patch_size
        self.use_conv_proj = use_conv_proj
        
        # Ensure output channels are compatible
        self.c2 = make_divisible(c2, 8)
        
        if use_conv_proj:
            # Use convolution for projection (more flexible)
            self.proj = nn.Conv2d(c1, self.c2, kernel_size=patch_size, stride=patch_size, bias=False)
        else:
            # Use linear projection
            self.proj = nn.Linear(c1, self.c2, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(self.c2)
        
        # Calculate output resolution
        self.output_resolution = (
            self.img_size[0] // patch_size,
            self.img_size[1] // patch_size
        )
    
    def forward(self, x):
        """
        Convert YOLO format to SWIN format with projection
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Tensor in SWIN format (B, H*W, C) and output resolution
        """
        B, C, H, W = x.shape
        
        if self.use_conv_proj:
            # Apply convolution projection
            x = self.proj(x)  # (B, C2, H//patch_size, W//patch_size)
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C2)
        else:
            # Convert to SWIN format first, then project
            x = yolo_to_swin_format(x)  # (B, H, W, C)
            x = x.view(B, H * W, C)  # (B, H*W, C)
            x = self.proj(x)  # (B, H*W, C2)
        
        # Apply normalization
        x = self.norm(x)
        
        return x, self.output_resolution

class SWINReverseAdapter(nn.Module):
    """
    Reverse adapter to convert SWIN output back to YOLO format
    """
    def __init__(self, c1, c2, output_size, use_conv_proj=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.output_size = output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size)
        self.use_conv_proj = use_conv_proj
        
        # Ensure input channels are compatible
        self.c1 = make_divisible(c1, 8)
        
        if use_conv_proj:
            # Use 1x1 convolution for reverse projection
            self.proj = nn.Conv2d(self.c1, c2, kernel_size=1, bias=False)
        else:
            # Use linear projection
            self.proj = nn.Linear(self.c1, c2, bias=False)
        
        # Normalization
        self.norm = nn.BatchNorm2d(c2) if use_conv_proj else nn.LayerNorm(c2)
    
    def forward(self, x, original_size):
        """
        Convert SWIN format back to YOLO format
        Args:
            x: Input tensor (B, H*W, C)
            original_size: Original spatial size (H, W)
        Returns:
            Tensor in YOLO format (B, C, H, W)
        """
        B, L, C = x.shape
        H, W = original_size
        
        if self.use_conv_proj:
            # Convert to spatial format
            x = x.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
            x = self.proj(x)  # (B, C2, H, W)
            x = self.norm(x)
        else:
            # Apply linear projection first
            x = self.proj(x)  # (B, H*W, C2)
            # Convert back to YOLO format
            x = x.view(B, H, W, -1)  # (B, H, W, C2)
            x = self.norm(x)
            x = swin_to_yolo_format(x)  # (B, C2, H, W)
        
        return x

class HybridC3F(nn.Module):
    """
    Hybrid C3F module that integrates SWIN Transformer with original C3F structure
    
    This module replaces the original C3F in YOLO v8 with enhanced processing:
    1. Split input through two paths: C3F path and SWIN path
    2. SWIN path uses SWIN Transformer blocks with Focal Modulation
    3. Fuse both paths for final output
    
    Args:
        c1: Input channels
        c2: Output channels  
        n: Number of bottleneck layers in C3F path
        swin_depth: Number of SWIN transformer blocks
        num_heads: Number of attention heads for SWIN
        window_size: Window size for SWIN attention
        img_size: Input image size for SWIN processing
        shortcut: Whether to use shortcut in bottlenecks
        g: Groups for convolution
        e: Expansion ratio
        use_focal_modulation: Whether to use focal modulation in SWIN blocks
    """
    
    def __init__(self, c1, c2, n=1, swin_depth=2, num_heads=None, window_size=7, 
                 img_size=56, shortcut=True, g=1, e=0.5, use_focal_modulation=True):
        super().__init__()
        
        # Calculate hidden channels
        c_ = int(c2 * e)  # hidden channels for C3F path
        swin_c = make_divisible(c_, 8)  # SWIN channels (ensure divisible by 8)
        
        # Auto-calculate num_heads if not provided
        if num_heads is None:
            num_heads = max(1, swin_c // 32)  # Rule of thumb: 32 channels per head
        
        self.c1 = c1
        self.c2 = c2
        self.swin_depth = swin_depth
        self.use_focal_modulation = use_focal_modulation
        
        # C3F path (preserve original YOLO processing)
        self.cv1 = Conv(c1, c_, 1, 1)  # Input projection for C3F
        self.cv2 = Conv(c1, c_, 1, 1)  # Shortcut projection for C3F
        
        # C3F bottleneck layers
        self.m = nn.Sequential(*(
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)
        ))
        
        # SWIN path
        self.swin_input_proj = SWINAdapter(c1, swin_c, img_size)
        
        # SWIN Transformer blocks
        img_size_tuple = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=swin_c,
                input_resolution=img_size_tuple,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                use_focal_modulation=use_focal_modulation
            ) for i in range(swin_depth)
        ])
        
        # SWIN output projection
        self.swin_output_proj = SWINReverseAdapter(swin_c, c_, img_size_tuple)
        
        # Feature fusion
        self.fusion_conv = Conv(3 * c_, c2, 1, 1)  # Fuse C3F main + shortcut + SWIN
        
        # Optional attention-based fusion
        self.use_attention_fusion = True
        if self.use_attention_fusion:
            self.fusion_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(3 * c_, (3 * c_) // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d((3 * c_) // 4, 3, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        """
        Forward pass of Hybrid C3F
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C2, H, W)
        """
        B, C, H, W = x.shape
        
        # C3F path
        c3f_main = self.m(self.cv1(x))  # Main path through bottlenecks
        c3f_shortcut = self.cv2(x)      # Shortcut path
        
        # SWIN path
        swin_input, swin_resolution = self.swin_input_proj(x)
        
        # Apply SWIN blocks with updated resolution
        swin_features = swin_input
        for block in self.swin_blocks:
            # Update block's input resolution to match actual input
            actual_L = swin_features.shape[1]
            import math
            sqrt_L = int(math.sqrt(actual_L))
            if sqrt_L * sqrt_L == actual_L:
                block.input_resolution = (sqrt_L, sqrt_L)
            else:
                block.input_resolution = (H, W)
            swin_features = block(swin_features)
        
        # Convert SWIN output back to YOLO format
        swin_output = self.swin_output_proj(swin_features, (H, W))
        
        # Ensure all tensors have the same spatial size
        if swin_output.shape[2:] != (H, W):
            swin_output = F.interpolate(
                swin_output, size=(H, W), 
                mode='bilinear', align_corners=False
            )
        
        # Feature fusion
        if self.use_attention_fusion:
            # Concatenate features
            fused_features = torch.cat([c3f_main, c3f_shortcut, swin_output], dim=1)
            
            # Attention-based weighting
            attention_weights = self.fusion_attention(fused_features)  # (B, 3, 1, 1)
            
            # Apply attention weights (no need to unsqueeze)
            weighted_features = torch.cat([
                c3f_main * attention_weights[:, 0:1],
                c3f_shortcut * attention_weights[:, 1:2], 
                swin_output * attention_weights[:, 2:3]
            ], dim=1)
            
            # Final fusion
            output = self.fusion_conv(weighted_features)
        else:
            # Simple concatenation and fusion
            fused_features = torch.cat([c3f_main, c3f_shortcut, swin_output], dim=1)
            output = self.fusion_conv(fused_features)
        
        return output
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization/analysis
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary of feature maps
        """
        B, C, H, W = x.shape
        
        # C3F path
        c3f_main = self.m(self.cv1(x))
        c3f_shortcut = self.cv2(x)
        
        # SWIN path
        swin_input, swin_resolution = self.swin_input_proj(x)
        swin_features = swin_input
        swin_intermediate = []
        
        for i, block in enumerate(self.swin_blocks):
            swin_features = block(swin_features)
            swin_intermediate.append(swin_features.clone())
        
        swin_output = self.swin_output_proj(swin_features, (H, W))
        
        return {
            'c3f_main': c3f_main,
            'c3f_shortcut': c3f_shortcut,
            'swin_input': swin_input,
            'swin_intermediate': swin_intermediate,
            'swin_output': swin_output,
            'swin_resolution': swin_resolution
        }

class AdaptiveHybridC3F(HybridC3F):
    """
    Adaptive version of HybridC3F that can handle different input sizes
    Automatically adjusts SWIN parameters based on input resolution
    """
    
    def __init__(self, c1, c2, n=1, swin_depth=2, num_heads=None, 
                 base_window_size=7, shortcut=True, g=1, e=0.5, 
                 use_focal_modulation=True, **kwargs):
        
        # Extract only valid parameters for parent init
        valid_params = {
            'c1': c1, 'c2': c2, 'n': n, 'swin_depth': swin_depth, 
            'num_heads': num_heads, 'window_size': base_window_size,
            'img_size': 56, 'shortcut': shortcut, 'g': g, 'e': e, 
            'use_focal_modulation': use_focal_modulation
        }
        
        # Initialize with base parameters
        super().__init__(**valid_params)
        
        self.base_window_size = base_window_size
        self.last_input_size = None
        self.cached_swin_blocks = {}
    
    def _get_adaptive_swin_blocks(self, H, W):
        """
        Get SWIN blocks adapted to current input size
        """
        input_key = f"{H}x{W}"
        
        if input_key not in self.cached_swin_blocks:
            # Calculate adaptive window size that fits the input dimensions
            adaptive_window_size = min(self.base_window_size, min(H, W))
            
            # Ensure window size divides evenly into H and W
            while adaptive_window_size > 1:
                if H % adaptive_window_size == 0 and W % adaptive_window_size == 0:
                    break
                adaptive_window_size -= 1
            
            # Minimum window size of 1
            if adaptive_window_size < 1:
                adaptive_window_size = 1
            
            # Create new SWIN blocks for this resolution
            swin_c = self.swin_input_proj.c2
            num_heads = max(1, swin_c // 32)
            
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=swin_c,
                    input_resolution=(H, W),
                    num_heads=num_heads,
                    window_size=adaptive_window_size,
                    shift_size=0 if (i % 2 == 0) else adaptive_window_size // 2,
                    use_focal_modulation=self.use_focal_modulation
                ) for i in range(self.swin_depth)
            ])
            
            # Move to same device as main model
            blocks = blocks.to(next(self.parameters()).device)
            self.cached_swin_blocks[input_key] = blocks
        
        return self.cached_swin_blocks[input_key]
    
    def forward(self, x):
        """
        Adaptive forward pass
        """
        B, C, H, W = x.shape
        
        # Use adaptive SWIN blocks
        adaptive_blocks = self._get_adaptive_swin_blocks(H, W)
        
        # C3F path (unchanged)
        c3f_main = self.m(self.cv1(x))
        c3f_shortcut = self.cv2(x)
        
        # SWIN path dengan adaptive input handling
        actual_H, actual_W = H, W
        self.swin_input_proj.img_size = (actual_H, actual_W)
        self.swin_input_proj.output_resolution = (actual_H, actual_W)
        swin_input, _ = self.swin_input_proj(x)
        
        # Apply adaptive SWIN blocks
        swin_features = swin_input
        for block in adaptive_blocks:
            # Update block's input resolution to match actual input
            block.input_resolution = (actual_H, actual_W)
            swin_features = block(swin_features)
        
        # Convert back
        swin_output = self.swin_output_proj(swin_features, (H, W))
        
        # Ensure spatial compatibility
        if swin_output.shape[2:] != (H, W):
            swin_output = F.interpolate(
                swin_output, size=(H, W), 
                mode='bilinear', align_corners=False
            )
        
        # Feature fusion (same as parent)
        if self.use_attention_fusion:
            fused_features = torch.cat([c3f_main, c3f_shortcut, swin_output], dim=1)
            attention_weights = self.fusion_attention(fused_features)
            
            weighted_features = torch.cat([
                c3f_main * attention_weights[:, 0:1],
                c3f_shortcut * attention_weights[:, 1:2], 
                swin_output * attention_weights[:, 2:3]
            ], dim=1)
            
            output = self.fusion_conv(weighted_features)
        else:
            fused_features = torch.cat([c3f_main, c3f_shortcut, swin_output], dim=1)
            output = self.fusion_conv(fused_features)
        
        return output

# Factory functions
def create_hybrid_c3f(
    c1: int, 
    c2: int, 
    variant: str = "standard",
    n: int = 1,
    swin_depth: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different variants of Hybrid C3F
    
    Args:
        c1: Input channels
        c2: Output channels
        variant: "standard", "adaptive", or "lightweight"
        n: Number of bottleneck layers
        swin_depth: SWIN transformer depth
        **kwargs: Additional arguments
    
    Returns:
        Hybrid C3F module
    """
    # Filter valid parameters for each variant
    valid_standard_params = [
        'num_heads', 'window_size', 'img_size', 'shortcut', 'g', 'e', 'use_focal_modulation'
    ]
    valid_adaptive_params = [
        'num_heads', 'base_window_size', 'shortcut', 'g', 'e', 'use_focal_modulation'
    ]
    
    if variant == "standard":
        # Filter kwargs for standard variant
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_standard_params}
        # Set default use_focal_modulation if not provided
        if 'use_focal_modulation' not in filtered_kwargs:
            filtered_kwargs['use_focal_modulation'] = True
        return HybridC3F(c1, c2, n=n, swin_depth=swin_depth, **filtered_kwargs)
    elif variant == "adaptive":
        # Filter kwargs for adaptive variant, handle window_size -> base_window_size
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_adaptive_params}
        if 'window_size' in kwargs:
            filtered_kwargs['base_window_size'] = kwargs['window_size']
        # Set default use_focal_modulation if not provided
        if 'use_focal_modulation' not in filtered_kwargs:
            filtered_kwargs['use_focal_modulation'] = True
        return AdaptiveHybridC3F(c1, c2, n=n, swin_depth=swin_depth, **filtered_kwargs)
    elif variant == "lightweight":
        # Lightweight version with reduced parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_standard_params}
        # Set default use_focal_modulation if not provided
        if 'use_focal_modulation' not in filtered_kwargs:
            filtered_kwargs['use_focal_modulation'] = True
        return HybridC3F(
            c1, c2, n=max(1, n//2), swin_depth=max(1, swin_depth//2), 
            e=0.25, **filtered_kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

# Testing utilities
if __name__ == "__main__":
    print("Testing Hybrid C3F components...")
    
    # Test standard HybridC3F
    print("\n1. Testing HybridC3F...")
    hybrid_c3f = HybridC3F(c1=64, c2=128, n=2, swin_depth=2, img_size=56)
    x = torch.randn(2, 64, 56, 56)
    out = hybrid_c3f(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in hybrid_c3f.parameters()):,}")
    
    # Test adaptive version
    print("\n2. Testing AdaptiveHybridC3F...")
    adaptive_c3f = AdaptiveHybridC3F(c1=64, c2=128, n=2, swin_depth=2)
    
    # Test with different input sizes
    for size in [28, 56, 112]:
        x_test = torch.randn(1, 64, size, size)
        out_test = adaptive_c3f(x_test)
        print(f"Size {size}: {x_test.shape} → {out_test.shape}")
    
    # Test factory function
    print("\n3. Testing factory function...")
    variants = ["standard", "adaptive", "lightweight"]
    for variant in variants:
        module = create_hybrid_c3f(64, 128, variant=variant, n=2, swin_depth=2)
        x_test = torch.randn(1, 64, 56, 56)
        out_test = module(x_test)
        params = sum(p.numel() for p in module.parameters())
        print(f"{variant}: {x_test.shape} → {out_test.shape}, Params: {params:,}")
    
    print("\nAll Hybrid C3F tests completed!")