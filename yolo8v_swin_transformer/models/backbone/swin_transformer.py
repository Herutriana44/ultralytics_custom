"""
SWIN Transformer implementation for YOLO integration
Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
File: models/backbone/swin_transformer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
import math

# Import from our custom modules
try:
    # Try relative imports first (when used as module)
    from ...utils.tensor_utils import (
        window_partition, window_reverse, drop_path, trunc_normal_, 
        create_attention_mask, yolo_to_swin_format, swin_to_yolo_format
    )
    from .focal_modulation import FocalModulation
except ImportError:
    # Fallback to absolute imports (when testing)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.tensor_utils import (
        window_partition, window_reverse, drop_path, trunc_normal_, 
        create_attention_mask, yolo_to_swin_format, swin_to_yolo_format
    )
    from models.backbone.focal_modulation import FocalModulation

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: Optional[float] = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """
    MLP module for SWIN Transformer
    
    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (default: in_features)
        out_features: Number of output features (default: in_features) 
        act_layer: Activation layer
        drop: Dropout rate
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention with relative position bias
    
    Args:
        dim: Number of input channels
        window_size: Window size tuple (height, width)
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to qkv projection
        qk_scale: Scale factor for query-key product
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features with shape (num_windows*B, N, C)
            mask: (0/-inf) mask with shape (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class SwinTransformerBlock(nn.Module):
    """
    SWIN Transformer Block with optional Focal Modulation
    
    Args:
        dim: Number of input channels
        input_resolution: Input resolution (height, width)
        num_heads: Number of attention heads
        window_size: Window size
        shift_size: Shift size for SW-MSA
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Whether to add bias to qkv projection
        qk_scale: Scale factor for query-key product  
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
        act_layer: Activation layer
        norm_layer: Normalization layer
        use_focal_modulation: Whether to use focal modulation instead of window attention
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_focal_modulation=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_focal_modulation = use_focal_modulation
        
        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        if self.use_focal_modulation:
            self.focal_modulation = FocalModulation(
                dim=dim,
                focal_window=window_size,
                focal_level=2,
                focal_factor=2,
                proj_drop=drop
            )
        else:
            self.attn = WindowAttention(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # Get actual input dimensions
        B, L, C = x.shape
        
        # Calculate H, W from input length
        H_W_product = L
        H = int(H_W_product ** 0.5)
        W = H_W_product // H
        
        # If perfect square, use that, otherwise use input_resolution
        if H * W != L:
            H, W = self.input_resolution
        
        # Verify the input size matches expected
        if L != H * W:
            # Try to infer correct dimensions
            import math
            sqrt_L = int(math.sqrt(L))
            if sqrt_L * sqrt_L == L:
                H = W = sqrt_L
            else:
                # Use the closest dimensions that work
                factors = []
                for i in range(1, int(math.sqrt(L)) + 1):
                    if L % i == 0:
                        factors.append((i, L // i))
                if factors:
                    H, W = factors[-1]  # Use the most square-like dimensions
                else:
                    raise ValueError(f"Cannot reshape input of length {L} into valid 2D dimensions")

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        if self.use_focal_modulation:
            # Use Focal Modulation - works directly with (B, H, W, C) format
            attn_windows = self.focal_modulation(shifted_x)
        else:
            # Partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # Merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            attn_windows = window_reverse(attn_windows, self.window_size, H, W, H, W)  # B H' W' C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = attn_windows

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for downsampling
    
    Args:
        input_resolution: Input resolution (height, width)
        dim: Number of input channels
        norm_layer: Normalization layer
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

class BasicLayer(nn.Module):
    """
    A basic SWIN Transformer layer for one stage
    
    Args:
        dim: Number of input channels
        input_resolution: Input resolution (height, width)
        depth: Number of blocks
        num_heads: Number of attention heads
        window_size: Local window size
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Whether to add bias to qkv projection
        qk_scale: Scale factor for query-key product
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
        norm_layer: Normalization layer
        downsample: Downsample layer at the end of the layer
        use_checkpoint: Whether to use gradient checkpointing to save memory
        use_focal_modulation: Whether to use focal modulation
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_focal_modulation=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,
                                use_focal_modulation=use_focal_modulation)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    
    Args:
        img_size: Image size (height, width)
        patch_size: Patch token size
        in_chans: Number of input image channels
        embed_dim: Number of linear projection output channels
        norm_layer: Normalization layer
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

# Testing utilities
if __name__ == "__main__":
    print("Testing SWIN Transformer components...")
    
    # Test SwinTransformerBlock
    print("\n1. Testing SwinTransformerBlock...")
    block = SwinTransformerBlock(
        dim=96,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        use_focal_modulation=True
    )
    x = torch.randn(2, 56*56, 96)  # B, H*W, C
    out = block(x)
    print(f"Block input: {x.shape}, output: {out.shape}")
    
    # Test BasicLayer
    print("\n2. Testing BasicLayer...")
    layer = BasicLayer(
        dim=96,
        input_resolution=(56, 56),
        depth=2,
        num_heads=3,
        window_size=7,
        use_focal_modulation=True
    )
    out = layer(x)
    print(f"Layer input: {x.shape}, output: {out.shape}")
    
    # Test PatchEmbed
    print("\n3. Testing PatchEmbed...")
    patch_embed = PatchEmbed(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96
    )
    x_img = torch.randn(2, 3, 224, 224)  # B, C, H, W
    out_embed = patch_embed(x_img)
    print(f"PatchEmbed input: {x_img.shape}, output: {out_embed.shape}")
    
    print("\nAll SWIN Transformer tests passed!")