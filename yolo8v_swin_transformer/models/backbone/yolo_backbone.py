"""
Modified YOLO v8 Backbone with SWIN Transformer Integration
This backbone replaces original C3F modules with HybridC3F modules
File: models/backbone/yolo_backbone.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import math

# Import from our custom modules
try:
    from ...utils.tensor_utils import make_divisible, ensure_tensor_compatibility
    from .hybrid_c3f import HybridC3F, AdaptiveHybridC3F, create_hybrid_c3f, Conv
except ImportError:
    # Fallback imports for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.tensor_utils import make_divisible, ensure_tensor_compatibility
    from models.backbone.hybrid_c3f import HybridC3F, AdaptiveHybridC3F, create_hybrid_c3f, Conv

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv8
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLOSWINBackbone(nn.Module):
    """
    YOLO v8 Backbone with SWIN Transformer Integration
    
    Architecture:
    - Input: 3-channel RGB image
    - Stem: Initial convolution layers
    - Stage 1-4: HybridC3F modules with different scales
    - SPPF: Spatial Pyramid Pooling
    
    Args:
        width_multiple: Width scaling factor (0.25, 0.5, 0.75, 1.0, 1.25)
        depth_multiple: Depth scaling factor (0.33, 0.67, 1.0, 1.33)
        max_channels: Maximum number of channels
        use_swin_in_stages: List of stages to use SWIN (default: [1,2,3])
        swin_config: Configuration for SWIN components
    """
    
    def __init__(
        self,
        width_multiple: float = 1.0,
        depth_multiple: float = 1.0, 
        max_channels: int = 1024,
        use_swin_in_stages: List[int] = [1, 2, 3],
        swin_config: Optional[Dict] = None,
        input_channels: int = 3
    ):
        super().__init__()
        
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple
        self.max_channels = max_channels
        self.use_swin_in_stages = use_swin_in_stages
        
        # Default SWIN configuration
        if swin_config is None:
            swin_config = {
                'swin_depth': 2,
                'use_focal_modulation': True,
                'window_size': 7,
                'variant': 'adaptive'  # Use adaptive version
            }
        self.swin_config = swin_config
        
        # Calculate channel configurations for different stages
        # Based on YOLOv8 architecture: [64, 128, 256, 512, 1024]
        base_channels = [64, 128, 256, 512, 1024]
        self.channels = [
            min(max_channels, make_divisible(ch * width_multiple, 8)) 
            for ch in base_channels
        ]
        
        # Calculate depth (number of layers) for each stage
        base_depths = [1, 2, 2, 1]  # n parameter for each C3F stage
        self.depths = [
            max(1, round(d * depth_multiple)) 
            for d in base_depths
        ]
        
        # Build network layers
        self._build_stem(input_channels)
        self._build_stages()
        self._build_head()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_stem(self, input_channels):
        """Build stem layers (initial convolutions)"""
        self.stem = nn.Sequential(
            Conv(input_channels, self.channels[0], 3, 2),  # /2
            Conv(self.channels[0], self.channels[1], 3, 2), # /4
        )
    
    def _build_stages(self):
        """Build main backbone stages"""
        self.stages = nn.ModuleList()
        
        # Stage 1: /8 resolution
        stage1_conv = Conv(self.channels[1], self.channels[1], 3, 2)  # /8
        if 1 in self.use_swin_in_stages:
            stage1_swin = self._create_swin_stage(
                self.channels[1], self.channels[1], 
                self.depths[0], img_size=80  # 640/8 = 80
            )
        else:
            stage1_swin = self._create_c3f_stage(
                self.channels[1], self.channels[1], self.depths[0]
            )
        self.stages.append(nn.Sequential(stage1_conv, stage1_swin))
        
        # Stage 2: /16 resolution  
        stage2_conv = Conv(self.channels[1], self.channels[2], 3, 2)  # /16
        if 2 in self.use_swin_in_stages:
            stage2_swin = self._create_swin_stage(
                self.channels[2], self.channels[2], 
                self.depths[1], img_size=40  # 640/16 = 40
            )
        else:
            stage2_swin = self._create_c3f_stage(
                self.channels[2], self.channels[2], self.depths[1]
            )
        self.stages.append(nn.Sequential(stage2_conv, stage2_swin))
        
        # Stage 3: /32 resolution
        stage3_conv = Conv(self.channels[2], self.channels[3], 3, 2)  # /32
        if 3 in self.use_swin_in_stages:
            stage3_swin = self._create_swin_stage(
                self.channels[3], self.channels[3], 
                self.depths[2], img_size=20  # 640/32 = 20
            )
        else:
            stage3_swin = self._create_c3f_stage(
                self.channels[3], self.channels[3], self.depths[2]
            )
        self.stages.append(nn.Sequential(stage3_conv, stage3_swin))
        
        # Stage 4: /32 resolution (no downsampling)
        if 4 in self.use_swin_in_stages:
            stage4 = self._create_swin_stage(
                self.channels[3], self.channels[4], 
                self.depths[3], img_size=20  # Same as stage 3
            )
        else:
            stage4 = self._create_c3f_stage(
                self.channels[3], self.channels[4], self.depths[3]
            )
        self.stages.append(stage4)
    
    def _build_head(self):
        """Build head layers (SPPF)"""
        self.sppf = SPPF(self.channels[4], self.channels[4])
    
    def _create_swin_stage(self, c1, c2, n, img_size):
        """Create a stage with HybridC3F (SWIN-enhanced)"""
        # Prepare parameters, avoiding duplicates
        swin_params = {
            'c1': c1,
            'c2': c2,
            'variant': self.swin_config['variant'],
            'n': n,
            'swin_depth': self.swin_config['swin_depth'],
            'img_size': img_size
        }
        
        # Add window_size/base_window_size based on variant
        if 'window_size' in self.swin_config:
            if self.swin_config['variant'] == 'adaptive':
                swin_params['base_window_size'] = self.swin_config['window_size']
            else:
                swin_params['window_size'] = self.swin_config['window_size']
        
        # Add use_focal_modulation if specified
        if 'use_focal_modulation' in self.swin_config:
            swin_params['use_focal_modulation'] = self.swin_config['use_focal_modulation']
        
        return create_hybrid_c3f(**swin_params)
    
    def _create_c3f_stage(self, c1, c2, n):
        """Create a stage with original C3F"""
        from .hybrid_c3f import C3F
        return C3F(c1, c2, n=n)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            List of feature maps from different scales for FPN
        """
        # Stem
        x = self.stem(x)  # /4
        
        # Stages with feature collection
        features = []
        
        # Stage 1: /8
        x = self.stages[0](x)
        features.append(x)  # P3: /8
        
        # Stage 2: /16
        x = self.stages[1](x)
        features.append(x)  # P4: /16
        
        # Stage 3: /32
        x = self.stages[2](x)
        features.append(x)  # P5: /32
        
        # Stage 4: /32 (same resolution)
        x = self.stages[3](x)
        
        # SPPF
        x = self.sppf(x)
        features.append(x)  # P6: /32 with SPPF
        
        return features
    
    def forward_with_features(self, x):
        """
        Forward pass with detailed intermediate features
        
        Returns:
            Dict with all intermediate features for analysis
        """
        features_dict = {'input': x}
        
        # Stem
        x = self.stem(x)
        features_dict['stem'] = x
        
        # Stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features_dict[f'stage_{i+1}'] = x
        
        # SPPF
        x = self.sppf(x)
        features_dict['sppf'] = x
        
        # Collect P3, P4, P5 features for FPN
        fpn_features = [
            features_dict['stage_1'],  # P3: /8
            features_dict['stage_2'],  # P4: /16  
            features_dict['stage_3'],  # P5: /32
            features_dict['sppf']      # P6: /32 + SPPF
        ]
        
        features_dict['fpn_features'] = fpn_features
        
        return features_dict
    
    def get_feature_channels(self):
        """Get number of channels for each FPN level"""
        return {
            'P3': self.channels[1],  # /8
            'P4': self.channels[2],  # /16
            'P5': self.channels[3],  # /32
            'P6': self.channels[4]   # /32 + SPPF
        }
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        swin_params = 0
        for name, module in self.named_modules():
            if 'swin' in name.lower() or isinstance(module, (HybridC3F, AdaptiveHybridC3F)):
                swin_params += sum(p.numel() for p in module.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'swin_parameters': swin_params,
            'swin_percentage': (swin_params / total_params) * 100,
            'channels': self.channels,
            'depths': self.depths,
            'swin_stages': self.use_swin_in_stages,
            'width_multiple': self.width_multiple,
            'depth_multiple': self.depth_multiple
        }

class YOLOSWINBackboneFactory:
    """Factory class for creating different YOLO-SWIN backbone variants"""
    
    # Predefined configurations for different model sizes
    CONFIGS = {
        'nano': {
            'width_multiple': 0.25,
            'depth_multiple': 0.33,
            'max_channels': 1024,
            'use_swin_in_stages': [2, 3],  # Use SWIN in fewer stages
            'swin_config': {
                'swin_depth': 1,
                'window_size': 7,
                'variant': 'lightweight'
            }
        },
        'small': {
            'width_multiple': 0.5,
            'depth_multiple': 0.33,
            'max_channels': 1024,
            'use_swin_in_stages': [1, 2, 3],
            'swin_config': {
                'swin_depth': 1,
                'window_size': 7,
                'variant': 'adaptive'
            }
        },
        'medium': {
            'width_multiple': 0.75,
            'depth_multiple': 0.67,
            'max_channels': 1024,
            'use_swin_in_stages': [1, 2, 3],
            'swin_config': {
                'swin_depth': 2,
                'window_size': 7,
                'variant': 'adaptive'
            }
        },
        'large': {
            'width_multiple': 1.0,
            'depth_multiple': 1.0,
            'max_channels': 1024,
            'use_swin_in_stages': [1, 2, 3, 4],
            'swin_config': {
                'swin_depth': 2,
                'window_size': 7,
                'variant': 'standard'
            }
        },
        'xlarge': {
            'width_multiple': 1.25,
            'depth_multiple': 1.33,
            'max_channels': 1280,
            'use_swin_in_stages': [1, 2, 3, 4],
            'swin_config': {
                'swin_depth': 3,
                'window_size': 7,
                'variant': 'standard'
            }
        }
    }
    
    @classmethod
    def create_backbone(cls, model_size: str = 'medium', **kwargs) -> YOLOSWINBackbone:
        """
        Create YOLO-SWIN backbone with predefined configuration
        
        Args:
            model_size: One of 'nano', 'small', 'medium', 'large', 'xlarge'
            **kwargs: Override any configuration parameters
            
        Returns:
            YOLOSWINBackbone instance
        """
        if model_size not in cls.CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(cls.CONFIGS.keys())}")
        
        config = cls.CONFIGS[model_size].copy()
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            if key == 'swin_config' and isinstance(value, dict):
                config['swin_config'].update(value)
            else:
                config[key] = value
        
        return YOLOSWINBackbone(**config)
    
    @classmethod
    def create_medical_backbone(cls, input_channels: int = 1, model_size: str = 'medium') -> YOLOSWINBackbone:
        """
        Create backbone optimized for medical imaging
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            model_size: Model size variant
            
        Returns:
            YOLOSWINBackbone optimized for medical images
        """
        # Medical imaging optimizations
        medical_config = {
            'input_channels': input_channels,
            'use_swin_in_stages': [1, 2, 3, 4],  # Use SWIN in all stages for better fine-grained detection
            'swin_config': {
                'swin_depth': 2,
                'window_size': 7,  # Good for medical detail preservation
                'variant': 'adaptive'
                # Note: use_focal_modulation default is True, so don't duplicate it
            }
        }
        
        return cls.create_backbone(model_size, **medical_config)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model configurations"""
        return list(cls.CONFIGS.keys())
    
    @classmethod
    def get_model_config(cls, model_size: str) -> Dict:
        """Get configuration for a specific model size"""
        if model_size not in cls.CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}")
        return cls.CONFIGS[model_size].copy()

# Convenience functions
def yolo_swin_nano(**kwargs):
    """Create YOLO-SWIN nano model"""
    return YOLOSWINBackboneFactory.create_backbone('nano', **kwargs)

def yolo_swin_small(**kwargs):
    """Create YOLO-SWIN small model"""
    return YOLOSWINBackboneFactory.create_backbone('small', **kwargs)

def yolo_swin_medium(**kwargs):
    """Create YOLO-SWIN medium model"""
    return YOLOSWINBackboneFactory.create_backbone('medium', **kwargs)

def yolo_swin_large(**kwargs):
    """Create YOLO-SWIN large model"""
    return YOLOSWINBackboneFactory.create_backbone('large', **kwargs)

def yolo_swin_xlarge(**kwargs):
    """Create YOLO-SWIN xlarge model"""
    return YOLOSWINBackboneFactory.create_backbone('xlarge', **kwargs)

def yolo_swin_medical(input_channels=1, model_size='medium', **kwargs):
    """Create YOLO-SWIN model for medical imaging"""
    return YOLOSWINBackboneFactory.create_medical_backbone(input_channels, model_size, **kwargs)

# Testing utilities
if __name__ == "__main__":
    print("Testing YOLO-SWIN Backbone...")
    
    # Test different model sizes
    print("\n1. Testing different model sizes...")
    sizes = ['nano', 'small', 'medium', 'large']
    
    for size in sizes:
        print(f"\n   Testing {size} model:")
        model = YOLOSWINBackboneFactory.create_backbone(size)
        
        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        features = model(x)
        
        # Get model info
        info = model.get_model_info()
        
        print(f"     Input: {x.shape}")
        print(f"     Features: {[f.shape for f in features]}")
        print(f"     Parameters: {info['total_parameters']:,}")
        print(f"     SWIN %: {info['swin_percentage']:.1f}%")
        print(f"     Channels: {info['channels']}")
    
    # Test medical backbone
    print("\n2. Testing medical backbone...")
    medical_model = yolo_swin_medical(input_channels=1, model_size='medium')
    x_medical = torch.randn(1, 1, 512, 512)  # Grayscale medical image
    features_medical = medical_model(x_medical)
    
    print(f"   Medical input: {x_medical.shape}")
    print(f"   Medical features: {[f.shape for f in features_medical]}")
    
    # Test feature extraction
    print("\n3. Testing detailed feature extraction...")
    model = yolo_swin_medium()
    x = torch.randn(1, 3, 640, 640)
    features_dict = model.forward_with_features(x)
    
    print(f"   Available features: {list(features_dict.keys())}")
    print(f"   FPN feature shapes: {[f.shape for f in features_dict['fpn_features']]}")
    
    # Test factory methods
    print("\n4. Testing factory methods...")
    print(f"   Available models: {YOLOSWINBackboneFactory.list_available_models()}")
    
    config = YOLOSWINBackboneFactory.get_model_config('medium')
    print(f"   Medium config: {config}")
    
    print("\nAll YOLO-SWIN Backbone tests completed!")