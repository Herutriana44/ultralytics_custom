"""
Test script for Step 1 foundation components
File: step1_test.py (place in project root)

Run this to verify all Step 1 components work correctly:
python step1_test.py
"""
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(project_path)))

def test_tensor_utils():
    """Test tensor utility functions"""
    print("=" * 50)
    print("Testing utils/tensor_utils.py")
    print("=" * 50)

    try:
        from utils.tensor_utils import (
            window_partition, window_reverse, yolo_to_swin_format,
            swin_to_yolo_format, make_divisible, drop_path,
            calculate_feature_stats, ensure_tensor_compatibility
        )

        # Test format conversion
        print("1. Testing format conversion...")
        x_yolo = torch.randn(2, 96, 56, 56)  # B, C, H, W
        x_swin = yolo_to_swin_format(x_yolo)  # B, H, W, C
        x_back = swin_to_yolo_format(x_swin)  # B, C, H, W

        assert x_yolo.shape == x_back.shape, f"Format conversion failed: {x_yolo.shape} != {x_back.shape}"
        assert torch.allclose(x_yolo, x_back), "Format conversion values don't match"
        print(f"   ✅ YOLO format: {x_yolo.shape}")
        print(f"   ✅ SWIN format: {x_swin.shape}")
        print(f"   ✅ Back to YOLO: {x_back.shape}")

        # Test window partition
        print("2. Testing window partition...")
        window_size = 7
        windows = window_partition(x_swin, window_size)
        x_restored = window_reverse(windows, window_size, x_swin.shape[1], x_swin.shape[2])

        assert x_swin.shape == x_restored.shape, f"Window partition failed: {x_swin.shape} != {x_restored.shape}"
        assert torch.allclose(x_swin, x_restored), "Window partition values don't match"
        print(f"   ✅ Original: {x_swin.shape}")
        print(f"   ✅ Windows: {windows.shape}")
        print(f"   ✅ Restored: {x_restored.shape}")

        # Test make_divisible
        print("3. Testing make_divisible...")
        result1 = make_divisible(97, 8)
        result2 = make_divisible(96, 8)
        print(f"   🔸 make_divisible(97, 8) = {result1} (expected: 96 or 104)")
        print(f"   🔸 make_divisible(96, 8) = {result2} (expected: 96)")

        assert result1 % 8 == 0, f"make_divisible(97, 8) not divisible by 8: {result1}"
        assert result2 == 96, f"make_divisible(96, 8) should be 96, got {result2}"
        print(f"   ✅ make_divisible working correctly")

        # Test feature stats
        print("4. Testing feature statistics...")
        mean, var = calculate_feature_stats(x_yolo)
        print(f"   ✅ Feature mean shape: {mean.shape}")
        print(f"   ✅ Feature var shape: {var.shape}")

        # Test tensor compatibility
        print("5. Testing tensor compatibility...")
        x_resized = ensure_tensor_compatibility(x_yolo, (112, 112))
        print(f"   ✅ Original: {x_yolo.shape}")
        print(f"   ✅ Resized: {x_resized.shape}")

        print("✅ All tensor_utils tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ tensor_utils test failed: {e}")
        return False

def test_focal_modulation():
    """Test focal modulation layers"""
    print("=" * 50)
    print("Testing models/backbone/focal_modulation.py")
    print("=" * 50)

    try:
        # Add fallback import handling
        try:
            from models.backbone.focal_modulation import (
                FocalModulation, FocalModulationYOLO, create_focal_modulation
            )
        except ImportError:
            # Try alternative import path
            sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'backbone'))
            from focal_modulation import (
                FocalModulation, FocalModulationYOLO, create_focal_modulation
            )

        # Test FocalModulation (SWIN format)
        print("1. Testing FocalModulation (SWIN format)...")
        focal_swin = FocalModulation(dim=96, focal_level=2, focal_window=7)
        x_swin = torch.randn(2, 56, 56, 96)  # B, H, W, C
        out_swin = focal_swin(x_swin)

        assert out_swin.shape == x_swin.shape, f"FocalModulation output shape mismatch: {out_swin.shape} != {x_swin.shape}"
        print(f"   ✅ Input: {x_swin.shape}")
        print(f"   ✅ Output: {out_swin.shape}")
        print(f"   ✅ Parameters: {sum(p.numel() for p in focal_swin.parameters()):,}")

        # Test FocalModulationYOLO (YOLO format)
        print("2. Testing FocalModulationYOLO (YOLO format)...")
        focal_yolo = FocalModulationYOLO(dim=96, focal_level=2, focal_window=7)
        x_yolo = torch.randn(2, 96, 56, 56)  # B, C, H, W
        out_yolo = focal_yolo(x_yolo)

        assert out_yolo.shape == x_yolo.shape, f"FocalModulationYOLO output shape mismatch: {out_yolo.shape} != {x_yolo.shape}"
        print(f"   ✅ Input: {x_yolo.shape}")
        print(f"   ✅ Output: {out_yolo.shape}")
        print(f"   ✅ Parameters: {sum(p.numel() for p in focal_yolo.parameters()):,}")

        # Test factory function
        print("3. Testing factory function...")
        focal_factory_swin = create_focal_modulation(dim=96, input_format="swin")
        focal_factory_yolo = create_focal_modulation(dim=96, input_format="yolo")

        assert isinstance(focal_factory_swin, FocalModulation), "Factory function failed for SWIN"
        assert isinstance(focal_factory_yolo, FocalModulationYOLO), "Factory function failed for YOLO"
        print(f"   ✅ Factory SWIN: {type(focal_factory_swin).__name__}")
        print(f"   ✅ Factory YOLO: {type(focal_factory_yolo).__name__}")

        # Test gradient flow
        print("4. Testing gradient flow...")
        x_test = torch.randn(1, 28, 28, 48, requires_grad=True)
        focal_test = FocalModulation(dim=48, focal_level=1)
        out_test = focal_test(x_test)
        loss = out_test.sum()
        loss.backward()

        assert x_test.grad is not None, "Gradient flow failed"
        print(f"   ✅ Gradient flow working")

        print("✅ All focal_modulation tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ focal_modulation test failed: {e}")
        return False

def test_swin_transformer():
    """Test SWIN transformer components"""
    print("=" * 50)
    print("Testing models/backbone/swin_transformer.py")
    print("=" * 50)

    try:
        # Add fallback import handling
        try:
            from models.backbone.swin_transformer import (
                SwinTransformerBlock, BasicLayer, PatchEmbed,
                WindowAttention, Mlp, PatchMerging
            )
        except ImportError:
            # Try alternative import path
            sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'backbone'))
            from swin_transformer import (
                SwinTransformerBlock, BasicLayer, PatchEmbed,
                WindowAttention, Mlp, PatchMerging
            )

        # Test SwinTransformerBlock with Focal Modulation
        print("1. Testing SwinTransformerBlock (with Focal Modulation)...")
        block_focal = SwinTransformerBlock(
            dim=96,
            input_resolution=(56, 56),
            num_heads=3,
            window_size=7,
            use_focal_modulation=True
        )
        x_block = torch.randn(2, 56*56, 96)  # B, H*W, C
        out_block_focal = block_focal(x_block)

        assert out_block_focal.shape == x_block.shape, f"SwinTransformerBlock output shape mismatch: {out_block_focal.shape} != {x_block.shape}"
        print(f"   ✅ Input: {x_block.shape}")
        print(f"   ✅ Output: {out_block_focal.shape}")
        print(f"   ✅ Using Focal Modulation: {block_focal.use_focal_modulation}")

        # Test SwinTransformerBlock with Window Attention
        print("2. Testing SwinTransformerBlock (with Window Attention)...")
        block_attn = SwinTransformerBlock(
            dim=96,
            input_resolution=(56, 56),
            num_heads=3,
            window_size=7,
            use_focal_modulation=False
        )
        out_block_attn = block_attn(x_block)

        assert out_block_attn.shape == x_block.shape, f"SwinTransformerBlock output shape mismatch: {out_block_attn.shape} != {x_block.shape}"
        print(f"   ✅ Input: {x_block.shape}")
        print(f"   ✅ Output: {out_block_attn.shape}")
        print(f"   ✅ Using Focal Modulation: {block_attn.use_focal_modulation}")

        # Test BasicLayer
        print("3. Testing BasicLayer...")
        layer = BasicLayer(
            dim=96,
            input_resolution=(56, 56),
            depth=2,
            num_heads=3,
            window_size=7,
            downsample=PatchMerging,
            use_focal_modulation=True
        )
        out_layer = layer(x_block)

        expected_shape = (2, (56//2)*(56//2), 96*2)  # Downsampled by PatchMerging
        assert out_layer.shape == expected_shape, f"BasicLayer output shape mismatch: {out_layer.shape} != {expected_shape}"
        print(f"   ✅ Input: {x_block.shape}")
        print(f"   ✅ Output: {out_layer.shape}")
        print(f"   ✅ Depth: {layer.depth}")

        # Test PatchEmbed
        print("4. Testing PatchEmbed...")
        patch_embed = PatchEmbed(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96
        )
        x_img = torch.randn(2, 3, 224, 224)  # B, C, H, W
        out_embed = patch_embed(x_img)

        expected_patches = (224//4) * (224//4)  # 56*56 = 3136
        expected_shape = (2, expected_patches, 96)
        assert out_embed.shape == expected_shape, f"PatchEmbed output shape mismatch: {out_embed.shape} != {expected_shape}"
        print(f"   ✅ Input: {x_img.shape}")
        print(f"   ✅ Output: {out_embed.shape}")
        print(f"   ✅ Patches: {patch_embed.num_patches}")

        # Test Mlp
        print("5. Testing Mlp...")
        mlp = Mlp(in_features=96, hidden_features=384)
        x_mlp = torch.randn(2, 100, 96)
        out_mlp = mlp(x_mlp)

        assert out_mlp.shape == x_mlp.shape, f"Mlp output shape mismatch: {out_mlp.shape} != {x_mlp.shape}"
        print(f"   ✅ Input: {x_mlp.shape}")
        print(f"   ✅ Output: {out_mlp.shape}")

        # Test WindowAttention
        print("6. Testing WindowAttention...")
        win_attn = WindowAttention(dim=96, window_size=(7, 7), num_heads=3)
        x_win = torch.randn(8, 49, 96)  # nW*B, window_size*window_size, C
        out_win = win_attn(x_win)

        assert out_win.shape == x_win.shape, f"WindowAttention output shape mismatch: {out_win.shape} != {x_win.shape}"
        print(f"   ✅ Input: {x_win.shape}")
        print(f"   ✅ Output: {out_win.shape}")

        print("✅ All swin_transformer tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ swin_transformer test failed: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("=" * 50)
    print("Testing Component Integration")
    print("=" * 50)

    try:
        # Add fallback import handling
        try:
            from utils.tensor_utils import yolo_to_swin_format, swin_to_yolo_format
            from models.backbone.focal_modulation import FocalModulation
            from models.backbone.swin_transformer import SwinTransformerBlock
        except ImportError:
            # Try alternative import paths
            from tensor_utils import yolo_to_swin_format, swin_to_yolo_format
            from focal_modulation import FocalModulation
            from swin_transformer import SwinTransformerBlock

        print("1. Testing YOLO → SWIN → Focal Modulation pipeline...")

        # Start with YOLO format
        x_yolo = torch.randn(2, 96, 56, 56)  # B, C, H, W
        print(f"   🔸 Original YOLO: {x_yolo.shape}")

        # Convert to SWIN format
        x_swin = yolo_to_swin_format(x_yolo)
        print(f"   🔸 Converted to SWIN: {x_swin.shape}")

        # Apply Focal Modulation
        focal_mod = FocalModulation(dim=96, focal_level=2)
        x_focal = focal_mod(x_swin)
        print(f"   🔸 After Focal Modulation: {x_focal.shape}")

        # Convert back to YOLO
        x_back = swin_to_yolo_format(x_focal)
        print(f"   🔸 Back to YOLO: {x_back.shape}")

        assert x_yolo.shape == x_back.shape, "Integration pipeline shape mismatch"
        print("   ✅ Integration pipeline successful!")

        print("2. Testing SWIN Block with different configurations...")

        # Test with different resolutions
        resolutions = [(28, 28), (56, 56), (112, 112)]
        dims = [96, 192, 384]

        for i, (res, dim) in enumerate(zip(resolutions, dims)):
            print(f"   🔸 Testing resolution {res} with dim {dim}...")

            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=res,
                num_heads=dim//32,
                window_size=7,
                use_focal_modulation=True
            )

            x_test = torch.randn(1, res[0]*res[1], dim)
            out_test = block(x_test)

            assert out_test.shape == x_test.shape, f"Block test failed for {res}x{dim}"
            print(f"     ✅ {x_test.shape} → {out_test.shape}")

        print("✅ All integration tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_memory_and_performance():
    """Test memory usage and basic performance"""
    print("=" * 50)
    print("Testing Memory & Performance")
    print("=" * 50)

    try:
        import time
        # Add fallback import handling
        try:
            from models.backbone.focal_modulation import FocalModulation
            from models.backbone.swin_transformer import SwinTransformerBlock
        except ImportError:
            from focal_modulation import FocalModulation
            from swin_transformer import SwinTransformerBlock

        print("1. Testing memory efficiency...")

        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 56, 56, 96)
            focal = FocalModulation(dim=96)

            # Measure memory before
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            out = focal(x)

            print(f"   ✅ Batch size {batch_size}: {x.shape} → {out.shape}")

        print("2. Testing inference speed...")

        # Warm up
        x = torch.randn(2, 56, 56, 96)
        focal = FocalModulation(dim=96)
        for _ in range(5):
            _ = focal(x)

        # Measure speed
        start_time = time.time()
        num_runs = 50
        for _ in range(num_runs):
            _ = focal(x)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        print(f"   ✅ Average inference time: {avg_time:.2f}ms")

        print("3. Testing gradient memory...")

        x = torch.randn(1, 28, 28, 48, requires_grad=True)
        block = SwinTransformerBlock(
            dim=48,
            input_resolution=(28, 28),
            num_heads=3,
            window_size=7,
            use_focal_modulation=True
        )

        out = block(x.view(1, 28*28, 48))
        loss = out.sum()
        loss.backward()

        # Check gradient exists
        assert x.grad is not None, "Gradient computation failed"
        print(f"   ✅ Gradient shape: {x.grad.shape}")

        print("✅ All memory & performance tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Memory & performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Step 1 Foundation Components Test")
    print("=" * 70)

    test_results = []

    # Run all tests
    test_results.append(("Tensor Utils", test_tensor_utils()))
    test_results.append(("Focal Modulation", test_focal_modulation()))
    test_results.append(("SWIN Transformer", test_swin_transformer()))
    test_results.append(("Integration", test_integration()))
    test_results.append(("Memory & Performance", test_memory_and_performance()))

    # Summary
    print("=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1

    print("=" * 70)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Step 1 foundation components are working correctly!")
        print("✨ Ready to proceed to Step 2: Integration Layer")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("🔧 Fix the issues before proceeding to Step 2.")

    print("=" * 70)

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)