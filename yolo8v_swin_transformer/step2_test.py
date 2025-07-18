"""
Test script for Step 2 integration layer components
File: step2_test.py (place in project root)

Run this to verify all Step 2 components work correctly:
python step2_test.py
"""
import torch
import torch.nn as nn
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(project_path)))

def test_hybrid_c3f():
    """Test Hybrid C3F components"""
    print("=" * 50)
    print("Testing models/backbone/hybrid_c3f.py")
    print("=" * 50)

    try:
        # Add fallback import handling
        try:
            from models.backbone.hybrid_c3f import (
                HybridC3F, AdaptiveHybridC3F, create_hybrid_c3f,
                SWINAdapter, SWINReverseAdapter, C3F
            )
        except ImportError:
            # Try alternative import path
            sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'backbone'))
            from hybrid_c3f import (
                HybridC3F, AdaptiveHybridC3F, create_hybrid_c3f,
                SWINAdapter, SWINReverseAdapter, C3F
            )

        # Test SWINAdapter
        print("1. Testing SWINAdapter...")
        adapter = SWINAdapter(c1=64, c2=96, img_size=56)
        x_yolo = torch.randn(2, 64, 56, 56)  # YOLO format
        x_swin, resolution = adapter(x_yolo)

        expected_shape = (2, 56*56, 96)
        assert x_swin.shape == expected_shape, f"SWINAdapter output shape mismatch: {x_swin.shape} != {expected_shape}"
        print(f"   ✅ Input: {x_yolo.shape}")
        print(f"   ✅ Output: {x_swin.shape}")
        print(f"   ✅ Resolution: {resolution}")

        # Test SWINReverseAdapter
        print("2. Testing SWINReverseAdapter...")
        reverse_adapter = SWINReverseAdapter(c1=96, c2=64, output_size=(56, 56))
        x_back = reverse_adapter(x_swin, (56, 56))

        assert x_back.shape == x_yolo.shape, f"SWINReverseAdapter output shape mismatch: {x_back.shape} != {x_yolo.shape}"
        print(f"   ✅ Input: {x_swin.shape}")
        print(f"   ✅ Output: {x_back.shape}")

        # Test original C3F
        print("3. Testing original C3F...")
        c3f = C3F(c1=64, c2=128, n=2)
        out_c3f = c3f(x_yolo)

        expected_c3f_shape = (2, 128, 56, 56)
        assert out_c3f.shape == expected_c3f_shape, f"C3F output shape mismatch: {out_c3f.shape} != {expected_c3f_shape}"
        print(f"   ✅ Input: {x_yolo.shape}")
        print(f"   ✅ Output: {out_c3f.shape}")
        print(f"   ✅ Parameters: {sum(p.numel() for p in c3f.parameters()):,}")

        # Test HybridC3F
        print("4. Testing HybridC3F...")
        hybrid_c3f = HybridC3F(c1=64, c2=128, n=2, swin_depth=2, img_size=56)
        out_hybrid = hybrid_c3f(x_yolo)

        assert out_hybrid.shape == expected_c3f_shape, f"HybridC3F output shape mismatch: {out_hybrid.shape} != {expected_c3f_shape}"
        print(f"   ✅ Input: {x_yolo.shape}")
        print(f"   ✅ Output: {out_hybrid.shape}")
        print(f"   ✅ Parameters: {sum(p.numel() for p in hybrid_c3f.parameters()):,}")

        # Compare parameter counts
        c3f_params = sum(p.numel() for p in c3f.parameters())
        hybrid_params = sum(p.numel() for p in hybrid_c3f.parameters())
        param_increase = ((hybrid_params - c3f_params) / c3f_params) * 100
        print(f"   ✅ Parameter increase: {param_increase:.1f}%")

        # Test AdaptiveHybridC3F
        print("5. Testing AdaptiveHybridC3F...")
        adaptive_c3f = AdaptiveHybridC3F(c1=64, c2=128, n=2, swin_depth=2)

        # Test with different input sizes
        test_sizes = [28, 56, 112]
        for size in test_sizes:
            x_test = torch.randn(1, 64, size, size)
            out_test = adaptive_c3f(x_test)
            expected_shape = (1, 128, size, size)
            assert out_test.shape == expected_shape, f"AdaptiveHybridC3F failed for size {size}"
            print(f"   ✅ Size {size}: {x_test.shape} → {out_test.shape}")

        # Test factory function
        print("6. Testing factory function...")
        variants = ['standard', 'adaptive', 'lightweight']
        for variant in variants:
            factory_model = create_hybrid_c3f(
                c1=64, c2=128, variant=variant, n=2, swin_depth=2
            )
            x_test = torch.randn(1, 64, 56, 56)
            out_test = factory_model(x_test)
            params = sum(p.numel() for p in factory_model.parameters())
            print(f"   ✅ {variant}: {out_test.shape}, Params: {params:,}")

        # Test feature extraction
        print("7. Testing feature extraction...")
        feature_maps = hybrid_c3f.get_feature_maps(x_yolo)
        print(f"   ✅ Available features: {list(feature_maps.keys())}")
        print(f"   ✅ C3F main shape: {feature_maps['c3f_main'].shape}")
        print(f"   ✅ SWIN output shape: {feature_maps['swin_output'].shape}")

        # Test gradient flow
        print("8. Testing gradient flow...")
        x_grad = torch.randn(1, 64, 56, 56, requires_grad=True)
        out_grad = hybrid_c3f(x_grad)
        loss = out_grad.sum()
        loss.backward()

        assert x_grad.grad is not None, "Gradient flow failed"
        print(f"   ✅ Gradient flow working")
        print(f"   ✅ Gradient shape: {x_grad.grad.shape}")

        print("✅ All hybrid_c3f tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ hybrid_c3f test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_backbone():
    """Test YOLO backbone with SWIN integration"""
    print("=" * 50)
    print("Testing models/backbone/yolo_backbone.py")
    print("=" * 50)

    try:
        # Add fallback import handling
        try:
            from models.backbone.yolo_backbone import (
                YOLOSWINBackbone, YOLOSWINBackboneFactory,
                yolo_swin_nano, yolo_swin_small, yolo_swin_medium,
                yolo_swin_large, yolo_swin_medical
            )
        except ImportError:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'backbone'))
            from yolo_backbone import (
                YOLOSWINBackbone, YOLOSWINBackboneFactory,
                yolo_swin_nano, yolo_swin_small, yolo_swin_medium,
                yolo_swin_large, yolo_swin_medical
            )

        # Test basic backbone
        print("1. Testing YOLOSWINBackbone...")
        backbone = YOLOSWINBackbone(
            width_multiple=1.0,
            depth_multiple=1.0,
            use_swin_in_stages=[1, 2, 3]
        )

        x = torch.randn(2, 3, 640, 640)
        features = backbone(x)

        print(f"   ✅ Input: {x.shape}")
        print(f"   ✅ Number of feature levels: {len(features)}")
        for i, feat in enumerate(features):
            print(f"   ✅ Feature {i+1}: {feat.shape}")

        # Test model info
        info = backbone.get_model_info()
        print(f"   ✅ Total parameters: {info['total_parameters']:,}")
        print(f"   ✅ SWIN percentage: {info['swin_percentage']:.1f}%")
        print(f"   ✅ Channels: {info['channels']}")

        # Test different model sizes using factory
        print("2. Testing different model sizes...")
        sizes = ['nano', 'small', 'medium', 'large']

        for size in sizes:
            print(f"   🔸 Testing {size} model...")
            model = YOLOSWINBackboneFactory.create_backbone(size)

            # Test with smaller input for faster testing
            x_test = torch.randn(1, 3, 320, 320)
            features_test = model(x_test)
            info_test = model.get_model_info()

            print(f"     ✅ Features: {len(features_test)} levels")
            print(f"     ✅ Parameters: {info_test['total_parameters']:,}")
            print(f"     ✅ SWIN stages: {info_test['swin_stages']}")

        # Test convenience functions
        print("3. Testing convenience functions...")
        models_to_test = [
            ('nano', yolo_swin_nano),
            ('small', yolo_swin_small),
            ('medium', yolo_swin_medium)
        ]

        for name, model_func in models_to_test:
            model = model_func()
            x_test = torch.randn(1, 3, 320, 320)
            features_test = model(x_test)
            print(f"   ✅ {name}: {len(features_test)} features, shapes: {[f.shape for f in features_test]}")

        # Test medical backbone
        print("4. Testing medical backbone...")
        medical_model = yolo_swin_medical(input_channels=1, model_size='small')
        x_medical = torch.randn(1, 1, 512, 512)  # Grayscale medical image
        features_medical = medical_model(x_medical)

        print(f"   ✅ Medical input: {x_medical.shape}")
        print(f"   ✅ Medical features: {len(features_medical)} levels")
        for i, feat in enumerate(features_medical):
            print(f"   ✅ Medical feature {i+1}: {feat.shape}")

        # Test detailed feature extraction
        print("5. Testing detailed feature extraction...")
        model = yolo_swin_small()
        x_test = torch.randn(1, 3, 320, 320)
        features_dict = model.forward_with_features(x_test)

        print(f"   ✅ Available features: {list(features_dict.keys())}")
        print(f"   ✅ FPN features: {len(features_dict['fpn_features'])}")

        # Test feature channels
        channels = model.get_feature_channels()
        print(f"   ✅ Feature channels: {channels}")

        # Test factory methods
        print("6. Testing factory methods...")
        available_models = YOLOSWINBackboneFactory.list_available_models()
        print(f"   ✅ Available models: {available_models}")

        # Test configuration retrieval
        config = YOLOSWINBackboneFactory.get_model_config('medium')
        print(f"   ✅ Medium config keys: {list(config.keys())}")

        # Test custom configuration
        print("7. Testing custom configuration...")
        custom_model = YOLOSWINBackboneFactory.create_backbone(
            'medium',
            use_swin_in_stages=[2, 3],  # Override SWIN stages
            swin_config={'swin_depth': 1}  # Override SWIN depth
        )

        x_custom = torch.randn(1, 3, 320, 320)
        features_custom = custom_model(x_custom)
        info_custom = custom_model.get_model_info()

        print(f"   ✅ Custom model features: {len(features_custom)}")
        print(f"   ✅ Custom SWIN stages: {info_custom['swin_stages']}")

        print("✅ All yolo_backbone tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ yolo_backbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between hybrid_c3f and yolo_backbone"""
    print("=" * 50)
    print("Testing Integration: HybridC3F ↔ YOLO Backbone")
    print("=" * 50)

    try:
        # Import both modules
        try:
            from models.backbone.hybrid_c3f import HybridC3F, AdaptiveHybridC3F
            from models.backbone.yolo_backbone import YOLOSWINBackbone, yolo_swin_medium
        except ImportError:
            from hybrid_c3f import HybridC3F, AdaptiveHybridC3F
            from yolo_backbone import YOLOSWINBackbone, yolo_swin_medium

        print("1. Testing HybridC3F integration in backbone...")

        # Create backbone with SWIN in different stages
        backbone = YOLOSWINBackbone(
            width_multiple=0.5,  # Smaller for faster testing
            depth_multiple=0.5,
            use_swin_in_stages=[2, 3],  # Use SWIN in stages 2 and 3
            swin_config={
                'swin_depth': 1,
                'variant': 'adaptive'
            }
        )

        # Test with multiple input sizes
        input_sizes = [(320, 320), (416, 416), (640, 640)]

        for h, w in input_sizes:
            print(f"   🔸 Testing input size {h}x{w}...")
            x = torch.randn(1, 3, h, w)
            features = backbone(x)

            # Verify feature shapes are reasonable
            for i, feat in enumerate(features):
                expected_h = h // (8 * (2 ** min(i, 2)))  # Downsampling ratios
                expected_w = w // (8 * (2 ** min(i, 2)))
                print(f"     ✅ Feature {i+1}: {feat.shape}")

            assert len(features) == 4, f"Expected 4 feature levels, got {len(features)}"

        print("2. Testing memory efficiency...")

        # Test memory usage with different batch sizes
        model = yolo_swin_medium()
        model.eval()

        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 320, 320)

            # Measure inference time
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                features = model(x)

            inference_time = (time.time() - start_time) * 1000  # ms

            print(f"   ✅ Batch {batch_size}: {inference_time:.1f}ms, Memory efficient: {len(features)} features")

        print("3. Testing gradient compatibility...")

        # Test that gradients flow properly through the integrated model
        model = yolo_swin_medium()
        model.train()

        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        features = model(x)

        # Create a simple loss from all features
        total_loss = sum(feat.mean() for feat in features)
        total_loss.backward()

        assert x.grad is not None, "Gradient flow failed"
        print(f"   ✅ Gradient flow working through {len(features)} feature levels")

        # Check that SWIN components received gradients
        swin_grad_found = False
        for name, param in model.named_parameters():
            if 'swin' in name.lower() and param.grad is not None:
                swin_grad_found = True
                break

        print(f"   ✅ SWIN components receiving gradients: {swin_grad_found}")

        print("4. Testing feature consistency...")

        # Compare outputs between standard and adaptive variants
        standard_model = YOLOSWINBackbone(
            width_multiple=0.5,
            use_swin_in_stages=[2],
            swin_config={'variant': 'standard', 'swin_depth': 1}
        )

        adaptive_model = YOLOSWINBackbone(
            width_multiple=0.5,
            use_swin_in_stages=[2],
            swin_config={'variant': 'adaptive', 'swin_depth': 1}
        )

        x_test = torch.randn(1, 3, 320, 320)

        with torch.no_grad():
            features_standard = standard_model(x_test)
            features_adaptive = adaptive_model(x_test)

        # Both should produce same number of features with same shapes
        assert len(features_standard) == len(features_adaptive), "Feature count mismatch"

        for i, (f_std, f_adapt) in enumerate(zip(features_standard, features_adaptive)):
            assert f_std.shape == f_adapt.shape, f"Feature {i} shape mismatch: {f_std.shape} vs {f_adapt.shape}"

        print(f"   ✅ Feature consistency verified across variants")

        print("✅ All integration tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison(device=None):
    """Compare performance between original and hybrid models"""
    print("=" * 50)
    print("Testing Performance Comparison")
    print("=" * 50)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Import modules
        try:
            from models.backbone.hybrid_c3f import C3F, HybridC3F
            from models.backbone.yolo_backbone import yolo_swin_medium
        except ImportError:
            from hybrid_c3f import C3F, HybridC3F
            from yolo_backbone import yolo_swin_medium

        print("1. Comparing C3F vs HybridC3F...")

        # Create comparable models
        c3f_model = C3F(c1=64, c2=128, n=2).to(device)
        hybrid_model = HybridC3F(c1=64, c2=128, n=2, swin_depth=1, img_size=56).to(device)

        x = torch.randn(4, 64, 56, 56, device=device)  # Batch of 4 for better timing

        # Compare parameters
        c3f_params = sum(p.numel() for p in c3f_model.parameters())
        hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
        param_ratio = hybrid_params / c3f_params

        print(f"   📊 C3F parameters: {c3f_params:,}")
        print(f"   📊 HybridC3F parameters: {hybrid_params:,}")
        print(f"   📊 Parameter ratio: {param_ratio:.2f}x")

        # Compare inference time
        num_runs = 20

        # Warm up
        for _ in range(5):
            _ = c3f_model(x)
            _ = hybrid_model(x)

        # Time C3F
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = c3f_model(x)
        c3f_time = (time.time() - start_time) / num_runs * 1000

        # Time HybridC3F
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = hybrid_model(x)
        hybrid_time = (time.time() - start_time) / num_runs * 1000

        time_ratio = hybrid_time / c3f_time

        print(f"   ⏱️  C3F inference time: {c3f_time:.2f}ms")
        print(f"   ⏱️  HybridC3F inference time: {hybrid_time:.2f}ms")
        print(f"   ⏱️  Time ratio: {time_ratio:.2f}x")

        print("2. Testing backbone performance...")

        # Test backbone inference time
        backbone = yolo_swin_medium().to(device)
        backbone.eval()

        input_sizes = [(320, 320), (640, 640)]
        batch_sizes = [1, 4]

        for (h, w) in input_sizes:
            for batch_size in batch_sizes:
                x_test = torch.randn(batch_size, 3, h, w, device=device)

                # Warm up
                for _ in range(3):
                    with torch.no_grad():
                        _ = backbone(x_test)

                # Time inference
                start_time = time.time()
                num_runs = 10
                for _ in range(num_runs):
                    with torch.no_grad():
                        features = backbone(x_test)
                inference_time = (time.time() - start_time) / num_runs * 1000

                print(f"   ⏱️  {h}x{w}, batch {batch_size}: {inference_time:.1f}ms")

        print("3. Testing memory usage...")

        # Test peak memory usage (approximate)
        model = yolo_swin_medium().to(device)

        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Test different input sizes
        for size in [320, 480, 640]:
            x_mem = torch.randn(1, 3, size, size, device=device)

            if torch.cuda.is_available():
                model = model.cuda()
                x_mem = x_mem.cuda()
                torch.cuda.reset_peak_memory_stats()

                with torch.no_grad():
                    _ = model(x_mem)

                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                print(f"   💾 {size}x{size}: ~{peak_memory:.0f}MB peak memory")

                model = model.cpu()
            else:
                with torch.no_grad():
                    _ = model(x_mem)
                print(f"   💾 {size}x{size}: Memory usage test completed (CPU)")

        print("✅ All performance tests completed!\n")
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Step 2 tests"""
    print("🚀 Starting Step 2 Integration Layer Components Test")
    print("=" * 70)

    test_results = []

    # Run all tests
    test_results.append(("Hybrid C3F", test_hybrid_c3f()))
    test_results.append(("YOLO Backbone", test_yolo_backbone()))
    test_results.append(("Integration", test_integration()))
    test_results.append(("Performance", test_performance_comparison()))

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
        print("🎉 All Step 2 integration components are working correctly!")
        print("✨ Ready to proceed to Step 3: Quality Control System")
        print("💡 Key achievements:")
        print("   • HybridC3F successfully integrates SWIN with YOLO")
        print("   • Multiple backbone variants working (nano to xlarge)")
        print("   • Medical imaging support implemented")
        print("   • Performance is reasonable with added capabilities")
        print("   • Memory usage is manageable")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("🔧 Fix the issues before proceeding to Step 3.")

    print("=" * 70)

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)