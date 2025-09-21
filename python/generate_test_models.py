#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Generate Test Models for ExecuTorch Flutter Plugin
#
# This script generates sample models for testing the Flutter plugin
# across different platforms and backends.

import os
import sys

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import torchvision
        import executorch
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("\nPlease install requirements:")
        print("pip install -r requirements.txt")
        return False

def generate_classification_models():
    """Generate image classification models for testing."""
    print("📱 Generating classification models...")

    from executorch_exporter import ExecuTorchExporter, ExportConfig
    import torchvision.models as models
    import torch

    # MobileNetV3 Small - Good for mobile testing
    model = models.mobilenet_v3_small(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    exporter = ExecuTorchExporter()

    # iOS optimized model
    ios_config = ExportConfig(
        model_name="mobilenet_v3_small_ios",
        backends=["coreml", "mps", "xnnpack"][:2],  # Use available backends
        output_dir="../example/assets/models",
        quantize=False
    )

    # Android CPU optimized model
    android_cpu_config = ExportConfig(
        model_name="mobilenet_v3_small_android_cpu",
        backends=["xnnpack"],
        output_dir="../example/assets/models",
        quantize=False
    )

    # Android GPU optimized model (Vulkan)
    android_gpu_config = ExportConfig(
        model_name="mobilenet_v3_small_android_gpu",
        backends=["vulkan"],
        output_dir="../example/assets/models",
        quantize=False
    )

    # Export models
    ios_results = exporter.export_model(model, sample_inputs, ios_config)
    android_cpu_results = exporter.export_model(model, sample_inputs, android_cpu_config)
    android_gpu_results = exporter.export_model(model, sample_inputs, android_gpu_config)

    all_results = ios_results + android_cpu_results + android_gpu_results
    successful = sum(1 for r in all_results if r.success)
    print(f"✓ Generated {successful} classification models")

def generate_simple_demo_model():
    """Generate a very simple model for basic testing."""
    print("🔧 Generating simple demo model...")

    from executorch_exporter import ExecuTorchExporter, ExportConfig
    import torch
    import torch.nn as nn

    # Simple linear model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval()
    sample_inputs = (torch.randn(1, 10),)

    exporter = ExecuTorchExporter()
    config = ExportConfig(
        model_name="simple_demo",
        backends=["portable", "xnnpack"][:1],  # Use available
        output_dir="../example/assets/models",
        quantize=False
    )

    results = exporter.export_model(model, sample_inputs, config)
    successful = sum(1 for r in results if r.success)
    print(f"✓ Generated {successful} demo models")

def generate_models():
    """Generate all test models."""
    print("🚀 ExecuTorch Test Model Generation")
    print("=" * 50)

    # Create output directory
    os.makedirs("../example/assets/models", exist_ok=True)

    try:
        generate_simple_demo_model()
        generate_classification_models()

        print("\n✅ Model generation completed!")
        print("Models saved to: ../example/assets/models/")
        print("\nNext steps:")
        print("1. Add models to your Flutter app's pubspec.yaml")
        print("2. Test loading with ExecutorchManager.instance.loadModel()")

    except Exception as e:
        print(f"\n❌ Model generation failed: {e}")
        return 1

    return 0

def main():
    if not check_requirements():
        return 1

    return generate_models()

if __name__ == "__main__":
    exit(main())