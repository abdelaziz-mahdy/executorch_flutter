#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# ExecuTorch Export Examples
#
# This script demonstrates how to use the generic ExecuTorch exporter
# with various model types and configurations.

import torch
import torch.nn as nn
import torchvision.models as models
from executorch_exporter import ExecuTorchExporter, ExportConfig


def example_custom_model_export():
    """Example: Export a custom PyTorch model."""
    print("🔧 Example: Custom Model Export")
    print("=" * 50)

    # Define a custom model
    class SimpleClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.features(x)

    # Create and export the model
    model = SimpleClassifier(num_classes=1000).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    exporter = ExecuTorchExporter()
    config = ExportConfig(
        model_name="simple_classifier",
        backends=["xnnpack", "portable"],
        output_dir="./examples_output",
        quantize=True
    )

    results = exporter.export_model(model, sample_inputs, config)
    print(f"Exported {len([r for r in results if r.success])} models successfully\n")


def example_pretrained_export():
    """Example: Export a pretrained model from torchvision."""
    print("📱 Example: Pretrained Model Export")
    print("=" * 50)

    # Load a pretrained model
    model = models.efficientnet_b0(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    exporter = ExecuTorchExporter()

    # Get recommended backends for mobile
    ios_backends = exporter.get_recommended_backends("ios")
    android_backends = exporter.get_recommended_backends("android")

    print(f"iOS recommended backends: {ios_backends}")
    print(f"Android recommended backends: {android_backends}")

    # Export for iOS
    config = ExportConfig(
        model_name="efficientnet_b0_ios",
        backends=ios_backends[:3],  # Limit to top 3
        output_dir="./examples_output",
        quantize=False
    )

    results = exporter.export_model(model, sample_inputs, config)
    print(f"iOS export: {len([r for r in results if r.success])}/{len(results)} successful\n")


def example_segmentation_export():
    """Example: Export a segmentation model."""
    print("🎯 Example: Segmentation Model Export")
    print("=" * 50)

    # Load segmentation model
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 512, 512),)

    exporter = ExecuTorchExporter()
    config = ExportConfig(
        model_name="deeplabv3_mobile",
        backends=["xnnpack", "vulkan"],  # GPU acceleration
        output_dir="./examples_output",
        quantize=True  # Reduce model size
    )

    results = exporter.export_model(model, sample_inputs, config)

    # Create detailed summary
    summary_path = "./examples_output/deeplabv3_mobile_summary.json"
    exporter.create_export_summary(results, summary_path)
    print(f"Detailed summary saved to: {summary_path}\n")


def example_multi_input_export():
    """Example: Export a model with multiple inputs."""
    print("🔗 Example: Multi-Input Model Export")
    print("=" * 50)

    # Define a model with multiple inputs
    class MultiInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.text_encoder = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU()
            )
            self.classifier = nn.Linear(128, 10)

        def forward(self, image, text_features):
            img_features = self.image_encoder(image)
            txt_features = self.text_encoder(text_features)
            combined = torch.cat([img_features, txt_features], dim=1)
            return self.classifier(combined)

    model = MultiInputModel().eval()
    sample_inputs = (
        torch.randn(1, 3, 224, 224),  # Image input
        torch.randn(1, 100)           # Text features
    )

    exporter = ExecuTorchExporter()
    config = ExportConfig(
        model_name="multi_input_model",
        backends=["portable", "xnnpack"],
        output_dir="./examples_output"
    )

    results = exporter.export_model(model, sample_inputs, config)
    print(f"Multi-input export: {len([r for r in results if r.success])}/{len(results)} successful\n")


def example_platform_specific_export():
    """Example: Platform-specific optimizations."""
    print("🎨 Example: Platform-Specific Export")
    print("=" * 50)

    model = models.mobilenet_v3_small(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    exporter = ExecuTorchExporter()

    # Export optimized for different platforms
    platforms = ["ios", "android", "embedded"]

    for platform in platforms:
        backends = exporter.get_recommended_backends(platform)
        if not backends:
            print(f"No backends available for {platform}")
            continue

        config = ExportConfig(
            model_name=f"mobilenet_v3_{platform}",
            backends=backends[:2],  # Top 2 backends
            output_dir="./examples_output",
            quantize=platform in ["android", "embedded"]  # Quantize for resource-constrained platforms
        )

        results = exporter.export_model(model, sample_inputs, config)
        successful = len([r for r in results if r.success])
        print(f"{platform.capitalize()} export: {successful}/{len(results)} successful")

    print()


def example_from_saved_model():
    """Example: Export from a saved PyTorch model file."""
    print("💾 Example: Export from Saved Model")
    print("=" * 50)

    # First, save a model (this would normally be done elsewhere)
    model = models.resnet18(weights='DEFAULT')
    model_path = "./examples_output/resnet18.pth"
    torch.save(model, model_path)

    # Now load and export it
    from executorch_exporter import load_model_from_path, create_sample_inputs

    loaded_model = load_model_from_path(model_path)
    sample_inputs = create_sample_inputs(
        input_shapes=[[1, 3, 224, 224]],
        input_dtypes=["float32"]
    )

    exporter = ExecuTorchExporter()
    config = ExportConfig(
        model_name="resnet18_from_file",
        backends=["xnnpack", "portable"],
        output_dir="./examples_output"
    )

    results = exporter.export_model(loaded_model, sample_inputs, config)
    print(f"Loaded model export: {len([r for r in results if r.success])}/{len(results)} successful\n")


def show_available_backends():
    """Display available backends and their information."""
    print("🔍 Available Backends Information")
    print("=" * 50)

    exporter = ExecuTorchExporter()

    print("Backend availability:")
    for backend, available in exporter.available_backends.items():
        status = "✓" if available else "✗"
        info = exporter.BACKEND_INFO.get(backend, {})
        platforms = ", ".join(info.get("platforms", []))
        description = info.get("description", "No description")
        print(f"  {status} {backend:10} - {description} ({platforms})")

    print(f"\nTotal available backends: {sum(exporter.available_backends.values())}")
    print()


def main():
    """Run all examples."""
    print("🚀 ExecuTorch Generic Exporter Examples")
    print("=" * 60)
    print()

    # Show available backends first
    show_available_backends()

    # Run examples
    example_custom_model_export()
    example_pretrained_export()
    example_segmentation_export()
    example_multi_input_export()
    example_platform_specific_export()
    example_from_saved_model()

    print("✅ All examples completed!")
    print("Check './examples_output/' directory for exported models")


if __name__ == "__main__":
    main()