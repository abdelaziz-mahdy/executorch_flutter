# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Progressive MobileNet V3 Small debug script for Vulkan backend.
# Exports partial models to find which operator produces NaN/incorrect outputs.
#
# Usage:
#   # Export all partial models (saves .pte files to output dir)
#   python mobilenet_debug.py --output-dir /path/to/output
#
#   # Export and test on host (no device needed)
#   python mobilenet_debug.py --test-host
#
#   # Export and test on Android device via adb
#   python mobilenet_debug.py --test-device

import argparse
import logging
import os
import subprocess
import tempfile

import torch
import torch.nn as nn
import torchvision.models as models

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


# =============================================================================
# Partial model wrappers - progressively larger slices of MobileNet V3 Small
# =============================================================================


class FeaturesSlice(nn.Module):
    """First N feature blocks of MobileNet V3 Small."""

    def __init__(self, model, end_idx):
        super().__init__()
        self.features = nn.Sequential(*list(model.features.children())[:end_idx])

    def forward(self, x):
        return self.features(x)


class FeaturesOnly(nn.Module):
    """All 13 feature blocks (full backbone)."""

    def __init__(self, model):
        super().__init__()
        self.features = model.features

    def forward(self, x):
        return self.features(x)


class FeaturesAndPool(nn.Module):
    """Features + adaptive avg pool."""

    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ClassifierOnly(nn.Module):
    """Just the classifier (takes flattened 576-d input)."""

    def __init__(self, model):
        super().__init__()
        self.classifier = model.classifier

    def forward(self, x):
        return self.classifier(x)


class FullModel(nn.Module):
    """Complete MobileNet V3 Small."""

    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# Export and test helpers
# =============================================================================


def export_partial(name, model, example_inputs, output_dir, vulkan_options=None):
    """Export a partial model to .pte with Vulkan delegate."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    from executorch.extension.export_util.utils import save_pte_program
    from torch.export import export

    model.eval()
    safe_name = name.lower().replace(" ", "_").replace("+", "_")

    logging.info(f"Exporting: {name}")

    program = export(model, example_inputs, strict=True)

    if vulkan_options is None:
        vulkan_options = {
            "texture_limits": (2048, 2048, 2048),
            "force_fp16": True,
        }

    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[VulkanPartitioner(vulkan_options)],
    )

    exec_prog = edge_program.to_executorch()

    filename = f"mobilenet_debug_{safe_name}_vulkan"
    os.makedirs(output_dir, exist_ok=True)
    save_pte_program(exec_prog, filename, output_dir)
    filepath = os.path.join(output_dir, f"{filename}.pte")
    logging.info(f"  Saved: {filepath} ({os.path.getsize(filepath) / 1024:.0f} KB)")

    return exec_prog, filepath


def test_on_host(name, model, exec_prog, example_inputs, atol=1e-3, rtol=1e-1):
    """Test partial model on host device (CPU Vulkan or fallback)."""
    import executorch.backends.vulkan.test.utils as test_utils

    logging.info(f"Testing on host: {name}")

    result = test_utils.run_and_check_output(
        reference_model=model,
        executorch_program=exec_prog,
        sample_inputs=example_inputs,
        atol=atol,
        rtol=rtol,
    )

    if result:
        logging.info(f"  [PASS] {name}")
    else:
        logging.error(f"  [FAIL] {name}")

    return result


def test_on_device(name, pte_path, example_inputs, reference_outputs):
    """Push .pte to Android device via adb and check outputs."""
    logging.info(f"Testing on device: {name}")

    device_dir = "/data/local/tmp/mobilenet_debug"
    device_pte = f"{device_dir}/{os.path.basename(pte_path)}"

    # Push model
    subprocess.run(["adb", "shell", f"mkdir -p {device_dir}"], check=True)
    subprocess.run(["adb", "push", pte_path, device_pte], check=True)

    # Save input tensor
    input_tensor = example_inputs[0]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        input_bytes = input_tensor.numpy().tobytes()
        f.write(input_bytes)
        local_input = f.name

    device_input = f"{device_dir}/input.bin"
    subprocess.run(["adb", "push", local_input, device_input], check=True)
    os.unlink(local_input)

    # Run executor_runner on device
    result = subprocess.run(
        [
            "adb",
            "shell",
            f"cd {device_dir} && ./executor_runner "
            f"--model_path {device_pte} "
            f"--input_path {device_input}",
        ],
        capture_output=True,
        text=True,
    )

    logging.info(f"  stdout: {result.stdout[:500]}")
    if result.returncode != 0:
        logging.error(f"  [FAIL] {name} - executor_runner failed: {result.stderr[:500]}")
        return False

    # Check for NaN in output
    if "nan" in result.stdout.lower() or "inf" in result.stdout.lower():
        logging.error(f"  [FAIL] {name} - NaN/Inf detected in output")
        return False

    logging.info(f"  [PASS] {name} - executor_runner completed")
    return True


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="MobileNet V3 Small Vulkan debug")
    parser.add_argument(
        "--output-dir",
        default="./mobilenet_debug_models",
        help="Directory to save .pte files",
    )
    parser.add_argument(
        "--test-host",
        action="store_true",
        help="Test exported models on host",
    )
    parser.add_argument(
        "--test-device",
        action="store_true",
        help="Test exported models on Android device via adb",
    )
    args = parser.parse_args()

    logging.info("Loading MobileNet V3 Small (pretrained)...")
    base_model = models.mobilenet_v3_small(weights="DEFAULT").eval()

    # Standard input
    example_input = torch.randn(1, 3, 224, 224)

    # Compute intermediate outputs for testing isolated components
    with torch.no_grad():
        features_out = base_model.features(example_input)
        pooled = base_model.avgpool(features_out)
        flattened = torch.flatten(pooled, 1)

    # Define progressive test cases:
    # (name, model_wrapper, input_tuple)
    test_cases = [
        # First conv block only
        ("features_0_1", FeaturesSlice(base_model, 1), (example_input,)),
        # First 2 blocks
        ("features_0_2", FeaturesSlice(base_model, 2), (example_input,)),
        # First 4 blocks (through first channel expansion)
        ("features_0_4", FeaturesSlice(base_model, 4), (example_input,)),
        # First 7 blocks
        ("features_0_7", FeaturesSlice(base_model, 7), (example_input,)),
        # First 9 blocks
        ("features_0_9", FeaturesSlice(base_model, 9), (example_input,)),
        # All 13 feature blocks
        ("features_all", FeaturesOnly(base_model), (example_input,)),
        # Features + avgpool + flatten
        ("features_pool", FeaturesAndPool(base_model), (example_input,)),
        # Classifier only (small, should definitely work)
        ("classifier_only", ClassifierOnly(base_model), (flattened,)),
        # Full model
        ("full_model", FullModel(base_model), (example_input,)),
    ]

    results = {}
    first_failure = None

    for name, model, inputs in test_cases:
        model.eval()

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*inputs)

        # Export
        exec_prog, pte_path = export_partial(name, model, inputs, args.output_dir)

        # Test on host if requested
        if args.test_host:
            passed = test_on_host(name, model, exec_prog, inputs)
            results[name] = passed
            if not passed and first_failure is None:
                first_failure = name

        # Test on device if requested
        if args.test_device:
            passed = test_on_device(name, pte_path, inputs, ref_output)
            results[name] = passed
            if not passed and first_failure is None:
                first_failure = name

    # Summary
    logging.info(f"\n{'=' * 60}")
    logging.info("SUMMARY")
    logging.info(f"{'=' * 60}")

    if results:
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logging.info(f"  {name}: {status}")

        if first_failure:
            logging.info(f"\nFirst failure: {first_failure}")
        else:
            logging.info("\nAll partial models passed!")
    else:
        logging.info(f"Exported {len(test_cases)} models to {args.output_dir}")
        logging.info("Use --test-host or --test-device to run tests")

    # Print paths for Flutter integration test
    logging.info(f"\n{'=' * 60}")
    logging.info("EXPORTED MODELS (for Flutter integration test)")
    logging.info(f"{'=' * 60}")
    for name, _, _ in test_cases:
        safe_name = name.lower().replace(" ", "_").replace("+", "_")
        logging.info(f"  mobilenet_debug_{safe_name}_vulkan.pte")


if __name__ == "__main__":
    with torch.no_grad():
        main()
