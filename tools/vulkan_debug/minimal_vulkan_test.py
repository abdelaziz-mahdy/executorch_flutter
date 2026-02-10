# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Minimal Vulkan test models for debugging PowerVR GPU issues.
# Exports the simplest possible .pte models with Vulkan backend to isolate
# which operations produce incorrect results on PowerVR GPUs.
#
# Usage:
#   python examples/vulkan/minimal_vulkan_test.py
#   python examples/vulkan/minimal_vulkan_test.py --output-dir /path/to/output

import argparse
import logging
import os

import torch
import torch.nn as nn

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


# =============================================================================
# Minimal test modules
# =============================================================================


class BareAdd(nn.Module):
    """Simplest possible operation: x + 1.0"""

    def forward(self, x):
        return x + 1.0


class BareConv2d(nn.Module):
    """Single Conv2d without bias, weights set to averaging filter."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        # Initialize to averaging filter: 1/(3*3*3) = 1/27
        nn.init.constant_(self.conv.weight, 1.0 / (3 * 3 * 3))

    def forward(self, x):
        return self.conv(x)


class BareConv2dBias(nn.Module):
    """Single Conv2d with bias."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        nn.init.constant_(self.conv.weight, 1.0 / (3 * 3 * 3))
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class BareLinear(nn.Module):
    """Single Linear layer."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 4)
        nn.init.constant_(self.linear.weight, 0.1)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)


# =============================================================================
# Export helper
# =============================================================================


def export_model(name, model, example_inputs, output_dir):
    """Export a model to .pte with Vulkan delegate."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    from executorch.extension.export_util.utils import save_pte_program
    from torch.export import export

    model.eval()

    logging.info(f"Exporting: {name}")

    # Compute reference output on CPU
    with torch.no_grad():
        ref_output = model(*example_inputs)

    # Print reference output summary
    if ref_output.numel() <= 32:
        logging.info(f"  Reference output: {ref_output.flatten().tolist()}")
    else:
        flat = ref_output.flatten()
        logging.info(
            f"  Reference output (first 8): {flat[:8].tolist()}"
        )
        logging.info(
            f"  Reference output (last 8):  {flat[-8:].tolist()}"
        )
    logging.info(f"  Output shape: {list(ref_output.shape)}")
    logging.info(f"  Output mean: {ref_output.mean().item():.6f}")
    logging.info(f"  Output std:  {ref_output.std().item():.6f}")
    logging.info(f"  Output min:  {ref_output.min().item():.6f}")
    logging.info(f"  Output max:  {ref_output.max().item():.6f}")

    # Export with torch.export
    program = export(model, example_inputs, strict=True)

    # Lower with Vulkan partitioner
    vulkan_options = {
        "texture_limits": (2048, 2048, 2048),
        "force_fp16": True,
    }

    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[VulkanPartitioner(compile_options=vulkan_options)],
    )

    exec_prog = edge_program.to_executorch()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_pte_program(exec_prog, name, output_dir)
    filepath = os.path.join(output_dir, f"{name}.pte")
    size_kb = os.path.getsize(filepath) / 1024
    logging.info(f"  Saved: {filepath} ({size_kb:.1f} KB)")

    return filepath


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Export minimal Vulkan test models for PowerVR debugging"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/minimal_vulkan_models",
        help="Directory to save .pte files (default: /tmp/minimal_vulkan_models)",
    )
    args = parser.parse_args()

    # Use fixed seed for reproducible reference outputs
    torch.manual_seed(42)

    # Define test cases: (filename, model, example_inputs)
    test_cases = [
        (
            "bare_add",
            BareAdd(),
            (torch.ones(1, 3, 8, 8),),
        ),
        (
            "bare_conv2d",
            BareConv2d(),
            (torch.ones(1, 3, 8, 8),),
        ),
        (
            "bare_conv2d_bias",
            BareConv2dBias(),
            (torch.ones(1, 3, 8, 8),),
        ),
        (
            "bare_linear",
            BareLinear(),
            (torch.ones(1, 16),),
        ),
    ]

    logging.info(f"Exporting {len(test_cases)} minimal Vulkan test models")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("")

    exported_files = []
    for name, model, inputs in test_cases:
        filepath = export_model(name, model, inputs, args.output_dir)
        exported_files.append(filepath)
        logging.info("")

    # Summary
    logging.info("=" * 60)
    logging.info("EXPORT SUMMARY")
    logging.info("=" * 60)
    for filepath in exported_files:
        size_kb = os.path.getsize(filepath) / 1024
        logging.info(f"  {os.path.basename(filepath):30s} {size_kb:8.1f} KB")

    logging.info("")
    logging.info("Expected results with input=ones:")
    logging.info("  bare_add:        all values should be 2.0")
    logging.info("  bare_conv2d:     all interior values ~1.0 (avg of 27 ones * 1/27)")
    logging.info("  bare_conv2d_bias: interior ~1.5 (conv ~1.0 + bias 0.5)")
    logging.info("  bare_linear:     all values should be 1.6 (16 * 0.1 + 0.0)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
