"""Export targeted conv2d diagnostic models to isolate PowerVR divergence.

On PowerVR GPU (Pixel 10 Pro), Conv2d(3->16, 3x3, stride=2, padding=1)
with real MobileNet weights produces XNNPACK mean=1.17 vs Vulkan mean=-0.03.

This script exports a suite of minimal conv2d models with controlled weights
so we can pinpoint exactly where the divergence happens:

  1. Identity weights + zero bias     -> tests basic convolution correctness
  2. All-ones weights + zero bias     -> output = sum of input region (verifiable)
  3. All-ones weights + known bias    -> tests bias addition path
  4. 1x1 pointwise (no spatial)       -> tests weight loading without spatial conv
  5. Depthwise conv2d                 -> tests groups code path
  6. Small constant weights + bias    -> controllable non-trivial case

Each model is exported with both Vulkan and XNNPACK backends. Uses a fixed
all-ones input so expected outputs can be computed analytically.

Usage:
  cd /Users/AbdelazizMahdy/flutter_projects/executorch/executorch
  python3 /path/to/conv_diagnostic.py
"""

import os
import sys
import json
import logging

import torch
import torch.nn as nn

# Add ExecuTorch source to path
sys.path.insert(0, '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch')

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

OUTPUT_DIR = '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/assets/debug_models'


# =============================================================================
# Diagnostic conv2d modules
# =============================================================================


class IdentityConv2d(nn.Module):
    """Conv2d 3->3, 3x3, padding=1, identity-like weights, zero bias.

    For a 3-channel input, each output channel copies one input channel.
    Weight shape: [3, 3, 3, 3]. We set the center tap of the spatial kernel
    to 1.0 for the matching channel, 0 elsewhere.

    With ones input, each output element should be exactly 1.0.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        nn.init.zeros_(self.conv.weight)
        # Set center spatial tap: weight[oc, ic, 1, 1] = 1.0 when oc == ic
        for c in range(3):
            self.conv.weight.data[c, c, 1, 1] = 1.0

    def forward(self, x):
        return self.conv(x)


class OnesConv2d(nn.Module):
    """Conv2d 3->4, 3x3, stride=2, padding=1, all-ones weights, zero bias.

    Matches MobileNet first layer shape: Conv2d(3, 16, 3, stride=2, padding=1)
    but uses 4 output channels to keep output small.

    With ones input (1, 3, 8, 8):
    - Interior (no padding): each output = 3 * 3 * 3 = 27.0
    - Edges/corners will be less due to zero-padding.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class OnesConv2dWithBias(nn.Module):
    """Conv2d 3->4, 3x3, stride=2, padding=1, all-ones weights, bias=1.0.

    Same as OnesConv2d but with a known bias of 1.0 per channel.
    Interior output = 27.0 + 1.0 = 28.0.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=True)
        nn.init.ones_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 1.0)

    def forward(self, x):
        return self.conv(x)


class PointwiseConv2d(nn.Module):
    """Conv2d 3->4, 1x1 (pointwise), all-ones weights, zero bias.

    No spatial computation - just channel mixing. Tests weight loading
    without the complexity of spatial convolution kernels.

    With ones input: each output = 3 * 1.0 = 3.0.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2d(nn.Module):
    """Depthwise Conv2d 3->3, 3x3, padding=1, groups=3, all-ones weights, zero bias.

    Each channel is convolved independently (groups=in_channels).

    With ones input (interior): each output = 1 * 3 * 3 = 9.0.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class SmallWeightsConv2d(nn.Module):
    """Conv2d 3->4, 3x3, stride=2, padding=1, weight=0.1, bias=0.5.

    A non-trivial but controllable case. With ones input:
    - Interior: 27 * 0.1 + 0.5 = 3.2
    - This tests that both weight multiplication and bias addition work.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class MobileNetFirstLayerConv2d(nn.Module):
    """Conv2d(3, 16, 3, stride=2, padding=1) with real MobileNet fused-BN weights.

    This is the exact layer that diverges. Exported for direct comparison.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT,
        )
        mobilenet.eval()
        conv = mobilenet.features[0][0]
        bn = mobilenet.features[0][1]
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)

        self.conv = nn.Conv2d(
            fused.in_channels, fused.out_channels,
            fused.kernel_size, stride=fused.stride,
            padding=fused.padding, bias=True,
        )
        self.conv.weight.data.copy_(fused.weight.data)
        self.conv.bias.data.copy_(fused.bias.data)

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# Export helpers
# =============================================================================


def export_model(name, model, example_input, backend):
    """Export model with given backend. Returns (path, ref_output) or (None, None) on failure."""
    model.eval()

    with torch.no_grad():
        ref_output = model(example_input)

    try:
        program = export(model, (example_input,), strict=True)

        if backend == "vulkan":
            vulkan_options = {"texture_limits": (2048, 2048, 2048)}
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[VulkanPartitioner(compile_options=vulkan_options)],
            )
        else:
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[XnnpackPartitioner()],
            )

        exec_prog = edge.to_executorch()

        filename = f"diag_{name}_{backend}.pte"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(exec_prog.buffer)

        size_kb = os.path.getsize(path) / 1024
        logging.info(f"  [{backend:7s}] {filename} ({size_kb:.1f} KB)")
        return path, ref_output

    except Exception as e:
        logging.error(f"  [{backend:7s}] FAILED: {e}")
        return None, ref_output


def print_ref_output(ref_output, label=""):
    """Print summary of reference output tensor."""
    flat = ref_output.flatten()
    logging.info(f"  {label}shape: {list(ref_output.shape)}")
    logging.info(f"  {label}mean:  {ref_output.mean().item():.6f}")
    logging.info(f"  {label}min:   {ref_output.min().item():.6f}")
    logging.info(f"  {label}max:   {ref_output.max().item():.6f}")
    if flat.numel() <= 16:
        logging.info(f"  {label}values: {flat.tolist()}")
    else:
        logging.info(f"  {label}first8: {flat[:8].tolist()}")
        logging.info(f"  {label}last8:  {flat[-8:].tolist()}")


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fixed input: all ones, small spatial size for fast inference
    input_8x8 = torch.ones(1, 3, 8, 8)

    # For MobileNet layer: use 224x224 to match real usage, but also 8x8
    input_224 = torch.ones(1, 3, 224, 224)

    # Collect expected outputs for the Flutter test
    expected = {}

    test_cases = [
        # (name, model_class, input, description)
        ("identity", IdentityConv2d(), input_8x8,
         "Identity weights: each output = matching input channel. Expected: all 1.0"),

        ("ones_nobias", OnesConv2d(), input_8x8,
         "All-ones weights, no bias. Interior expected: 27.0"),

        ("ones_bias1", OnesConv2dWithBias(), input_8x8,
         "All-ones weights, bias=1.0. Interior expected: 28.0"),

        ("pointwise", PointwiseConv2d(), input_8x8,
         "1x1 pointwise, ones weights. Expected: all 3.0"),

        ("depthwise", DepthwiseConv2d(), input_8x8,
         "Depthwise 3x3, ones weights. Interior expected: 9.0"),

        ("small_weights", SmallWeightsConv2d(), input_8x8,
         "Weight=0.1, bias=0.5. Interior expected: 3.2"),

        ("mobilenet_real", MobileNetFirstLayerConv2d(), input_8x8,
         "Real MobileNet fused-BN weights, small input. Compare Vulkan vs XNNPACK."),

        ("mobilenet_real_224", MobileNetFirstLayerConv2d(), input_224,
         "Real MobileNet fused-BN weights, 224x224 input. Full-size comparison."),
    ]

    logging.info(f"Exporting {len(test_cases)} diagnostic conv2d models")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Input: all-ones tensor")
    logging.info("")

    for name, model, inp, desc in test_cases:
        logging.info(f"{'=' * 60}")
        logging.info(f"Model: {name}")
        logging.info(f"  {desc}")
        logging.info(f"  Input shape: {list(inp.shape)}")

        # Compute and record reference output
        model.eval()
        with torch.no_grad():
            ref = model(inp)
        print_ref_output(ref, label="CPU ref ")

        expected[name] = {
            "shape": list(ref.shape),
            "mean": round(ref.mean().item(), 6),
            "min": round(ref.min().item(), 6),
            "max": round(ref.max().item(), 6),
        }

        # Export both backends
        for backend in ["vulkan", "xnnpack"]:
            export_model(name, model, inp, backend)
        logging.info("")

    # Save expected values as JSON for the Flutter test
    expected_path = os.path.join(OUTPUT_DIR, "diag_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    logging.info(f"Saved expected values: {expected_path}")

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("EXPECTED OUTPUTS (with all-ones input):")
    logging.info("=" * 60)
    logging.info("  identity:       all values = 1.0")
    logging.info("  ones_nobias:    interior = 27.0, edges < 27.0")
    logging.info("  ones_bias1:     interior = 28.0, edges < 28.0")
    logging.info("  pointwise:      all values = 3.0")
    logging.info("  depthwise:      interior = 9.0, edges < 9.0")
    logging.info("  small_weights:  interior = 3.2, edges < 3.2")
    logging.info("  mobilenet_real: compare Vulkan vs XNNPACK outputs")
    logging.info("")
    logging.info("Run on PowerVR device:")
    logging.info("  cd example")
    logging.info("  flutter test integration_test/vulkan_conv_diagnostic_test.dart -d <device>")


if __name__ == "__main__":
    with torch.no_grad():
        main()
