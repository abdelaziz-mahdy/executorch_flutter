"""Comprehensive Vulkan diagnostic: test ALL variants to isolate PowerVR failure.

Test Matrix:
  1. Dtype: FP32 vs FP16 (force_fp16)
  2. Bias: with bias vs without bias
  3. Conv type: standard 3x3, pointwise 1x1, depthwise
  4. Progressive MobileNet: layers 0-N slices

This systematically isolates which variable causes the divergence.

Usage:
  cd /Users/AbdelazizMahdy/flutter_projects/executorch/executorch
  python3 /path/to/comprehensive_diagnostic.py
"""

import os
import sys
import json
import logging
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch')

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

OUTPUT_DIR = '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/assets/debug_models'


# =============================================================================
# Diagnostic models
# =============================================================================

class Conv2dNoBias(nn.Module):
    """Conv2d 3->4, 3x3, stride=2, padding=1, weight=0.1, NO bias."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
        nn.init.constant_(self.conv.weight, 0.1)

    def forward(self, x):
        return self.conv(x)


class Conv2dWithBias(nn.Module):
    """Conv2d 3->4, 3x3, stride=2, padding=1, weight=0.1, bias=0.5."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class Conv2dLargeBias(nn.Module):
    """Conv2d 3->32, 3x3, stride=2, padding=1, weight=0.1, bias=0.5.
    32 output channels means bias texture is 8x1 (more texels to test)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class PointwiseNoBias(nn.Module):
    """Conv2d 3->4, 1x1, weight=0.1, NO bias."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        nn.init.constant_(self.conv.weight, 0.1)

    def forward(self, x):
        return self.conv(x)


class PointwiseWithBias(nn.Module):
    """Conv2d 3->4, 1x1, weight=0.1, bias=0.5."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class DepthwiseNoBias(nn.Module):
    """Depthwise Conv2d 4->4, 3x3, groups=4, weight=0.1, NO bias."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
        nn.init.constant_(self.conv.weight, 0.1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseWithBias(nn.Module):
    """Depthwise Conv2d 4->4, 3x3, groups=4, weight=0.1, bias=0.5."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class AddBiasOnly(nn.Module):
    """Just adds a constant (0.5) to input. Tests if scalar/constant loading works."""
    def __init__(self):
        super().__init__()
        self.register_buffer('bias', torch.full((1, 4, 1, 1), 0.5))

    def forward(self, x):
        return x + self.bias


class MobileNetSlice(nn.Module):
    """MobileNet V3 Small sliced at layer N."""
    def __init__(self, n_layers):
        super().__init__()
        import torchvision.models as models
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT,
        )
        mobilenet.eval()
        self.features = nn.Sequential(*list(mobilenet.features.children())[:n_layers])

    def forward(self, x):
        return self.features(x)


# =============================================================================
# Export helpers
# =============================================================================

def export_model(name, model, example_input, backend, force_fp16=False):
    """Export model with given backend. Returns (path, ref_output) or (None, None)."""
    model.eval()
    with torch.no_grad():
        ref_output = model(example_input)

    try:
        program = export(model, (example_input,), strict=True)

        if backend == "vulkan":
            vulkan_options = {"texture_limits": (2048, 2048, 2048)}
            if force_fp16:
                vulkan_options["force_fp16"] = True
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

        fp_suffix = "_fp16" if force_fp16 else "_fp32"
        filename = f"v2_{name}{fp_suffix}_{backend}.pte"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(exec_prog.buffer)

        size_kb = os.path.getsize(path) / 1024
        logging.info(f"  [{backend:7s}] {filename} ({size_kb:.1f} KB)")
        return path, ref_output

    except Exception as e:
        logging.error(f"  [{backend:7s}] FAILED: {e}")
        traceback.print_exc()
        return None, ref_output


def ref_stats(ref_output):
    """Return stats dict for reference output."""
    flat = ref_output.flatten()
    return {
        "shape": list(ref_output.shape),
        "mean": round(ref_output.mean().item(), 6),
        "min": round(ref_output.min().item(), 6),
        "max": round(ref_output.max().item(), 6),
        "first8": [round(v, 6) for v in flat[:8].tolist()],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_8x8_3ch = torch.ones(1, 3, 8, 8)
    input_8x8_4ch = torch.ones(1, 4, 8, 8)
    input_224 = torch.ones(1, 3, 224, 224)

    # Collect all test metadata
    manifest = {"tests": []}

    # =========================================================================
    # PART 1: Conv variants - FP32 and FP16 x with/without bias
    # =========================================================================
    conv_tests = [
        # (name, model, input, description, expected_interior)
        ("conv3x3_nobias", Conv2dNoBias(), input_8x8_3ch,
         "3x3 stride=2, w=0.1, NO bias", 2.7),
        ("conv3x3_bias", Conv2dWithBias(), input_8x8_3ch,
         "3x3 stride=2, w=0.1, bias=0.5", 3.2),
        ("conv3x3_bias32ch", Conv2dLargeBias(), input_8x8_3ch,
         "3x3 stride=2, w=0.1, bias=0.5, 32 out channels", 3.2),
        ("pw_nobias", PointwiseNoBias(), input_8x8_3ch,
         "1x1 pointwise, w=0.1, NO bias", 0.3),
        ("pw_bias", PointwiseWithBias(), input_8x8_3ch,
         "1x1 pointwise, w=0.1, bias=0.5", 0.8),
        ("dw_nobias", DepthwiseNoBias(), input_8x8_4ch,
         "Depthwise 3x3, w=0.1, NO bias", None),
        ("dw_bias", DepthwiseWithBias(), input_8x8_4ch,
         "Depthwise 3x3, w=0.1, bias=0.5", None),
        ("add_bias_only", AddBiasOnly(), input_8x8_4ch,
         "Just add constant 0.5 (no conv)", 1.5),
    ]

    logging.info("=" * 70)
    logging.info("PART 1: Conv2d variants - FP32 and FP16")
    logging.info("=" * 70)

    for name, model, inp, desc, expected in conv_tests:
        for fp16 in [False, True]:
            dtype_label = "FP16" if fp16 else "FP32"
            full_name = name
            logging.info(f"\n--- {full_name} ({dtype_label}) ---")
            logging.info(f"  {desc}")

            model.eval()
            with torch.no_grad():
                ref = model(inp)
            stats = ref_stats(ref)
            logging.info(f"  CPU ref: mean={stats['mean']}, range=[{stats['min']}, {stats['max']}]")

            test_entry = {
                "name": full_name,
                "dtype": dtype_label,
                "description": desc,
                "input_shape": list(inp.shape),
                "expected_interior": expected,
                "cpu_ref": stats,
            }

            # Export Vulkan (both FP32 and FP16)
            vk_path, _ = export_model(full_name, model, inp, "vulkan", force_fp16=fp16)
            test_entry["vulkan_asset"] = os.path.basename(vk_path) if vk_path else None

            # Export XNNPACK (only FP32 - XNNPACK doesn't have force_fp16)
            if not fp16:
                xn_path, _ = export_model(full_name, model, inp, "xnnpack", force_fp16=False)
                test_entry["xnnpack_asset"] = os.path.basename(xn_path) if xn_path else None

            manifest["tests"].append(test_entry)

    # =========================================================================
    # PART 2: Progressive MobileNet slices
    # =========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("PART 2: Progressive MobileNet V3 Small slices")
    logging.info("=" * 70)

    # MobileNet V3 Small has 13 feature blocks (features[0] through features[12])
    # features[0] = Conv2d + BN + Hardswish
    # features[1] = InvertedResidual
    # features[2] = InvertedResidual
    # ...
    for n_layers in [1, 2, 3, 4]:
        name = f"mobilenet_slice_{n_layers}"
        logging.info(f"\n--- {name} (layers 0-{n_layers-1}) ---")

        try:
            model = MobileNetSlice(n_layers)
            model.eval()

            with torch.no_grad():
                ref = model(input_224)
            stats = ref_stats(ref)
            logging.info(f"  Output shape: {list(ref.shape)}")
            logging.info(f"  CPU ref: mean={stats['mean']}, range=[{stats['min']}, {stats['max']}]")

            for fp16 in [False, True]:
                dtype_label = "FP16" if fp16 else "FP32"
                test_entry = {
                    "name": name,
                    "dtype": dtype_label,
                    "description": f"MobileNet V3 Small, first {n_layers} blocks",
                    "input_shape": [1, 3, 224, 224],
                    "expected_interior": None,
                    "cpu_ref": stats,
                }

                vk_path, _ = export_model(name, model, input_224, "vulkan", force_fp16=fp16)
                test_entry["vulkan_asset"] = os.path.basename(vk_path) if vk_path else None

                if not fp16:
                    xn_path, _ = export_model(name, model, input_224, "xnnpack", force_fp16=False)
                    test_entry["xnnpack_asset"] = os.path.basename(xn_path) if xn_path else None

                manifest["tests"].append(test_entry)

        except Exception as e:
            logging.error(f"  FAILED: {e}")
            traceback.print_exc()

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "v2_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"\nSaved manifest: {manifest_path}")

    # Print summary
    logging.info("\n" + "=" * 70)
    logging.info("EXPORT SUMMARY")
    logging.info("=" * 70)
    total = len(manifest["tests"])
    exported = sum(1 for t in manifest["tests"] if t.get("vulkan_asset"))
    logging.info(f"  Total test configs: {total}")
    logging.info(f"  Successfully exported: {exported}")
    logging.info(f"\nRun on device:")
    logging.info(f"  cd example")
    logging.info(f"  flutter test integration_test/vulkan_comprehensive_test.dart -d <device>")


if __name__ == "__main__":
    with torch.no_grad():
        main()
