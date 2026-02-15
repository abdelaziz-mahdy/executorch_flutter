#!/usr/bin/env python3
"""Export larger conv models to test spatial size dependency on PowerVR."""

import os
import json
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../example/assets/debug_models",
)


class LargeConv(nn.Module):
    """Single conv with configurable input size."""
    def __init__(self, in_ch=3, out_ch=16, kernel=3, stride=2, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2, bias=bias)
        nn.init.constant_(self.conv.weight, 0.1)
        if bias:
            nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        return self.conv(x)


class ConvHardswish(nn.Module):
    """Conv followed by Hardswish."""
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.conv(x))


class ConvBNHardswish(nn.Module):
    """Conv + BatchNorm + Hardswish (like MobileNet stem)."""
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MobileNetStem(nn.Module):
    """Exact MobileNet V3 Small stem (real weights)."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT,
        )
        mobilenet.eval()
        # features[0] is Conv2dNormActivation = Conv2d + BN + Hardswish
        self.stem = mobilenet.features[0]

    def forward(self, x):
        return self.stem(x)


def export_model(name, model, example_input, backend, force_fp16=False):
    model.eval()
    with torch.no_grad():
        ref = model(example_input)

    try:
        program = export(model, (example_input,), strict=True)

        if backend == "vulkan":
            opts = {"texture_limits": (2048, 2048, 2048)}
            if force_fp16:
                opts["force_fp16"] = True
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[VulkanPartitioner(compile_options=opts)],
            )
        else:
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[XnnpackPartitioner()],
            )

        exec_prog = edge.to_executorch()
        fp_label = "fp16" if force_fp16 else "fp32"
        filename = f"v3_{name}_{fp_label}_{backend}.pte"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(exec_prog.buffer)

        size_kb = os.path.getsize(path) / 1024
        logging.info(f"  [{backend:7s}] {filename} ({size_kb:.1f} KB)")
        return path, ref
    except Exception as e:
        logging.error(f"  [{backend:7s}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, ref


def ref_stats(t):
    flat = t.flatten()
    return {
        "shape": list(t.shape),
        "mean": round(t.mean().item(), 6),
        "min": round(t.min().item(), 6),
        "max": round(t.max().item(), 6),
        "first8": [round(v, 6) for v in flat[:8].tolist()],
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = {"tests": []}

    tests = [
        # Test 1: Large conv, uniform weights, no activation (spatial size test)
        ("large_conv_nobias", LargeConv(bias=False),
         torch.ones(1, 3, 224, 224),
         "3x3 stride=2, w=0.1, NO bias, 224x224 input"),

        # Test 2: Large conv with bias
        ("large_conv_bias", LargeConv(bias=True),
         torch.ones(1, 3, 224, 224),
         "3x3 stride=2, w=0.1, bias=0.5, 224x224 input"),

        # Test 3: Conv + Hardswish (is Hardswish the problem?)
        ("conv_hardswish", ConvHardswish(),
         torch.ones(1, 3, 224, 224),
         "3x3 stride=2, w=0.1, bias=0.5, Hardswish, 224x224"),

        # Test 4: Conv + BN + Hardswish with random weights (like stem)
        ("conv_bn_hs", ConvBNHardswish(),
         torch.ones(1, 3, 224, 224),
         "3x3 stride=2, BN, Hardswish, random weights, 224x224"),

        # Test 5: Exact MobileNet stem (real pretrained weights)
        ("mobilenet_stem", MobileNetStem(),
         torch.ones(1, 3, 224, 224),
         "MobileNet V3 Small stem (real weights), 224x224"),

        # Test 6: Medium size (64x64) - is there a spatial threshold?
        ("medium_conv", LargeConv(bias=True),
         torch.ones(1, 3, 64, 64),
         "3x3 stride=2, w=0.1, bias=0.5, 64x64 input"),
    ]

    for name, model, inp, desc in tests:
        logging.info(f"\n{'='*60}")
        logging.info(f"{name}: {desc}")

        model.eval()
        with torch.no_grad():
            ref = model(inp)
        stats = ref_stats(ref)
        logging.info(f"  CPU ref: shape={stats['shape']}, mean={stats['mean']}, "
                     f"range=[{stats['min']}, {stats['max']}]")
        logging.info(f"  first8: {stats['first8']}")

        test_entry = {
            "name": name,
            "description": desc,
            "input_shape": list(inp.shape),
            "cpu_ref": stats,
        }

        # Export Vulkan FP32
        vk_path, _ = export_model(name, model, inp, "vulkan", force_fp16=False)
        test_entry["vulkan_asset"] = os.path.basename(vk_path) if vk_path else None

        # Export XNNPACK FP32
        xn_path, _ = export_model(name, model, inp, "xnnpack")
        test_entry["xnnpack_asset"] = os.path.basename(xn_path) if xn_path else None

        manifest["tests"].append(test_entry)

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "v3_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
