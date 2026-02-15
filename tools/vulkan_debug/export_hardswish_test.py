#!/usr/bin/env python3
"""Export minimal test models to isolate Hardswish failure on PowerVR."""

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


class JustHardswish(nn.Module):
    """Just a Hardswish activation, nothing else."""
    def forward(self, x):
        return nn.functional.hardswish(x)


class JustRelu(nn.Module):
    """Just a ReLU activation for comparison."""
    def forward(self, x):
        return nn.functional.relu(x)


class JustClamp(nn.Module):
    """Just clamp(x, 0, 6) for comparison."""
    def forward(self, x):
        return torch.clamp(x, 0, 6)


class ConvThenHardswish(nn.Module):
    """Conv2d followed by Hardswish (separate, not fused)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=True)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.5)

    def forward(self, x):
        y = self.conv(x)
        return nn.functional.hardswish(y)


class ConvHardswishFused(nn.Module):
    """Conv2d + BN + Hardswish (MobileNet stem pattern)."""
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )
        # Set fixed weights for reproducibility
        nn.init.constant_(self.block[0].weight, 0.1)
        # Set BN to identity-like
        nn.init.constant_(self.block[1].weight, 1.0)
        nn.init.constant_(self.block[1].bias, 0.0)
        nn.init.constant_(self.block[1].running_mean, 0.0)
        nn.init.constant_(self.block[1].running_var, 1.0)

    def forward(self, x):
        return self.block(x)


def export_model(name, model, example_input, backend):
    model.eval()
    with torch.no_grad():
        ref = model(example_input)

    try:
        program = export(model, (example_input,), strict=True)

        if backend == "vulkan":
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[VulkanPartitioner(
                    compile_options={"texture_limits": (2048, 2048, 2048)}
                )],
            )
        else:
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[XnnpackPartitioner()],
            )

        exec_prog = edge.to_executorch()
        filename = f"v5_{name}_{backend}.pte"
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

    # Test cases: model, name, input shape, description
    tests = [
        (JustHardswish(), "just_hardswish_small",
         [1, 4, 8, 8], "Pure Hardswish, small (8x8)"),
        (JustHardswish(), "just_hardswish_large",
         [1, 16, 112, 112], "Pure Hardswish, large (112x112)"),
        (JustRelu(), "just_relu",
         [1, 16, 112, 112], "Pure ReLU, large (112x112)"),
        (JustClamp(), "just_clamp",
         [1, 16, 112, 112], "Pure clamp(0,6), large (112x112)"),
        (ConvThenHardswish(), "conv_then_hs",
         [1, 3, 224, 224], "Conv2d then Hardswish (separate)"),
        (ConvHardswishFused(), "conv_bn_hs_fixed",
         [1, 3, 224, 224], "Conv2d+BN+Hardswish (fixed weights)"),
    ]

    for model, name, input_shape, desc in tests:
        logging.info(f"\n{'='*60}")
        logging.info(f"{name}: {desc}")

        example_input = torch.ones(*input_shape)
        model.eval()
        with torch.no_grad():
            ref = model(example_input)
        stats = ref_stats(ref)
        logging.info(f"  Output: {stats['shape']}")
        logging.info(f"  mean={stats['mean']}, range=[{stats['min']}, {stats['max']}]")
        logging.info(f"  first8: {stats['first8']}")

        test_entry = {
            "name": name,
            "description": desc,
            "input_shape": input_shape,
            "cpu_ref": stats,
        }

        # Vulkan
        vk_path, _ = export_model(name, model, example_input, "vulkan")
        test_entry["vulkan_asset"] = os.path.basename(vk_path) if vk_path else None

        # XNNPACK
        xn_path, _ = export_model(name, model, example_input, "xnnpack")
        test_entry["xnnpack_asset"] = os.path.basename(xn_path) if xn_path else None

        manifest["tests"].append(test_entry)

    manifest_path = os.path.join(OUTPUT_DIR, "v5_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
