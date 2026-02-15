#!/usr/bin/env python3
"""Export models to isolate scalar constant prepacking on PowerVR.

Hardswish decomposes into add(x,3) + clamp + mul + div(x,6).
The constants 3 and 6 are int32 scalars that get prepacked then cast to float32.
This script isolates each operation to find which one fails.
"""

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


class AddScalar(nn.Module):
    """x + 3 (int constant, will be prepacked as int32 then cast to float32)."""
    def forward(self, x):
        return x + 3


class AddFloatScalar(nn.Module):
    """x + 3.0 (float constant)."""
    def forward(self, x):
        return x + 3.0


class DivScalar(nn.Module):
    """x / 6 (int constant)."""
    def forward(self, x):
        return x / 6


class DivFloatScalar(nn.Module):
    """x / 6.0 (float constant)."""
    def forward(self, x):
        return x / 6.0


class MulScalar(nn.Module):
    """x * 2 (int constant)."""
    def forward(self, x):
        return x * 2


class MulFloatScalar(nn.Module):
    """x * 0.5 (float constant)."""
    def forward(self, x):
        return x * 0.5


class AddThenDiv(nn.Module):
    """(x + 3) / 6 - core of hardswish without clamp/mul."""
    def forward(self, x):
        return (x + 3) / 6


class ManualHardswish(nn.Module):
    """Manual hardswish: x * clamp(x + 3, 0, 6) / 6."""
    def forward(self, x):
        return x * torch.clamp(x + 3, 0, 6) / 6


class ClampWithTensorMax(nn.Module):
    """clamp(x, 0, tensor(6)) - using tensor constant for max."""
    def forward(self, x):
        six = torch.tensor(6.0)
        return torch.clamp(x, min=0.0, max=six)


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
        filename = f"v6_{name}_{backend}.pte"
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
        (AddScalar(), "add_int3",
         [1, 4, 8, 8], "x + 3 (int scalar constant)"),
        (AddFloatScalar(), "add_float3",
         [1, 4, 8, 8], "x + 3.0 (float scalar constant)"),
        (DivScalar(), "div_int6",
         [1, 4, 8, 8], "x / 6 (int scalar constant)"),
        (DivFloatScalar(), "div_float6",
         [1, 4, 8, 8], "x / 6.0 (float scalar constant)"),
        (MulScalar(), "mul_int2",
         [1, 4, 8, 8], "x * 2 (int scalar constant)"),
        (MulFloatScalar(), "mul_float05",
         [1, 4, 8, 8], "x * 0.5 (float scalar constant)"),
        (AddThenDiv(), "add3_div6",
         [1, 4, 8, 8], "(x + 3) / 6"),
        (ManualHardswish(), "manual_hardswish",
         [1, 4, 8, 8], "x * clamp(x+3, 0, 6) / 6"),
        (ClampWithTensorMax(), "clamp_tensor_max",
         [1, 4, 8, 8], "clamp(x, 0, tensor(6))"),
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

        vk_path, _ = export_model(name, model, example_input, "vulkan")
        test_entry["vulkan_asset"] = os.path.basename(vk_path) if vk_path else None

        xn_path, _ = export_model(name, model, example_input, "xnnpack")
        test_entry["xnnpack_asset"] = os.path.basename(xn_path) if xn_path else None

        manifest["tests"].append(test_entry)

    manifest_path = os.path.join(OUTPUT_DIR, "v6_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
