#!/usr/bin/env python3
"""Export models to isolate the multi-op chain failure on PowerVR.

Single scalar ops pass but combining them fails.
This script tests various combinations to find the exact failure point.
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


class AddFloatDivFloat(nn.Module):
    """(x + 3.0) / 6.0 - both float constants, no dim_order_copy."""
    def forward(self, x):
        return (x + 3.0) / 6.0


class AddIntDivFloat(nn.Module):
    """(x + 3) / 6.0 - int add, float div."""
    def forward(self, x):
        return (x + 3) / 6.0


class AddFloatDivInt(nn.Module):
    """(x + 3.0) / 6 - float add, int div."""
    def forward(self, x):
        return (x + 3.0) / 6


class TwoAdds(nn.Module):
    """(x + 3) + 2 - two int add ops."""
    def forward(self, x):
        return (x + 3) + 2


class TwoAddsFloat(nn.Module):
    """(x + 3.0) + 2.0 - two float add ops."""
    def forward(self, x):
        return (x + 3.0) + 2.0


class AddThenMul(nn.Module):
    """(x + 1.0) * 2.0 - float add then float mul."""
    def forward(self, x):
        return (x + 1.0) * 2.0


class MulThenAdd(nn.Module):
    """x * 2.0 + 1.0 - float mul then float add."""
    def forward(self, x):
        return x * 2.0 + 1.0


class ThreeOpsFloat(nn.Module):
    """(x + 1.0) * 2.0 - 3.0 - three float ops."""
    def forward(self, x):
        return (x + 1.0) * 2.0 - 3.0


class AddSelf(nn.Module):
    """x + x - no constants at all, just binary op on same tensor."""
    def forward(self, x):
        return x + x


class MulSelf(nn.Module):
    """x * x - no constants."""
    def forward(self, x):
        return x * x


class AddSelfThenDiv(nn.Module):
    """(x + x) / 2.0 - binary without constants, then div with float constant."""
    def forward(self, x):
        return (x + x) / 2.0


class AddSelfThenDivInt(nn.Module):
    """(x + x) / 2 - binary without constants, then div with int constant."""
    def forward(self, x):
        return (x + x) / 2


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
        filename = f"v7_{name}_{backend}.pte"
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
        (AddFloatDivFloat(), "add_f_div_f",
         [1, 4, 8, 8], "(x+3.0)/6.0 - both float"),
        (AddIntDivFloat(), "add_i_div_f",
         [1, 4, 8, 8], "(x+3)/6.0 - int add, float div"),
        (AddFloatDivInt(), "add_f_div_i",
         [1, 4, 8, 8], "(x+3.0)/6 - float add, int div"),
        (TwoAdds(), "two_adds_int",
         [1, 4, 8, 8], "(x+3)+2 - two int adds"),
        (TwoAddsFloat(), "two_adds_float",
         [1, 4, 8, 8], "(x+3.0)+2.0 - two float adds"),
        (AddThenMul(), "add_then_mul",
         [1, 4, 8, 8], "(x+1.0)*2.0 - add then mul"),
        (MulThenAdd(), "mul_then_add",
         [1, 4, 8, 8], "x*2.0+1.0 - mul then add"),
        (ThreeOpsFloat(), "three_ops",
         [1, 4, 8, 8], "(x+1.0)*2.0-3.0 - three float ops"),
        (AddSelf(), "add_self",
         [1, 4, 8, 8], "x+x - no constants"),
        (MulSelf(), "mul_self",
         [1, 4, 8, 8], "x*x - no constants"),
        (AddSelfThenDiv(), "add_self_div_f",
         [1, 4, 8, 8], "(x+x)/2.0 - self add then float div"),
        (AddSelfThenDivInt(), "add_self_div_i",
         [1, 4, 8, 8], "(x+x)/2 - self add then int div"),
    ]

    for model, name, input_shape, desc in tests:
        logging.info(f"\n{'='*60}")
        logging.info(f"{name}: {desc}")

        example_input = torch.ones(*input_shape)
        model.eval()
        with torch.no_grad():
            ref = model(example_input)
        stats = ref_stats(ref)
        logging.info(f"  mean={stats['mean']}, first8: {stats['first8']}")

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

    manifest_path = os.path.join(OUTPUT_DIR, "v7_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
