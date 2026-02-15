#!/usr/bin/env python3
"""Re-export MobileNet V3 Small slices with current ExecuTorch source."""

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
        filename = f"v4_mobilenet_slice_{name}_{fp_label}_{backend}.pte"
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
    input_224 = torch.ones(1, 3, 224, 224)
    manifest = {"tests": []}

    for n_layers in [1, 2, 3, 4]:
        logging.info(f"\n{'='*60}")
        logging.info(f"MobileNet slice {n_layers} (features[0:{n_layers}])")

        model = MobileNetSlice(n_layers)
        model.eval()
        with torch.no_grad():
            ref = model(input_224)
        stats = ref_stats(ref)
        logging.info(f"  Output: {stats['shape']}")
        logging.info(f"  mean={stats['mean']}, range=[{stats['min']}, {stats['max']}]")
        logging.info(f"  first8: {stats['first8']}")

        test_entry = {
            "name": f"mobilenet_slice_{n_layers}",
            "description": f"MobileNet V3 Small, features[0:{n_layers}]",
            "input_shape": [1, 3, 224, 224],
            "cpu_ref": stats,
        }

        # Vulkan FP32
        vk_path, _ = export_model(str(n_layers), model, input_224, "vulkan")
        test_entry["vulkan_asset"] = os.path.basename(vk_path) if vk_path else None

        # XNNPACK FP32
        xn_path, _ = export_model(str(n_layers), model, input_224, "xnnpack")
        test_entry["xnnpack_asset"] = os.path.basename(xn_path) if xn_path else None

        manifest["tests"].append(test_entry)

    manifest_path = os.path.join(OUTPUT_DIR, "v4_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
