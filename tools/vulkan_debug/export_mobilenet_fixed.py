"""Export full MobileNet V3 Small with native hardswish fix.

The key fix is in vulkan_partitioner.py: ops_not_to_decompose now includes
hardswish and hardsigmoid, so to_edge_transform_and_lower() automatically
preserves these ops instead of decomposing them into mul/add/clamp/div.

Exports:
1. Full MobileNet - Vulkan FP32 (with native hardswish/hardsigmoid)
2. Full MobileNet - XNNPACK (reference)
"""
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models

# Add executorch to path
sys.path.insert(0, '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch')

OUTPUT_DIR = '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/assets/debug_models'


class FullMobileNet(nn.Module):
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading MobileNet V3 Small (pretrained)...")
    base_model = models.mobilenet_v3_small(weights="DEFAULT").eval()
    model = FullMobileNet(base_model)
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)

    # Verify reference output
    with torch.no_grad():
        ref_output = model(example_input)
        top5 = torch.topk(ref_output[0], 5)
        print(f"Reference top-5 classes: {top5.indices.tolist()}")
        print(f"Reference top-5 scores: {[f'{s:.3f}' for s in top5.values.tolist()]}")

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from torch.export import export

    program = export(model, (example_input,), strict=True)

    # 1. Vulkan FP32
    print("\n1. Exporting Vulkan FP32 (with native hardswish)...")
    vulkan_options_fp32 = {
        "texture_limits": (2048, 2048, 2048),
    }
    try:
        vk_edge = to_edge_transform_and_lower(
            program,
            partitioner=[VulkanPartitioner(compile_options=vulkan_options_fp32)],
        )
        vk_exec = vk_edge.to_executorch()
        vk_path = os.path.join(OUTPUT_DIR, "mobilenet_fixed_vulkan.pte")
        with open(vk_path, "wb") as f:
            f.write(vk_exec.buffer)
        print(f"  Saved: {vk_path} ({os.path.getsize(vk_path) / 1024:.1f} KB)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. Vulkan FP16
    print("\n2. Exporting Vulkan FP16 (with native hardswish)...")
    vulkan_options_fp16 = {
        "texture_limits": (2048, 2048, 2048),
        "force_fp16": True,
    }
    try:
        vk_edge_fp16 = to_edge_transform_and_lower(
            program,
            partitioner=[VulkanPartitioner(compile_options=vulkan_options_fp16)],
        )
        vk_exec_fp16 = vk_edge_fp16.to_executorch()
        vk_path_fp16 = os.path.join(OUTPUT_DIR, "mobilenet_fixed_fp16_vulkan.pte")
        with open(vk_path_fp16, "wb") as f:
            f.write(vk_exec_fp16.buffer)
        print(f"  Saved: {vk_path_fp16} ({os.path.getsize(vk_path_fp16) / 1024:.1f} KB)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3. XNNPACK reference
    print("\n3. Exporting XNNPACK reference...")
    try:
        xn_edge = to_edge_transform_and_lower(
            program,
            partitioner=[XnnpackPartitioner()],
        )
        xn_exec = xn_edge.to_executorch()
        xn_path = os.path.join(OUTPUT_DIR, "mobilenet_fixed_xnnpack.pte")
        with open(xn_path, "wb") as f:
            f.write(xn_exec.buffer)
        print(f"  Saved: {xn_path} ({os.path.getsize(xn_path) / 1024:.1f} KB)")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nDone! Run the Flutter integration test on device to verify.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
