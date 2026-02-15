"""Export minimal single-op models to isolate PowerVR NaN source.

Exports:
1. Conv2d only (with real MobileNet fused-BN weights) - no activation
2. Hardswish only
3. Conv2d + Hardswish (no BN) with random weights
4. Conv2d + Hardswish with real MobileNet fused weights
"""
import torch
import torch.nn as nn
import os
import sys

# Add executorch to path
sys.path.insert(0, '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch')

from executorch.exir import to_edge, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
import torchvision.models as models

OUTPUT_DIR = '/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/assets/debug_models'

def get_mobilenet_first_layer_weights():
    """Extract the first conv layer's fused BN weights from MobileNetV3 Small."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()

    # First layer: features[0] = ConvBNActivation(Conv2d + BN + Hardswish)
    conv = model.features[0][0]  # Conv2d(3, 16, 3, stride=2, padding=1)
    bn = model.features[0][1]    # BatchNorm2d(16)

    # Fuse BN into Conv2d (same as ExecuTorch export does)
    fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
    return fused


class ConvOnly(nn.Module):
    """Just Conv2d with real fused-BN weights, no activation."""
    def __init__(self, fused_conv):
        super().__init__()
        self.conv = nn.Conv2d(
            fused_conv.in_channels, fused_conv.out_channels,
            fused_conv.kernel_size, stride=fused_conv.stride,
            padding=fused_conv.padding, bias=True,
        )
        self.conv.weight.data.copy_(fused_conv.weight.data)
        self.conv.bias.data.copy_(fused_conv.bias.data)

    def forward(self, x):
        return self.conv(x)


class HardswishOnly(nn.Module):
    """Just Hardswish activation."""
    def forward(self, x):
        return nn.functional.hardswish(x)


class ConvHardswishRandom(nn.Module):
    """Conv2d + Hardswish with random weights."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=True)

    def forward(self, x):
        return nn.functional.hardswish(self.conv(x))


class ConvHardswishReal(nn.Module):
    """Conv2d + Hardswish with real MobileNet fused weights."""
    def __init__(self, fused_conv):
        super().__init__()
        self.conv = nn.Conv2d(
            fused_conv.in_channels, fused_conv.out_channels,
            fused_conv.kernel_size, stride=fused_conv.stride,
            padding=fused_conv.padding, bias=True,
        )
        self.conv.weight.data.copy_(fused_conv.weight.data)
        self.conv.bias.data.copy_(fused_conv.bias.data)

    def forward(self, x):
        return nn.functional.hardswish(self.conv(x))


def export_model(model, name, example_input):
    """Export model to both Vulkan and XNNPACK."""
    model.eval()

    vulkan_opts = {"texture_limits": (2048, 2048, 2048)}

    for backend_name, partitioner_cls, kwargs in [
        ("vulkan", VulkanPartitioner, {"compile_options": vulkan_opts}),
        ("xnnpack", XnnpackPartitioner, {}),
    ]:
        try:
            exported = torch.export.export(model, (example_input,), strict=False)
            edge = to_edge(exported, compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                preserve_ops=[torch.ops.aten.hardswish.default],
            ))
            lowered = edge.to_backend(partitioner_cls(**kwargs))
            et_prog = lowered.to_executorch()

            path = os.path.join(OUTPUT_DIR, f"{name}_{backend_name}.pte")
            with open(path, "wb") as f:
                f.write(et_prog.buffer)
            size_kb = os.path.getsize(path) / 1024
            print(f"  [{backend_name}] {path} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  [{backend_name}] FAILED: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    example_input = torch.randn(1, 3, 224, 224)
    fused_conv = get_mobilenet_first_layer_weights()

    print("\n1. Conv2d only (real MobileNet fused-BN weights)")
    export_model(ConvOnly(fused_conv), "isolate_conv_real", example_input)

    print("\n2. Hardswish only (16ch, 112x112 - matches conv output size)")
    hs_input = torch.randn(1, 16, 112, 112)
    export_model(HardswishOnly(), "isolate_hardswish", hs_input)

    print("\n3. Conv2d + Hardswish (random weights)")
    export_model(ConvHardswishRandom(), "isolate_conv_hs_random", example_input)

    print("\n4. Conv2d + Hardswish (real MobileNet fused weights)")
    export_model(ConvHardswishReal(fused_conv), "isolate_conv_hs_real", example_input)

    print("\nDone! Run integration test on device to check each.")


if __name__ == "__main__":
    main()
