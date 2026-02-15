## Update: Root Cause Found - Hardswish/Hardsigmoid Decomposition

### TL;DR

The NaN is caused by `aten.hardswish` and `aten.hardsigmoid` being **decomposed into primitive ops** (`mul/add/clamp/div` with constant tensors) before reaching the Vulkan backend. The Vulkan backend has native GLSL shaders for both ops that work correctly on PowerVR, but they never get used because the ops are decomposed during `to_edge()`.

### Root Cause

PyTorch's default decomposition table decomposes:
- `hardswish(x)` → `x * clamp(x + 3, 0, 6) / 6`
- `hardsigmoid(x)` → `clamp(x + 3, 0, 6) / 6`

The decomposed path loads constant scalars (3 and 6) via `dim_order_ops._to_dim_order_copy` (buffer→texture conversion). On PowerVR GPUs, this produces `Inf` values, which cascade into NaN through the rest of the network.

The Vulkan backend **already has native GLSL shaders** for both ops in `activations.h` and they're registered via `VK_REGISTER_OP` in `UnaryOp.cpp`. However, `vulkan_partitioner.py`'s `ops_not_to_decompose` list only contains `upsample_nearest2d.vec` - it's missing `hardswish` and `hardsigmoid` (and other ops with native shaders that get decomposed: `hardshrink`, `silu`).

### How I Found It

1. **Isolation testing** - Exported 4 minimal single-op models:
   - Conv2d only → OK (no NaN)
   - **Hardswish only → NaN** (7403/200704 values, first4=[Inf, Inf, Inf, Inf])
   - Conv2d + Hardswish → ALL NaN
   - Conv2d + Hardswish (real weights) → ALL NaN

2. **Native shader test** - Re-exported with `preserve_ops=[torch.ops.aten.hardswish.default]` in `EdgeCompileConfig`:
   - Hardswish only → **PERFECT MATCH** (maxDiff=0.000000, 0 NaN)
   - Conv2d + Hardswish → **0 NaN**

3. **Full MobileNet test** - Exported full MobileNet V3 Small using `to_edge_transform_and_lower()` after adding hardswish/hardsigmoid to `ops_not_to_decompose`:
   - Before fix: **ALL 1000 outputs = NaN**
   - After fix: **0 NaN** (both FP32 and FP16)

### The Fix

In `backends/vulkan/partitioner/vulkan_partitioner.py`, add the decomposed ops to `ops_not_to_decompose`:

```python
ops_not_to_decompose = [
    torch.ops.aten.upsample_nearest2d.vec,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardshrink.default,
    torch.ops.aten.silu.default,
]
```

When `to_edge_transform_and_lower()` is used (the recommended API), it calls `partitioner.ops_to_not_decompose()` which returns this list, preventing decomposition and letting the native GLSL shaders handle these ops.

**5 ops** in the Vulkan op registry have native shader implementations but get decomposed by PyTorch's default decomposition table: `hardswish`, `hardsigmoid`, `hardshrink`, `silu`, `mish`. The first two are critical for MobileNetV3.

### Remaining Issue: Conv2d Accuracy

After fixing the NaN, the full MobileNet now produces 0 NaN but outputs are all near-zero (all 1000 logits ≈ 0.0). This is a separate **convolution accuracy issue** on PowerVR where single Conv2d outputs diverge from XNNPACK (e.g., XNNPACK mean=1.17 vs Vulkan mean=-0.03 for the first layer). Small per-layer errors compound through 13 feature blocks, collapsing the final output.

This Conv2d accuracy issue is separate from the decomposition bug and likely relates to how PowerVR handles the convolution shader's texture memory layout or precision.

### Test Environment

- Device: Pixel 10 Pro (PowerVR D-Series DXT-48-1536 MC1)
- ExecuTorch: main branch (`38ebc35e13`)
- Android 16 (API 36)

### Next Steps

I'll open a PR to add the missing ops to `ops_not_to_decompose`. The conv2d accuracy issue will need separate investigation.
