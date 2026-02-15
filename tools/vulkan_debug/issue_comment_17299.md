# Issue Comment Draft for #17299

---

## Root Cause Found — Command Buffer Batching in `ComputeGraph::prepack()`

@SS-JIA I found the root cause. It is a driver bug, but there's a clean workaround.

### The Problem

`ComputeGraph::prepack()` encodes all prepack compute shader dispatches into a single command buffer, then submits it once. On PowerVR, when multiple prepack dispatches share one command buffer, only the first prepacked constant gets correct data. Every subsequent constant reads as zero.

This single bug explains every issue I reported in this thread:

- **FP16 +0.5 bias offset**: Conv2d bias was the second prepacked tensor (after weights). It read as zero instead of the intended values, so the convolution added uninitialized data
- **Hardswish producing Inf**: Hardswish decomposes into `x * clamp(x + 3, 0, 6) / 6`. The constants 3 and 6 are each prepacked scalar tensors. The first constant (3) was correct, but the second (6) read as zero, causing division by zero
- **MobileNet NaN**: The Inf from Hardswish cascaded through batch normalization (division by near-zero variance), producing NaN everywhere
- **1D bias appearing "dropped"** (PR #17467): The bias was simply the second prepacked tensor in the command buffer — same corruption, not a coordinate math issue

### How I Isolated It

I built a series of minimal test models (single op, each under 3KB) and ran them on the Pixel 10 Pro:

1. **Single-constant models** (e.g., `x + 3.0`, `x / 6.0`, `x * 2.0`): All produce correct results — both int and float constants, with and without `dim_order_copy`
2. **Two-constant models** (e.g., `(x + 3.0) / 6.0`, `(x + 3.0) + 2.0`, `(x + 1.0) * 2.0`): All fail. The first constant works, the second reads as zero
   - `(x + 3.0) / 6.0` with input 1.0 → expected 0.667, got Inf (div by 0)
   - `(x + 3.0) + 2.0` with input 1.0 → expected 6.0, got 4.0 (second const is 0)
   - `(x + 1.0) * 2.0` with input 1.0 → expected 4.0, got 0.0 (second const is 0)
3. **No-constant models** (e.g., `x + x`, `x * x`): All pass — the issue is specific to prepacked constants, not compute shaders in general

The pattern is clear: models with 0 or 1 prepacked constants pass, models with 2+ fail. The first constant is always correct.

### The Fix

In `ComputeGraph::prepack()`, serialize prepack dispatches on PowerVR. After each prepack node, submit the command buffer and wait before encoding the next one:

```cpp
const bool serialize_prepack = device_is_powervr();
// ... inside the prepack loop, after node->encode(this):
if (serialize_prepack && i < static_cast<int>(prepack_nodes_.size())) {
  submit_current_cmd_and_wait();
  context_->flush();
  staging_nbytes_in_cmd_ = 0;
  context_->set_cmd();
}
```

This uses the `device_is_powervr()` method from PR #17323, so it only affects PowerVR devices.

### Test Results (Pixel 10 Pro, with only this fix + PR #17323)

All previous workarounds reverted (no CPU prepack, no 1D direct copy, no GLSL changes). Only the serialize fix applied:

| Test | Result |
|------|--------|
| 12 two/three-constant models (add+div, add+mul, etc.) | All PASS |
| Hardswish (standalone, with conv, with conv+batchnorm) | All PASS |
| MobileNet V3 Small FP32 — cat.jpg classification | PASS — class 281 tabby cat (37.09%), exact match with XNNPACK |
| MobileNet V3 Small FP16 — cat.jpg classification | PASS — class 281 tabby cat (37.19%), correct within FP16 precision |

### What This Means for Open PRs

- **PR #17467** (1D bias direct copy): Not needed. The bias was corrupted by batched dispatches, not coordinate math. I'll close it.
- **PR #17323** (PowerVR device detection): Still needed — the serialize fix depends on `device_is_powervr()`.

### Trade-off

Serializing prepack dispatches increases model load time on PowerVR since each constant requires a separate command buffer submission. For MobileNet V3 Small, load time went from ~50ms to ~200ms. This is a one-time cost at model load, not per-inference.

I'll open a PR with the fix on top of #17323.
