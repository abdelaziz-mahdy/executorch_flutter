# PR Draft: Serialize Prepack Dispatches on PowerVR

## Title
[ET-VK] Serialize prepack dispatches on PowerVR GPUs

## Summary

Fixes all Vulkan backend failures on PowerVR GPUs (Pixel 10 Pro) by serializing prepack compute shader dispatches.

PowerVR corrupts prepacked constant data when multiple prepack compute dispatches are batched in a single command buffer. Only the first constant is correct; subsequent constants read as zero. This caused MobileNet to produce NaN (via division-by-zero in Hardswish decomposition) and FP16 convolution to show a +0.5 bias offset.

The fix submits and waits after each prepack node on PowerVR, ensuring each constant is fully consumed by the GPU before the next staging buffer is created.

## Changes

- `ComputeGraph.cpp` — Add `serialize_prepack` flag for PowerVR that submits and waits after each prepack node in the `prepack()` loop

## Test Results (Pixel 10 Pro, PowerVR D-Series DXT-48-1536 MC1)

| Test | Status |
|------|--------|
| 12 multi-op chain models (2+ prepacked constants) | All PASS |
| 6 Hardswish isolation models | All PASS |
| 9 single scalar-constant models | All PASS |
| MobileNet V3 Small FP32 (cat.jpg) | PASS - class 281 (37.09%), exact match with XNNPACK |
| MobileNet V3 Small FP16 (cat.jpg) | PASS - class 281 (37.19%), correct within FP16 precision |

## Trade-off

Serializing prepack dispatches increases model load time on PowerVR since each constant requires a separate command buffer submission. For MobileNet V3 Small, load time increases from ~50ms to ~200ms. This only affects model loading (one-time cost), not inference latency.

## Related

- Builds on #17323 (PowerVR device detection, workgroup tuning)
- Supersedes #17467 (1D bias direct copy workaround - same root cause)
- Fixes #17299 (Vulkan backend all-zero outputs on PowerVR)

## Test Plan

- [x] 12 multi-op chain models pass on Pixel 10 Pro (PowerVR)
- [x] 6 Hardswish isolation models pass on Pixel 10 Pro
- [x] MobileNet V3 Small FP32 and FP16 produce correct classification on Pixel 10 Pro
- [ ] Verify no regression on Adreno/Mali (serialize_prepack is guarded by `device_is_powervr()`)

## Diff

```cpp
// In ComputeGraph::prepack():
const bool serialize_prepack = device_is_powervr();
// ... inside the prepack loop, after node->encode(this):
if (serialize_prepack && i < static_cast<int>(prepack_nodes_.size())) {
  submit_current_cmd_and_wait();
  context_->flush();
  staging_nbytes_in_cmd_ = 0;
  context_->set_cmd();
}
```
