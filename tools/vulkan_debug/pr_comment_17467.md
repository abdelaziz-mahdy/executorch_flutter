# PR Comment Draft for #17467

## Closing - Superseded by Serialize Prepack Fix

I'm closing this PR. The 1D bias prepacking issue on PowerVR turned out to be a symptom of a broader problem, not a coordinate math issue.

### What I Found

The actual root cause is that PowerVR corrupts prepacked constant data when multiple prepack compute dispatches are batched in a single Vulkan command buffer. Only the first constant in the batch is correct; subsequent constants read as zero.

For conv2d bias, this meant the bias tensor (typically the second prepacked tensor after weights) read as zero, producing the incorrect output I attributed to out-of-bounds texture writes.

### The Real Fix

A change in `ComputeGraph::prepack()` that serializes prepack dispatches on PowerVR. This fixes bias prepacking, weight prepacking, scalar constant prepacking, and every other multi-constant model — without touching `PrepackNode`, `CommandBuffer`, or `Allocator`.

I tested MobileNet V3 Small on Pixel 10 Pro with only this fix (all changes from this PR reverted): FP32 and FP16 both produce correct classification results, matching XNNPACK output.

Details in #17299.
