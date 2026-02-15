## Investigation update (Feb 10-14)

Since my last comment, I dug deeper into why the bias gets dropped and why the full MobileNet still fails even after fixing the decomposition issue. Found two more root causes — one is fixed, one is still open.

| # | Problem | Status |
|---|---------|--------|
| 1 | 1D bias tensor coordinate mismatch in `nchw_to_image` | Draft PR #17467 |
| 2 | `imageStore` in weight prepack compute shaders writes wrong data | Investigating |

After fix 1 (plus the `vmaFlushAllocation` fix you already have), isolated conv2d ops (with and without bias) produce correct results on PowerVR. But the full MobileNet still outputs near-zero logits, which points to issue 2.

---

### 1D bias tensor coordinate mismatch

This is the root cause of the "bias completely dropped" behavior I reported earlier.

Conv2d bias tensors are 1D (`[out_channels]`). The prepacking pipeline pads them to 4D, assigns `kTexture2D` storage, and runs `nchw_to_image` to write data into a texture. The shader uses `axis_map` coordinate remapping designed for multi-dimensional tensors. For a 1D tensor with three fake dimensions of size 1, the coordinate math produces invalid (x,y) texture coordinates. PowerVR strictly validates texture bounds — out-of-bounds writes are silently dropped, so bias values never reach the texture. Adreno and Mali silently wrap coordinates, hiding the bug.

**Fix**: In `PrepackNode::encode()`, I detect 1D width-packed tensors and use `vkCmdCopyBufferToImage` (DMA transfer) instead of the `nchw_to_image` compute shader. The staging buffer already has correctly ordered data — it just needs to reach the texture without the coordinate remapping.

Draft PR: #17467

---

### `imageStore` in weight prepack shaders (investigating)

After the above fixes, isolated conv2d ops pass. But the full MobileNet still fails. Progressive layer testing shows the first conv layer already produces Infinity, meaning weight prepacking writes wrong data. The weight prepack shaders (`conv2d_prepack_weights`, `conv2d_dw_prepack_weights`) use `imageStore` to write repacked weights into 3D textures.

Normally, prepacking is one pass: the compute shader reads weight data from a staging buffer, rearranges it into GPU-optimized layout, and writes directly to a 3D texture via `imageStore`. On PowerVR, this standard path produces wrong results (Infinity in the first conv layer output).

To work around this, I tried a two-pass approach:
1. The prepack compute shader does the same rearrangement but writes to an SSBO (plain buffer) instead of a texture, avoiding `imageStore` entirely.
2. `vkCmdCopyBufferToImage` then copies that buffer into the final 3D texture via DMA.

The two-pass path doesn't crash, but also produces wrong results. The likely cause is that the shader's linear buffer indexing doesn't match the texture memory layout that `vkCmdCopyBufferToImage` expects, or the rearrangement logic produces different results when targeting a linear buffer vs a tiled texture.

When I later tried to disable the two-pass path to go back and test the standard `imageStore` path in isolation, it crashed in `vkUpdateDescriptorSets` during descriptor set binding for the image output. So the standard path may have an additional driver-level issue on PowerVR beyond the wrong output.

I added logcat debug logging to `PrepackNode.cpp` and all parameters look correct (shader names, image extents, workgroup sizes, push constants). Still working through why both paths produce wrong results.


---

### What I think we should do next

1. The 1D bias direct copy fix is in draft PR #17467.
2. For the `imageStore` issue, I want to try CPU prepack as a fallback — do weight repacking on CPU and upload via `vkCmdCopyBufferToImage`. Model loading would be slower (one-time cost, not per-inference), but it would be correct.
3. The `vkUpdateDescriptorSets` crash on the standard `imageStore` path is likely a driver bug. Might be worth reporting to Imagination Technologies with a minimal repro.
