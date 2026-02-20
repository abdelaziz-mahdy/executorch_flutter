# PowerVR Prepack Corruption - Complete Findings

## Summary for PR Reply (pytorch/executorch#17468)

### Root Cause

On PowerVR GPUs (tested on Pixel 10 Pro, `powervr d-series dxt-48-1536 mc1`), batching multiple prepack compute dispatches in a single Vulkan command buffer causes state corruption. The first dispatch in a command buffer executes correctly, but subsequent dispatches read corrupted data (constants read as 0, output buffers zeroed).

This is a **known PowerVR driver bug** - documented on Imagination's developer forum as affecting push constant updates within the same command buffer.

### What We Tested

We implemented and tested **11 different synchronization strategies** between prepack dispatches, cycling through them automatically on each Vulkan model load. All tests used MobileNet V3 Small, comparing Vulkan output against XNNPACK reference.

| Mode | Strategy | Load Time | NaN Count | Max Diff | Top-1 | Result |
|------|----------|-----------|-----------|----------|-------|--------|
| 0 | No barrier (baseline) | ~100ms | 1000 | N/A | NO | **FAIL** |
| 1 | Execution barrier only | ~100ms | 1000 | N/A | NO | **FAIL** |
| 2 | Full memory barrier (compute→compute) | ~100ms | 1000 | N/A | NO | **FAIL** |
| 3 | Submit + wait per node | ~8200ms | 0 | 0.0039 | YES | **PASS** |
| 4 | Submit without wait (new cmd buffer, no CPU stall) | ~7800ms | 0 | 0.0039 | YES | **PASS** |
| 5 | Submit without wait + flush staging | ~8200ms | 0 | 0.0039 | YES | **PASS** |
| 6 | Batch every 2 nodes | ~4080ms | 0 | 50.79 | NO | **FAIL** |
| 7 | Batch every 4 nodes | ~2100ms | 0 | all zeros | NO | **FAIL** |
| 8 | Batch every 8 nodes | ~1000ms | 0 | all zeros | NO | **FAIL** |
| 9 | Batch every 16 nodes | ~600ms | 0 | all zeros | NO | **FAIL** |
| 10 | Hybrid: UBO for standard prepack, serialize only push-constant nodes | ~7600ms | 0 | 4.27 | NO | **FAIL** |

### Key Findings

1. **Pipeline barriers are insufficient** (modes 0-2): Neither execution-only barriers nor full memory barriers (with proper src/dst access masks) fix the corruption. The GPU reports the barriers as complete but the data is still wrong.

2. **Only command buffer boundary fixes it** (modes 3-5): Submitting the command buffer to the queue and starting a new one is the only reliable fix. Interestingly, we don't need to CPU-wait for completion (mode 4 works as well as mode 3), suggesting the issue is command buffer-internal state, not a timing/race condition.

3. **Batching any >1 nodes corrupts** (modes 6-9): Even batching just 2 nodes per command buffer produces incorrect output (mode 6: maxDiff=50). Larger batches produce all-zero output.

4. **The bug is NOT limited to push constants** (mode 10): We replaced push constants with UBOs for standard prepack and bias prepack shaders (using existing `no_pc` shader variants + `sizes_ubo()`). Nodes using only UBOs (no push constants at all) still produce wrong output when batched together. The corruption affects ALL dispatch types sharing a command buffer on PowerVR.

5. **Performance cost**: Full serialization increases MobileNet load time from ~100ms to ~8200ms (~80x slower). This only affects model loading, not inference speed.

### Conclusion

The PowerVR driver has a fundamental issue with multiple compute dispatches sharing a command buffer during prepack. The only working fix is **1 dispatch per command buffer** (submit after every prepack node). This is not a push-constant-specific issue - it affects all dispatches regardless of how parameters are passed.

### Recommended Fix

The simplest and most reliable approach:

```cpp
void ComputeGraph::prepack() {
  for (auto& node : prepack_nodes()) {
    node->encode(this);

    if (device_is_powervr()) {
      // PowerVR driver bug: multiple dispatches in same CB corrupt state.
      // Submit after each node to isolate in its own command buffer.
      submit_current_cmd_and_wait(/* ... */);
      context_->set_cmd();
    }
  }
  // ... existing submit for non-PowerVR
}
```

This is a ~12-line change that's fully isolated to PowerVR devices and has no impact on other GPUs.

### Open Questions

- Is the ~8s load time acceptable for PowerVR devices? (inference speed is unaffected)
- Should we investigate `vkQueueSubmit` without `vkQueueWaitIdle` (mode 4) to potentially reduce overhead?
- Are there PowerVR driver updates that fix this? (The forum post suggests it may be a long-standing issue)

---

## GPU Concepts Explained (For Non-GPU Programmers)

### The Basics: How GPU Work Gets Done

Think of the GPU as a restaurant kitchen:

- **Command Buffer** = An order ticket. You write down all the dishes (GPU operations) you want on one ticket, then hand the whole ticket to the kitchen at once.
- **Submit** = Handing the ticket to the kitchen. The kitchen starts working on everything on that ticket.
- **Wait** = Standing at the counter until the kitchen finishes that ticket.
- **Dispatch** = One specific dish/operation on the ticket (e.g., "transform these weights into GPU format").

Normally, you put MANY operations on one ticket for efficiency - the kitchen can work through them without stopping to check in with you. This is called **batching**.

### What "Prepack" Means

Before a neural network can run on the GPU, its weights (the learned numbers that define the model) need to be reformatted into a layout the GPU can efficiently use. This is called **prepacking**.

A model like MobileNet has ~100 weight tensors. Each one needs a separate "reformat" operation (dispatch). Normally, you'd put all 100 on one command buffer (one order ticket) and submit them all at once. Fast and efficient.

### The PowerVR Bug

On PowerVR GPUs (found in Pixel 10 Pro), when you put multiple prepack operations on the same order ticket, the kitchen messes up. The first dish comes out fine, but every dish after that uses the wrong ingredients (reads zeros instead of the actual data).

It's like a kitchen that can only focus on one order at a time - if you put two dishes on the same ticket, they'll make the first one right but completely botch the second.

### What Each Mode Does

#### Mode 0: No Barrier (Baseline)
**Analogy**: Write all 100 dishes on one ticket. Hand it to the kitchen. Hope for the best.
**Technical**: All prepack dispatches in one command buffer with no synchronization between them.
**Result**: FAIL - Kitchen botches everything after the first dish.

#### Mode 1: Execution Barrier
**Analogy**: Same one ticket, but you add notes saying "finish dish 1 before starting dish 2." The kitchen knows the ORDER but doesn't double-check ingredients between dishes.
**Technical**: Insert a pipeline barrier with `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` dependency between dispatches. This ensures execution ordering but doesn't guarantee memory visibility.
**Result**: FAIL - The order is right but the ingredients (data) are still corrupted.

#### Mode 2: Memory Barrier
**Analogy**: Same one ticket, but with stronger notes: "finish dish 1, make sure all ingredients are put back on the shelf correctly, THEN start dish 2." The kitchen should check that everything is properly stored between dishes.
**Technical**: Full `VkMemoryBarrier` with `VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT` between dispatches. This should guarantee both execution order AND memory visibility.
**Result**: FAIL - Even with proper memory guarantees, the state is still corrupted. This proves it's a driver bug, not a synchronization issue.

#### Mode 3: Full Submit + Wait (Current Fix)
**Analogy**: Write ONE dish per ticket. Hand it to the kitchen. Wait until they're completely done. Then write the next dish on a NEW ticket. Repeat 100 times.
**Technical**: After each prepack dispatch, submit the command buffer via `vkQueueSubmit`, then `vkQueueWaitIdle` to wait for GPU completion, then allocate a new command buffer for the next dispatch.
**Result**: PASS - Each dish gets its own fresh ticket and the kitchen's full attention. Slowest approach (~8 seconds vs ~100ms).

#### Mode 4: Submit Without Wait
**Analogy**: Write ONE dish per ticket. Hand it to the kitchen. DON'T wait - immediately start writing the next dish on a NEW ticket. Hand that one in too. The kitchen has a queue of single-dish tickets.
**Technical**: Submit command buffer via `vkQueueSubmit` but skip `vkQueueWaitIdle`. Immediately allocate a new command buffer. The GPU processes submissions in order but the CPU doesn't stall.
**Result**: PASS - This works! The key insight: the bug is about SHARING a command buffer, not about timing. Each dispatch just needs its own ticket, even if we don't wait between them.

#### Mode 5: Submit Without Wait + Flush Staging
**Analogy**: Same as Mode 4, but also clean the prep counter between dishes (free up temporary workspace memory).
**Technical**: Same as Mode 4, plus recycle staging buffers after each submit to reduce peak memory usage.
**Result**: PASS - Works the same as Mode 4 with slightly better memory management.

#### Mode 6: Batch Every 2 Nodes
**Analogy**: Put 2 dishes per ticket instead of 1. Still uses separate tickets, but each has 2 dishes.
**Technical**: Submit command buffer after every 2 prepack dispatches.
**Result**: FAIL - Even just 2 dispatches per command buffer causes corruption (maxDiff=50). The second dish on each ticket is wrong.

#### Modes 7-9: Batch Every 4/8/16 Nodes
**Analogy**: Put 4, 8, or 16 dishes per ticket.
**Technical**: Submit command buffer after every N dispatches.
**Result**: FAIL - All produce worse results than mode 6. More dishes per ticket = more corruption. Output is all zeros.

#### Mode 10: Hybrid UBO (Serialize Push-Constant Nodes Only)
**Analogy**: We thought the problem was with one specific type of ingredient label (push constants). So we switched most dishes to use a different label system (UBOs). Then we only gave separate tickets to the few dishes still using the old labels.
**Technical**: Replaced push constants with Uniform Buffer Objects (UBOs) for standard prepack and bias prepack shaders. Only conv weight prepack nodes (which still use push constants because no UBO variant exists) got their own command buffers. UBO-only nodes were batched together.
**Result**: FAIL - The UBO-only nodes ALSO produce wrong output when batched! This proves the bug affects ALL dispatch types, not just those using push constants.

### The Bottom Line

The PowerVR GPU driver has a bug where it can't properly handle multiple compute shader dispatches in the same command buffer during prepack. The ONLY fix is to give each operation its own command buffer (Mode 3, 4, or 5). This costs about 8 seconds of extra model loading time, but doesn't affect inference speed at all.

The fix is simple (~12 lines of code), only affects PowerVR devices, and is completely transparent to users.
