## Overview

Extensive testing on Pixel 10 Pro (`powervr d-series dxt-48-1536 mc1`) confirms this is a broader PowerVR driver bug, not limited to push constants.

This matches a [known issue on Imagination's developer forum](https://forums.imgtec.com/t/vulkan-problem-on-powervr-ge8300-update-push-constants-multiple-times-inside-render-pass-constants-at-gpu-side-messed-up/2931) where push constants get corrupted when updated multiple times within the same command buffer.

## Test Results

I tested **11 different synchronization strategies** between prepack dispatches, using MobileNet V3 Small (Vulkan output vs XNNPACK reference):

| Mode | Strategy | Load Time | Max Diff | NaN | Top-1 | Result |
|:----:|----------|----------:|---------:|----:|:-----:|:------:|
| 0 | No barrier (baseline) | ~100ms | N/A | 1000 | NO | FAIL |
| 1 | Execution barrier | ~100ms | N/A | 1000 | NO | FAIL |
| 2 | Memory barrier (compute->compute) | ~100ms | N/A | 1000 | NO | FAIL |
| **3** | **Submit + wait per node** | **~8200ms** | **0.50** | **0** | **YES** | **PASS** |
| **4** | **Submit, no wait (new CB)** | **~7800ms** | **0.50** | **0** | **YES** | **PASS** |
| **5** | **Submit, no wait + flush** | **~8200ms** | **0.50** | **0** | **YES** | **PASS** |
| 6 | Batch every 2 nodes | ~4080ms | 50.12 | 0 | NO | FAIL |
| 7 | Batch every 4 nodes | ~2100ms | 4.27 | 0 | NO | FAIL |
| 8 | Batch every 8 nodes | ~1100ms | 4.27 | 0 | NO | FAIL |
| 9 | Batch every 16 nodes | ~600ms | 4.27 | 0 | NO | FAIL |
| 10 | Hybrid: UBO + serialize PC-only | ~7600ms | 4.27 | 0 | NO | FAIL |

## Findings

1. **Barriers don't help** (modes 0-2) - Neither execution-only nor full memory barriers fix it, ruling out a synchronization issue.

2. **Only new command buffers work** (modes 3-5) - Submitting and starting a fresh CB after each node is the only fix. Mode 4 shows I don't even need `vkQueueWaitIdle`, just a CB boundary - pointing to internal CB state corruption.

3. **Even 2 nodes per CB corrupts** (mode 6) - No middle ground. Strictly 1 dispatch per CB required.

4. **Not limited to push constants** (mode 10) - I replaced push constants with UBOs for standard/bias prepack shaders (using the existing `no_pc` shader variants + `sizes_ubo()`). UBO-only nodes still corrupt when batched. The bug affects all dispatch types sharing a command buffer.

## Impact

- ~12 line fix, PowerVR-only, zero impact on other GPUs
- Load time: ~100ms to ~8200ms (one-time at model load, inference unaffected)
- Mode 4 (submit without CPU wait) could reduce overhead while still fixing the bug
