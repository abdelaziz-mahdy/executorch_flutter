# GPU Concepts - Visual Explainer

## How GPU Work Gets Done

Think of the GPU as a restaurant kitchen.

```
  YOU (CPU)                          KITCHEN (GPU)
  =========                          =============

  Write order ticket    ------>      Receives ticket
  (command buffer)                   (command buffer)

  "Submit" = hand                    Kitchen works through
  the ticket over                    all items on the ticket

  "Wait" = stand at                  Rings bell when
  counter until done                 ticket is complete
```

**Command Buffer** = an order ticket. You write all the GPU operations on one ticket, then hand the whole ticket to the kitchen at once.

**Dispatch** = one item on the ticket (e.g., "reformat these weights for GPU use").

**Submit** = handing the ticket to the kitchen.

**Wait** = standing at the counter until the kitchen finishes that ticket.

Normally, you put MANY items on one ticket for efficiency. The kitchen works through them without stopping to check in with you. This is called **batching**.

---

## What "Prepack" Means

Before a neural network can run on the GPU, its weights (the learned numbers that define the model) need to be reformatted into a layout the GPU can use. This reformatting is called **prepacking**.

```
  CPU Memory (raw weights)           GPU Memory (optimized layout)
  ========================           ============================
  [w1, w2, w3, w4, ...]   ---->     [w1 w3 | w2 w4 | ...]
                            ^
                            |
                     prepack dispatch
                     (one per weight tensor)
```

A model like MobileNet has ~100 weight tensors. Each needs a separate reformat operation (dispatch). Normally, you put all 100 on one command buffer and submit them at once.

---

## The PowerVR Bug

On PowerVR GPUs (Pixel 10 Pro), when you put multiple prepack operations on the same ticket, the kitchen messes up. The first item comes out fine. Every item after that reads zeros instead of the actual data.

```
  NORMAL GPU (Adreno, Mali, etc.)        POWERVR (Pixel 10 Pro)
  ===============================        =======================

  Ticket: [A, B, C, D]                  Ticket: [A, B, C, D]
           |  |  |  |                            |  |  |  |
           v  v  v  v                            v  v  v  v
           OK OK OK OK                          OK  X  X  X
                                                     ^
                                                     |
                                              reads zeros instead
                                              of actual data
```

---

## What Each Mode Does

### Modes 0-2: Barriers (all FAIL)

Barriers are notes on the ticket telling the kitchen to finish one item before starting the next.

```
  Mode 0: No barrier                 Mode 1: Execution barrier
  ========================           ========================
  Ticket: [A, B, C]                  Ticket: [A, WAIT, B, WAIT, C]
  "do all of these"                  "do A, then B, then C"

  Mode 2: Memory barrier
  ========================
  Ticket: [A, SYNC, B, SYNC, C]
  "do A, flush memory, then B, flush memory, then C"
```

All three fail because the corruption is **inside the command buffer** itself. No amount of ordering or memory flushing within the same ticket fixes it.

```
  Result for modes 0-2:
  +------------------+
  | Ticket [A, B, C] |  <-- same command buffer = same corruption
  |   A = OK         |
  |   B = ZEROS      |      regardless of barriers between them
  |   C = ZEROS      |
  +------------------+
```

### Modes 3-5: Submit Per Node (all PASS)

Each item gets its own ticket. The kitchen processes one ticket at a time.

```
  Mode 3: Submit + wait              Mode 4: Submit, no wait
  ========================           ========================
  Ticket 1: [A]  --> submit          Ticket 1: [A]  --> submit
  wait...                            Ticket 2: [B]  --> submit (immediately)
  Ticket 2: [B]  --> submit          Ticket 3: [C]  --> submit (immediately)
  wait...                            (GPU processes queue in order,
  Ticket 3: [C]  --> submit           CPU doesn't stall between submits)
  wait...

  Mode 5: Same as mode 4, but also recycles temp memory between submits
```

Mode 4 is the interesting one: no CPU-wait needed, just a new command buffer. This proves the bug is about **sharing a command buffer**, not about timing or races.

```
  Result for modes 3-5:
  +-----------+  +-----------+  +-----------+
  | Ticket [A]|  | Ticket [B]|  | Ticket [C]|
  |   A = OK  |  |   B = OK  |  |   C = OK  |
  +-----------+  +-----------+  +-----------+
       ^              ^              ^
       |              |              |
       separate command buffers = no corruption
```

### Modes 6-9: Batching N Nodes (all FAIL)

Put 2, 4, 8, or 16 items per ticket. A middle ground between "all on one" and "one per ticket."

```
  Mode 6 (batch 2):                  Mode 7 (batch 4):
  ========================           ========================
  Ticket 1: [A, B]  --> submit       Ticket 1: [A, B, C, D]  --> submit
  Ticket 2: [C, D]  --> submit       Ticket 2: [E, F, G, H]  --> submit

  Result:                            Result:
  Ticket 1: A=OK, B=WRONG           Ticket 1: A=OK, B/C/D=ZEROS
  Ticket 2: C=OK, D=WRONG           Ticket 2: E=OK, F/G/H=ZEROS
```

Even 2 items per ticket corrupts. There is no safe batch size above 1.

### Mode 10: Hybrid UBO (FAIL)

The initial theory: maybe the bug is specific to **push constants** (a way to pass small parameters to GPU shaders). So I switched most prepack shaders to use **UBOs** (Uniform Buffer Objects, an alternative parameter mechanism) and only serialized the few nodes that still had push constants.

```
  Push Constants vs UBOs:
  ========================
  Push Constants:  global to command buffer, updated via vkCmdPushConstants()
  UBOs:            per-descriptor-set, bound via vkCmdBindDescriptorSets()

  Mode 10 approach:
  +------------------------------------------+
  | UBO nodes batched:  [A, B, C]  (no PC)   |  <-- expected to work
  +------------------------------------------+
  +-----------+  +-----------+
  | PC node D |  | PC node E |  <-- serialized (isolated CBs)
  +-----------+  +-----------+

  Actual result: UBO nodes A, B, C STILL CORRUPT
```

This proves the bug is not about push constants. It affects all dispatch types when they share a command buffer on PowerVR.

---

## Summary

```
  What works:        1 dispatch per command buffer (modes 3, 4, 5)
  What doesn't:      Barriers, batching, UBO replacement
  Performance cost:  ~100ms -> ~8200ms model load (inference unaffected)
  Scope:             PowerVR only, ~12 line fix, transparent to other GPUs
```
