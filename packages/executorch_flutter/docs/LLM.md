# On-device LLM (Gemma 4) with executorch_flutter

`executorch_flutter` can run a generative LLM — **Google Gemma 4 E2B** — fully
on-device, with token-by-token streaming, via ExecuTorch's `extension/llm/runner`.
This is separate from the vision/tensor API (`ExecuTorchModel`): an LLM is driven
by a stateful decode loop, a real tokenizer, and a KV cache, and is loaded from
**file paths** (weights are 1+ GB) rather than `Uint8List`.

> **Status:** experimental. Validated on **macOS** (XNNPACK CPU + MLX GPU). Other
> platforms get the XNNPACK CPU runner; coverage expands over time.

---

## TL;DR

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

final llm = await ExecuTorchLLM.load(
  modelPath: '/path/gemma-4-E2B-it_xnnpack.pte',
  tokenizerPath: '/path/gemma-4-E2B-it_tokenizer.json',
  // MLX (Apple-GPU) models ONLY — point at the mlx.metallib you shipped:
  // mlxMetallibPath: '/path/mlx.metallib',
);

final prompt = '<bos><|turn>user\nExplain Flutter in one line.<turn|>\n<|turn>model\n';
await for (final piece in llm.generate(
  prompt,
  config: const GenConfig(maxNewTokens: 512, temperature: 0), // 0 = greedy
)) {
  stdout.write(piece);
}
await llm.dispose();
```

See `example/lib/screens/llm_chat_screen.dart` for a complete streaming chat
screen (multi-turn, stop, reset, settings, MLX metallib field).

---

## 1. Get the model

LLM weights are large (1+ GB) and carry their own license terms, so — unlike the
vision models — **nothing is pre-hosted**: there is no download URL to point at.
You export the model yourself, once, with the scripts below.

Sources you'll need:

- **Base model + `tokenizer.json`**: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) on Hugging Face (accept the Gemma terms)
- **Export scripts**: [`executorch_flutter_models/python`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/tree/main/python)
- **`mlx.metallib`** (MLX only): [`executorch_native` releases](https://github.com/abdelaziz-mahdy/executorch_native/releases)

The example app's LLM screen lists these same links next to the file pickers.

You need **three** files (for MLX) or **two** (for XNNPACK):

| File | XNNPACK | MLX | What it is |
|------|:------:|:---:|------------|
| `gemma-4-E2B-it_{xnnpack,mlx}.pte` | ✅ | ✅ | the model |
| `gemma-4-E2B-it_tokenizer.json` | ✅ | ✅ | HF tokenizer (from the model repo) |
| `mlx.metallib` | — | ✅ | MLX's Metal GPU kernels |

### Export it yourself

The export scripts live in the models repo (`models/python/`):

```bash
# (in a Python env with optimum-executorch + executorch installed)
python models/python/export_gemma4_xnnpack.py   # -> CPU model (all platforms)
python models/python/export_gemma4_mlx.py        # -> Apple-GPU model (macOS)
```

Key recipe details (already handled by the scripts):
- **XNNPACK:** `qlinear="8da4w"`, **no** `qembedding` (quantizing the embedding
  breaks Gemma 4's `pad_embedding` op during export), static decode-only export.
- **MLX:** `qlinear="4w"`, MLX partitioner, dynamic.
- Both **rename `get_eos_id` → `get_eos_ids`** and embed `[1, 106]` so the runner
  stops on `<end_of_turn>` (106). (For the XNNPACK export that omits it, the
  example app stops on the `<turn|>` marker instead — see §4.)

The `tokenizer.json` comes from the HF model repo (`google/gemma-4-E2B-it`).
The `mlx.metallib` is produced by the MLX native build and is published as a
release artifact of `executorch_native`.

---

## 2. Enable the LLM build

The LLM runner + MLX backend are shipped as **dedicated prebuilt variants**
(`xnnpack-llm`, and `xnnpack-mlx-llm` on macOS-arm64). Enable them in your app's
`pubspec.yaml`:

```yaml
hooks:
  user_defines:
    executorch_dart:
      llm: true
      backends:
        - xnnpack          # CPU LLM — all platforms
        - mlx              # Apple-Silicon GPU LLM — macOS arm64 only
```

### macOS requirements (MLX)

- **Deployment target ≥ 14.0.** MLX needs macOS 14. Set it in your app:
  - `macos/Podfile`: `platform :osx, '14.0'`
  - `macos/Runner.xcodeproj`: `MACOSX_DEPLOYMENT_TARGET = 14.0`
  - (The native library is already built for 14; the package logs a warning if
    your app target is lower.)
- **Sandbox + the metallib (important — see §3).**
- Building MLX **from source** additionally needs the Xcode Metal Toolchain:
  `xcodebuild -downloadComponent MetalToolchain`.

### Source build (advanced)

If a prebuilt LLM variant isn't published for your platform yet, build from
source:

```yaml
      build_mode: "source"
      # optional, faster than fetching: export EXECUTORCH_SOURCE_DIR=/path/to/executorch
```

First source build is slow (~30–60 min). XNNPACK LLM builds on every platform;
MLX is macOS-arm64 only.

---

## 3. MLX metallib — a required shipping step

**This only applies to the MLX (Apple-GPU) backend. XNNPACK needs nothing here.**

MLX loads its Metal kernels from `mlx.metallib` at runtime. A sandboxed app can't
read the copy next to the native library, so **your app must ship the metallib and
tell the runner where it is**:

1. Get `mlx.metallib` (from the `executorch_native` release, or your MLX build).
2. Bundle it with your app — e.g. as a Flutter asset:
   ```yaml
   flutter:
     assets:
       - assets/mlx/mlx.metallib
   ```
3. At runtime, copy it to a writable, readable path and pass it to `load`:
   ```dart
   final dir = await getApplicationSupportDirectory();
   final dest = File('${dir.path}/mlx.metallib');
   if (!dest.existsSync()) {
     final bytes = await rootBundle.load('assets/mlx/mlx.metallib');
     await dest.writeAsBytes(bytes.buffer.asUint8List());
   }
   final llm = await ExecuTorchLLM.load(
     modelPath: ..., tokenizerPath: ..., mlxMetallibPath: dest.path,
   );
   ```

Without this, an MLX model fails to load with `Error 0x23` (`MLXBackend` could not
load its metallib).

> The example app instead lets you **pick the metallib file** and disables its
> sandbox for convenience — that's a dev shortcut; real apps keep the sandbox and
> use the bundle-and-path approach above.

---

## 4. Using the API

### The Gemma 4 chat template

Gemma 4 (instruction-tuned) expects turn markers, or it produces degenerate
output. The runner tokenizes the prompt verbatim, so **you format it**:

```
<bos><|turn>user\n{message}<turn|>\n<|turn>model\n
```

For multi-turn, re-send the whole conversation each turn and call `llm.reset()`
between turns (the prompt carries the full history). The turn tokens are the
literal strings `<|turn>` (start) and `<turn|>` (end) — **not**
`<start_of_turn>`/`<end_of_turn>`, which this tokenizer splits into many tokens.

### Stopping

MLX models embed `get_eos_ids=[1,106]`, so the runner stops on `<end_of_turn>`
itself. The XNNPACK export only declares `<eos>`, so the model keeps emitting
`<turn|>` — handle it at the app level (strip it from display and stop):

```dart
.listen((piece) {
  if (piece.contains('<turn|>') || piece.contains('<end_of_turn>')) {
    // strip the marker, append the rest, then:
    llm.stop();
    return;
  }
  // append piece
});
```

### GenConfig

Temperature-only sampling (no top-p/top-k — the runner doesn't support them).
`temperature: 0` is greedy (deterministic, most reliable). `maxNewTokens`,
`seqLen`, `echo`, `ignoreEos` mirror the native `GenerationConfig`.

### Lifecycle

```dart
llm.generate(prompt, config: ...)  // Stream<String>
llm.stop();                         // cooperative cancel
llm.reset();                        // clear KV cache / new conversation
await llm.dispose();                // release native resources
```

---

## Platform support

| Platform | XNNPACK (CPU) LLM | MLX (GPU) LLM |
|----------|:----------------:|:-------------:|
| macOS (arm64) | ✅ | ✅ (needs metallib + macOS 14) |
| macOS (x64) | ✅ | — |
| iOS / Android / Linux / Windows | ✅ (CPU) | — |

Performance (Gemma 4 E2B, M-series, approx): XNNPACK ~4 tok/s decode; MLX
~30–40 tok/s decode.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Error 0x23` `MLXBackend` metallib | metallib not shipped — pass `mlxMetallibPath` (§3) |
| `Error 0x10` resize static tensor | older build without auto-detect prefill; update the native lib |
| Garbage / repeated tokens | wrong chat template — use `<bos><|turn>user\n…<turn|>\n<|turn>model\n` |
| Never stops | XNNPACK model lacks `<turn|>` in eos — stop at app level (§4) |
| MLX build: "cannot execute tool 'metal'" | `xcodebuild -downloadComponent MetalToolchain` |
| App won't launch (macOS, MLX) | app deployment target < 14.0 (§2) |
