# On-device LLMs with executorch_flutter

`executorch_flutter` can run a generative LLM fully on-device, with
token-by-token streaming, via ExecuTorch's `extension/llm/runner`. This is
separate from the vision/tensor API (`ExecuTorchModel`): an LLM is driven by a
stateful decode loop, a real tokenizer, and a KV cache, and is loaded from
**file paths** (weights are hundreds of MB) rather than `Uint8List`.

> **Status:** experimental. Validated on **macOS** (XNNPACK CPU + MLX GPU),
> **Android** and **web** (XNNPACK CPU). Other platforms get the XNNPACK CPU
> runner too; coverage expands over time.

---

## TL;DR

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

final llm = await ExecuTorchLLM.load(
  modelPath: '/path/qwen3_0.6b_xnnpack.pte',
  tokenizerPath: '/path/qwen3_0.6b_tokenizer.json',
  // MLX (Apple-GPU) models ONLY — point at the mlx.metallib you shipped:
  // mlxMetallibPath: '/path/mlx.metallib',
);

const prompt = '<|im_start|>user\nExplain Flutter in one line.<|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\n';
await for (final piece in llm.generate(
  prompt,
  config: const GenConfig(maxNewTokens: 512, temperature: 0), // 0 = greedy
)) {
  stdout.write(piece.replaceAll('<|im_end|>', ''));
}
await llm.dispose();
```

The example app's **Language Models** screen
(`example/lib/screens/llm_chat_screen.dart`) downloads a hosted model with one
tap and chats with it: multi-turn, stop, reset, generation settings, and a
"use your own model" panel for files you exported yourself.

---

## 1. Get a model

### Hosted: Qwen3 0.6B

Ready-made exports of [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
(Apache 2.0) are published on the models repo's release for each ExecuTorch
version, and listed with sizes and SHA-256 hashes in
[`<version>/llms.json`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/blob/main/1.6.0/llms.json).
For ExecuTorch 1.6.0:

| File | Backend | Size |
|------|---------|-----:|
| [`qwen3_0.6b_xnnpack.pte`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases/download/v1.6.0/qwen3_0.6b_xnnpack.pte) | XNNPACK (CPU, every native platform) | 470 MB |
| [`qwen3_0.6b_mlx.pte`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases/download/v1.6.0/qwen3_0.6b_mlx.pte) | MLX (Apple-Silicon GPU, macOS) | 617 MB |
| [`qwen3_0.6b_tokenizer.json`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases/download/v1.6.0/qwen3_0.6b_tokenizer.json) | both | 11 MB |
| [`mlx.metallib`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases/download/v1.6.0/mlx.metallib) | MLX only (§3) | 1.4 MB |

The license text ships next to them as
[`Qwen3-LICENSE`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases/download/v1.6.0/Qwen3-LICENSE).
An export only runs on the ExecuTorch version it was exported for, so take the
files from the release that matches `executorchVersion`.

### Export your own

[`export_llm.py`](https://github.com/abdelaziz-mahdy/executorch_flutter_models/blob/main/python/export_llm.py)
exports a Hugging Face causal LM the same way the hosted files were made:

```bash
# (in a Python env with executorch + optimum-executorch installed)
python export_llm.py --model Qwen/Qwen3-0.6B --backend xnnpack \
  --out qwen3_0.6b_xnnpack.pte --extra-eos 151645 151643
python export_llm.py --model Qwen/Qwen3-0.6B --backend mlx \
  --out qwen3_0.6b_mlx.pte --extra-eos 151645 151643
```

- **XNNPACK:** `8da4w` linears, 8-bit embedding, dynamic shapes (parallel
  prefill).
- **MLX:** 4-bit linears, MLX partitioner.
- Both embed the stop set as `get_eos_ids` (the runner reads the plural name;
  optimum writes `get_eos_id`), so the runner stops by itself. `--extra-eos`
  adds the end-of-turn token.
- No custom ops, so the prebuilt LLM runtimes can run the result.

Take `tokenizer.json` from the model's Hugging Face repo. Gemma 4 E2B has its
own recipes (`export_gemma4_{xnnpack,mlx}.py` in the same folder), because its
multimodal checkpoint needs the text model extracted first.

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
read the copy next to the native library, so **your app must provide the
metallib and tell the runner where it is**:

1. Get `mlx.metallib` from the models release (table in §1). It is the same
   file the `xnnpack-mlx-llm` prebuilt carries in its `lib/`, and must match the
   runtime's ExecuTorch version.
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

> The example app downloads the metallib with the hosted MLX model, or lets you
> **pick the metallib file** for your own model, and disables its sandbox for
> convenience — that's a dev shortcut; real apps keep the sandbox and use the
> bundle-and-path approach above.

---

## 4. Using the API

### Chat templates

Instruction-tuned models expect their own turn markers, or they produce
degenerate output. The runner tokenizes the prompt verbatim, so **you format
it**. For multi-turn, re-send the whole conversation each turn and call
`llm.reset()` between turns (the prompt carries the full history).

**Qwen3 (ChatML)**, with thinking turned off by the empty think block:

```
<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

**Gemma 4:**

```
<bos><|turn>user\n{message}<turn|>\n<|turn>model\n
```

The Gemma turn tokens are the literal strings `<|turn>` (start) and `<turn|>`
(end) — **not** `<start_of_turn>`/`<end_of_turn>`, which this tokenizer splits
into many tokens.

The hosted catalog (`llms.json`) carries each model's template as data
(`chatFormat`), which is how the example app's chat screen stays
model-agnostic.

### Stopping

When an export declares its end-of-turn token in `get_eos_ids` (the hosted
Qwen3 files and the MLX Gemma export do), the runner stops by itself, though
the marker's text may still arrive as the last piece. The XNNPACK Gemma export
only declares `<eos>`, so the model keeps emitting `<turn|>`. Handle both at the
app level — strip the marker from display and stop:

```dart
.listen((piece) {
  if (piece.contains('<|im_end|>')) {
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

## 5. Web

The same API works in the browser for XNNPACK exports. The LLM runner is its
own WebAssembly module (`wasm/executorch_llm.{js,wasm}`), separate from the
vision one, and runs in a Web Worker (`js/executorch_llm_worker.js`) so the
page keeps painting while tokens are generated. `dart run
executorch_flutter:setup_web` copies all three into your app's `web/`.

```dart
final llm = await ExecuTorchLLM.load(
  modelPath: 'models/qwen3_0.6b_xnnpack.pte',      // URLs on web
  tokenizerPath: 'models/qwen3_0.6b_tokenizer.json',
);
```

- **Paths are URLs**, resolved against the page. Browsers enforce CORS, and
  GitHub release assets send no CORS header, so serve the files from your own
  origin or a host that sends `Access-Control-Allow-Origin` (see the README's
  "Serving models on web").
- **Caching:** the worker reads a URL from the origin private file system
  (OPFS) when it is stored there — directory `executorch-llm`, file name
  `Uri.encodeComponent(url)` — and downloads it otherwise. Store the files
  there yourself to download once with your own progress UI and keep them
  across reloads; the example app's `llm_files_web.dart` does this. (Cache
  Storage doesn't work for this: browsers cap a single entry, some at a few
  hundred MB.)
- **Large files on capped hosts:** if the model URL answers 404, the worker
  looks for `<url>.parts.json` (`{"size": bytes, "parts": [names relative to
  it]}`) and joins the parts in order. GitHub Pages deploys refuse files over
  100 MB, so this repo's web demo serves the 470 MB model split into 90 MB
  parts; the example's download handles the same layout.
- **Memory:** the weights live in the Wasm heap (up to 4 GB), so a 0.6B model
  fits easily; much larger models will not.
- **Speed:** one thread, since multi-threaded Wasm needs cross-origin isolation
  headers that GitHub Pages and most static hosts can't send. Qwen3 0.6B
  generates about 3 tokens/s on an M2 Pro.
- **MLX** and separate `.ptd` weight files are not available on web.

---

## Platform support

| Platform | XNNPACK (CPU) LLM | MLX (GPU) LLM |
|----------|:----------------:|:-------------:|
| macOS (arm64) | ✅ | ✅ (needs metallib + macOS 14) |
| macOS (x64) | ✅ | — |
| iOS / Android / Linux / Windows | ✅ (CPU) | — |
| Web | ✅ (CPU, one thread; see §5) | — |

Performance on an M2 Pro (release build, approx):

| Model | Backend | Prefill | Decode |
|-------|---------|--------:|-------:|
| Qwen3 0.6B | XNNPACK | ~75 tok/s | ~8 tok/s |
| Qwen3 0.6B | MLX | ~590 tok/s | ~155 tok/s |
| Gemma 4 E2B | XNNPACK | — | ~4 tok/s |
| Gemma 4 E2B | MLX | — | ~30–40 tok/s |

The `debug: true` build option makes the native library log every XNNPACK
operator, which slows generation by an order of magnitude; measure with it off.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Error 0x23` `MLXBackend` metallib | metallib not shipped — pass `mlxMetallibPath` (§3) |
| `Error 0x10` resize static tensor | older build without auto-detect prefill; update the native lib |
| Garbage / repeated tokens | wrong chat template for the model (§4) |
| Never stops | the export lacks the end-of-turn token in `get_eos_ids` — stop at app level (§4) |
| Very slow generation | native library built with `debug: true` |
| MLX build: "cannot execute tool 'metal'" | `xcodebuild -downloadComponent MetalToolchain` |
| App won't launch (macOS, MLX) | app deployment target < 14.0 (§2) |
