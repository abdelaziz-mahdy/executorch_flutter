# executorch_dart

[![pub package](https://img.shields.io/pub/v/executorch_dart.svg)](https://pub.dev/packages/executorch_dart)
[![build](https://github.com/abdelaziz-mahdy/executorch_flutter/actions/workflows/release.yml/badge.svg)](https://github.com/abdelaziz-mahdy/executorch_flutter/actions/workflows/release.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Dart on-device ML inference with PyTorch ExecuTorch. **No Flutter SDK
required** — runs in any Dart program, including servers and command-line
tools, on Android, iOS, macOS, Linux, and Windows.

**[pub.dev](https://pub.dev/packages/executorch_dart)** | **[Example (CLI)](example/)**

Building a Flutter app instead? Use
**[`executorch_flutter`](https://pub.dev/packages/executorch_flutter)** — it
wraps this package and adds Flutter asset-bundle loading plus Web support.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Platform Support](#platform-support)
- [API Reference](#api-reference)
- [Build Configuration](#build-configuration)
- [Example](#example)
- [License](#license)

---

## Overview

`executorch_dart` loads and runs ExecuTorch `.pte` models directly from Dart
via `dart:ffi` and native assets — no platform channel, no Flutter engine, no
widget tree. It's the pure-Dart core that `executorch_flutter` builds on; use
it directly whenever your code doesn't need Flutter, such as a CLI tool or a
Dart server.

- **Vision inference**: load a model, run `forward()`, get tensors back.
- **On-device LLM (experimental)**: streaming text generation with Gemma 4 —
  see `ExecuTorchLLM` below.
- **Backends**: XNNPACK (all platforms), CoreML + Metal + MLX (Apple), Vulkan
  (opt-in).

## Installation

**Requirements:** Dart SDK 3.10+ with native assets support. `dart run`,
`dart test`, and `dart build` all invoke this package's build hook
automatically — there's nothing to compile by hand.

<!-- Not auto-updated by CI (update-readme.yml only rewrites
     packages/executorch_flutter/README.md) — bump this by hand on release. -->
```yaml
dependencies:
  executorch_dart: ^0.6.2
```

## Quick Start

### 1. Load a Model

```dart
import 'package:executorch_dart/executorch_dart.dart';

final model = await ExecuTorchModel.load('/path/to/model.pte');
```

### 2. Run Inference

```dart
final inputTensor = TensorData(
  shape: [1, 3, 224, 224],
  dataType: TensorType.float32,
  data: yourImageBytes,
);

final outputs = await model.forward([inputTensor]);

for (var output in outputs) {
  print('Shape: ${output.shape}, Type: ${output.dataType}');
}
```

### 3. Clean Up

```dart
await model.dispose();
```

### Model Loading Options

| Method | Use Case |
|--------|----------|
| `ExecuTorchModel.load(filePath)` | Load from a file path |
| `ExecuTorchModel.loadFromBytes(bytes)` | Downloaded/generated model bytes |

```dart
// From bytes
final bytes = await File('/path/to/model.pte').readAsBytes();
final model = await ExecuTorchModel.loadFromBytes(bytes);
```

Flutter apps should load from the asset bundle with `loadModelFromAsset` from
`package:executorch_flutter/executorch_flutter.dart` instead — this package
has no dependency on `dart:ui` or `rootBundle`.

---

## Platform Support

| Platform | Min Version | Architectures | Backends |
|----------|-------------|---------------|----------|
| **Android** | API 23 | arm64-v8a, armeabi-v7a, x86_64, x86 | XNNPACK, Vulkan* |
| **iOS** | 13.0+ | arm64, x86_64+arm64 (sim) | XNNPACK, CoreML, Vulkan* |
| **macOS** | 11.0+ | arm64, x86_64 | XNNPACK, CoreML, Metal, MLX*, Vulkan* |
| **Windows** | 10+ | x64 | XNNPACK, Vulkan* |
| **Linux** | Ubuntu 20.04+ | x64, arm64 | XNNPACK, Vulkan* |

*\*Opt-in. MLX is the Apple-Silicon GPU runtime used by the LLM path (macOS
14+, arm64).*

There's no Web target here — `dart:ffi` doesn't run in a browser. Use
`executorch_flutter` for Web; it ships a WebAssembly build behind the same
API.

---

## API Reference

### ExecuTorchModel

The entire model API is four instance members plus two static loaders:

```dart
// Load (static factories)
static Future<ExecuTorchModel> load(String filePath)
static Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes)

// Inference
Future<List<TensorData>> forward(List<TensorData> inputs)

// Lifecycle
Future<void> dispose()
bool get isDisposed
String get modelId
```

No options, no timeouts, no singleton manager — you own the model's
lifecycle.

### TensorData

```dart
final tensor = TensorData(
  shape: [1, 3, 224, 224],       // Dimensions
  dataType: TensorType.float32,  // See dtype table below
  data: Uint8List(...),          // Raw bytes, little-endian
  name: 'input_0',               // Optional
);
```

**Supported dtypes** — all 13 map 1:1 to ExecuTorch's native types:

| Dtype | Bytes | Dtype | Bytes | Dtype | Bytes |
|-------|-------|-------|-------|-------|-------|
| `float32` | 4 | `int8` | 1 | `uint32` | 4 |
| `float64` | 8 | `int16` | 2 | `uint64` | 8 |
| `float16` | 2 | `int32` | 4 | `bool_` | 1 |
| `bfloat16` | 2 | `int64` | 8 | | |
| `uint8` | 1 | `uint16` | 2 | | |

Building `data` by hand is error-prone, so use `ExecutorchManager` to encode
numeric lists — it handles float16/bfloat16 conversion (round-to-nearest-even)
and endianness:

```dart
final tensor = ExecutorchManager.instance.createTensorData(
  shape: [1, 4],
  dataType: TensorType.float16,
  data: [1.0, 2.5, -3.25, 0.5],
);
```

### BackendQuery

Query available backends at runtime:

```dart
if (BackendQuery.isAvailable(Backend.coreml)) {
  model = await ExecuTorchModel.load('/path/model_coreml.pte');
} else {
  model = await ExecuTorchModel.load('/path/model_xnnpack.pte');
}

final backends = BackendQuery.available;
print('Available: ${backends.map((b) => b.displayName).join(", ")}');
```

### ExecuTorchLLM (experimental)

On-device generative text — **Google Gemma 4 E2B** — with token-by-token
streaming, separate from the tensor API (`load` / `generate` `Stream<String>`
/ `stop` / `reset` / `dispose`). See **[executorch_flutter's LLM
guide](https://github.com/abdelaziz-mahdy/executorch_flutter/blob/main/packages/executorch_flutter/docs/LLM.md)**
for model export and setup — the API is identical from pure Dart; just import
`package:executorch_dart/executorch_dart.dart` instead of the Flutter
package.

### Exception Hierarchy

```
ExecuTorchException (base)
├── ExecuTorchModelException      // Model loading/lifecycle
├── ExecuTorchInferenceException  // Inference execution
├── ExecuTorchValidationException // Tensor validation
├── ExecuTorchMemoryException     // Memory/resources
├── ExecuTorchIOException         // File I/O
└── ExecuTorchPlatformException   // Platform communication
```

---

## Build Configuration

The native library is downloaded (or compiled) by a native-assets build
hook. Configure it under `hooks: user_defines: executorch_dart:` in **your
application's** `pubspec.yaml` — the package that will actually run, not a
library you merely depend on:

<!-- The prebuilt_version below is not auto-updated by CI — bump it by hand
     on release, alongside _defaultPrebuiltVersion in run_build.dart. -->
```yaml
hooks:
  user_defines:
    executorch_dart:
      debug: false                  # Debug logging + debug binaries
      build_mode: "prebuilt"        # "prebuilt", "local", or "source"
      # prebuilt_version: "1.3.1.9" # Optional: pin a specific native release
      llm: false                    # Opt in to the LLM runner (ExecuTorchLLM)
      # For source mode: build from local ExecuTorch checkout
      # build_mode: "source"
      # executorch_source: "/path/to/executorch"
      # For local mode: point at pre-compiled libraries
      # local_lib_dir: "/path/to/compiled/libs"
      backends:
        - xnnpack
        - coreml
        - metal
```

### All seven keys

| Key | Default | Description |
|-----|---------|--------------|
| `build_mode` | `"prebuilt"` | `"prebuilt"` (download, fast), `"local"` (your own compiled libs), or `"source"` (compile ExecuTorch from a checkout) |
| `backends` | Platform-specific | Backends to enable. Listing `backends:` replaces the defaults entirely — `vulkan`, `mlx`, and `qnn` are never on by default |
| `llm` | `false` | Build the LLM runner (`ExecuTorchLLM`). Selects the `xnnpack-llm` / `xnnpack-mlx-llm` prebuilt variants |
| `debug` | `false` | Enable native debug logging and use debug binaries |
| `local_lib_dir` | - | Path to precompiled `lib/` + `include/` (`build_mode: "local"` only) |
| `executorch_source` | - | Path to a local ExecuTorch checkout (`build_mode: "source"` only) |
| `prebuilt_version` | Current release (e.g. `1.3.1.9`) | Pin a specific prebuilt native release instead of this package version's default |

### Default backends by platform

| Platform | Defaults |
|----------|----------|
| Android | xnnpack |
| iOS | xnnpack, coreml |
| macOS | xnnpack, coreml, metal |
| Windows/Linux | xnnpack |

See **[executorch_flutter's
README](https://github.com/abdelaziz-mahdy/executorch_flutter/blob/main/packages/executorch_flutter/README.md#build-configuration)**
for the environment-variable overrides and troubleshooting — it's the same
build hook underneath.

---

## Example

A minimal CLI example lives in [`example/`](example/):

```bash
cd example
dart run bin/infer.dart /path/to/model.pte
```

Or compile it to a self-contained bundle with the native library included:

```bash
dart build cli
./build/cli/*/bundle/bin/infer /path/to/model.pte
```

---

## License

MIT License - see [LICENSE](LICENSE).

---

Built with love for the Dart and PyTorch communities.
