# ExecuTorch Flutter

[![pub package](https://img.shields.io/pub/v/executorch_flutter.svg)](https://pub.dev/packages/executorch_flutter)
[![build](https://github.com/abdelaziz-mahdy/executorch_flutter/actions/workflows/build.yml/badge.svg)](https://github.com/abdelaziz-mahdy/executorch_flutter/actions/workflows/build.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Flutter plugin for on-device ML inference using PyTorch ExecuTorch, supporting Android, iOS, macOS, Windows, Linux, and Web.

**[pub.dev](https://pub.dev/packages/executorch_flutter)** | **[Live Demo](https://abdelaziz-mahdy.github.io/executorch_flutter/)** | **[Example App](example/)**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Platform Support](#platform-support)
- [API Reference](#api-reference)
- [On-device LLM (Gemma 4)](docs/LLM.md)
- [Build Configuration](#build-configuration)
- [Advanced Usage](#advanced-usage)
- [Web Platform](#web-platform)
- [Example Application](#example-application)
- [Model Export](#converting-pytorch-models-to-executorch)
- [Troubleshooting](#troubleshooting)
- [Vulkan Backend (Experimental)](#experimental-vulkan-backend)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ExecuTorch Flutter provides a simple Dart API for loading and running ExecuTorch models (`.pte` files) in your Flutter applications. The package handles all native platform integration, providing you with a straightforward interface for on-device machine learning inference.

Writing a Dart server or command-line tool instead — no Flutter SDK involved?
Use **[`executorch_dart`](https://pub.dev/packages/executorch_dart)** directly;
this package is a thin wrapper over it that adds Flutter asset-bundle loading
and Web support.

## Features

- **Cross-Platform**: Android (API 23+), iOS (13.0+), macOS (11.0+), Windows, Linux, and Web
- **Type-Safe API**: dart:ffi bindings with type-safe Dart wrapper classes
- **Async Operations**: Non-blocking model loading and inference
- **Multiple Models**: Support for concurrent model instances
- **Error Handling**: Structured exception handling with clear error messages
- **Backend Support**: XNNPACK (all platforms), CoreML (Apple), Metal + MLX (macOS), Vulkan (opt-in)
- **13 Tensor Dtypes**: float32/64, float16, bfloat16, int8/16/32/64, uint8/16/32/64, bool
- **On-device LLM (experimental)**: streaming text generation with Gemma 4 (XNNPACK CPU + MLX Apple-GPU) — see **[docs/LLM.md](docs/LLM.md)**
- **Live Camera**: Real-time inference with camera stream support

### Library Size by Backend

<!-- NATIVE_VERSION_START -->
📊 **[Download Release Size Comparison (SVG)](https://github.com/abdelaziz-mahdy/executorch_native/releases/download/v1.3.1.9/size-report-release.svg)** | **[Download Debug Size Comparison (SVG)](https://github.com/abdelaziz-mahdy/executorch_native/releases/download/v1.3.1.9/size-report-debug.svg)** | **[JSON Report](https://github.com/abdelaziz-mahdy/executorch_native/releases/download/v1.3.1.9/size-report.json)**
<!-- NATIVE_VERSION_END -->

---

## Installation

**Requirements:** Flutter 3.38+ (first version with native assets hooks)

<!-- PACKAGE_VERSION_START -->
```yaml
dependencies:
  executorch_flutter: ^0.6.0
```
<!-- PACKAGE_VERSION_END -->

---

## Quick Start

### 1. Load a Model

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// Load from Flutter assets (recommended - works on all platforms)
final model = await loadModelFromAsset('assets/models/model.pte');
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

| Method | Platforms | Use Case |
|--------|-----------|----------|
| `loadModelFromAsset(path)` | All (including web) | Bundled assets |
| `ExecuTorchModel.loadFromBytes(bytes)` | All (including web) | Downloaded/cached models |
| `ExecuTorchModel.load(filePath)` | Native only | External file paths |

`loadModelFromAsset` is a top-level function, not a static method on
`ExecuTorchModel` — this is what changed in 0.6.0 (see
[CHANGELOG.md](CHANGELOG.md)).

```dart
// From bytes
final byteData = await rootBundle.load('assets/models/model.pte');
final model = await ExecuTorchModel.loadFromBytes(byteData.buffer.asUint8List());

// From file path (native platforms only)
final model = await ExecuTorchModel.load('/path/to/model.pte');
```

---

## Platform Support

| Platform | Min Version | Architectures | Backends |
|----------|-------------|---------------|----------|
| **Android** | API 23 | arm64-v8a, armeabi-v7a, x86_64, x86 | XNNPACK, Vulkan* |
| **iOS** | 13.0+ | arm64, x86_64+arm64 (sim) | XNNPACK, CoreML, Vulkan* |
| **macOS** | 11.0+ | arm64, x86_64 | XNNPACK, CoreML, Metal, MLX*, Vulkan* |
| **Windows** | 10+ | x64 | XNNPACK, Vulkan* |
| **Linux** | Ubuntu 20.04+ | x64, arm64 | XNNPACK, Vulkan* |
| **Web** | Modern browsers | WebAssembly | XNNPACK (Wasm SIMD) |

*\*Opt-in. Vulkan is experimental — see [Vulkan Backend](#experimental-vulkan-backend). MLX is the Apple-Silicon GPU runtime used by the LLM path (macOS 14+, arm64) — see [docs/LLM.md](docs/LLM.md).*

### Platform Configuration

If you encounter deployment target errors, update your project settings:

<details>
<summary><b>iOS Deployment Target (iOS 13.0+)</b></summary>

1. Open `ios/Runner.xcworkspace` in Xcode
2. Select **Runner** target → **Build Settings**
3. Search "iOS Deployment Target" → Set to **13.0**
</details>

<details>
<summary><b>macOS Deployment Target (macOS 11.0+)</b></summary>

1. Open `macos/Runner.xcworkspace` in Xcode
2. Select **Runner** target → **Build Settings**
3. Search "macOS Deployment Target" → Set to **11.0**
</details>

After updating, run:
```bash
flutter clean && flutter pub get && flutter build <platform>
```

---

## API Reference

### ExecuTorchModel

```dart
// Top-level loader — Flutter asset bundle, all platforms including web
Future<ExecuTorchModel> loadModelFromAsset(String assetPath)

// ExecuTorchModel static factories
static Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes)
static Future<ExecuTorchModel> load(String filePath)  // Native only

// Inference
Future<List<TensorData>> forward(List<TensorData> inputs)

// Lifecycle
Future<void> dispose()
bool get isDisposed
String get modelId
```

`ExecutorchManager.instance` also has `loadModelFromAssets(assetPath)` — an
extension method that does the same thing but caches the model in the
manager, like its `loadModel`/`loadModelFromBytes` counterparts.

### ExecuTorchLLM (experimental)

On-device generative text — **Google Gemma 4 E2B** — with token-by-token
streaming, separate from the tensor API. Loaded from **file paths** (weights are
1+ GB) and driven by a stateful decode loop + tokenizer + KV cache. Backends:
**XNNPACK** (CPU, all platforms) and **MLX** (Apple-Silicon GPU, macOS arm64).

```dart
// Load (file paths; mlxMetallibPath is MLX-only)
static Future<ExecuTorchLLM> load({
  required String modelPath,
  required String tokenizerPath,
  String? dataPath,
  String? mlxMetallibPath,
})

// Stream tokens as they decode
Stream<String> generate(String prompt, {GenConfig config})

// Control / lifecycle
void stop();              // cooperative cancel mid-generation
void reset();             // clear KV cache / start a new conversation
Future<void> dispose();   // release native resources

// GenConfig — temperature-only sampling (no top-p/top-k)
const GenConfig({int maxNewTokens, int seqLen, double temperature, bool echo, bool ignoreEos});
```

```dart
final llm = await ExecuTorchLLM.load(
  modelPath: '/path/gemma-4-E2B-it_xnnpack.pte',
  tokenizerPath: '/path/gemma-4-E2B-it_tokenizer.json',
);
// Gemma 4 needs its turn markers around the message:
final prompt = '<bos><|turn>user\nExplain Flutter in one line.<turn|>\n<|turn>model\n';
await for (final piece in llm.generate(prompt,
    config: const GenConfig(maxNewTokens: 512, temperature: 0))) {
  stdout.write(piece);
}
await llm.dispose();
```

Enable it in `pubspec.yaml` (`hooks.user_defines.executorch_dart`):

```yaml
llm: true
backends: [xnnpack, mlx]   # mlx is auto-dropped off macOS-arm64
```

> 📖 **Full guide: [docs/LLM.md](docs/LLM.md)** — model **export** (the Gemma 4
> scripts), the chat template, the MLX `mlx.metallib` shipping step, stopping,
> platform support, and troubleshooting. A complete streaming chat screen is in
> [`example/lib/screens/llm_chat_screen.dart`](example/lib/screens/llm_chat_screen.dart).

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
// Check specific backend
if (BackendQuery.isAvailable(Backend.coreml)) {
  model = await loadModelFromAsset('assets/model_coreml.pte');
} else {
  model = await loadModelFromAsset('assets/model_xnnpack.pte');
}

// List all available backends
final backends = BackendQuery.available;
print('Available: ${backends.map((b) => b.displayName).join(", ")}');
```

| Backend | Display Name | Platforms |
|---------|--------------|-----------|
| `Backend.xnnpack` | XNNPACK | All |
| `Backend.coreml` | CoreML | iOS, macOS |
| `Backend.metal` | Metal | macOS |
| `Backend.vulkan` | Vulkan | Android, iOS, macOS, Windows, Linux |
| `Backend.qnn` | Qualcomm QNN | Android |
| `Backend.mps` | *(deprecated — use `metal`)* | macOS |

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

Configure the native build in your app's `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    executorch_dart:
      debug: false              # Enable debug logging
      build_mode: "prebuilt"    # "prebuilt", "local", or "source"
      # prebuilt_version: "1.3.1.9"  # Optional: pin specific native version
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

The key under `user_defines:` is the package that owns the native build —
`executorch_dart`, even though you depend on `executorch_flutter`. This
package used to own the build directly and read `executorch_flutter:` here;
see the 0.6.0 entry in [CHANGELOG.md](CHANGELOG.md) if you're migrating.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `debug` | `false` | Debug logging + debug binaries |
| `build_mode` | `"prebuilt"` | `"prebuilt"` (fast), `"local"` (pre-compiled), or `"source"` (from source) |
| `prebuilt_version` | Current | Prebuilt release version |
| `executorch_source` | - | Path to local ExecuTorch checkout (source mode) |
| `local_lib_dir` | - | Path to pre-compiled libraries (local mode) |
| `backends` | Platform-specific | Backends to enable |

### Default Backends by Platform

| Platform | Defaults |
|----------|----------|
| Android | xnnpack |
| iOS | xnnpack, coreml |
| macOS | xnnpack, coreml, metal |
| Windows/Linux | xnnpack |

Listing `backends:` replaces the defaults entirely — include every backend you
want. `vulkan`, `mlx`, and `qnn` are never on by default. A legacy `mps` entry
is accepted and treated as `metal` on macOS.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `EXECUTORCH_BUILD_MODE` | Override build mode (`prebuilt`, `local`, `source`) |
| `EXECUTORCH_SOURCE_DIR` | Path to local ExecuTorch checkout (source mode) |
| `EXECUTORCH_INSTALL_DIR` | Path to pre-compiled libraries (local mode) |
| `EXECUTORCH_CACHE_DIR` | Custom cache directory for source builds |
| `EXECUTORCH_DISABLE_DOWNLOAD` | Skip prebuilt download |

---

## Advanced Usage

### Preprocessing Strategies

The example app demonstrates three preprocessing approaches:

| Strategy | Performance | Platforms | Dependencies |
|----------|-------------|-----------|--------------|
| **GPU Shader** | ~75ms (web), comparable to OpenCV (native) | All | None |
| **OpenCV** | Very fast | Native only | opencv_dart |
| **CPU (image lib)** | ~560ms (web), slower | All | image |

**[GPU Preprocessing Tutorial](example/GPU_PREPROCESSING.md)** - Step-by-step guide with GLSL shader examples.

---

## Web Platform

Web runs via WebAssembly with XNNPACK backend.

### Performance

| Metric | Native | Web (Wasm) |
|--------|--------|------------|
| YOLO11n Inference | ~50-100ms | ~622ms |
| Total E2E | ~150-200ms | ~855ms |

**When to use Web:**
- Demos and prototyping
- Interactive inference (sub-second)
- No app install required

**Not recommended for:**
- Real-time camera inference
- High-throughput batch processing

### Setup

1. Run setup script:
   ```bash
   dart run executorch_flutter:setup_web
   ```

2. Add to `web/index.html`:
   ```html
   <head>
     <script src="js/executorch_wrapper.js"></script>
   </head>
   ```

3. Use XNNPACK models (same as native).

### Serving models on web

**Host your models on an origin you control, or bundle them as Flutter assets.**

On native you can download a `.pte` from anywhere. On web the browser enforces
CORS, and a cross-origin fetch only succeeds if the server sends
`Access-Control-Allow-Origin`. GitHub release assets do not send it, so
fetching a model straight from a GitHub release fails in every browser:

```
Access to fetch at 'https://github.com/.../releases/download/...pte'
blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present
```

Three approaches that work:

- **Bundle the model as an asset** and load it with `loadModelFromAsset` — the
  simplest option, and the right one when the model ships with the app.
- **Serve it from your own origin**, same host as the app, so CORS never
  applies.
- **Use a CDN or object store that sends `Access-Control-Allow-Origin`** if the
  model must be fetched cross-origin.

Remember that label files and any other side-car assets go through the same
check — it is easy to fix the model download and still fail on labels.

This repo's own web demo takes the second approach: its deploy workflow copies
the models into the published site so they are served from the app's origin.

---

## Example Application

The `example/` directory includes:

- **Unified Model Playground** - Multiple model types in one interface
- **MobileNet V3** - Image classification (1000 ImageNet classes)
- **YOLO** - Object detection (v5, v8, v11)
- **Camera Mode** - Real-time inference
- **Settings** - Thresholds, preprocessing, performance overlay

```bash
cd example
flutter run -d macos  # or ios, android, windows, linux, chrome
```

---

## Converting PyTorch Models to ExecuTorch

Convert your PyTorch models to `.pte` format:

**[Official ExecuTorch Export Guide](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html)**

**Example app models** are hosted at [executorch_flutter_models](https://github.com/abdelaziz-mahdy/executorch_flutter_models) and downloaded automatically.

To export manually:
```bash
cd models/python
python3 main.py
```

**LLM (Gemma 4)** models are exported with dedicated scripts (they need a
tokenizer + quantization recipe, not the tensor export path):

```bash
python models/python/export_gemma4_xnnpack.py   # CPU model (all platforms)
python models/python/export_gemma4_mlx.py        # Apple-GPU model (macOS)
```

See **[docs/LLM.md](docs/LLM.md)** for the full export recipe, the required
`tokenizer.json` / `mlx.metallib`, and how to load them with `ExecuTorchLLM`.

---

## Troubleshooting

<details>
<summary><b>Model loading fails</b></summary>

- Verify asset is listed in `pubspec.yaml`
- Check model bytes: `modelBytes.lengthInBytes > 0`
- Re-export with correct ExecuTorch version
</details>

<details>
<summary><b>Inference returns error</b></summary>

- Double-check the `shape:`/`dataType:` you pass match what the model was
  exported with — `ExecuTorchModel` doesn't expose shape introspection, so
  this has to come from the export script or the model's documentation
- Ensure shapes match exactly (including batch dimension)
- Verify the dtype matches what the model was exported with — a `data size
  mismatch` error means `data.length` != `elementCount * dataType.sizeInBytes`
- Prefer `ExecutorchManager.instance.createTensorData(...)` over packing bytes
  by hand, especially for `float16`/`bfloat16`
</details>

<details>
<summary><b>Edits to native C++ code have no effect</b></summary>

The default `prebuilt` build mode downloads an already-compiled library, so
local changes under `native/` are ignored. Use `build_mode: "source"` with
`executorch_source:` pointing at a local ExecuTorch checkout. See
[CONTRIBUTING.md](../../CONTRIBUTING.md#source-build-from-local-executorch-checkout).
</details>

<details>
<summary><b>Memory issues</b></summary>

- Always call `dispose()` when done
- Don't load too many models simultaneously
</details>

---

## Experimental: Vulkan Backend

> **Warning**: Vulkan is experimental and opt-in.

### Status

| Platform | Status |
|----------|--------|
| Android | Works on most devices; see [#26](https://github.com/abdelaziz-mahdy/executorch_flutter/issues/26) for PowerVR GPU status |
| Windows/Linux | Generally functional |
| macOS/iOS | Works via MoltenVK (Vulkan-to-Metal translation) |

### Enable Vulkan

```yaml
hooks:
  user_defines:
    executorch_dart:
      backends:
        - xnnpack
        - vulkan
```

### Vulkan Troubleshooting

<details>
<summary><b>"uniform data allocation exceeded" on Android</b></summary>

This can occur when Vulkan tensor metadata exceeds the per-tensor uniform buffer limit. Fix submitted upstream: [pytorch/executorch#17294](https://github.com/pytorch/executorch/pull/17294).
</details>

<details>
<summary><b>Vulkan on PowerVR GPUs</b></summary>

Some PowerVR devices may produce incorrect Vulkan results due to texture dimension limits. Being tracked upstream: [pytorch/executorch#17299](https://github.com/pytorch/executorch/issues/17299). XNNPACK is recommended as a fallback.
</details>

### Recommendations

- **Production**: Use XNNPACK (stable everywhere)
- **Apple platforms**: Use CoreML (iOS/macOS) or Metal (macOS) instead of Vulkan
- **Testing**: Report issues with device info and logs

**[Report Vulkan Issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues)**

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Acknowledgments

- **[opencv_dart](https://github.com/rainyl/opencv_dart)** - Referenced for understanding Flutter native assets build patterns and cross-platform FFI packaging

## License

MIT License - see [LICENSE](LICENSE).

## Support

- [Official ExecuTorch Documentation](https://pytorch.org/executorch/stable/getting-started-architecture.html)
- [Report Issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues)
- [Roadmap](../../ROADMAP.md)

---

Built with love for the Flutter and PyTorch communities.
