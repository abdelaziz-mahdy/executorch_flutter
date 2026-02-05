# ExecuTorch Flutter

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

## Features

- **Cross-Platform**: Android (API 23+), iOS (13.0+), macOS (11.0+), Windows, Linux, and Web
- **Type-Safe API**: dart:ffi bindings with type-safe Dart wrapper classes
- **Async Operations**: Non-blocking model loading and inference
- **Multiple Models**: Support for concurrent model instances
- **Error Handling**: Structured exception handling with clear error messages
- **Backend Support**: XNNPACK, CoreML, MPS, Vulkan backends
- **Live Camera**: Real-time inference with camera stream support

### Library Size by Backend

<!-- NATIVE_VERSION_START -->
📊 **[Download Release Size Comparison (SVG)](https://github.com/abdelaziz-mahdy/executorch_native/releases/download/v1.1.0.7/size-report-release.svg)** | **[Download Debug Size Comparison (SVG)](https://github.com/abdelaziz-mahdy/executorch_native/releases/download/v1.1.0.7/size-report-debug.svg)** | **[JSON Report](https://github.com/abdelaziz-mahdy/executorch_native/releases/download/v1.1.0.7/size-report.json)**
<!-- NATIVE_VERSION_END -->

---

## Installation

**Requirements:** Flutter 3.38+ (first version with native assets hooks)

<!-- PACKAGE_VERSION_START -->
```yaml
dependencies:
  executorch_flutter: ^0.3.0
```
<!-- PACKAGE_VERSION_END -->

---

## Quick Start

### 1. Load a Model

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// Load from Flutter assets (recommended - works on all platforms)
final model = await ExecuTorchModel.loadFromAsset('assets/models/model.pte');
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
| `loadFromAsset(path)` | All (including web) | Bundled assets |
| `loadFromBytes(bytes)` | All (including web) | Downloaded/cached models |
| `load(filePath)` | Native only | External file paths |

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
| **macOS** | 11.0+ | arm64, x86_64 | XNNPACK, CoreML, MPS, Vulkan* |
| **Windows** | 10+ | x64 | XNNPACK, Vulkan* |
| **Linux** | Ubuntu 20.04+ | x64, arm64 | XNNPACK, Vulkan* |
| **Web** | Modern browsers | WebAssembly | XNNPACK (Wasm SIMD) |

*\*Vulkan is opt-in and experimental. See [Vulkan Backend](#experimental-vulkan-backend).*

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
// Load methods
static Future<ExecuTorchModel> loadFromAsset(String assetPath)
static Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes)
static Future<ExecuTorchModel> load(String filePath)  // Native only

// Inference
Future<List<TensorData>> forward(List<TensorData> inputs)

// Lifecycle
Future<void> dispose()
bool get isDisposed
```

### TensorData

```dart
final tensor = TensorData(
  shape: [1, 3, 224, 224],       // Dimensions
  dataType: TensorType.float32,  // float32, int32, int8, uint8
  data: Uint8List(...),          // Raw bytes
  name: 'input_0',               // Optional
);
```

### BackendQuery

Query available backends at runtime:

```dart
// Check specific backend
if (BackendQuery.isAvailable(Backend.coreml)) {
  model = await ExecuTorchModel.loadFromAsset('assets/model_coreml.pte');
} else {
  model = await ExecuTorchModel.loadFromAsset('assets/model_xnnpack.pte');
}

// List all available backends
final backends = BackendQuery.available;
print('Available: ${backends.map((b) => b.displayName).join(", ")}');
```

| Backend | Display Name | Platforms |
|---------|--------------|-----------|
| `Backend.xnnpack` | XNNPACK | All |
| `Backend.coreml` | CoreML | iOS, macOS |
| `Backend.mps` | Metal Performance Shaders | macOS |
| `Backend.vulkan` | Vulkan | Android, iOS, macOS, Windows, Linux |

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
    executorch_flutter:
      debug: false              # Enable debug logging
      build_mode: "prebuilt"    # "prebuilt" or "source"
      prebuilt_version: "1.0.1.21"
      backends:
        - xnnpack
        - coreml
        - mps
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `debug` | `false` | Debug logging + debug binaries |
| `build_mode` | `"prebuilt"` | `"prebuilt"` (fast) or `"source"` (custom) |
| `prebuilt_version` | Current | Prebuilt release version |
| `backends` | Platform-specific | Backends to enable |

### Default Backends by Platform

| Platform | Defaults |
|----------|----------|
| Android | xnnpack |
| iOS | xnnpack, coreml |
| macOS | xnnpack, coreml, mps |
| Windows/Linux | xnnpack |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `EXECUTORCH_BUILD_MODE` | Override build mode |
| `EXECUTORCH_CACHE_DIR` | Custom cache directory |
| `EXECUTORCH_DISABLE_DOWNLOAD` | Skip prebuilt download |
| `EXECUTORCH_INSTALL_DIR` | Local ExecuTorch path |

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

- Check `model.inputShapes` / `model.outputShapes`
- Verify tensor data types match expectations
- Ensure shapes match exactly (including batch dimension)
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
| Android | Works on most devices; some UBO size issues |
| Windows/Linux | Generally functional |
| macOS/iOS | **Not functional** - MoltenVK crashes |

### Enable Vulkan

```yaml
hooks:
  user_defines:
    executorch_flutter:
      backends:
        - xnnpack
        - vulkan
```

### Recommendations

- **Production**: Use XNNPACK (stable everywhere)
- **Apple platforms**: Use CoreML or MPS instead of Vulkan
- **Testing**: Report issues with device info and logs

**[Report Vulkan Issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues)**

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- **[opencv_dart](https://github.com/rainyl/opencv_dart)** - Referenced for understanding Flutter native assets build patterns and cross-platform FFI packaging

## License

MIT License - see [LICENSE](LICENSE).

## Support

- [Official ExecuTorch Documentation](https://pytorch.org/executorch/stable/getting-started-architecture.html)
- [Report Issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues)
- [Roadmap](ROADMAP.md)

---

Built with love for the Flutter and PyTorch communities.
