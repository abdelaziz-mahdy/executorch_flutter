# ExecuTorch Flutter

A Flutter plugin package using ExecuTorch to allow model inference on Android, iOS, macOS, and Web platforms.

**📦 [pub.dev](https://pub.dev/packages/executorch_flutter)** | **🌐 [Live Demo](https://abdelaziz-mahdy.github.io/executorch_flutter/)** | **🔧 [Example App](example/)**

## Overview

ExecuTorch Flutter provides a simple Dart API for loading and running ExecuTorch models (`.pte` files) in your Flutter applications. The package handles all native platform integration, providing you with a straightforward interface for on-device machine learning inference.

## Features

- ✅ **Cross-Platform Support**: Android (API 23+), iOS (17.0+), macOS (12.0+ Apple Silicon), and Web
- ✅ **Type-Safe API**: Generated with Pigeon for reliable cross-platform communication
- ✅ **Async Operations**: Non-blocking model loading and inference execution
- ✅ **Multiple Models**: Support for concurrent model instances
- ✅ **Error Handling**: Structured exception handling with clear error messages
- ✅ **Backend Support**: XNNPACK, CoreML, MPS backends
- ✅ **Live Camera**: Real-time inference with camera stream support

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  executorch_flutter: ^0.0.1
```

## Basic Usage

The package provides a simple, intuitive API that matches native ExecuTorch patterns:

### 1. Load a Model

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// Recommended: Load from Flutter assets (works on all platforms including web)
final model = await ExecuTorchModel.loadFromAsset('assets/models/model.pte');
```

### 2. Run Inference

```dart
// Prepare input tensor
final inputTensor = TensorData(
  shape: [1, 3, 224, 224],
  dataType: TensorType.float32,
  data: yourImageBytes,
);

// Run inference
final outputs = await model.forward([inputTensor]);

// Process outputs
for (var output in outputs) {
  print('Output shape: ${output.shape}');
  print('Output type: ${output.dataType}');
}

// Clean up when done
await model.dispose();
```

### 3. Loading Models

**From Assets (Recommended - works on all platforms including web):**
```dart
// Simply use the asset path - no manual file copying needed
final model = await ExecuTorchModel.loadFromAsset('assets/models/model.pte');
final outputs = await model.forward([inputTensor]);
await model.dispose();
```

**From Bytes (works on all platforms):**
```dart
import 'package:flutter/services.dart' show rootBundle;

final byteData = await rootBundle.load('assets/models/model.pte');
final model = await ExecuTorchModel.loadFromBytes(byteData.buffer.asUint8List());
final outputs = await model.forward([inputTensor]);
await model.dispose();
```

**From File Path (native platforms only):**
```dart
// Only available on Android, iOS, macOS - not on web
final model = await ExecuTorchModel.load('/path/to/model.pte');
final outputs = await model.forward([inputTensor]);
await model.dispose();
```

### Complete Examples

See the `example/` directory for a full working application:

- **[Unified Model Playground](example/lib/screens/unified_model_playground.dart)** - Complete app with MobileNet classification and YOLO object detection, supporting both static images and live camera

## Supported Model Formats

- **ExecuTorch (.pte)**: Optimized PyTorch models converted to ExecuTorch format
- **Input Types**: float32, int8, int32, uint8 tensors
- **Model Size**: Tested with models up to 500MB

> 📖 **Need to export your PyTorch models?** See the [Official ExecuTorch Export Guide](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html) for converting PyTorch models to ExecuTorch format with platform-specific optimizations.

## Platform Requirements

### Android
- **Minimum SDK**: API 23 (Android 6.0)
- **Architecture**: arm64-v8a
- **Supported Backends**: XNNPACK

### iOS
- **Minimum Version**: iOS 17.0+
- **Architecture**: arm64 (device only)
  - ⚠️ **iOS Simulator (x86_64) is NOT supported**
- **Supported Backends**: XNNPACK, CoreML, MPS

### macOS
- **Minimum Version**: macOS 12.0+ (Monterey)
- **Architecture**: **arm64 only** (Apple Silicon)
  - ⚠️ **Intel Macs (x86_64) are NOT supported**
- **Supported Backends**: XNNPACK, CoreML, MPS

### Web
- **Status**: Supported via WebAssembly
- **Runtime**: WebAssembly (Wasm) with XNNPACK backend
- **Supported Backends**: XNNPACK (Wasm SIMD)

#### Web Performance

Web with XNNPACK is ~6-10x slower than native, but fully functional for interactive use.

| Platform | Backend | YOLO11n Inference | Total (E2E) |
|----------|---------|-------------------|-------------|
| Native (macOS/iOS/Android) | XNNPACK | ~50-100ms | ~150-200ms |
| Web | XNNPACK (Wasm SIMD) | ~622ms | ~855ms |

**Web Performance Breakdown (YOLO11n):**

| Stage | Time | % of Total |
|-------|------|------------|
| Preprocessing | ~154ms | 18% |
| Inference | ~622ms | 73% |
| Postprocessing | ~79ms | 9% |

**When to use Web:**
- ✅ Demos and prototyping
- ✅ Interactive inference (sub-second response)
- ✅ Accessibility (no app install required)
- ❌ Real-time camera inference
- ❌ High-throughput batch processing

#### Web Setup

1. **Run the setup script** to copy required JavaScript and Wasm files:

```bash
dart run executorch_flutter:setup_web
```

2. **Add the script tag** to your `web/index.html` (before the Flutter bootstrap script):

```html
<head>
  <!-- ... other head elements ... -->

  <!-- ExecuTorch Wasm wrapper - must load before Flutter -->
  <script src="js/executorch_wrapper.js"></script>
</head>
<body>
  <script src="flutter_bootstrap.js" async></script>
</body>
```

3. **Use XNNPACK models** - Web uses the same XNNPACK-exported models as native platforms.

> **Note**: File-based model loading is not supported on web - models are loaded from bytes or remote URLs.

#### macOS Build Limitations

**Debug Builds**: ✅ Work by default on Apple Silicon Macs

**Release Builds**: ⚠️ **Currently NOT working**

> macOS release builds are not supported due to Flutter's build system forcing universal binaries (arm64 + x86_64), but ExecuTorch only provides arm64 libraries.
>
> 🔗 **Tracking**: [Flutter Issue #176605](https://github.com/flutter/flutter/issues/176605)

## Platform Configuration

When adding `executorch_flutter` to an existing Flutter project, you may need to update the minimum deployment targets. If you see build errors mentioning platform versions, follow these steps:

### iOS Deployment Target (iOS 17.0+)

If you get an error like: `The package product 'executorch-flutter' requires minimum platform version 17.0 for the iOS platform`

**Update using Xcode (Recommended):**
1. Open your Flutter project in Xcode:
   - Navigate to your project folder
   - Open `ios/Runner.xcworkspace` (NOT the `.xcodeproj` file)
2. In Xcode's left sidebar, click on **Runner** (the blue project icon at the top)
3. Make sure **Runner** is selected under "TARGETS" (not under "PROJECT")
4. Click the **Build Settings** tab at the top
5. In the search bar, type: `iOS Deployment Target`
6. You'll see "iOS Deployment Target" with a version number (like 13.0)
7. Click on the version number and change it to **17.0**
8. Close Xcode

### macOS Deployment Target (macOS 12.0+)

If you get an error like: `The package product 'executorch-flutter' requires minimum platform version 12.0 for the macOS platform`

**Update using Xcode (Recommended):**
1. Open your Flutter project in Xcode:
   - Navigate to your project folder
   - Open `macos/Runner.xcworkspace` (NOT the `.xcodeproj` file)
2. In Xcode's left sidebar, click on **Runner** (the blue project icon at the top)
3. Make sure **Runner** is selected under "TARGETS" (not under "PROJECT")
4. Click the **Build Settings** tab at the top
5. In the search bar, type: `macOS Deployment Target`
6. You'll see "macOS Deployment Target" with a version number (like 10.15)
7. Click on the version number and change it to **12.0**
8. Close Xcode

### Verification

After updating deployment targets, clean and rebuild:

```bash
# Clean build artifacts
flutter clean

# Get dependencies
flutter pub get

# Build for your target platform
flutter build ios --debug --no-codesign  # For iOS
flutter build macos --debug               # For macOS
flutter build apk --debug                 # For Android
```

## Build Configuration

The package supports build-time configuration through Flutter's `hooks` system in your app's `pubspec.yaml`. This allows you to customize the build behavior without modifying package code.

### Configuration Options

Add the following to your **app's** `pubspec.yaml` (not the package):

```yaml
hooks:
  user_defines:
    executorch_flutter:
      # Enable debug logging and use Debug prebuilt binaries
      debug: true

      # Build mode: "prebuilt" (default) or "source"
      build_mode: "prebuilt"

      # ExecuTorch source version - for source builds (default: "1.0.1")
      executorch_version: "1.0.1"

      # Prebuilt release version - for prebuilt downloads (default: "1.0.1.8")
      prebuilt_version: "1.0.1.8"

      # Backend selection (platform-specific defaults apply)
      backends:
        - xnnpack
        - coreml
        - mps
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `debug` | `bool` | `false` | Enables native debug logging and selects Debug prebuilt binaries (useful for debugging crashes) |
| `build_mode` | `string` | `"prebuilt"` | `"prebuilt"` downloads pre-compiled binaries (fast, recommended). `"source"` builds from source (slower, requires Python 3.8+ with pyyaml) |
| `executorch_version` | `string` | `"1.0.1"` | ExecuTorch source version (for source builds) |
| `prebuilt_version` | `string` | `"1.0.1.8"` | Prebuilt release version (for prebuilt downloads) |
| `backends` | `list` | Platform-specific | List of backends to enable. Options: `xnnpack`, `coreml`, `mps`, `vulkan`, `qnn` |

### Backend Defaults by Platform

If `backends` is not specified, the following defaults are used:

| Platform | Default Backends |
|----------|------------------|
| Android | xnnpack |
| iOS | xnnpack, coreml |
| macOS | xnnpack, coreml, mps |
| Windows | xnnpack |
| Linux | xnnpack |

### Prebuilt Binaries

Prebuilt binaries are maintained by the package maintainer and hosted at [executorch_native](https://github.com/abdelaziz-mahdy/executorch_native). The prebuilt mode downloads these pre-compiled libraries automatically during the Flutter build process.

**Available prebuilt configurations:**
- Release and Debug builds for each platform
- Platform-specific backend combinations (see table above)

**Request new backends or platforms:** If you need a backend or platform configuration that isn't currently available in prebuilt mode, please [open an issue](https://github.com/abdelaziz-mahdy/executorch_native/issues) on the executorch_native repository.

### Environment Variables

You can also override configuration via environment variables:

| Variable | Description |
|----------|-------------|
| `EXECUTORCH_BUILD_MODE` | Override build mode ("prebuilt" or "source") |
| `EXECUTORCH_CACHE_DIR` | Custom cache directory for downloaded/built artifacts |
| `EXECUTORCH_DISABLE_DOWNLOAD` | Set to "1" to skip prebuilt download (requires `EXECUTORCH_INSTALL_DIR`) |
| `EXECUTORCH_INSTALL_DIR` | Path to local ExecuTorch installation |

### Example Configurations

**Debug build with verbose logging:**
```yaml
hooks:
  user_defines:
    executorch_flutter:
      debug: true
```

**Build from source with specific backends:**
```yaml
hooks:
  user_defines:
    executorch_flutter:
      build_mode: "source"
      backends:
        - xnnpack
        - vulkan
```

**Use a specific prebuilt version:**
```yaml
hooks:
  user_defines:
    executorch_flutter:
      prebuilt_version: "1.0.1.8"
```

## Advanced Usage

### Preprocessing Strategies

The example app demonstrates three preprocessing approaches for common model types:

#### 1. GPU Preprocessing (Default, Recommended)
Hardware-accelerated preprocessing using Flutter Fragment Shaders:
- **Performance**: ~75ms on web, comparable to OpenCV on native
- **Platform Support**: All platforms including web
- **Dependencies**: None (native Flutter APIs)
- **Use Case**: Real-time camera inference, high frame rates, web apps

**📖 [Complete GPU Preprocessing Tutorial](example/GPU_PREPROCESSING.md)** - Step-by-step guide with GLSL shader examples

**Reference implementations:** [example/lib/processors/shaders/](example/lib/processors/shaders/)

#### 2. OpenCV Preprocessing
High-performance C++ library preprocessing:
- **Performance**: High-performance (very close to GPU on native)
- **Platform Support**: Native platforms only (not available on web)
- **Dependencies**: opencv_dart package
- **Use Case**: Advanced image processing, computer vision operations

See **[OpenCV Processors](example/lib/processors/opencv/)** for implementations.

#### 3. CPU Preprocessing (image library)
Pure Dart image processing:
- **Performance**: ~560ms on web, slower than GPU/OpenCV
- **Platform Support**: All platforms
- **Dependencies**: image package
- **Use Case**: Simple preprocessing, debugging, fallback option

See the [example app](example/) for complete processor implementations using the strategy pattern.

## Example Application

The `example/` directory contains a comprehensive demo app showcasing:

- **[Unified Model Playground](example/lib/screens/unified_model_playground.dart)** - Main playground supporting multiple model types
  - MobileNet V3 image classification
  - YOLO object detection (v5, v8, v11)
  - Static image and live camera modes
  - Reactive settings (thresholds, top-K, preprocessing providers)
  - Performance monitoring and metrics

## Converting PyTorch Models to ExecuTorch

To use your PyTorch models with this package, convert them to ExecuTorch format (`.pte` files).

**📖 Official ExecuTorch Export Guide**: [PyTorch ExecuTorch Documentation](https://pytorch.org/executorch/stable/getting-started-architecture.html)

**Key Resources:**
- [ExecuTorch Export Tutorial](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html)
- [XNNPACK Backend Delegation](https://pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)
- [Supported Operators](https://pytorch.org/executorch/stable/ir-ops-set-definition.html)

**Example App Models:**

Models are automatically downloaded from GitHub on first use. To export models manually:

```bash
# Export all models with all available backends
cd models/python
python3 main.py
```

This will:
- ✅ Export MobileNet V3 (all backends: XNNPACK, CoreML, MPS, Vulkan)
- ✅ Export YOLO11n, YOLOv8n, YOLOv5n (all backends)
- ✅ Generate labels files
- ✅ Generate index.json for dynamic model discovery

**Model Hosting:**

Models are hosted in a separate repository for faster cloning:
- Repository: [executorch_flutter_models](https://github.com/abdelaziz-mahdy/executorch_flutter_models)
- Models are downloaded and cached locally on first use
- index.json provides model metadata (size, hash, platforms)

## Development Status

This project is actively developed following these principles:

- **Test-First Development**: Comprehensive testing before implementation
- **Platform Parity**: Consistent behavior across Android and iOS
- **Performance-First**: Optimized for mobile device constraints
- **Documentation-Driven**: Clear examples and API documentation

## API Reference

### Core Classes

#### ExecuTorchModel

The primary class for model management and inference.

```dart
// Load from Flutter asset bundle (recommended - all platforms including web)
static Future<ExecuTorchModel> loadFromAsset(String assetPath)

// Load from bytes (all platforms including web)
static Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes)

// Load from file path (native platforms only - Android, iOS, macOS)
static Future<ExecuTorchModel> load(String filePath)

// Execute inference (matches native module.forward())
Future<List<TensorData>> forward(List<TensorData> inputs)

// Release model resources
Future<void> dispose()

// Check if model is disposed
bool get isDisposed
```

**Native API Mapping:**
- **Android (Kotlin)**: `Module.load()` → `module.forward()`
- **iOS/macOS (Swift)**: `Module()` + `load("forward")` → `module.forward()`

#### TensorData

Input/output tensor representation:

```dart
final tensor = TensorData(
  shape: [1, 3, 224, 224],           // Tensor dimensions
  dataType: TensorType.float32,      // Data type (float32, int32, int8, uint8)
  data: Uint8List(...),              // Raw bytes
  name: 'input_0',                   // Optional tensor name
);
```

### Exception Hierarchy

```dart
ExecuTorchException              // Base exception
├── ExecuTorchModelException     // Model loading/lifecycle errors
├── ExecuTorchInferenceException // Inference execution errors
├── ExecuTorchValidationException // Tensor validation errors
├── ExecuTorchMemoryException    // Memory/resource errors
├── ExecuTorchIOException        // File I/O errors
└── ExecuTorchPlatformException  // Platform communication errors
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup and prerequisites
- Automated Pigeon code generation script
- Integration testing workflow
- Code standards and PR process
- Platform-specific guidelines

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- 📖 Check the [Official ExecuTorch Documentation](https://pytorch.org/executorch/stable/getting-started-architecture.html)
- 🐛 [Report issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues) on GitHub for bugs and feature requests

## Roadmap

See our [Roadmap](ROADMAP.md) for planned features and improvements, including:
- Additional model type examples (segmentation, pose estimation)
- Windows and Linux platform support
- Performance optimizations and more

---

Built with ❤️ for the Flutter and PyTorch communities.