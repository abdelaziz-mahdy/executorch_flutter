# ExecuTorch Flutter

A Flutter plugin package using ExecuTorch to allow model inference on Android, iOS, and macOS platforms.

## Overview

ExecuTorch Flutter provides a simple Dart API for loading and running ExecuTorch models (`.pte` files) in your Flutter applications. The package handles all native platform integration, providing you with a straightforward interface for on-device machine learning inference.

## Features

- ✅ **Cross-Platform Support**: Android (API 23+), iOS (13.0+), and macOS (12.0+ Apple Silicon)
- ✅ **Type-Safe API**: Generated with Pigeon for reliable cross-platform communication
- ✅ **Async Operations**: Non-blocking model loading and inference execution
- ✅ **Multiple Models**: Support for concurrent model instances
- ✅ **Error Handling**: Structured exception handling with clear error messages
- ✅ **Backend Support**: XNNPACK, CoreML, MPS backends

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

// Load a model from file path
final model = await ExecuTorchModel.load('/path/to/model.pte');
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

### 3. Loading Models from Assets

```dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'dart:io';

// Load model from assets
final byteData = await rootBundle.load('assets/models/model.pte');
final tempDir = await getTemporaryDirectory();
final file = File('${tempDir.path}/model.pte');
await file.writeAsBytes(byteData.buffer.asUint8List());

// Load and run inference
final model = await ExecuTorchModel.load(file.path);
final outputs = await model.forward([inputTensor]);

// Dispose when done
await model.dispose();
```

### Complete Examples

See the `example/` directory for full working applications:

- **[Image Classification](example/lib/screens/image_classification_demo.dart)** - MobileNet V3 image classification
- **[Object Detection](example/lib/screens/object_detection_demo.dart)** - YOLO object detection with bounding boxes
- **[Model Management](example/lib/screens/model_manager.dart)** - Loading models from assets, storage, and network

## Supported Model Formats

- **ExecuTorch (.pte)**: Optimized PyTorch models converted to ExecuTorch format
- **Input Types**: float32, int8, int32, uint8 tensors
- **Model Size**: Tested with models up to 500MB

> 📖 **Need to export your PyTorch models?** See our comprehensive **[Model Export Guide](MODEL_EXPORT_GUIDE.md)** for step-by-step instructions on converting PyTorch models to ExecuTorch format with platform-specific optimizations.
>
> 📱 **For example app setup:** See [Example App Model Guide](example/MODEL_EXPORT_GUIDE.md) for complete integration examples.

## Platform Requirements

### Android
- **Minimum SDK**: API 23 (Android 6.0)
- **Architecture**: arm64-v8a
- **Supported Backends**: XNNPACK

### iOS
- **Minimum Version**: iOS 13.0+
- **Architecture**: arm64 (device only)
  - ⚠️ **iOS Simulator (x86_64) is NOT supported**
- **Supported Backends**: XNNPACK, CoreML, MPS

### macOS
- **Minimum Version**: macOS 12.0+ (Monterey)
- **Architecture**: **arm64 only** (Apple Silicon)
  - ⚠️ **Intel Macs (x86_64) are NOT supported**
- **Supported Backends**: XNNPACK, CoreML, MPS

#### macOS Build Limitations

**Debug Builds**: ✅ Work by default on Apple Silicon Macs

**Release Builds**: ⚠️ **Currently NOT working**

> macOS release builds are not supported due to Flutter's build system forcing universal binaries (arm64 + x86_64), but ExecuTorch only provides arm64 libraries.
>
> 🔗 **Tracking**: [Flutter Issue #176605](https://github.com/flutter/flutter/issues/176605)

## Advanced Usage

### Processor Interfaces

The package includes high-level processor interfaces that handle preprocessing and postprocessing:

- **[ImageNetProcessor](example/lib/processors/imagenet_processor.dart)** - Image classification with MobileNet
- **[YOLOProcessor](example/lib/processors/yolo_processor.dart)** - Object detection with bounding boxes
- **[CustomProcessor](example/lib/processors/base_processor.dart)** - Create your own processors

See the example app for complete implementations of these processors.

## Example Applications

The `example/` directory contains complete working demos:

- **[Image Classification Demo](example/lib/screens/image_classification_demo.dart)** - MobileNet V3 classification
- **[Object Detection Demo](example/lib/screens/object_detection_demo.dart)** - YOLO object detection
- **[Model Manager](example/lib/screens/model_manager.dart)** - Load models from different sources

## Converting PyTorch Models to ExecuTorch

To use your PyTorch models with this package, convert them to ExecuTorch format (`.pte` files).

**📖 Complete guide**: [Model Export Guide](MODEL_EXPORT_GUIDE.md)

**Quick reference**: [ExecuTorch Documentation](https://pytorch.org/executorch/)

**Quick Setup for Example App:**

```bash
# One-command setup: installs dependencies and exports all models
cd python
python3 setup_models.py
```

This will:
- ✅ Install all required dependencies (torch, ultralytics, executorch)
- ✅ Export MobileNet V3 for image classification
- ✅ Export YOLO11n for object detection
- ✅ Generate COCO labels file
- ✅ Verify all models are ready

**Manual export:**
```bash
cd python
python3 export_models.py  # MobileNet V3 + COCO labels
python3 export_yolo.py     # YOLO models (v5, v8, v11)
```

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
// Static factory method to load a model
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

## Architecture

```
Flutter App (Dart)
       ↓
ExecuTorchModel (Simple wrapper over native API)
       ↓
Pigeon Generated APIs (Type-safe communication)
       ↓
Platform Channel Layer
   ↓              ↓              ↓
Android         iOS           macOS
(Kotlin)      (Swift)        (Swift)
   ↓              ↓              ↓
ExecuTorch    ExecuTorch    ExecuTorch
1.0.0-rc2     SPM 1.0.0     SPM 1.0.0
   ↓              ↓              ↓
XNNPACK       CoreML/MPS    CoreML/MPS
Backends      Backends      Backends
```

### Key Components

- **ExecuTorchModel**: Primary class for model loading and inference (matches native APIs)
  - `load()` → Native `Module.load()` / `Module()`
  - `forward()` → Native `module.forward()`
- **TensorData**: Input/output tensor representation with shape, type, and raw data
- **Pigeon Interface**: Type-safe method channel communication (no manual platform channels)
- **Native Model Managers**: Platform-specific model lifecycle and inference handling
- **Backend Integration**: Optimized ExecuTorch backends for each platform

### Design Philosophy

This package follows a **thin wrapper** philosophy:
- Dart API directly mirrors native ExecuTorch APIs
- Minimal abstractions for predictable behavior
- User controls model lifecycle explicitly (load/dispose)
- No hidden state or automatic memory management

### Thread Safety

- **iOS & macOS**: Actor-based concurrency with Swift async/await
- **Android**: Kotlin coroutines with structured concurrency
- **Flutter**: All operations return Futures for non-blocking UI

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- 📖 Check the [Model Export Guide](MODEL_EXPORT_GUIDE.md)
- 🐛 [Report issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues) on GitHub
- 💬 [Discussions](https://github.com/abdelaziz-mahdy/executorch_flutter/discussions) for questions and feature requests

## Roadmap

See our [Roadmap](ROADMAP.md) for planned features and improvements, including:
- Live camera integration example
- Additional preprocessing/postprocessing options for different model types
- Platform expansion and more

---

Built with ❤️ for the Flutter and PyTorch communities.