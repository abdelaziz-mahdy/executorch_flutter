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
- ✅ **Live Camera**: Real-time inference with camera stream support

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  executorch_flutter: ^0.0.1
```

## Basic Usage

The package provides a simple workflow:

### 1. Initialize and Load a Model

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// Initialize the manager (call once)
await ExecutorchManager.instance.initialize();

// Load a model from file path
final model = await ExecutorchManager.instance.loadModel('/path/to/model.pte');
```

### 2. Run Inference

```dart
// Prepare input tensor
final inputTensor = TensorData(
  shape: [1, 3, 224, 224],
  dataType: TensorType.float32,
  data: yourImageBytes, // FlutterStandardTypedData
);

// Run inference
final result = await model.runInference(inputs: [inputTensor]);

// Access outputs
if (result.status == InferenceStatus.success) {
  final output = result.outputs?.first;
  print('Inference time: ${result.executionTimeMs}ms');
}

// Clean up when done
await model.dispose();
```

### Complete Example

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

The example app demonstrates processor strategies for common model types:

- **[Image Classification](example/lib/processors/image_processor.dart)** - ImageNet preprocessing and postprocessing for MobileNet
- **[Object Detection](example/lib/processors/yolo_processor.dart)** - YOLO preprocessing, NMS, and bounding box extraction
- **[OpenCV Processors](example/lib/processors/opencv_processors.dart)** - High-performance OpenCV-based preprocessing

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

The example app includes scripts for exporting reference models (MobileNet, YOLO):

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

## Development Status

This project is actively developed following these principles:

- **Test-First Development**: Comprehensive testing before implementation
- **Platform Parity**: Consistent behavior across Android and iOS
- **Performance-First**: Optimized for mobile device constraints
- **Documentation-Driven**: Clear examples and API documentation

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
- 🐛 [Report issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues) on GitHub
- 💬 [Discussions](https://github.com/abdelaziz-mahdy/executorch_flutter/discussions) for questions and feature requests

## Roadmap

See our [Roadmap](ROADMAP.md) for planned features and improvements, including:
- Additional model type examples (segmentation, pose estimation)
- Windows and Linux platform support
- Performance optimizations and more

---

Built with ❤️ for the Flutter and PyTorch communities.