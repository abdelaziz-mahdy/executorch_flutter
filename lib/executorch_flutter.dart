/// ExecuTorch Flutter Plugin - On-device ML inference with ExecuTorch
///
/// This package provides Flutter developers with the ability to run
/// ExecuTorch machine learning models on Android, iOS, and macOS platforms
/// with high performance and low latency.
///
/// ## Key Features
///
/// - **High Performance**: Optimized for mobile inference with ExecuTorch runtime via FFI
/// - **Type Safe**: Direct C interop with Dart FFI for zero-overhead communication
/// - **Cross Platform**: Identical APIs across Android, iOS, and macOS platforms
/// - **User-Controlled Resources**: Explicit model lifecycle management with load/dispose
/// - **Easy Integration**: Simple API for loading models and running inference
/// - **Automatic Cleanup**: NativeFinalizer ensures models are freed on garbage collection
///
/// ## Quick Start
///
/// ```dart
/// import 'package:executorch_flutter/executorch_flutter.dart';
///
/// // Load a model (FFI initializes automatically on first use)
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
///
/// // Prepare input data
/// final inputTensor = TensorData(
///   shape: [1, 3, 224, 224],
///   dataType: TensorType.float32,
///   data: imageBytes,
///   name: 'input',
/// );
///
/// // Run inference
/// final outputs = await model.forward([inputTensor]);
///
/// // Process outputs (List<TensorData>)
/// for (var output in outputs) {
///   print('Output shape: ${output.shape}');
/// }
///
/// // Clean up (automatic via finalizer, but recommended for explicit cleanup)
/// await model.dispose();
/// ```
///
/// ## Main Classes
///
/// - [ExecuTorchModel]: Represents a loaded model instance (load/forward/dispose)
/// - [TensorData]: Tensor data representation (shape, type, data bytes)
///
/// ## Processors
///
/// - [ExecuTorchPreprocessor]: Base class for input preprocessing
/// - [ExecuTorchPostprocessor]: Base class for output postprocessing
/// - [ExecuTorchProcessor]: Combined preprocessing and postprocessing
/// - [ImageClassificationProcessor]: Ready-to-use image classification
/// - [TextClassificationProcessor]: Ready-to-use text classification
/// - [AudioClassificationProcessor]: Ready-to-use audio classification
///
/// ## Platform Support
///
/// - **Android**: API 23+ (Android 6.0+), arm64-v8a architecture
/// - **iOS**: iOS 13.0+, arm64 (device only, simulator not supported)
/// - **macOS**: macOS 12.0+ (Monterey), arm64 only (Apple Silicon)
///
/// For detailed documentation and examples, see the individual class documentation.
library;

// Core API exports
export 'src/executorch_errors.dart';
export 'src/executorch_model.dart';

// Tensor types from Pigeon (still used for data structures)
export 'src/generated/executorch_api.dart' show TensorData, TensorType;

// Preprocessing and postprocessing utilities
export 'src/processors/processors.dart';

// Note: executorch_inference.dart (ExecutorchManager) is deprecated in favor of
// direct ExecuTorchModel.load() usage with FFI
