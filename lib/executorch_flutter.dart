
/// ExecuTorch Flutter Plugin - On-device ML inference with ExecuTorch
///
/// This package provides Flutter developers with the ability to run
/// ExecuTorch machine learning models on Android and iOS devices with
/// high performance and low latency.
///
/// ## Key Features
///
/// - **High Performance**: Optimized for mobile inference with ExecuTorch runtime
/// - **Type Safe**: Generated platform communication with Pigeon ensures type safety
/// - **Cross Platform**: Identical APIs across Android and iOS platforms
/// - **Resource Management**: Automatic memory management and model lifecycle
/// - **Easy Integration**: Simple API for loading models and running inference
///
/// ## Quick Start
///
/// ```dart
/// import 'package:executorch_flutter/executorch_flutter.dart';
///
/// // Initialize the manager
/// await ExecutorchManager.instance.initialize();
///
/// // Load a model
/// final model = await ExecutorchManager.instance.loadModel('path/to/model.pte');
///
/// // Prepare input data
/// final inputTensor = TensorDataWrapper(
///   shape: [1, 3, 224, 224],
///   dataType: TensorType.float32,
///   data: imageBytes,
///   name: 'input',
/// );
///
/// // Run inference
/// final result = await model.runInference(inputs: [inputTensor]);
///
/// if (result.isSuccess) {
///   print('Inference completed in ${result.executionTimeMs}ms');
///   // Process result.outputs
/// }
///
/// // Clean up
/// await model.dispose();
/// ```
///
/// ## Main Classes
///
/// - [ExecutorchManager]: Main entry point for ExecuTorch operations
/// - [ExecuTorchModel]: Represents a loaded model instance
/// - [TensorDataWrapper]: High-level tensor data representation
/// - [InferenceResultWrapper]: Inference execution results
/// - [TensorUtils]: Utility functions for tensor operations
///
/// ## Platform Support
///
/// - **Android**: API 23+ (Android 6.0+), arm64-v8a architecture
/// - **iOS**: iOS 13.0+, arm64 architecture (device and simulator)
///
/// ## Performance Targets
///
/// - Model Loading: <200ms for models up to 100MB
/// - Inference: <50ms for typical mobile models
/// - Memory: <100MB additional RAM during inference
///
/// For detailed documentation and examples, see the individual class documentation.
library executorch_flutter;

// Core API exports
export 'src/executorch_inference.dart' show
    ExecutorchManager,
    TensorUtils;

export 'src/executorch_model.dart' show
    ExecuTorchModel,
    ExecuTorchModelLoadException;

export 'src/executorch_types.dart' show
    TensorDataWrapper,
    ModelMetadataWrapper,
    InferenceRequestWrapper,
    InferenceResultWrapper;

// Error handling
export 'src/executorch_errors.dart';

// Generated Pigeon types (selective export of commonly used enums and classes)
export 'src/generated/executorch_api.dart' show
    // Enums
    TensorType,
    ModelState,
    InferenceStatus,

    // Core data classes that users might need to interact with
    TensorSpec,
    ModelMetadata,
    TensorData,
    InferenceRequest,
    InferenceResult,
    ModelLoadResult;

// Note: ExecutorchHostApi and ExecutorchFlutterApi are not exported
// as they are internal platform communication interfaces.
// Users should interact with the high-level wrapper classes instead.
