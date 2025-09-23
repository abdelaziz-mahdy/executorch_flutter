
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
/// final inputTensor = TensorData(
///   shape: [1, 3, 224, 224],
///   dataType: TensorType.float32,
///   data: imageBytes,
///   name: 'input',
/// );
///
/// // Run inference
/// final result = await model.runInference(inputs: [inputTensor]);
///
/// if (result.status == InferenceStatus.success) {
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
/// - [TensorData]: Tensor data representation
/// - [InferenceResult]: Inference execution results
/// - [ModelMetadata]: Model information and specifications
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
export 'src/executorch_inference.dart';
export 'src/executorch_model.dart';
export 'src/executorch_errors.dart';

// Generated Pigeon types - direct export for type safety
export 'src/generated/executorch_api.dart';

// Preprocessing and postprocessing utilities
export 'src/processors/processors.dart';
