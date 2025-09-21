// Pigeon API specification for Flutter ExecuTorch package
// This file defines the type-safe interface between Dart and native platforms
// Updated: 2025-09-20 - Verified against latest ExecuTorch stable documentation
// Android: ExecuTorch 0.7.0, iOS: Xcode 15+/Swift 5.9+

import 'package:pigeon/pigeon.dart';

@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  dartOptions: DartOptions(),
  kotlinOut: 'android/src/main/kotlin/com/executorch/flutter/generated/ExecutorchApi.kt',
  kotlinOptions: KotlinOptions(),
  swiftOut: 'ios/Classes/Generated/ExecutorchApi.swift',
  swiftOptions: SwiftOptions(),
  dartPackageName: 'executorch_flutter',
))

/// Tensor data type enumeration
enum TensorType {
  float32,
  int8,
  int32,
  uint8,
}

/// Model loading and execution states
enum ModelState {
  loading,
  ready,
  error,
  disposed,
}

/// Inference execution status
enum InferenceStatus {
  success,
  error,
  timeout,
  cancelled,
}

/// Tensor specification for input/output requirements
class TensorSpec {
  TensorSpec({
    required this.name,
    required this.shape,
    required this.dataType,
    required this.optional,
    this.validRange,
  });

  final String name;
  final List<int> shape; // -1 for dynamic dimensions
  final TensorType dataType;
  final bool optional;
  final List<int>? validRange; // [min, max] if specified
}

/// Model metadata and capabilities
class ModelMetadata {
  ModelMetadata({
    required this.modelName,
    required this.version,
    required this.inputSpecs,
    required this.outputSpecs,
    required this.estimatedMemoryMB,
    this.properties,
  });

  final String modelName;
  final String version;
  final List<TensorSpec> inputSpecs;
  final List<TensorSpec> outputSpecs;
  final int estimatedMemoryMB;
  final Map<String, Object>? properties;
}

/// Tensor data for input/output
class TensorData {
  TensorData({
    required this.shape,
    required this.dataType,
    required this.data,
    this.name,
  });

  final List<int> shape;
  final TensorType dataType;
  final Uint8List data;
  final String? name;
}

/// Inference request parameters
class InferenceRequest {
  InferenceRequest({
    required this.modelId,
    required this.inputs,
    this.options,
    this.timeoutMs,
    this.requestId,
  });

  final String modelId;
  final List<TensorData> inputs;
  final Map<String, Object>? options;
  final int? timeoutMs;
  final String? requestId;
}

/// Inference execution result
class InferenceResult {
  InferenceResult({
    required this.status,
    required this.executionTimeMs,
    this.requestId,
    this.outputs,
    this.errorMessage,
    this.metadata,
  });

  final InferenceStatus status;
  final double executionTimeMs;
  final String? requestId;
  final List<TensorData>? outputs;
  final String? errorMessage;
  final Map<String, Object>? metadata;
}

/// Model loading result
class ModelLoadResult {
  ModelLoadResult({
    required this.modelId,
    required this.state,
    this.metadata,
    this.errorMessage,
  });

  final String modelId;
  final ModelState state;
  final ModelMetadata? metadata;
  final String? errorMessage;
}

/// Host API - Called from Dart to native platforms
@HostApi()
abstract class ExecutorchHostApi {
  /// Load a model from the specified file path
  /// Returns a unique model ID for subsequent operations
  @async
  ModelLoadResult loadModel(String filePath);

  /// Run inference on a loaded model
  /// Returns inference results or error information
  @async
  InferenceResult runInference(InferenceRequest request);

  /// Get metadata for a loaded model
  ModelMetadata? getModelMetadata(String modelId);

  /// Dispose a loaded model and free its resources
  void disposeModel(String modelId);

  /// Get list of currently loaded model IDs
  List<String> getLoadedModels();

  /// Check if a model is currently loaded and ready
  ModelState getModelState(String modelId);
}

/// Flutter API - Called from native platforms to Dart (optional)
@FlutterApi()
abstract class ExecutorchFlutterApi {
  /// Notify Dart about model loading progress (optional)
  void onModelLoadProgress(String modelId, double progress);

  /// Notify Dart about inference completion (optional)
  void onInferenceComplete(String requestId, InferenceResult result);
}

/// Exception classes for structured error handling
class ExecutorchException {
  ExecutorchException({
    required this.code,
    required this.message,
    this.details,
  });

  final String code;
  final String message;
  final Map<String, Object>? details;
}

/// Specific exception types
class ModelLoadException extends ExecutorchException {
  ModelLoadException({
    required String message,
    Map<String, Object>? details,
  }) : super(
          code: 'MODEL_LOAD_ERROR',
          message: message,
          details: details,
        );
}

class InferenceException extends ExecutorchException {
  InferenceException({
    required String message,
    Map<String, Object>? details,
  }) : super(
          code: 'INFERENCE_ERROR',
          message: message,
          details: details,
        );
}

class ValidationException extends ExecutorchException {
  ValidationException({
    required String message,
    Map<String, Object>? details,
  }) : super(
          code: 'VALIDATION_ERROR',
          message: message,
          details: details,
        );
}

class ResourceException extends ExecutorchException {
  ResourceException({
    required String message,
    Map<String, Object>? details,
  }) : super(
          code: 'RESOURCE_ERROR',
          message: message,
          details: details,
        );
}