// Pigeon API specification for Flutter ExecuTorch package
// This file defines the type-safe interface between Dart and native platforms
// Updated: 2025-09-20 - Verified against latest ExecuTorch stable documentation
// Android: ExecuTorch 0.7.0, iOS: Xcode 15+/Swift 5.9+

import 'package:pigeon/pigeon.dart';

@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  dartOptions: DartOptions(),
  kotlinOut: 'android/src/main/kotlin/com/zcreations/executorch_flutter/generated/ExecutorchApi.kt',
  kotlinOptions: KotlinOptions(
    package: 'com.zcreations.executorch_flutter.generated',
  ),
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

  String name;
  List<int?> shape; // Pigeon requires nullable generics
  TensorType dataType;
  bool optional;
  List<int?>? validRange; // [min, max] if specified
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

  String modelName;
  String version;
  List<TensorSpec?> inputSpecs; // Pigeon requires nullable generics
  List<TensorSpec?> outputSpecs; // Pigeon requires nullable generics
  int estimatedMemoryMB;
  Map<String?, Object?>? properties; // Pigeon requires nullable generics
}

/// Tensor data for input/output
class TensorData {
  TensorData({
    required this.shape,
    required this.dataType,
    required this.data,
    this.name,
  });

  List<int?> shape; // Pigeon requires nullable generics
  TensorType dataType;
  Uint8List data;
  String? name;
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

  String modelId;
  List<TensorData?> inputs; // Pigeon requires nullable generics
  Map<String?, Object?>? options; // Pigeon requires nullable generics
  int? timeoutMs;
  String? requestId;
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

  InferenceStatus status;
  double executionTimeMs;
  String? requestId;
  List<TensorData?>? outputs; // Pigeon requires nullable generics
  String? errorMessage;
  Map<String?, Object?>? metadata; // Pigeon requires nullable generics
}

/// Model loading result
class ModelLoadResult {
  ModelLoadResult({
    required this.modelId,
    required this.state,
    this.metadata,
    this.errorMessage,
  });

  String modelId;
  ModelState state;
  ModelMetadata? metadata;
  String? errorMessage;
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
  List<String?> getLoadedModels();

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