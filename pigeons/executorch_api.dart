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
  swiftOptions: SwiftOptions(
    includeErrorClass: false,
  ),
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

// Metadata removed - ExecuTorch doesn't provide runtime introspection
// Models should document their input/output specs externally

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
    this.errorMessage,
  });

  String modelId;
  ModelState state;
  String? errorMessage;
}


/// Host API - Called from Dart to native platforms
/// Simplified to core operations: load, inference, dispose
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

  /// Dispose a loaded model and free its resources
  /// User has full control over memory management
  void disposeModel(String modelId);

  /// Get list of currently loaded model IDs
  List<String?> getLoadedModels();

  /// Enable or disable ExecuTorch debug logging
  /// Only works in debug builds
  void setDebugLogging(bool enabled);
}

/// Flutter API - Called from native platforms to Dart (optional)
@FlutterApi()
abstract class ExecutorchFlutterApi {
  /// Notify Dart about model loading progress (optional)
  void onModelLoadProgress(String modelId, double progress);

  /// Notify Dart about inference completion (optional)
  void onInferenceComplete(String requestId, InferenceResult result);
}