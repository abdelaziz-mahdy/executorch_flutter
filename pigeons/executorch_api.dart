// Pigeon API specification for Flutter ExecuTorch package
// This file defines the type-safe interface between Dart and native platforms
// Updated: 2025-10-07 - Updated to ExecuTorch 1.0.0
// Android: ExecuTorch 1.0.0-rc2, iOS/macOS: SPM 1.0.0, Xcode 15+/Swift 5.9+

import 'package:pigeon/pigeon.dart';

@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  dartOptions: DartOptions(),
  kotlinOut: 'android/src/main/kotlin/com/zcreations/executorch_flutter/generated/ExecutorchApi.kt',
  kotlinOptions: KotlinOptions(
    package: 'com.zcreations.executorch_flutter.generated',
  ),
  swiftOut: 'darwin/Sources/executorch_flutter/Generated/ExecutorchApi.swift',
  swiftOptions: SwiftOptions(
    includeErrorClass: true,
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

// API uses exceptions for error handling instead of status enums
// On success: methods return result
// On failure: methods throw PlatformException with error details
// Dart wrappers catch and convert to typed ExecuTorch exceptions

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
/// On success: contains outputs and timing info
/// On failure: platform throws exception
class InferenceResult {
  InferenceResult({
    required this.outputs,
    required this.executionTimeMs,
    this.requestId,
  });

  List<TensorData?> outputs; // Pigeon requires nullable generics
  double executionTimeMs;
  String? requestId;
}

/// Model loading result
/// On success: returns unique model ID
/// On failure: platform throws exception
class ModelLoadResult {
  ModelLoadResult({
    required this.modelId,
  });

  String modelId;
}


/// Host API - Called from Dart to native platforms
/// All methods throw PlatformException on error
@HostApi()
abstract class ExecutorchHostApi {
  /// Load a model from the specified file path
  /// Returns a unique model ID for subsequent operations
  /// Throws: PlatformException if file not found or model loading fails
  @async
  ModelLoadResult loadModel(String filePath);

  /// Run inference on a loaded model
  /// Returns inference results with outputs and timing
  /// Throws: PlatformException if model not found or inference fails
  @async
  InferenceResult runInference(InferenceRequest request);

  /// Dispose a loaded model and free its resources
  /// User has full control over memory management
  /// Throws: PlatformException if model not found
  void disposeModel(String modelId);

  /// Get list of currently loaded model IDs
  /// Returns empty list if no models loaded
  List<String?> getLoadedModels();

  /// Enable or disable ExecuTorch debug logging
  /// Only works in debug builds
  void setDebugLogging(bool enabled);
}