/// ExecuTorch model wrapper class providing high-level model management
library;

import 'dart:async';

import 'executorch_errors.dart';
import 'generated/executorch_api.dart';

/// High-level wrapper for an ExecuTorch model instance
///
/// This class provides a convenient interface for loading, managing, and
/// running inference on ExecuTorch models. It handles model lifecycle
/// and provides validation and error handling.
///
/// Note: ExecuTorch doesn't provide runtime introspection for model metadata.
/// Input/output specs must be known externally (from model documentation).
class ExecuTorchModel {
  ExecuTorchModel._({
    required this.modelId,
    required this.hostApi,
  });

  /// Unique identifier for this model instance
  final String modelId;

  /// Reference to the host API for platform communication
  final ExecutorchHostApi hostApi;

  /// Whether this model has been disposed
  bool _isDisposed = false;

  /// Create and load an ExecuTorch model from a file path
  ///
  /// [filePath] must point to a valid ExecuTorch .pte model file.
  /// Returns the loaded model instance or throws an exception if loading fails.
  static Future<ExecuTorchModel> loadFromFile(String filePath) async {
    final hostApi = ExecutorchHostApi();
    final loadResult = await hostApi.loadModel(filePath);

    if (loadResult.state != ModelState.ready) {
      throw ExecuTorchException(
        'Failed to load model from $filePath: ${loadResult.errorMessage ?? "Unknown error"}',
      );
    }

    return ExecuTorchModel._(
      modelId: loadResult.modelId,
      hostApi: hostApi,
    );
  }

  /// Run inference on this model with the provided inputs
  ///
  /// [inputs] must match the model's input specifications.
  /// [options] can provide platform-specific execution options.
  /// [timeoutMs] sets a maximum execution time (optional).
  /// [requestId] provides a unique identifier for tracking (optional).
  ///
  /// Returns the inference result or throws an exception if inference fails.
  Future<InferenceResult> runInference({
    required List<TensorData> inputs,
    Map<String, Object>? options,
    int? timeoutMs,
    String? requestId,
  }) async {
    if (_isDisposed) {
      throw const ExecuTorchException('Model has been disposed and cannot be used');
    }

    final request = InferenceRequest(
      modelId: modelId,
      inputs: inputs.cast<TensorData?>(),
      options: options?.cast<String?, Object?>(),
      timeoutMs: timeoutMs,
      requestId: requestId,
    );

    final result = await hostApi.runInference(request);

    if (result.status != InferenceStatus.success) {
      throw ExecuTorchInferenceException(
        'Inference failed: ${result.errorMessage ?? "Unknown error"}',
        result.errorMessage,
      );
    }

    return result;
  }

  /// Dispose this model and free its resources
  ///
  /// Call this when you're done with the model to free platform resources.
  /// The user has full control over memory management.
  Future<void> dispose() async {
    if (_isDisposed) return;

    await hostApi.disposeModel(modelId);
    _isDisposed = true;
  }

  /// Check if this model has been disposed
  bool get isDisposed => _isDisposed;
}

