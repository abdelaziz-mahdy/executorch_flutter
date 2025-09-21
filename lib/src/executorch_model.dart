/// ExecuTorch model wrapper class providing high-level model management
library executorch_model;

import 'dart:async';
import 'package:meta/meta.dart';

import 'generated/executorch_api.dart';
import 'executorch_errors.dart';

/// High-level wrapper for an ExecuTorch model instance
///
/// This class provides a convenient interface for loading, managing, and
/// running inference on ExecuTorch models. It handles model lifecycle,
/// metadata access, and provides validation and error handling.
class ExecuTorchModel {
  ExecuTorchModel._({
    required this.modelId,
    required this.metadata,
    required this.hostApi,
  });

  /// Unique identifier for this model instance
  final String modelId;

  /// Metadata describing the model's structure and requirements
  final ModelMetadata metadata;

  /// Reference to the host API for platform communication
  final ExecutorchHostApi hostApi;

  /// Whether this model instance has been disposed
  bool _disposed = false;

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

    if (loadResult.metadata == null) {
      throw ExecuTorchException(
        'Model loaded but metadata is unavailable for $filePath',
      );
    }

    return ExecuTorchModel._(
      modelId: loadResult.modelId,
      metadata: loadResult.metadata!,
      hostApi: hostApi,
    );
  }

  /// Create an ExecuTorchModel instance from an existing loaded model
  ///
  /// This is useful when you have already loaded a model through the low-level API
  /// and want to wrap it in the high-level interface.
  static Future<ExecuTorchModel> fromExisting({
    required String modelId,
    required ExecutorchHostApi hostApi,
  }) async {
    final metadata = await hostApi.getModelMetadata(modelId);
    if (metadata == null) {
      throw ExecuTorchException(
        'Model $modelId not found or metadata unavailable',
      );
    }

    final state = await hostApi.getModelState(modelId);
    if (state != ModelState.ready) {
      throw ExecuTorchException(
        'Model $modelId is not in ready state: $state',
      );
    }

    return ExecuTorchModel._(
      modelId: modelId,
      metadata: metadata,
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
    _checkNotDisposed();
    _validateInputs(inputs);

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

  /// Run inference with a single input tensor (convenience method)
  ///
  /// This is a convenience method for models that take a single input.
  /// Equivalent to calling [runInference] with a single-element input list.
  Future<InferenceResult> runSingleInput({
    required TensorData input,
    Map<String, Object>? options,
    int? timeoutMs,
    String? requestId,
  }) async {
    return runInference(
      inputs: [input],
      options: options,
      timeoutMs: timeoutMs,
      requestId: requestId,
    );
  }

  /// Get the current state of this model
  Future<ModelState> getState() async {
    return hostApi.getModelState(modelId);
  }

  /// Check if this model is currently ready for inference
  Future<bool> get isReady async {
    if (_disposed) return false;
    final state = await getState();
    return state == ModelState.ready;
  }

  /// Check if this model has been disposed
  bool get isDisposed => _disposed;

  /// Get the primary (first) input specification
  ///
  /// This is a convenience property for models with a single primary input.
  TensorSpec? get primaryInputSpec => metadata.inputSpecs.isNotEmpty ? metadata.inputSpecs.first : null;

  /// Get the primary (first) output specification
  ///
  /// This is a convenience property for models with a single primary output.
  TensorSpec? get primaryOutputSpec => metadata.outputSpecs.isNotEmpty ? metadata.outputSpecs.first : null;

  /// Validate that inputs match the model's input specifications
  void _validateInputs(List<TensorData> inputs) {
    final inputSpecs = metadata.inputSpecs.whereType<TensorSpec>().toList();

    if (inputs.length != inputSpecs.length) {
      throw ExecuTorchValidationException(
        'Input count mismatch: expected ${inputSpecs.length}, got ${inputs.length}',
      );
    }

    for (int i = 0; i < inputs.length; i++) {
      final input = inputs[i];
      final spec = inputSpecs[i];

      // Validate data type
      if (input.dataType != spec.dataType) {
        throw ExecuTorchValidationException(
          'Input $i data type mismatch: expected ${spec.dataType}, got ${input.dataType}',
        );
      }

      // Validate shape (allowing dynamic dimensions marked as -1)
      final inputShape = input.shape.whereType<int>().toList();
      final specShape = spec.shape.whereType<int>().toList();

      if (inputShape.length != specShape.length) {
        throw ExecuTorchValidationException(
          'Input $i shape rank mismatch: expected ${specShape.length}D, got ${inputShape.length}D',
        );
      }

      for (int j = 0; j < inputShape.length; j++) {
        final inputDim = inputShape[j];
        final specDim = specShape[j];

        // Skip validation for dynamic dimensions (-1)
        if (specDim == -1) continue;

        if (inputDim != specDim) {
          throw ExecuTorchValidationException(
            'Input $i dimension $j mismatch: expected $specDim, got $inputDim',
          );
        }
      }

      // Basic tensor data validation - just check data is not empty
      if (input.data.isEmpty) {
        throw ExecuTorchValidationException(
          'Input $i has empty data',
        );
      }
    }
  }

  /// Check that this model hasn't been disposed
  void _checkNotDisposed() {
    if (_disposed) {
      throw ExecuTorchException(
        'Model $modelId has been disposed and cannot be used',
      );
    }
  }

  /// Dispose this model and free its resources
  ///
  /// After calling this method, the model cannot be used for inference.
  /// This is automatically called when the model is garbage collected,
  /// but it's recommended to call it explicitly when done with the model.
  Future<void> dispose() async {
    if (_disposed) return;

    try {
      await hostApi.disposeModel(modelId);
    } catch (e) {
      // Log but don't throw - disposal should be best-effort
      print('Warning: Failed to dispose model $modelId: $e');
    } finally {
      _disposed = true;
    }
  }

  @override
  String toString() {
    final disposedStr = _disposed ? ' (DISPOSED)' : '';
    return 'ExecuTorchModel($modelId: ${metadata.modelName})$disposedStr';
  }
}

