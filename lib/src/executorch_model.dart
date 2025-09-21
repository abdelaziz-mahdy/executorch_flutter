/// ExecuTorch model wrapper class providing high-level model management
library executorch_model;

import 'dart:async';
import 'package:meta/meta.dart';

import 'executorch_types.dart';
import '../src/generated/executorch_api.dart' as pigeon;

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
  final ModelMetadataWrapper metadata;

  /// Reference to the host API for platform communication
  final pigeon.ExecutorchHostApi hostApi;

  /// Whether this model instance has been disposed
  bool _disposed = false;

  /// Create and load an ExecuTorch model from a file path
  ///
  /// [filePath] must point to a valid ExecuTorch .pte model file.
  /// Returns the loaded model instance or throws an exception if loading fails.
  static Future<ExecuTorchModel> loadFromFile(String filePath) async {
    final hostApi = pigeon.ExecutorchHostApi();
    final loadResult = await hostApi.loadModel(filePath);

    if (loadResult.state != pigeon.ModelState.ready) {
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
      metadata: ModelMetadataWrapper.fromPigeon(loadResult.metadata!),
      hostApi: hostApi,
    );
  }

  /// Create an ExecuTorchModel instance from an existing loaded model
  ///
  /// This is useful when you have already loaded a model through the low-level API
  /// and want to wrap it in the high-level interface.
  static Future<ExecuTorchModel> fromExisting({
    required String modelId,
    required pigeon.ExecutorchHostApi hostApi,
  }) async {
    final metadata = await hostApi.getModelMetadata(modelId);
    if (metadata == null) {
      throw ExecuTorchException(
        'Model $modelId not found or metadata unavailable',
      );
    }

    final state = await hostApi.getModelState(modelId);
    if (state != pigeon.ModelState.ready) {
      throw ExecuTorchException(
        'Model $modelId is not in ready state: $state',
      );
    }

    return ExecuTorchModel._(
      modelId: modelId,
      metadata: ModelMetadataWrapper.fromPigeon(metadata),
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
  Future<InferenceResultWrapper> runInference({
    required List<TensorDataWrapper> inputs,
    Map<String, Object>? options,
    int? timeoutMs,
    String? requestId,
  }) async {
    _checkNotDisposed();
    _validateInputs(inputs);

    final request = InferenceRequestWrapper(
      modelId: modelId,
      inputs: inputs,
      options: options,
      timeoutMs: timeoutMs,
      requestId: requestId,
    );

    final pigeonResult = await hostApi.runInference(request.toPigeon());
    final result = InferenceResultWrapper.fromPigeon(pigeonResult);

    if (!result.isSuccess) {
      throw ExecuTorchInferenceException(
        'Inference failed: ${result.errorMessage ?? "Unknown error"}',
        result: result,
      );
    }

    return result;
  }

  /// Run inference with a single input tensor (convenience method)
  ///
  /// This is a convenience method for models that take a single input.
  /// Equivalent to calling [runInference] with a single-element input list.
  Future<InferenceResultWrapper> runSingleInput({
    required TensorDataWrapper input,
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
  Future<pigeon.ModelState> getState() async {
    return hostApi.getModelState(modelId);
  }

  /// Check if this model is currently ready for inference
  Future<bool> get isReady async {
    if (_disposed) return false;
    final state = await getState();
    return state == pigeon.ModelState.ready;
  }

  /// Check if this model has been disposed
  bool get isDisposed => _disposed;

  /// Get the primary (first) input specification
  ///
  /// This is a convenience property for models with a single primary input.
  pigeon.TensorSpec? get primaryInputSpec => metadata.primaryInput;

  /// Get the primary (first) output specification
  ///
  /// This is a convenience property for models with a single primary output.
  pigeon.TensorSpec? get primaryOutputSpec => metadata.primaryOutput;

  /// Validate that inputs match the model's input specifications
  void _validateInputs(List<TensorDataWrapper> inputs) {
    if (inputs.length != metadata.inputSpecs.length) {
      throw ExecuTorchValidationException(
        'Input count mismatch: expected ${metadata.inputSpecs.length}, got ${inputs.length}',
      );
    }

    for (int i = 0; i < inputs.length; i++) {
      final input = inputs[i];
      final spec = metadata.inputSpecs[i];

      // Validate data type
      if (input.dataType != spec.dataType) {
        throw ExecuTorchValidationException(
          'Input $i data type mismatch: expected ${spec.dataType}, got ${input.dataType}',
        );
      }

      // Validate shape (allowing dynamic dimensions marked as -1)
      if (input.shape.length != spec.shape.length) {
        throw ExecuTorchValidationException(
          'Input $i shape rank mismatch: expected ${spec.shape.length}D, got ${input.shape.length}D',
        );
      }

      for (int j = 0; j < input.shape.length; j++) {
        final inputDim = input.shape[j];
        final specDim = spec.shape[j];

        // Skip validation for dynamic dimensions (-1)
        if (specDim == -1) continue;

        if (inputDim != specDim) {
          throw ExecuTorchValidationException(
            'Input $i dimension $j mismatch: expected $specDim, got $inputDim',
          );
        }
      }

      // Validate tensor data integrity
      if (!input.isValid) {
        throw ExecuTorchValidationException(
          'Input $i has invalid data: expected ${input.expectedSizeBytes} bytes, got ${input.data.length} bytes',
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
    return 'ExecuTorchModel($modelId: ${metadata.description})$disposedStr';
  }
}

/// Base exception class for ExecuTorch-related errors
class ExecuTorchException implements Exception {
  const ExecuTorchException(this.message, {this.details});

  final String message;
  final Map<String, Object>? details;

  @override
  String toString() => 'ExecuTorchException: $message';
}

/// Exception thrown when model validation fails
class ExecuTorchValidationException extends ExecuTorchException {
  const ExecuTorchValidationException(String message, {Map<String, Object>? details})
      : super(message, details: details);

  @override
  String toString() => 'ExecuTorchValidationException: $message';
}

/// Exception thrown when inference execution fails
class ExecuTorchInferenceException extends ExecuTorchException {
  const ExecuTorchInferenceException(String message, {
    Map<String, Object>? details,
    this.result,
  }) : super(message, details: details);

  final InferenceResultWrapper? result;

  @override
  String toString() => 'ExecuTorchInferenceException: $message';
}

/// Exception thrown when model loading fails
class ExecuTorchModelLoadException extends ExecuTorchException {
  const ExecuTorchModelLoadException(String message, {Map<String, Object>? details})
      : super(message, details: details);

  @override
  String toString() => 'ExecuTorchModelLoadException: $message';
}