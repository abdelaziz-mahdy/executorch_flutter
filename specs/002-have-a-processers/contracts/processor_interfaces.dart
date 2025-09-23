/// API Contract: Processor Interfaces
///
/// This file defines the contracts for processor interfaces without implementation.
/// These contracts must be fulfilled by the implementation for type safety and consistency.

import 'dart:typed_data';

// Note: These are contract definitions only - actual types come from the main package
abstract class TensorData {
  List<int?> get shape;
  TensorType get dataType;
  Uint8List get data;
  String? get name;
}

abstract class ModelMetadata {
  String get name;
  String get version;
  Map<String, dynamic> get properties;
}

enum TensorType { float32, int32, int8, uint8 }

/// Contract: Base preprocessor interface
abstract class ExecuTorchPreprocessor<T> {
  /// Transform input data into TensorData for model inference
  ///
  /// Contract requirements:
  /// - Must validate input using validateInput() first
  /// - Must return non-empty List<TensorData>
  /// - Must complete within 50ms for typical inputs
  /// - Must throw PreprocessingException on validation failure
  Future<List<TensorData>> preprocess(T input, {ModelMetadata? metadata});

  /// Get the expected input type name for debugging/logging
  ///
  /// Contract requirements:
  /// - Must return non-empty descriptive string
  /// - Should be consistent across instances
  String get inputTypeName;

  /// Validate that the input data is compatible with this preprocessor
  ///
  /// Contract requirements:
  /// - Must return true for valid inputs, false for invalid
  /// - Must not throw exceptions
  /// - Must be called before preprocess()
  bool validateInput(T input);
}

/// Contract: Base postprocessor interface
abstract class ExecuTorchPostprocessor<T> {
  /// Transform inference outputs into meaningful results
  ///
  /// Contract requirements:
  /// - Must validate outputs using validateOutputs() first
  /// - Must return valid T instance
  /// - Must complete within 50ms for typical outputs
  /// - Must throw PostprocessingException on validation failure
  Future<T> postprocess(List<TensorData> outputs, {ModelMetadata? metadata});

  /// Get the expected output type name for debugging/logging
  ///
  /// Contract requirements:
  /// - Must return non-empty descriptive string
  /// - Should be consistent across instances
  String get outputTypeName;

  /// Validate that the outputs are compatible with this postprocessor
  ///
  /// Contract requirements:
  /// - Must return true for valid outputs, false for invalid
  /// - Must not throw exceptions
  /// - Must be called before postprocess()
  bool validateOutputs(List<TensorData> outputs);
}

/// Contract: Combined processor interface
abstract class ExecuTorchProcessor<TInput, TOutput> {
  /// The preprocessor for input transformation
  ///
  /// Contract requirements:
  /// - Must return valid ExecuTorchPreprocessor<TInput> instance
  /// - Must be consistent across calls
  ExecuTorchPreprocessor<TInput> get preprocessor;

  /// The postprocessor for output transformation
  ///
  /// Contract requirements:
  /// - Must return valid ExecuTorchPostprocessor<TOutput> instance
  /// - Must be consistent across calls
  ExecuTorchPostprocessor<TOutput> get postprocessor;

  /// Run the complete pipeline: preprocess → inference → postprocess
  ///
  /// Contract requirements:
  /// - Must call preprocessor.preprocess() with input
  /// - Must call model.runInference() with preprocessed tensors
  /// - Must call postprocessor.postprocess() with inference results
  /// - Must handle InferenceStatus.success vs failure
  /// - Must throw Exception on inference failure
  /// - Must complete entire pipeline within performance targets
  Future<TOutput> process(TInput input, dynamic model, {ModelMetadata? metadata});
}

/// Contract: Exception for preprocessing failures
abstract class PreprocessingException implements Exception {
  String get message;
  Map<String, dynamic>? get details;
}

/// Contract: Exception for postprocessing failures
abstract class PostprocessingException implements Exception {
  String get message;
  Map<String, dynamic>? get details;
}

/// Contract: Tensor utility functions
abstract class ProcessorTensorUtils {
  /// Create a TensorData from a list of values
  ///
  /// Contract requirements:
  /// - shape must have positive dimensions
  /// - data length must match calculated element count
  /// - dataType must be compatible with data values
  /// - Must return valid TensorData instance
  static TensorData createTensor({
    required List<int> shape,
    required TensorType dataType,
    required List<num> data,
    String? name,
  }) {
    throw UnimplementedError('Contract only - implementation required');
  }

  /// Extract numeric data from a float32 tensor
  ///
  /// Contract requirements:
  /// - tensor.dataType must be TensorType.float32
  /// - Must return List<double> with correct element count
  /// - Must throw ArgumentError for incompatible tensor type
  static List<double> extractFloat32Data(TensorData tensor) {
    throw UnimplementedError('Contract only - implementation required');
  }

  /// Extract integer data from an int32 tensor
  ///
  /// Contract requirements:
  /// - tensor.dataType must be TensorType.int32
  /// - Must return List<int> with correct element count
  /// - Must throw ArgumentError for incompatible tensor type
  static List<int> extractInt32Data(TensorData tensor) {
    throw UnimplementedError('Contract only - implementation required');
  }

  /// Calculate total number of elements from shape
  ///
  /// Contract requirements:
  /// - Must handle empty shape (return 1)
  /// - Must handle multi-dimensional shapes correctly
  /// - Must return positive integer result
  static int calculateElementCount(List<int> shape) {
    throw UnimplementedError('Contract only - implementation required');
  }
}