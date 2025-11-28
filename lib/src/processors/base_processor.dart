import 'dart:typed_data';

import '../executorch_errors.dart';
import '../executorch_model.dart';
import '../generated/executorch_api.dart';

/// Utility class for tensor operations in processors
class ProcessorTensorUtils {
  ProcessorTensorUtils._();

  /// Creates a tensor from numeric data with specified shape and type
  static TensorData createTensor({
    required List<int> shape,
    required TensorType dataType,
    required List<num> data,
    String? name,
  }) {
    final elementCount = calculateElementCount(shape);

    if (data.length != elementCount) {
      throw ArgumentError(
        'Data length (${data.length}) does not match '
        'shape element count ($elementCount)',
      );
    }

    Uint8List bytes;
    switch (dataType) {
      case TensorType.float32:
        final float32List = Float32List.fromList(data.cast<double>());
        bytes = float32List.buffer.asUint8List();
        break;
      case TensorType.int32:
        final int32List = Int32List.fromList(data.cast<int>());
        bytes = int32List.buffer.asUint8List();
        break;
      case TensorType.uint8:
        bytes = Uint8List.fromList(data.cast<int>());
        break;
      default:
        throw UnsupportedError('Tensor type $dataType not supported');
    }

    return TensorData(
      shape: shape.cast<int?>(),
      dataType: dataType,
      data: bytes,
      name: name,
    );
  }

  /// Extracts Float32 data from a tensor
  static Float32List extractFloat32Data(TensorData tensor) {
    if (tensor.dataType != TensorType.float32) {
      throw ArgumentError('Expected Float32 tensor, got ${tensor.dataType}');
    }
    return tensor.data.buffer.asFloat32List();
  }

  /// Extracts Int32 data from a tensor
  static Int32List extractInt32Data(TensorData tensor) {
    if (tensor.dataType != TensorType.int32) {
      throw ArgumentError('Expected Int32 tensor, got ${tensor.dataType}');
    }
    return tensor.data.buffer.asInt32List();
  }

  /// Extracts Uint8 data from a tensor
  static Uint8List extractUint8Data(TensorData tensor) {
    if (tensor.dataType != TensorType.uint8) {
      throw ArgumentError('Expected Uint8 tensor, got ${tensor.dataType}');
    }
    return tensor.data;
  }

  /// Calculates the total number of elements from a shape
  static int calculateElementCount(List<int> shape) {
    if (shape.isEmpty) return 0;
    return shape.reduce((a, b) => a * b);
  }

  /// Validates tensor shape matches expected dimensions
  static bool validateTensorShape(TensorData tensor, List<int> expectedShape) {
    if (tensor.shape.length != expectedShape.length) return false;

    for (var i = 0; i < expectedShape.length; i++) {
      if (expectedShape[i] != -1 && tensor.shape[i] != expectedShape[i]) {
        return false;
      }
    }
    return true;
  }
}

/// Base exception class for processor-related errors.
///
/// This is the parent class for all processor-specific exceptions.
/// It provides a consistent error handling mechanism across all
/// preprocessing and postprocessing operations.
abstract class ProcessorException implements Exception {
  /// Creates a processor exception with the given [message] and optional
  /// [cause].
  const ProcessorException(this.message, [this.cause]);

  /// The error message describing what went wrong.
  final String message;

  /// The underlying cause of this exception, if any.
  final Object? cause;

  @override
  String toString() => 'ProcessorException: $message';
}

/// Generic processor exception implementation.
///
/// Used for unexpected or general errors that don't fit specific categories.
class GenericProcessorException extends ProcessorException {
  /// Creates a generic processor exception with the given [message].
  const GenericProcessorException(super.message, [super.cause]);
}

/// Exception thrown during preprocessing operations.
///
/// Thrown when input data cannot be successfully converted to tensors.
class PreprocessingException extends ProcessorException {
  /// Creates a preprocessing exception with the given [message].
  const PreprocessingException(super.message, [super.cause]);

  @override
  String toString() => 'PreprocessingException: $message';
}

/// Exception thrown during postprocessing operations.
///
/// Thrown when model output tensors cannot be converted to results.
class PostprocessingException extends ProcessorException {
  /// Creates a postprocessing exception with the given [message].
  const PostprocessingException(super.message, [super.cause]);

  @override
  String toString() => 'PostprocessingException: $message';
}

/// Exception thrown for invalid processor input.
///
/// Thrown when input validation fails before preprocessing.
class InvalidInputException extends ProcessorException {
  /// Creates an invalid input exception with the given [message].
  const InvalidInputException(super.message, [super.cause]);

  @override
  String toString() => 'InvalidInputException: $message';
}

/// Exception thrown for invalid processor output.
///
/// Thrown when model output validation fails before postprocessing.
class InvalidOutputException extends ProcessorException {
  /// Creates an invalid output exception with the given [message].
  const InvalidOutputException(super.message, [super.cause]);

  @override
  String toString() => 'InvalidOutputException: $message';
}

/// Abstract base class for input preprocessing
///
/// This class defines the contract for preprocessing input data before
/// inference. Implementations should convert domain-specific input (images,
/// text, audio) into tensors suitable for model execution.
///
/// Type parameter [T] represents the input data type (e.g., Uint8List
/// for images, String for text, Float32List for audio).
abstract class ExecuTorchPreprocessor<T> {
  /// Human-readable name for the input type this preprocessor handles
  String get inputTypeName;

  /// Validates that the input data is suitable for preprocessing
  ///
  /// This method should perform basic validation without expensive operations.
  /// Return true if the input can be processed, false otherwise.
  bool validateInput(T input);

  /// Preprocesses input data into tensors for model inference
  ///
  /// This method performs the actual conversion from input data to tensors.
  /// It should handle normalization, resizing, tokenization, or other
  /// domain-specific transformations.
  ///
  /// Throws [PreprocessingException] if processing fails.
  /// Throws [InvalidInputException] if input validation fails.
  Future<List<TensorData>> preprocess(T input);
}

/// Abstract base class for output postprocessing
///
/// This class defines the contract for postprocessing model outputs into
/// domain-specific results. Implementations should convert raw tensor outputs
/// into meaningful results (classification labels, detected objects, etc.).
///
/// Type parameter [R] represents the result type (e.g., ClassificationResult,
/// DetectionResult, String).
abstract class ExecuTorchPostprocessor<R> {
  /// Human-readable name for the output type this postprocessor produces
  String get outputTypeName;

  /// Validates that the output tensors are suitable for postprocessing
  ///
  /// This method should verify tensor shapes, types, and counts match
  /// expectations. Return true if outputs can be processed, false otherwise.
  bool validateOutputs(List<TensorData> outputs);

  /// Postprocesses model output tensors into domain-specific results
  ///
  /// This method performs the actual conversion from raw tensors to meaningful
  /// results. It should handle softmax application, label mapping, coordinate
  /// transformation, or other domain-specific operations.
  ///
  /// Throws [PostprocessingException] if processing fails.
  /// Throws [InvalidOutputException] if output validation fails.
  Future<R> postprocess(List<TensorData> outputs);
}

/// Abstract base class combining preprocessing and postprocessing
///
/// This class provides a complete processing pipeline that handles input
/// preprocessing, model inference, and output postprocessing in a single
/// convenient interface.
///
/// Type parameters:
/// - [T]: Input data type (e.g., Uint8List, String, Float32List)
/// - [R]: Result data type (e.g., ClassificationResult, String)
abstract class ExecuTorchProcessor<T, R> {
  /// The preprocessor for handling input data
  ExecuTorchPreprocessor<T> get preprocessor;

  /// The postprocessor for handling output data
  ExecuTorchPostprocessor<R> get postprocessor;

  /// Processes input through the complete pipeline
  ///
  /// This method orchestrates the full processing pipeline:
  /// 1. Validates and preprocesses input data
  /// 2. Runs model inference
  /// 3. Validates and postprocesses outputs
  /// 4. Returns the final result
  ///
  /// This is the primary method users should call for end-to-end processing.
  ///
  /// Throws [InvalidInputException] if input validation fails.
  /// Throws [PreprocessingException] if preprocessing fails.
  /// Throws [PostprocessingException] if postprocessing fails.
  /// Throws [ExecuTorchException] if model inference fails.
  Future<R> process(T input, ExecuTorchModel model) async {
    try {
      // Validate input
      if (!preprocessor.validateInput(input)) {
        throw InvalidInputException(
            'Input validation failed for ${preprocessor.inputTypeName}');
      }

      // Preprocess input
      final inputs = await preprocessor.preprocess(input);

      if (inputs.isEmpty) {
        throw const PreprocessingException(
            'Preprocessing produced no output tensors');
      }

      // Run inference (throws exception on failure)
      final outputs = await model.forward(inputs);

      // Validate outputs
      if (!postprocessor.validateOutputs(outputs)) {
        throw InvalidOutputException(
          'Model outputs validation failed for '
          '${postprocessor.outputTypeName}',
        );
      }

      // Postprocess outputs
      final result = await postprocessor.postprocess(outputs);

      return result;
    } catch (e) {
      if (e is ProcessorException || e is ExecuTorchException) {
        rethrow;
      }
      throw GenericProcessorException(
          'Unexpected error during processing: $e', e);
    }
  }
}
