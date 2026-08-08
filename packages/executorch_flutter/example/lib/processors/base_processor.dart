import 'package:executorch_flutter/executorch_flutter.dart';

/// Base class for input processors that convert model inputs to tensor data
///
/// Input processors are created with settings baked in and are responsible
/// for preprocessing model inputs (images, text, etc.) into tensor format.
abstract class InputProcessor<TInput> {
  const InputProcessor();

  /// Process the input and return tensor data ready for inference
  Future<List<TensorData>> process(TInput input);
}

/// Base class for output processors that convert tensor outputs to results
///
/// Output processors are created with settings baked in and are responsible
/// for postprocessing model outputs (tensors) into structured results.
abstract class OutputProcessor<TOutput> {
  const OutputProcessor();

  /// Process the output tensors and return structured result
  Future<TOutput> process(List<TensorData> outputs);
}
