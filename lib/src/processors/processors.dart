/// ExecuTorch Flutter Base Processors
///
/// This library provides the base classes and utilities for building
/// preprocessing and postprocessing components for ExecuTorch models.
/// These abstract classes define the interface for converting domain-specific
/// input into tensors and model outputs back into meaningful results.
///
/// ## Core Architecture
///
/// The processor architecture is based on three main abstract classes:
///
/// - [ExecuTorchPreprocessor]: Converts input data to tensors
/// - [ExecuTorchPostprocessor]: Converts output tensors to results
/// - [ExecuTorchProcessor]: Combines preprocessing and postprocessing
///
/// ## Utilities
/// - [ProcessorTensorUtils]: Tensor creation and manipulation utilities
/// - [ProcessorException]: Base exception class for processor errors
/// - [PreprocessingException]: Preprocessing-specific exceptions
/// - [PostprocessingException]: Postprocessing-specific exceptions
/// - [InvalidInputException]: Invalid input exceptions
/// - [InvalidOutputException]: Invalid output exceptions
///
/// ## Usage Examples
///
/// ### Creating a Custom Preprocessor
/// ```dart
/// class MyImagePreprocessor extends ExecuTorchPreprocessor<Uint8List> {
///   @override
///   String get inputTypeName => 'Image';
///
///   @override
///   bool validateInput(Uint8List input) => input.isNotEmpty;
///
///   @override
///   Future<List<TensorData>> preprocess(
///     Uint8List input, {ModelMetadata? metadata}
///   ) async {
///     // Convert image to tensor
///     final tensorData = ProcessorTensorUtils.createTensor(
///       shape: [1, 3, 224, 224],
///       dataType: TensorType.float32,
///       data: processedImageData,
///       name: 'input',
///     );
///     return [tensorData];
///   }
/// }
/// ```
///
/// ### Creating a Custom Postprocessor
/// ```dart
/// class MyClassificationPostprocessor
///     extends ExecuTorchPostprocessor<MyResult> {
///   @override
///   String get outputTypeName => 'Classification';
///
///   @override
///   bool validateOutputs(List<TensorData> outputs) => outputs.isNotEmpty;
///
///   @override
///   Future<MyResult> postprocess(
///     List<TensorData> outputs, {ModelMetadata? metadata}
///   ) async {
///     // Process model outputs
///     return MyResult(/* ... */);
///   }
/// }
/// ```
///
/// For complete processor implementations, see the example app which contains
/// reference implementations for image classification, object detection,
/// text processing, and audio processing.
library;

import 'package:executorch_flutter/executorch_flutter.dart'
    show
        ExecuTorchPreprocessor,
        ExecuTorchPostprocessor,
        ExecuTorchProcessor,
        ProcessorTensorUtils,
        ProcessorException,
        PreprocessingException,
        PostprocessingException,
        InvalidInputException,
        InvalidOutputException;
import 'package:executorch_flutter/src/processors/base_processor.dart'
    show
        ExecuTorchPreprocessor,
        ExecuTorchPostprocessor,
        ExecuTorchProcessor,
        ProcessorTensorUtils,
        ProcessorException,
        PreprocessingException,
        PostprocessingException,
        InvalidInputException,
        InvalidOutputException;
import 'package:executorch_flutter/src/processors/processors.dart'
    show
        ExecuTorchPreprocessor,
        ExecuTorchPostprocessor,
        ExecuTorchProcessor,
        ProcessorTensorUtils,
        ProcessorException,
        PreprocessingException,
        PostprocessingException,
        InvalidInputException,
        InvalidOutputException;

// Base processor classes and utilities only
export 'base_processor.dart';
