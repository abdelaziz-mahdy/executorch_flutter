// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Core types for ExecuTorch Flutter FFI bindings.
///
/// This file defines the fundamental types used throughout the library
/// for tensor representation and model loading results.
library;

import 'dart:typed_data';

/// Tensor data type enumeration.
///
/// Represents the data type of tensor elements.
enum TensorType {
  /// 32-bit floating point.
  float32,

  /// 8-bit signed integer.
  int8,

  /// 32-bit signed integer.
  int32,

  /// 8-bit unsigned integer.
  uint8,
}

/// Tensor data for input/output.
///
/// Represents a multi-dimensional array of numeric data used as
/// input to or output from model inference.
///
/// Example:
/// ```dart
/// final input = TensorData(
///   shape: [1, 3, 224, 224],  // NCHW format
///   dataType: TensorType.float32,
///   data: imageBytes,
///   name: 'input_0',
/// );
/// ```
class TensorData {
  /// Creates a new TensorData instance.
  ///
  /// [shape] - The dimensions of the tensor.
  /// [dataType] - The data type of tensor elements.
  /// [data] - The raw bytes of tensor data.
  /// [name] - Optional name for the tensor.
  TensorData({
    required this.shape,
    required this.dataType,
    required this.data,
    this.name,
  });

  /// The dimensions of the tensor.
  ///
  /// For example, `[1, 3, 224, 224]` represents a batch of 1 image
  /// with 3 channels and 224x224 pixels.
  List<int?> shape;

  /// The data type of tensor elements.
  TensorType dataType;

  /// The raw bytes of tensor data.
  ///
  /// The byte layout depends on [dataType]:
  /// - float32: 4 bytes per element (IEEE 754)
  /// - int32: 4 bytes per element (little-endian)
  /// - int8: 1 byte per element
  /// - uint8: 1 byte per element
  Uint8List data;

  /// Optional name for the tensor.
  ///
  /// Used for debugging and model introspection.
  String? name;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! TensorData) return false;

    // Compare shapes
    if (shape.length != other.shape.length) return false;
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] != other.shape[i]) return false;
    }

    // Compare dataType
    if (dataType != other.dataType) return false;

    // Compare data
    if (data.length != other.data.length) return false;
    for (var i = 0; i < data.length; i++) {
      if (data[i] != other.data[i]) return false;
    }

    // Compare name
    if (name != other.name) return false;

    return true;
  }

  @override
  int get hashCode => Object.hash(
        Object.hashAll(shape),
        dataType,
        Object.hashAll(data),
        name,
      );

  @override
  String toString() {
    final shapeStr = shape.map((d) => d?.toString() ?? '?').join(', ');
    return 'TensorData(shape: [$shapeStr], dataType: $dataType, '
        'data: ${data.length} bytes${name != null ? ', name: $name' : ''})';
  }
}

/// Model loading result.
///
/// Contains the unique model ID assigned when a model is loaded.
class ModelLoadResult {
  /// Creates a new ModelLoadResult.
  ///
  /// [modelId] - The unique identifier for the loaded model.
  ModelLoadResult({
    required this.modelId,
  });

  /// The unique identifier for the loaded model.
  ///
  /// This ID is used to reference the model in subsequent
  /// operations like inference and disposal.
  String modelId;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! ModelLoadResult) return false;
    return modelId == other.modelId;
  }

  @override
  int get hashCode => modelId.hashCode;

  @override
  String toString() => 'ModelLoadResult(modelId: $modelId)';
}
