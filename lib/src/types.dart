// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

// TensorData and ModelLoadResult are effectively immutable (all fields are
// final) but can't use @immutable + const constructors because Uint8List
// and List<int?> can't be compile-time constants. The == and hashCode
// overrides are safe because the fields are never modified after construction.
// ignore_for_file: avoid_equals_and_hash_code_on_mutable_classes

/// Core types for ExecuTorch Flutter FFI bindings.
///
/// This file defines the fundamental types used throughout the library
/// for tensor representation and model loading results.
library;

import 'dart:typed_data';

/// Hardware acceleration backends supported by ExecuTorch.
enum Backend {
  /// XNNPACK CPU optimization library (available on all platforms).
  xnnpack,

  /// Apple CoreML backend (iOS/macOS only).
  coreml,

  /// Apple Metal Performance Shaders backend (deprecated; replaced by [metal]).
  mps,

  /// Apple Metal GPU backend (macOS desktop only). Replaces the deprecated MPS.
  metal,

  /// Vulkan GPU compute backend (Android/iOS/macOS/Windows/Linux).
  vulkan,

  /// Qualcomm Neural Processing Unit backend (Android only).
  qnn;

  /// Human-readable display name for this backend.
  String get displayName => switch (this) {
        Backend.xnnpack => 'XNNPACK',
        Backend.coreml => 'CoreML',
        Backend.mps => 'Metal Performance Shaders',
        Backend.metal => 'Metal',
        Backend.vulkan => 'Vulkan',
        Backend.qnn => 'Qualcomm QNN',
      };
}

/// Tensor data type enumeration.
///
/// Represents the data type of tensor elements.
/// All 13 types are supported by the native FFI layer and map 1:1 to
/// ExecuTorch's [ETDType] enum in [executorch_ffi.h].
enum TensorType {
  /// 32-bit floating point (IEEE 754).
  float32,

  /// 64-bit floating point (IEEE 754).
  float64,

  /// 64-bit signed integer (two's complement).
  int64,

  /// 32-bit signed integer (two's complement).
  int32,

  /// 16-bit signed integer (two's complement).
  int16,

  /// 8-bit signed integer (two's complement).
  int8,

  /// 8-bit unsigned integer.
  uint8,

  /// Boolean (1 byte per element: 0 = false, 1 = true).
  bool_,

  /// 16-bit unsigned integer.
  uint16,

  /// 32-bit unsigned integer.
  uint32,

  /// 64-bit unsigned integer.
  uint64,

  /// 16-bit floating point (IEEE 754 half precision).
  float16,

  /// 16-bit brain floating point (Intel BF16).
  bfloat16;

  /// ExecuTorch native integer value.
  ///
  /// This value maps 1:1 to ExecuTorch's [ETDType] enum values:
  /// - float32 = 0 (ET_DTYPE_FLOAT32)
  /// - float64 = 1 (ET_DTYPE_FLOAT64)
  /// - int64 = 2 (ET_DTYPE_INT64)
  /// - int32 = 3 (ET_DTYPE_INT32)
  /// - int16 = 4 (ET_DTYPE_INT16)
  /// - int8 = 5 (ET_DTYPE_INT8)
  /// - uint8 = 6 (ET_DTYPE_UINT8)
  /// - bool_ = 7 (ET_DTYPE_BOOL)
  /// - uint16 = 8 (ET_DTYPE_UINT16)
  /// - uint32 = 9 (ET_DTYPE_UINT32)
  /// - uint64 = 10 (ET_DTYPE_UINT64)
  /// - float16 = 11 (ET_DTYPE_FLOAT16)
  /// - bfloat16 = 12 (ET_DTYPE_BFLOAT16)
  int get executorchValue => index;

  /// Create a [TensorType] from an ExecuTorch native integer value.
  ///
  /// Supports all 13 types (values 0–12). Throws [ArgumentError] for
  /// any other value so the developer is immediately notified of
  /// unsupported or unknown types rather than receiving a silent fallback.
  factory TensorType.fromExecuTorchValue(int value) {
    if (value < 0 || value >= TensorType.values.length) {
      throw ArgumentError.value(
        value,
        'value',
        'Unsupported ExecuTorch tensor type: $value. '
        'Supported range is 0–${TensorType.values.length - 1} '
        '(${TensorType.values.length} types).',
      );
    }
    return TensorType.values[value];
  }

  /// Human-readable display name.
  String get displayName => switch (this) {
        TensorType.float32 => 'Float32',
        TensorType.float64 => 'Float64',
        TensorType.int64 => 'Int64',
        TensorType.int32 => 'Int32',
        TensorType.int16 => 'Int16',
        TensorType.int8 => 'Int8',
        TensorType.uint8 => 'UInt8',
        TensorType.bool_ => 'Bool',
        TensorType.uint16 => 'UInt16',
        TensorType.uint32 => 'UInt32',
        TensorType.uint64 => 'UInt64',
        TensorType.float16 => 'Float16',
        TensorType.bfloat16 => 'BFloat16',
      };

  /// Size of one element in bytes.
  int get sizeInBytes => switch (this) {
        TensorType.float32 => 4,
        TensorType.float64 => 8,
        TensorType.int64 => 8,
        TensorType.int32 => 4,
        TensorType.int16 => 2,
        TensorType.int8 => 1,
        TensorType.uint8 => 1,
        TensorType.bool_ => 1,
        TensorType.uint16 => 2,
        TensorType.uint32 => 4,
        TensorType.uint64 => 8,
        TensorType.float16 => 2,
        TensorType.bfloat16 => 2,
      };
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
final class TensorData {
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
  final List<int?> shape;

  /// The data type of tensor elements.
  final TensorType dataType;

  /// The raw bytes of tensor data.
  ///
  /// The byte layout depends on [dataType]:
  /// - float32: 4 bytes per element (IEEE 754)
  /// - int32: 4 bytes per element (little-endian)
  /// - int8: 1 byte per element
  /// - uint8: 1 byte per element
  final Uint8List data;

  /// Optional name for the tensor.
  ///
  /// Used for debugging and model introspection.
  final String? name;

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
  int get hashCode =>
      Object.hash(Object.hashAll(shape), dataType, Object.hashAll(data), name);

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
final class ModelLoadResult {
  /// Creates a new ModelLoadResult.
  ///
  /// [modelId] - The unique identifier for the loaded model.
  ModelLoadResult({required this.modelId});

  /// The unique identifier for the loaded model.
  ///
  /// This ID is used to reference the model in subsequent
  /// operations like inference and disposal.
  final String modelId;

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
