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

  /// Apple Metal Performance Shaders backend (macOS only).
  mps,

  /// Vulkan GPU compute backend (Android/iOS/macOS/Windows/Linux).
  vulkan,

  /// Qualcomm Neural Processing Unit backend (Android only).
  qnn;

  /// Human-readable display name for this backend.
  String get displayName => switch (this) {
        Backend.xnnpack => 'XNNPACK',
        Backend.coreml => 'CoreML',
        Backend.mps => 'Metal Performance Shaders',
        Backend.vulkan => 'Vulkan',
        Backend.qnn => 'Qualcomm QNN',
      };
}

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

/// Extended tensor types supported by the FFI layer.
///
/// This enum provides all tensor types supported by ExecuTorch,
/// extending beyond the basic types in [TensorType].
enum ExtendedTensorType {
  /// 32-bit floating point (corresponds to TensorType.float32)
  float32,

  /// 64-bit floating point (new in FFI)
  float64,

  /// 64-bit signed integer (new in FFI)
  int64,

  /// 32-bit signed integer (corresponds to TensorType.int32)
  int32,

  /// 16-bit signed integer (new in FFI)
  int16,

  /// 8-bit signed integer (corresponds to TensorType.int8)
  int8,

  /// 8-bit unsigned integer (corresponds to TensorType.uint8)
  uint8,

  /// Boolean (new in FFI)
  bool_;

  /// Human-readable display name for this type.
  String get displayName => switch (this) {
        ExtendedTensorType.float32 => 'Float32',
        ExtendedTensorType.float64 => 'Float64',
        ExtendedTensorType.int64 => 'Int64',
        ExtendedTensorType.int32 => 'Int32',
        ExtendedTensorType.int16 => 'Int16',
        ExtendedTensorType.int8 => 'Int8',
        ExtendedTensorType.uint8 => 'UInt8',
        ExtendedTensorType.bool_ => 'Bool',
      };

  /// Size of this data type in bytes.
  int get sizeInBytes => switch (this) {
        ExtendedTensorType.float32 => 4,
        ExtendedTensorType.float64 => 8,
        ExtendedTensorType.int64 => 8,
        ExtendedTensorType.int32 => 4,
        ExtendedTensorType.int16 => 2,
        ExtendedTensorType.int8 => 1,
        ExtendedTensorType.uint8 => 1,
        ExtendedTensorType.bool_ => 1,
      };

  /// Convert to TensorType (with fallback for unsupported types).
  ///
  /// Types not directly supported by TensorType will use the closest available:
  /// - float64 → float32
  /// - int64 → int32
  /// - int16 → int32
  /// - bool → uint8
  TensorType toTensorType() => switch (this) {
        ExtendedTensorType.float32 => TensorType.float32,
        ExtendedTensorType.float64 => TensorType.float32,
        ExtendedTensorType.int64 => TensorType.int32,
        ExtendedTensorType.int32 => TensorType.int32,
        ExtendedTensorType.int16 => TensorType.int32,
        ExtendedTensorType.int8 => TensorType.int8,
        ExtendedTensorType.uint8 => TensorType.uint8,
        ExtendedTensorType.bool_ => TensorType.uint8,
      };

  /// Create from TensorType.
  static ExtendedTensorType fromTensorType(TensorType type) => switch (type) {
        TensorType.float32 => ExtendedTensorType.float32,
        TensorType.int32 => ExtendedTensorType.int32,
        TensorType.int8 => ExtendedTensorType.int8,
        TensorType.uint8 => ExtendedTensorType.uint8,
      };
}

/// Extension on TensorType for extended type conversions.
extension TensorTypeExtension on TensorType {
  /// Convert to ExtendedTensorType.
  ExtendedTensorType toExtended() => ExtendedTensorType.fromTensorType(this);
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
