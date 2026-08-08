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
/// ExecuTorch's ETDType enum in executorch_ffi.h.
enum TensorType {
  /// 32-bit floating point (IEEE 754).
  float32('Float32', 0, 4),

  /// 64-bit floating point (IEEE 754).
  float64('Float64', 1, 8),

  /// 64-bit signed integer (two's complement).
  int64('Int64', 2, 8),

  /// 32-bit signed integer (two's complement).
  int32('Int32', 3, 4),

  /// 16-bit signed integer (two's complement).
  int16('Int16', 4, 2),

  /// 8-bit signed integer (two's complement).
  int8('Int8', 5, 1),

  /// 8-bit unsigned integer.
  uint8('UInt8', 6, 1),

  /// Boolean (1 byte per element: 0 = false, 1 = true).
  bool_('Bool', 7, 1),

  /// 16-bit unsigned integer.
  uint16('UInt16', 8, 2),

  /// 32-bit unsigned integer.
  uint32('UInt32', 9, 4),

  /// 64-bit unsigned integer.
  uint64('UInt64', 10, 8),

  /// 16-bit floating point (IEEE 754 half precision).
  float16('Float16', 11, 2),

  /// 16-bit brain floating point (Intel BF16).
  bfloat16('BFloat16', 12, 2);

  const TensorType(this.displayName, this.executorchValue, this.sizeInBytes);

  /// Human-readable display name.
  final String displayName;

  /// ExecuTorch native dtype enum value (ET_DTYPE_*).
  final int executorchValue;

  /// Size of one element in bytes.
  final int sizeInBytes;

  /// Inverse map: executorchValue → TensorType (built once at load time).
  static final Map<int, TensorType> _fromValue = {
    for (final t in TensorType.values) t.executorchValue: t,
  };

  /// Create a [TensorType] from an ExecuTorch native integer value.
  static TensorType fromExecuTorchValue(int value) =>
      _fromValue[value] ??
      (throw ArgumentError.value(
        value,
        'value',
        'Unsupported ExecuTorch tensor type: $value. '
            'Supported values: ${_fromValue.keys.toList()}.',
      ));
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

/// Deprecated alias for [TensorType].
///
/// Previously a separate enum with only 8 types. Now simply re-exports
/// [TensorType] which covers all 13 ExecuTorch dtypes.
///
/// **Deprecated**: Use [TensorType] directly.
@Deprecated('Use TensorType instead')
typedef ExtendedTensorType = TensorType;

/// Deprecated extension providing backward compatibility for
/// [ExtendedTensorType].
///
/// **Deprecated**: Use [TensorType.displayName] and
/// [TensorType.executorchValue] directly.
@Deprecated('Use TensorType.displayName and TensorType.executorchValue instead')
extension TensorTypeExtension on TensorType {
  /// Returns a human-readable name for this tensor type.
  @Deprecated('Use TensorType.displayName instead')
  String get name => displayName;

  /// Returns the size in bytes of a single element.
  @Deprecated('Use TensorType.sizeInBytes instead')
  int get bytes => sizeInBytes;

  /// Converts this tensor type to its [TensorType] equivalent.
  ///
  /// Returns this value unchanged, since [ExtendedTensorType] is now an
  /// alias for [TensorType].
  @Deprecated('This is now a no-op; ExtendedTensorType is TensorType')
  TensorType get baseType => this;
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
