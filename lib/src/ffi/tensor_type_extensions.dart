// Copyright (c) 2024 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Extended tensor type support for FFI layer.
///
/// Provides additional tensor types beyond the Pigeon-generated
/// TensorType enum.
library;

import '../types.dart';
import '../generated/executorch_ffi.g.dart';

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

  /// Convert to native ETDType.
  ETDType toETDType() => switch (this) {
        ExtendedTensorType.float32 => ETDType.ET_DTYPE_FLOAT32,
        ExtendedTensorType.float64 => ETDType.ET_DTYPE_FLOAT64,
        ExtendedTensorType.int64 => ETDType.ET_DTYPE_INT64,
        ExtendedTensorType.int32 => ETDType.ET_DTYPE_INT32,
        ExtendedTensorType.int16 => ETDType.ET_DTYPE_INT16,
        ExtendedTensorType.int8 => ETDType.ET_DTYPE_INT8,
        ExtendedTensorType.uint8 => ETDType.ET_DTYPE_UINT8,
        ExtendedTensorType.bool_ => ETDType.ET_DTYPE_BOOL,
      };

  /// Create from native ETDType.
  static ExtendedTensorType fromETDType(ETDType dtype) => switch (dtype) {
        ETDType.ET_DTYPE_FLOAT32 => ExtendedTensorType.float32,
        ETDType.ET_DTYPE_FLOAT64 => ExtendedTensorType.float64,
        ETDType.ET_DTYPE_INT64 => ExtendedTensorType.int64,
        ETDType.ET_DTYPE_INT32 => ExtendedTensorType.int32,
        ETDType.ET_DTYPE_INT16 => ExtendedTensorType.int16,
        ETDType.ET_DTYPE_INT8 => ExtendedTensorType.int8,
        ETDType.ET_DTYPE_UINT8 => ExtendedTensorType.uint8,
        ETDType.ET_DTYPE_BOOL => ExtendedTensorType.bool_,
      };

  /// Convert to Pigeon TensorType (with fallback for unsupported types).
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

  /// Create from Pigeon TensorType.
  static ExtendedTensorType fromTensorType(TensorType type) => switch (type) {
        TensorType.float32 => ExtendedTensorType.float32,
        TensorType.int32 => ExtendedTensorType.int32,
        TensorType.int8 => ExtendedTensorType.int8,
        TensorType.uint8 => ExtendedTensorType.uint8,
      };
}

/// Extension on TensorType for FFI conversions.
extension TensorTypeFFI on TensorType {
  /// Convert to native ETDType.
  ETDType toETDType() => switch (this) {
        TensorType.float32 => ETDType.ET_DTYPE_FLOAT32,
        TensorType.int8 => ETDType.ET_DTYPE_INT8,
        TensorType.int32 => ETDType.ET_DTYPE_INT32,
        TensorType.uint8 => ETDType.ET_DTYPE_UINT8,
      };

  /// Convert to ExtendedTensorType.
  ExtendedTensorType toExtended() => ExtendedTensorType.fromTensorType(this);
}

/// Extension on ETDType for Dart type conversions.
extension ETDTypeDart on ETDType {
  /// Convert to Pigeon TensorType (with fallback).
  TensorType toTensorType() =>
      ExtendedTensorType.fromETDType(this).toTensorType();

  /// Convert to ExtendedTensorType.
  ExtendedTensorType toExtended() => ExtendedTensorType.fromETDType(this);
}
