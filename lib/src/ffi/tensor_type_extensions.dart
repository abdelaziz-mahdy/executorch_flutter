// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// FFI-specific tensor type conversions.
///
/// Provides extensions for converting between Dart types and native ETDType.
library;

import '../generated/executorch_ffi.g.dart';
import '../types.dart';

/// Extension on ExtendedTensorType for FFI conversions.
extension ExtendedTensorTypeFFI on ExtendedTensorType {
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
}

/// Extension on ETDType for Dart type conversions.
extension ETDTypeDart on ETDType {
  /// Convert to TensorType (with fallback).
  TensorType toTensorType() =>
      ExtendedTensorTypeFFI.fromETDType(this).toTensorType();

  /// Convert to ExtendedTensorType.
  ExtendedTensorType toExtended() => ExtendedTensorTypeFFI.fromETDType(this);
}
