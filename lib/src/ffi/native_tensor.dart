// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native tensor wrapper for FFI layer.
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
import '../generated/executorch_ffi.g.dart';
import '../types.dart';
import 'native_status.dart';

/// Wrapper around native ETTensor pointer with automatic memory management.
///
/// Uses NativeFinalizer for automatic cleanup when the Dart object is GC'd.
class NativeTensor implements ffi.Finalizable {
  /// Create a NativeTensor from a native pointer.
  ///
  /// Takes ownership of the pointer and will free it when disposed.
  /// Internal constructor - use [NativeTensor.fromTensorData] for public API.
  NativeTensor.fromPointer(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Create a NativeTensor from TensorData.
  ///
  /// Converts the Dart TensorData to a native ETTensor.
  factory NativeTensor.fromTensorData(TensorData tensorData) {
    // Convert shape to native array
    final rank = tensorData.shape.length;
    final shapePtr = calloc<ffi.Int64>(rank);
    for (var i = 0; i < rank; i++) {
      shapePtr[i] = tensorData.shape[i] ?? 0;
    }

    // Convert data type using shared conversion from TensorType
    final dtype = ETDType.fromValue(tensorData.dataType.executorchValue);

    // Allocate native data
    final dataSize = tensorData.data.length;
    final dataPtr = calloc<ffi.Uint8>(dataSize);
    dataPtr.asTypedList(dataSize).setAll(0, tensorData.data);

    // Create native tensor
    final outPtr = calloc<ffi.Pointer<ETTensor>>();
    try {
      final status = et_tensor_create(
        dataPtr.cast(),
        dataSize,
        shapePtr,
        rank,
        dtype,
        outPtr,
      );
      checkStatus(status);

      return NativeTensor.fromPointer(outPtr.value);
    } finally {
      // Free temporary allocations
      calloc
        ..free(shapePtr)
        ..free(dataPtr)
        ..free(outPtr);
    }
  }

  /// The native tensor pointer.
  final ffi.Pointer<ETTensor> _ptr;

  /// Whether this tensor has been disposed.
  bool _disposed = false;

  /// Finalizer for automatic cleanup.
  static final _finalizer = ffi.NativeFinalizer(
    addresses.et_tensor_free.cast(),
  );

  /// Convert this native tensor to TensorData.
  TensorData toTensorData() {
    _checkDisposed();

    // Get tensor properties
    final dtype = et_tensor_dtype(_ptr);
    final rank = et_tensor_rank(_ptr);
    final shapePtr = et_tensor_shape(_ptr);
    final dataSize = et_tensor_data_size(_ptr);
    final dataPtr = et_tensor_data(_ptr);

    // Convert shape
    final shape = <int?>[];
    for (var i = 0; i < rank; i++) {
      shape.add(shapePtr[i]);
    }

    // Copy data
    final data = Uint8List(dataSize);
    final nativeData = dataPtr.cast<ffi.Uint8>().asTypedList(dataSize);
    data.setAll(0, nativeData);

    return TensorData(
      shape: shape,
      dataType: TensorType.fromExecuTorchValue(dtype.value),
      data: data,
    );
  }

  /// Get the native pointer.
  ///
  /// Warning: The pointer is only valid while this NativeTensor is alive.
  ffi.Pointer<ETTensor> get ptr {
    _checkDisposed();
    return _ptr;
  }

  /// Whether this tensor has been disposed.
  bool get isDisposed => _disposed;

  /// Dispose this tensor and free native resources.
  ///
  /// Safe to call multiple times - subsequent calls are no-ops.
  void dispose() {
    if (!_disposed) {
      _finalizer.detach(this);
      et_tensor_free(_ptr);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) {
      throw const ExecuTorchMemoryException('Tensor has been disposed');
    }
  }

}
