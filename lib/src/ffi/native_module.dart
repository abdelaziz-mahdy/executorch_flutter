// Copyright (c) 2024 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native module wrapper for FFI layer.
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../generated/executorch_api.dart';
import '../generated/executorch_ffi.g.dart';
import 'native_status.dart';
import 'native_tensor.dart';

/// Wrapper around native ETModule pointer with automatic memory management.
///
/// Uses NativeFinalizer for automatic cleanup when the Dart object is GC'd.
class NativeModule implements ffi.Finalizable {
  /// Create a NativeModule from a native pointer.
  ///
  /// Takes ownership of the pointer and will free it when disposed.
  NativeModule._(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// The native module pointer.
  final ffi.Pointer<ETModule> _ptr;

  /// Whether this module has been disposed.
  bool _disposed = false;

  /// Finalizer for automatic cleanup.
  static final _finalizer = ffi.NativeFinalizer(
    addresses.et_module_free.cast(),
  );

  /// Load a model from memory buffer.
  ///
  /// [data] is the model data in .pte format.
  static NativeModule load(Uint8List data) {
    // Allocate native buffer
    final dataPtr = calloc<ffi.Uint8>(data.length);
    try {
      // Copy data to native buffer
      final nativeData = dataPtr.asTypedList(data.length);
      nativeData.setAll(0, data);

      // Allocate output pointer
      final outPtr = calloc<ffi.Pointer<ETModule>>();
      try {
        final status = et_module_load(dataPtr, data.length, outPtr);
        checkStatus(status);

        return NativeModule._(outPtr.value);
      } finally {
        calloc.free(outPtr);
      }
    } finally {
      calloc.free(dataPtr);
    }
  }

  /// Load a model from file path.
  ///
  /// [path] is the path to the .pte model file.
  static NativeModule loadFile(String path) {
    final pathPtr = path.toNativeUtf8().cast<ffi.Char>();
    try {
      final outPtr = calloc<ffi.Pointer<ETModule>>();
      try {
        final status = et_module_load_file(pathPtr, outPtr);
        checkStatus(status);

        return NativeModule._(outPtr.value);
      } finally {
        calloc.free(outPtr);
      }
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// Get the number of model inputs.
  int get inputCount {
    _checkDisposed();
    return et_module_input_count(_ptr);
  }

  /// Get the number of model outputs.
  int get outputCount {
    _checkDisposed();
    return et_module_output_count(_ptr);
  }

  /// Run forward pass (inference).
  ///
  /// [inputs] is a list of input tensors as TensorData.
  /// Returns a list of output tensors as TensorData.
  List<TensorData> forward(List<TensorData> inputs) {
    _checkDisposed();

    // Convert inputs to native tensors
    final nativeInputs = inputs.map(NativeTensor.fromTensorData).toList();

    try {
      // Allocate input array
      final inputPtrs = calloc<ffi.Pointer<ETTensor>>(nativeInputs.length);
      for (var i = 0; i < nativeInputs.length; i++) {
        inputPtrs[i] = nativeInputs[i].ptr;
      }

      // Allocate output pointers
      final outputsPtr =
          calloc<ffi.Pointer<ffi.Pointer<ETTensor>>>();
      final outputCountPtr = calloc<ffi.Int32>();

      try {
        // Run forward pass
        final status = et_module_forward(
          _ptr,
          inputPtrs,
          nativeInputs.length,
          outputsPtr,
          outputCountPtr,
        );
        checkStatus(status);

        // Convert outputs to TensorData
        final outputCount = outputCountPtr.value;
        final outputs = <TensorData>[];
        final outputArray = outputsPtr.value;

        for (var i = 0; i < outputCount; i++) {
          final nativeOutput = NativeTensor.fromPointer(outputArray[i]);
          outputs.add(nativeOutput.toTensorData());
          // Don't dispose - the array free will handle it
        }

        // Free the output array (but not the tensors - we've converted them)
        et_tensor_array_free(outputArray, outputCount);

        return outputs;
      } finally {
        calloc.free(inputPtrs);
        calloc.free(outputsPtr);
        calloc.free(outputCountPtr);
      }
    } finally {
      // Dispose input native tensors
      for (final tensor in nativeInputs) {
        tensor.dispose();
      }
    }
  }

  /// Whether this module has been disposed.
  bool get isDisposed => _disposed;

  /// Dispose this module and free native resources.
  ///
  /// Safe to call multiple times - subsequent calls are no-ops.
  void dispose() {
    if (!_disposed) {
      _finalizer.detach(this);
      et_module_free(_ptr);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('NativeModule has been disposed');
    }
  }
}
