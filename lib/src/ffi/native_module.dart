// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native module wrapper for FFI layer.
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
import '../generated/executorch_ffi.g.dart';
import '../types.dart';
import 'native_logging.dart';
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

  /// Load a model from memory buffer.
  ///
  /// [data] is the model data in .pte format.
  factory NativeModule.load(Uint8List data) {
    logDebug('NativeModule.load() called with ${data.length} bytes');

    // Allocate native buffer
    logDebug('Allocating native buffer...');
    final dataPtr = calloc<ffi.Uint8>(data.length);
    try {
      // Copy data to native buffer
      logDebug('Copying data to native buffer...');
      dataPtr.asTypedList(data.length).setAll(0, data);

      // Allocate output pointer
      final outPtr = calloc<ffi.Pointer<ETModule>>();
      try {
        logDebug('Calling et_module_load...');
        final status = et_module_load(dataPtr, data.length, outPtr);

        logDebug(
          'et_module_load returned, status code: ${status.ref.code}',
        );
        checkStatus(status);

        final module = NativeModule._(outPtr.value);
        logDebug(
          'Module loaded successfully! '
          'inputCount=${module.inputCount}, '
          'outputCount=${module.outputCount}',
        );
        return module;
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
  factory NativeModule.loadFile(String path) {
    logDebug('NativeModule.loadFile() called with path: $path');

    final pathPtr = path.toNativeUtf8().cast<ffi.Char>();
    try {
      final outPtr = calloc<ffi.Pointer<ETModule>>();
      try {
        logDebug('Calling et_module_load_file...');
        final status = et_module_load_file(pathPtr, outPtr);

        logDebug(
          'et_module_load_file returned, status code: ${status.ref.code}',
        );
        checkStatus(status);

        final module = NativeModule._(outPtr.value);
        logDebug(
          'Module loaded successfully! '
          'inputCount=${module.inputCount}, '
          'outputCount=${module.outputCount}',
        );
        return module;
      } finally {
        calloc.free(outPtr);
      }
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// The native module pointer.
  final ffi.Pointer<ETModule> _ptr;

  /// Whether this module has been disposed.
  bool _disposed = false;

  /// Finalizer for automatic cleanup.
  static final _finalizer = ffi.NativeFinalizer(
    addresses.et_module_free.cast(),
  );

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

    logDebug('forward() called with ${inputs.length} inputs');

    // Log input tensor details
    for (var i = 0; i < inputs.length; i++) {
      final input = inputs[i];
      logDebug(
        'Input[$i]: shape=${input.shape}, dtype=${input.dataType}, '
        'dataSize=${input.data.length} bytes',
      );
    }

    // Convert inputs to native tensors
    logDebug('Converting inputs to native tensors...');
    final nativeInputs = <NativeTensor>[];
    for (var i = 0; i < inputs.length; i++) {
      try {
        final native = NativeTensor.fromTensorData(inputs[i]);
        nativeInputs.add(native);
        logDebug('Input[$i] converted successfully');
      } catch (e) {
        logError('Failed to convert input[$i] to native tensor: $e');
        // Dispose already created tensors
        for (final tensor in nativeInputs) {
          tensor.dispose();
        }
        rethrow;
      }
    }

    try {
      // Allocate input array
      logDebug('Allocating input pointer array...');
      final inputPtrs = calloc<ffi.Pointer<ETTensor>>(nativeInputs.length);
      for (var i = 0; i < nativeInputs.length; i++) {
        inputPtrs[i] = nativeInputs[i].ptr;
        logDebug('Input[$i] ptr: ${nativeInputs[i].ptr.address}');
      }

      // Allocate output pointers
      final outputsPtr = calloc<ffi.Pointer<ffi.Pointer<ETTensor>>>();
      final outputCountPtr = calloc<ffi.Int32>();

      try {
        // Run forward pass
        logDebug('Calling et_module_forward...');
        final status = et_module_forward(
          _ptr,
          inputPtrs,
          nativeInputs.length,
          outputsPtr,
          outputCountPtr,
        );

        logDebug('et_module_forward returned, checking status...');
        logDebug(
          'Status code: ${status.ref.code}, '
          'message ptr: ${status.ref.message.address}',
        );

        checkStatus(status);
        logDebug('Forward pass completed successfully!');

        // Convert outputs to TensorData
        final outputCount = outputCountPtr.value;
        logDebug('Output count: $outputCount');
        final outputs = <TensorData>[];
        final outputArray = outputsPtr.value;

        for (var i = 0; i < outputCount; i++) {
          // Extract tensor data directly without creating NativeTensor
          // (to avoid double-free from finalizer + et_tensor_array_free)
          final tensorPtr = outputArray[i];
          logDebug('Extracting output[$i] from ptr: ${tensorPtr.address}');
          final tensorData = _extractTensorData(tensorPtr);
          logDebug(
            'Output[$i]: shape=${tensorData.shape}, '
            'dtype=${tensorData.dataType}, '
            'dataSize=${tensorData.data.length} bytes',
          );
          outputs.add(tensorData);
        }

        // Free the output array and all tensors
        logDebug('Freeing output tensor array...');
        et_tensor_array_free(outputArray, outputCount);

        logDebug('forward() completed with ${outputs.length} outputs');
        return outputs;
      } finally {
        calloc
          ..free(inputPtrs)
          ..free(outputsPtr)
          ..free(outputCountPtr);
      }
    } finally {
      // Dispose input native tensors
      logDebug('Disposing input native tensors...');
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
      throw const ExecuTorchModelException('Model has been disposed');
    }
  }

  /// Extract tensor data directly from native pointer without ownership.
  ///
  /// This copies the data and does NOT take ownership of the pointer.
  /// The caller is responsible for freeing the native tensor.
  static TensorData _extractTensorData(ffi.Pointer<ETTensor> ptr) {
    // Get tensor properties
    final dtype = et_tensor_dtype(ptr);
    final rank = et_tensor_rank(ptr);
    final shapePtr = et_tensor_shape(ptr);
    final dataSize = et_tensor_data_size(ptr);
    final dataPtr = et_tensor_data(ptr);

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
      dataType: _etDTypeToTensorType(dtype),
      data: data,
    );
  }

  /// Convert ETDType to TensorType.
  static TensorType _etDTypeToTensorType(ETDType dtype) => switch (dtype) {
        ETDType.ET_DTYPE_FLOAT32 => TensorType.float32,
        ETDType.ET_DTYPE_FLOAT64 => TensorType.float32, // Fallback
        ETDType.ET_DTYPE_INT64 => TensorType.int32, // Fallback
        ETDType.ET_DTYPE_INT32 => TensorType.int32,
        ETDType.ET_DTYPE_INT16 => TensorType.int32, // Fallback
        ETDType.ET_DTYPE_INT8 => TensorType.int8,
        ETDType.ET_DTYPE_UINT8 => TensorType.uint8,
        ETDType.ET_DTYPE_BOOL => TensorType.uint8, // Fallback
      };
}
