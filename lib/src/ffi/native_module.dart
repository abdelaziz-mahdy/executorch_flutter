// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native module wrapper for FFI layer.
library;

import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
import '../generated/executorch_ffi.g.dart';
import '../types.dart';
import 'native_logging.dart';
import 'native_status.dart';
import 'native_tensor.dart';

/// Native callback type for async functions that pass a void* result.
typedef ETNativeCallback
    = ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>;

/// Run a native FFI function asynchronously using NativeCallable.listener.
///
/// The native function spawns a thread, does work, and calls the callback
/// with a void* result pointer from that thread. NativeCallable.listener
/// marshals the callback onto the Dart event loop automatically.
///
/// [func] receives a native callback pointer and should call the async
/// C function (which returns immediately after spawning a thread).
///
/// [onComplete] receives the completer and the void* passed by C.
Future<T> etRunAsync<T>(
  void Function(ETNativeCallback callback) func,
  void Function(Completer<T> completer, ffi.Pointer<ffi.Void> result)
      onComplete,
) {
  final completer = Completer<T>();
  late final ffi.NativeCallable<ffi.Void Function(ffi.Pointer<ffi.Void>)>
      nativeCallable;
  void onResponse(ffi.Pointer<ffi.Void> result) {
    onComplete(completer, result);
    nativeCallable.close();
  }

  nativeCallable = ffi.NativeCallable.listener(onResponse);
  func(nativeCallable.nativeFunction);
  return completer.future;
}

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

        logDebug('et_module_load returned, status code: ${status.ref.code}');
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

  /// Load a model from file path asynchronously.
  ///
  /// Uses NativeCallable.listener + C-side threading to avoid blocking
  /// the UI thread during model loading.
  static Future<NativeModule> loadFileAsync(String path) async {
    logDebug('NativeModule.loadFileAsync() called with path: $path');

    final pathPtr = path.toNativeUtf8().cast<ffi.Char>();
    final outPtr = calloc<ffi.Pointer<ETModule>>();

    final module = await etRunAsync<NativeModule>(
      (callback) {
        et_module_load_file_async(pathPtr, outPtr, callback);
      },
      (completer, statusVoidPtr) {
        try {
          final statusPtr = statusVoidPtr.cast<ETStatus>();
          checkStatus(statusPtr);
          final module = NativeModule._(outPtr.value);
          logDebug(
            'Module loaded successfully (async file)! '
            'inputCount=${module.inputCount}, '
            'outputCount=${module.outputCount}',
          );
          completer.complete(module);
        } catch (e) {
          completer.completeError(e);
        } finally {
          calloc
            ..free(outPtr)
            ..free(pathPtr);
        }
      },
    );

    return module;
  }

  /// Load a model from memory buffer asynchronously.
  ///
  /// Uses NativeCallable.listener + C-side threading to avoid blocking
  /// the UI thread during model loading.
  static Future<NativeModule> loadAsync(Uint8List data) async {
    logDebug('NativeModule.loadAsync() called with ${data.length} bytes');

    final dataPtr = calloc<ffi.Uint8>(data.length);
    dataPtr.asTypedList(data.length).setAll(0, data);

    final outPtr = calloc<ffi.Pointer<ETModule>>();

    final module = await etRunAsync<NativeModule>(
      (callback) {
        // C side copies data internally, so dataPtr can be freed in callback
        et_module_load_async(dataPtr, data.length, outPtr, callback);
      },
      (completer, statusVoidPtr) {
        try {
          final statusPtr = statusVoidPtr.cast<ETStatus>();
          checkStatus(statusPtr);
          final module = NativeModule._(outPtr.value);
          logDebug(
            'Module loaded successfully (async)! '
            'inputCount=${module.inputCount}, '
            'outputCount=${module.outputCount}',
          );
          completer.complete(module);
        } catch (e) {
          completer.completeError(e);
        } finally {
          calloc
            ..free(outPtr)
            ..free(dataPtr);
        }
      },
    );

    return module;
  }

  /// The native module pointer.
  final ffi.Pointer<ETModule> _ptr;

  /// Whether this module has been disposed.
  bool _disposed = false;

  /// In-flight async forward operations.
  ///
  /// Awaited by [disposeAsync] so the native module is never freed while a
  /// forward pass is still running on a worker thread - doing so frees the
  /// model's weights mid-inference and crashes deep in the kernels (e.g.
  /// convolution_out) with a use-after-free.
  final List<Future<void>> _inFlight = [];

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

  /// Run forward pass asynchronously.
  ///
  /// Uses NativeCallable.listener + C-side threading to avoid blocking
  /// the UI thread during inference.
  Future<List<TensorData>> forwardAsync(List<TensorData> inputs) async {
    _checkDisposed();
    // Track this operation so disposeAsync() can wait for it to finish before
    // freeing the native module (prevents a use-after-free crash).
    final done = Completer<void>();
    _inFlight.add(done.future);
    try {
      return await _forwardAsyncImpl(inputs);
    } finally {
      done.complete();
      _inFlight.remove(done.future);
    }
  }

  Future<List<TensorData>> _forwardAsyncImpl(List<TensorData> inputs) async {
    logDebug('forwardAsync() called with ${inputs.length} inputs');

    final nativeInputs = <NativeTensor>[];
    for (var i = 0; i < inputs.length; i++) {
      try {
        nativeInputs.add(NativeTensor.fromTensorData(inputs[i]));
      } catch (e) {
        for (final tensor in nativeInputs) {
          tensor.dispose();
        }
        rethrow;
      }
    }

    final inputPtrs = calloc<ffi.Pointer<ETTensor>>(nativeInputs.length);
    for (var i = 0; i < nativeInputs.length; i++) {
      inputPtrs[i] = nativeInputs[i].ptr;
    }

    final outputsPtr = calloc<ffi.Pointer<ffi.Pointer<ETTensor>>>();
    final outputCountPtr = calloc<ffi.Int32>();

    final results = await etRunAsync<List<TensorData>>(
      (callback) {
        et_module_forward_async(
          _ptr,
          inputPtrs,
          nativeInputs.length,
          outputsPtr,
          outputCountPtr,
          callback,
        );
      },
      (completer, statusVoidPtr) {
        try {
          final statusPtr = statusVoidPtr.cast<ETStatus>();
          checkStatus(statusPtr);

          final outputCount = outputCountPtr.value;
          final outputs = <TensorData>[];
          final outputArray = outputsPtr.value;

          for (var i = 0; i < outputCount; i++) {
            outputs.add(_extractTensorData(outputArray[i]));
          }

          et_tensor_array_free(outputArray, outputCount);

          completer.complete(outputs);
        } catch (e) {
          completer.completeError(e);
        } finally {
          calloc
            ..free(inputPtrs)
            ..free(outputsPtr)
            ..free(outputCountPtr);
          for (final tensor in nativeInputs) {
            tensor.dispose();
          }
        }
      },
    );

    return results;
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

  /// Await any in-flight async operations, then dispose.
  ///
  /// Prefer this over [dispose] when async forward passes may still be running
  /// (e.g. tearing down a live-camera pipeline): it waits for them to complete
  /// so the native module is not freed mid-inference. Safe to call multiple
  /// times.
  Future<void> disposeAsync() async {
    if (_disposed) return;
    if (_inFlight.isNotEmpty) {
      // Ignore the results/errors of pending forwards; we only need them done.
      await Future.wait<void>(
        _inFlight.toList(),
      ).catchError((_) => const <void>[]);
    }
    dispose();
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
