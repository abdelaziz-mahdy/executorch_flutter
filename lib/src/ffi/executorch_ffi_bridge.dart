/**
 * @file executorch_ffi_bridge.dart
 * @brief FFI bridge providing Dart interface to C wrapper
 *
 * This module provides a high-level Dart API that wraps the C FFI bindings,
 * handling memory management, error conversion, and async execution via isolates.
 */

import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'dart:isolate';
import 'package:ffi/ffi.dart';

import '../generated/executorch_api.dart'; // Pigeon TensorData
import '../generated/executorch_ffi_bindings.dart';
import 'error_conversion.dart';
import 'tensor_conversion.dart';

/// FFI bridge for ExecuTorch C wrapper
///
/// This class provides the main Dart interface to the C wrapper, handling:
/// - Dynamic library loading
/// - Model loading and disposal
/// - Forward pass execution
/// - Memory management
/// - Error handling
class ExecuTorchFfiBridge {
  /// FFI bindings to C functions
  late final ExecutorchFfiBindings _bindings;

  /// Dynamic library handle
  late final ffi.DynamicLibrary _dylib;

  /// Whether the bridge has been initialized
  bool _initialized = false;

  /// Native finalizer token for automatic model cleanup
  late final ffi.Pointer<ffi.NativeFinalizerFunction> _finalizerToken;

  /// Initialize the FFI bridge
  ///
  /// Loads the native library and sets up bindings.
  /// Must be called before any other operations.
  ///
  /// Throws [UnsupportedError] if platform is not supported.
  void initialize() {
    if (_initialized) {
      return; // Already initialized
    }

    // Load platform-specific dynamic library
    _dylib = _loadLibrary();

    // Create FFI bindings
    _bindings = ExecutorchFfiBindings(_dylib);

    // Get finalizer function pointer for NativeFinalizer
    _finalizerToken = _dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
      'et_flutter_dispose_model',
    )
        .cast<ffi.NativeFinalizerFunction>();

    _initialized = true;
  }

  /// Get the native finalizer token for model cleanup
  ffi.Pointer<ffi.NativeFinalizerFunction> get finalizerToken {
    _ensureInitialized();
    return _finalizerToken;
  }

  /// Loads ExecuTorch model from file path
  ///
  /// Calls C et_flutter_load_model() and returns model handle.
  ///
  /// Throws appropriate exception if loading fails.
  /// Returns opaque pointer to model (to be used with [forward] and [dispose]).
  Future<ffi.Pointer<ffi.Void>> loadModel(String filePath) async {
    _ensureInitialized();

    // Run in isolate to avoid blocking UI thread
    return Isolate.run(() {
      // Convert Dart string to C string
      final filePathPtr = filePath.toNativeUtf8();

      try {
        // Call C function
        final result = _bindings.et_flutter_load_model(
          filePathPtr.cast<ffi.Char>(),
        );

        // Check for errors
        throwIfError(result.error);

        // Return model handle
        return result.model_handle;
      } finally {
        // Free C string
        malloc.free(filePathPtr);
      }
    });
  }

  /// Runs forward pass on loaded model
  ///
  /// Calls C et_flutter_forward() with input tensors and returns output tensors.
  ///
  /// [modelHandle] - Opaque pointer from [loadModel]
  /// [inputs] - List of input tensors
  ///
  /// Returns list of output tensors.
  /// Throws appropriate exception if inference fails.
  Future<List<TensorData>> forward(
    ffi.Pointer<ffi.Void> modelHandle,
    List<TensorData> inputs,
  ) async {
    _ensureInitialized();

    // Run in isolate to avoid blocking UI thread
    return Isolate.run(() {
      // Convert Dart inputs to C
      final (inputArrayPtr, inputCount) = toCTensorArray(inputs);

      // Allocate C input struct
      final cInput = calloc<ETFlutterForwardInput>();
      cInput.ref.num_inputs = inputCount;
      for (int i = 0; i < inputCount; i++) {
        cInput.ref.inputs[i] = inputArrayPtr[i];
      }

      try {
        // Call C function
        final output = _bindings.et_flutter_forward(
          modelHandle,
          cInput,
        );

        // Check for errors
        throwIfError(output.error);

        // Convert outputs to Dart (copies data from C to Dart)
        final dartOutputs = <TensorData>[];
        for (int i = 0; i < output.num_outputs; i++) {
          final cTensor = output.outputs[i];
          if (cTensor.address != 0) {
            dartOutputs.add(fromCTensor(cTensor));
          }
        }

        // Free C output memory (now that we've copied to Dart)
        final outputPtr = calloc<ETFlutterForwardOutput>();
        outputPtr.ref = output;
        _bindings.et_flutter_free_forward_output(outputPtr);
        calloc.free(outputPtr);

        return dartOutputs;
      } finally {
        // Free C input memory
        freeTensorArray(inputArrayPtr, inputCount);
        calloc.free(cInput);
      }
    });
  }

  /// Disposes a loaded model
  ///
  /// Calls C et_flutter_dispose_model() to free model resources.
  ///
  /// [modelHandle] - Opaque pointer from [loadModel]
  ///
  /// Safe to call multiple times (second call is no-op).
  Future<void> disposeModel(ffi.Pointer<ffi.Void> modelHandle) async {
    _ensureInitialized();

    if (modelHandle.address == 0) {
      return; // Already disposed or null
    }

    // Run in isolate to avoid blocking UI thread
    await Isolate.run(() {
      _bindings.et_flutter_dispose_model(modelHandle);
    });
  }

  /// Loads platform-specific dynamic library
  ffi.DynamicLibrary _loadLibrary() {
    if (Platform.isAndroid) {
      return ffi.DynamicLibrary.open('libexecutorch_flutter.so');
    } else if (Platform.isIOS || Platform.isMacOS) {
      // Try loading from the framework first
      try {
        return ffi.DynamicLibrary.open('executorch_flutter.framework/executorch_flutter');
      } catch (e) {
        // Fallback to process (for development/testing)
        return ffi.DynamicLibrary.process();
      }
    } else {
      throw UnsupportedError(
        'Platform ${Platform.operatingSystem} is not supported',
      );
    }
  }

  /// Ensures bridge is initialized
  void _ensureInitialized() {
    if (!_initialized) {
      throw StateError(
        'ExecuTorchFfiBridge not initialized. Call initialize() first.',
      );
    }
  }
}

/// Global singleton instance (lazy-initialized)
final execuTorchFfiBridge = ExecuTorchFfiBridge();
