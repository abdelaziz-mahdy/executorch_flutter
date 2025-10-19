/// ExecuTorch model wrapper class providing high-level model management
library;

import 'dart:async';
import 'dart:ffi' as ffi;

import 'executorch_errors.dart';
import 'generated/executorch_api.dart'; // TensorData from Pigeon
import 'ffi/executorch_ffi_bridge.dart';

/// High-level wrapper for an ExecuTorch model instance
///
/// This class provides a convenient interface for loading, managing, and
/// running inference on ExecuTorch models using Dart FFI for direct C interop.
///
/// ## Usage Pattern
///
/// ### Loading from File Path:
/// ```dart
/// // Load from file path
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
///
/// // Run inference
/// final outputs = await model.forward(inputs);
///
/// // Clean up (automatic via finalizer, but can call explicitly)
/// await model.dispose();
/// ```
///
/// ### Loading from Assets (Extract to temp file first):
/// ```dart
/// import 'dart:io';
/// import 'package:flutter/services.dart';
/// import 'package:path_provider/path_provider.dart';
///
/// // Extract asset to temporary file
/// final byteData = await rootBundle.load('assets/models/model.pte');
/// final tempDir = await getTemporaryDirectory();
/// final file = File('${tempDir.path}/model.pte');
/// await file.writeAsBytes(byteData.buffer.asUint8List());
///
/// // Load the model
/// final model = await ExecuTorchModel.load(file.path);
/// ```
///
/// ## Memory Management
///
/// Models are automatically disposed via [NativeFinalizer] when garbage collected,
/// but you can (and should) call [dispose] explicitly when done for immediate cleanup.
///
/// ## Native API Mapping
///
/// This Dart API uses FFI to directly call the C wrapper which integrates with:
/// - **C Wrapper**: `et_flutter_load_model()` → `et_flutter_forward()` → `et_flutter_dispose_model()`
/// - **ExecuTorch C++**: `Module::load()` → `module->forward()` → destructor
///
/// Note: ExecuTorch doesn't provide runtime introspection for model metadata.
/// Input/output specs must be known externally (from model documentation).
class ExecuTorchModel implements ffi.Finalizable {
  ExecuTorchModel._({
    required this.modelHandle,
    required this.filePath,
    required ffi.Pointer<ffi.Void> handle,
    required ffi.Pointer<ffi.NativeFinalizerFunction> finalizerToken,
  }) : _modelHandle = handle {
    // Initialize finalizer on first use
    _finalizer ??= ffi.NativeFinalizer(finalizerToken);

    // Register finalizer for automatic cleanup on garbage collection
    _finalizer!.attach(this, handle, detach: this);
  }

  /// Model handle (opaque pointer to C model data)
  final String modelHandle;

  /// File path of the loaded model
  final String filePath;

  /// Native model handle (C pointer)
  final ffi.Pointer<ffi.Void> _modelHandle;

  /// Whether this model has been disposed
  bool _isDisposed = false;

  /// Finalizer for automatic native resource cleanup
  /// This will be initialized lazily when first model is loaded
  static ffi.NativeFinalizer? _finalizer;

  /// Load an ExecuTorch model from a file path (static factory)
  ///
  /// This loads the model using FFI and returns a ready-to-use model instance.
  ///
  /// ### Loading from File System:
  /// ```dart
  /// final model = await ExecuTorchModel.load('/path/to/model.pte');
  /// ```
  ///
  /// ### Loading from Assets:
  /// ```dart
  /// import 'dart:io';
  /// import 'package:flutter/services.dart';
  /// import 'package:path_provider/path_provider.dart';
  ///
  /// // Extract asset to temporary file
  /// final byteData = await rootBundle.load('assets/models/model.pte');
  /// final tempDir = await getTemporaryDirectory();
  /// final file = File('${tempDir.path}/model.pte');
  /// await file.writeAsBytes(byteData.buffer.asUint8List());
  ///
  /// // Load the model
  /// final model = await ExecuTorchModel.load(file.path);
  /// ```
  ///
  /// ### Parameters:
  /// - [filePath]: Absolute path to a valid ExecuTorch `.pte` model file
  ///
  /// ### Returns:
  /// A loaded model instance ready for inference
  ///
  /// ### Throws:
  /// - [ExecuTorchModelException] if the file doesn't exist, is not readable,
  ///   or the model format is invalid
  /// - [ExecuTorchIOException] if file I/O fails
  ///
  /// ### Platform Requirements:
  /// - **Android**: API 23+, ExecuTorch AAR 1.0.0-rc2, arm64-v8a or x86_64
  /// - **iOS**: iOS 13.0+, arm64 only (no simulator)
  /// - **macOS**: macOS 11.0+, Apple Silicon (arm64) only
  static Future<ExecuTorchModel> load(String filePath) async {
    // Ensure FFI bridge is initialized
    execuTorchFfiBridge.initialize();

    // Load model via FFI
    final handle = await execuTorchFfiBridge.loadModel(filePath);

    // Create model instance with finalizer
    return ExecuTorchModel._(
      modelHandle: handle.address.toString(),
      filePath: filePath,
      handle: handle,
      finalizerToken: execuTorchFfiBridge.finalizerToken,
    );
  }

  /// Execute inference on the model
  ///
  /// This runs the model's forward pass using FFI to call the C wrapper.
  ///
  /// ### Example:
  /// ```dart
  /// // Prepare input tensor
  /// final input = TensorData(
  ///   shape: [1, 3, 224, 224],
  ///   dataType: TensorType.float32,
  ///   data: imageBytes,
  /// );
  ///
  /// // Run forward pass
  /// final outputs = await model.forward([input]);
  ///
  /// // Process output tensors
  /// for (var output in outputs) {
  ///   print('Shape: ${output.shape}, Type: ${output.dataType}');
  /// }
  /// ```
  ///
  /// ### Parameters:
  /// - [inputs]: List of input tensors matching the model's input specification
  ///
  /// ### Returns:
  /// List of output tensors from the model
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if the model has been disposed
  /// - [ExecuTorchInferenceException] if inference fails (invalid inputs,
  ///   runtime error, etc.)
  /// - [ExecuTorchValidationException] if tensor shapes or types are invalid
  ///
  /// ### Performance Tips:
  /// - Pre-allocate and reuse input tensors when possible
  /// - Ensure input shapes match model expectations exactly
  /// - Call [dispose] when done to free native resources immediately
  Future<List<TensorData>> forward(List<TensorData> inputs) async {
    if (_isDisposed) {
      throw const ExecuTorchException(
        'Model has been disposed and cannot be used',
      );
    }

    if (inputs.isEmpty) {
      throw const ExecuTorchValidationException(
        'Input list cannot be empty',
      );
    }

    // Run inference via FFI bridge
    return execuTorchFfiBridge.forward(_modelHandle, inputs);
  }

  /// Dispose this model and free its resources
  ///
  /// Call this when you're done with the model to free native resources immediately.
  /// If not called, the model will be automatically disposed when garbage collected
  /// (via [NativeFinalizer]), but explicit disposal is recommended for predictable
  /// memory management.
  ///
  /// ### Example:
  /// ```dart
  /// final model = await ExecuTorchModel.load('/path/to/model.pte');
  /// try {
  ///   final outputs = await model.forward(inputs);
  ///   // Use outputs...
  /// } finally {
  ///   await model.dispose(); // Always dispose
  /// }
  /// ```
  ///
  /// Safe to call multiple times (second call is no-op).
  Future<void> dispose() async {
    if (_isDisposed) return;

    // Detach finalizer (we're manually disposing)
    _finalizer?.detach(this);

    // Dispose via FFI
    await execuTorchFfiBridge.disposeModel(_modelHandle);

    _isDisposed = true;
  }

  /// Check if this model has been disposed
  bool get isDisposed => _isDisposed;
}
