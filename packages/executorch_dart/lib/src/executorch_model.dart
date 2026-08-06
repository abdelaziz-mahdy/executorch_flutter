/// ExecuTorch model interface and platform-specific implementations
library;

import 'dart:async';
import 'dart:typed_data';

import 'executorch_errors.dart';
import 'executorch_model_unsupported_stub.dart'
    if (dart.library.ffi) 'executorch_model_ffi_stub.dart' as stub;
import 'types.dart';

/// High-level wrapper for an ExecuTorch model instance
///
/// This is the main API for loading, managing, and running inference on
/// ExecuTorch models. It provides a consistent interface across all platforms.
///
/// Platform-specific implementations:
/// - **Native (Android, iOS, macOS, Windows, Linux)**: Uses dart:ffi with
///   native assets
/// - **Web**: Uses WebAssembly via JavaScript interop
///
/// ## Usage Pattern
///
/// ### Loading from a file (native platforms):
/// ```dart
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
/// final outputs = await model.forward(inputs);
/// await model.dispose();
/// ```
///
/// Flutter applications can load from the asset bundle with
/// `loadModelFromAsset` from
/// `package:executorch_flutter/executorch_flutter.dart`.
///
/// ### Loading from Bytes (works on all platforms):
/// ```dart
/// final bytes = await File('/path/to/model.pte').readAsBytes();
/// final model = await ExecuTorchModel.loadFromBytes(bytes);
/// final outputs = await model.forward(inputs);
/// await model.dispose();
/// ```
abstract class ExecuTorchModel {
  /// Unique identifier for this model instance
  String get modelId;

  /// Whether this model has been disposed
  bool get isDisposed;

  /// Load an ExecuTorch model from a file path (static factory)
  ///
  /// **Note**: Only available on native platforms (Android, iOS, macOS).
  /// On web, use [loadFromBytes] instead.
  ///
  /// ### Parameters:
  /// - [filePath]: Absolute path to a valid ExecuTorch `.pte` model file
  ///
  /// ### Returns:
  /// A loaded model instance ready for inference
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if the file doesn't exist, is not readable,
  ///   or the model format is invalid
  /// - [UnsupportedError] on web platform
  static Future<ExecuTorchModel> load(String filePath) => stub.load(filePath);

  /// Load an ExecuTorch model from bytes (works on all platforms)
  ///
  /// This method works on all platforms:
  /// - **Native (Android, iOS, macOS)**: Writes bytes to temp file, then loads
  /// - **Web**: Loads bytes directly into WebAssembly virtual filesystem
  ///
  /// ### Parameters:
  /// - [modelBytes]: Model file bytes in .pte format
  ///
  /// ### Returns:
  /// A loaded model instance ready for inference
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if model format is invalid or loading fails
  static Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes) =>
      stub.loadFromBytes(modelBytes);

  /// Execute inference on the model
  ///
  /// This is the primary inference method that runs the forward pass.
  ///
  /// ### Parameters:
  /// - [inputs]: List of input tensors matching the model's input specification
  ///
  /// ### Returns:
  /// List of output tensors from the model
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if the model has been disposed
  /// - [ExecuTorchInferenceException] if inference fails
  Future<List<TensorData>> forward(List<TensorData> inputs);

  /// Dispose this model and free its resources
  ///
  /// Call this when you're done with the model to free platform resources.
  Future<void> dispose();
}
