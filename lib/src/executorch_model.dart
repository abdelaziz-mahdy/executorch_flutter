/// ExecuTorch model interface and platform-specific implementations
library;

import 'dart:async';
import 'dart:typed_data';

import 'executorch_model_native_stub.dart' as stub
    if (dart.library.js_interop) 'executorch_model_web_stub.dart';
import 'generated/executorch_api.dart';

/// High-level wrapper for an ExecuTorch model instance
///
/// This is the main API for loading, managing, and running inference on
/// ExecuTorch models. It provides a consistent interface across all platforms.
///
/// Platform-specific implementations:
/// - **Native (Android, iOS, macOS)**: Uses Pigeon for platform communication
/// - **Web**: Uses WebAssembly via JavaScript interop
///
/// ## Usage Pattern
///
/// ### Loading from Assets (Recommended - works on all platforms):
/// ```dart
/// final model = await ExecuTorchModel.loadFromAsset('assets/models/model.pte');
/// final outputs = await model.forward(inputs);
/// await model.dispose();
/// ```
///
/// ### Loading from Bytes (works on all platforms):
/// ```dart
/// import 'package:flutter/services.dart';
///
/// final byteData = await rootBundle.load('assets/models/model.pte');
/// final model = await ExecuTorchModel.loadFromBytes(
///   byteData.buffer.asUint8List(),
/// );
/// final outputs = await model.forward(inputs);
/// await model.dispose();
/// ```
///
/// ### Loading from File System (native platforms only):
/// ```dart
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
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
  /// On web, use [loadFromAsset] or [loadFromBytes] instead.
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

  /// Load an ExecuTorch model from Flutter asset bundle (works on all platforms)
  ///
  /// This is a convenience method that loads model bytes from the asset bundle
  /// and then calls [loadFromBytes]. Works on all platforms including web.
  ///
  /// ### Parameters:
  /// - [assetPath]: Path to the model in the asset bundle
  ///
  /// ### Returns:
  /// A loaded model instance ready for inference
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if asset is not found or loading fails
  static Future<ExecuTorchModel> loadFromAsset(String assetPath) =>
      stub.loadFromAsset(assetPath);

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
