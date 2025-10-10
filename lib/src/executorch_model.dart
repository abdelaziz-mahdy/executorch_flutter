/// ExecuTorch model wrapper class providing high-level model management
library;

import 'dart:async';

import 'executorch_errors.dart';
import 'generated/executorch_api.dart';

/// High-level wrapper for an ExecuTorch model instance
///
/// This class provides a convenient interface for loading, managing, and
/// running inference on ExecuTorch models. It matches the native ExecuTorch API
/// pattern used in Kotlin and Swift for consistency.
///
/// ## Usage Pattern
///
/// ### Loading from Assets (Recommended):
/// ```dart
/// import 'dart:io';
/// import 'package:flutter/services.dart';
/// import 'package:path_provider/path_provider.dart';
///
/// // Load from assets
/// final byteData = await rootBundle.load('assets/models/model.pte');
/// final tempDir = await getTemporaryDirectory();
/// final file = File('${tempDir.path}/model.pte');
/// await file.writeAsBytes(byteData.buffer.asUint8List());
/// final model = await ExecuTorchModel.load(file.path);
///
/// // Run inference (calls native module.forward())
/// final outputs = await model.forward(inputs);
///
/// // Clean up
/// await model.dispose();
/// ```
///
/// ### Loading from File System Directly:
/// ```dart
/// // Load from file path
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
///
/// // Run inference
/// final outputs = await model.forward(inputs);
///
/// // Clean up
/// await model.dispose();
/// ```
///
/// ## Native API Mapping
///
/// This Dart API directly maps to the native ExecuTorch APIs:
/// - **Kotlin (Android)**: `Module.load()` → `module.forward()`
/// - **Swift (iOS/macOS)**: `Module()` + `load("forward")` → `module.forward()`
///
/// Note: ExecuTorch doesn't provide runtime introspection for model metadata.
/// Input/output specs must be known externally (from model documentation).
class ExecuTorchModel {
  ExecuTorchModel._({
    required this.modelId,
    required this.hostApi,
  });

  /// Unique identifier for this model instance
  final String modelId;

  /// Reference to the host API for platform communication
  final ExecutorchHostApi hostApi;

  /// Whether this model has been disposed
  bool _isDisposed = false;

  /// Load an ExecuTorch model from a file path (static factory)
  ///
  /// This is the primary way to load models. It matches the native pattern:
  /// - **Android**: Calls `Module.load(filePath)`
  /// - **iOS/macOS**: Calls `Module(filePath)` + `module.load("forward")`
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
  /// ### Loading from File System:
  /// ```dart
  /// final model = await ExecuTorchModel.load('/path/to/model.pte');
  /// ```
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
  ///
  /// ### Platform Requirements:
  /// - **Android**: API 23+, ExecuTorch AAR 1.0.0-rc2
  /// - **iOS**: iOS 13.0+, arm64 only
  /// - **macOS**: macOS 11.0+, Apple Silicon (arm64) only
  static Future<ExecuTorchModel> load(String filePath) async {
    final hostApi = ExecutorchHostApi();

    try {
      final loadResult = await hostApi.load(filePath);

      return ExecuTorchModel._(
        modelId: loadResult.modelId,
        hostApi: hostApi,
      );
    } catch (e) {
      throw ExecuTorchException(
        'Failed to load model from $filePath: $e',
      );
    }
  }

  /// Execute inference on the model (matches native `module.forward()`)
  ///
  /// This is the primary inference method that directly maps to the native
  /// APIs:
  /// - **Android**: Calls `module.forward(inputEValues)`
  /// - **iOS/macOS**: Calls `module.forward(inputValues)`
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
  ///
  /// ### Performance Tips:
  /// - Pre-allocate and reuse input tensors when possible
  /// - Ensure input shapes match model expectations exactly
  /// - Call [dispose] when done to free native resources immediately
  Future<List<TensorData>> forward(List<TensorData> inputs) async {
    if (_isDisposed) {
      throw const ExecuTorchException(
          'Model has been disposed and cannot be used');
    }

    try {
      final outputs = await hostApi.forward(modelId, inputs);
      return outputs.whereType<TensorData>().toList();
    } catch (e) {
      throw ExecuTorchInferenceException(
        'Forward pass failed: $e',
        e.toString(),
      );
    }
  }

  /// Dispose this model and free its resources
  ///
  /// Call this when you're done with the model to free platform resources.
  /// The user has full control over memory management.
  Future<void> dispose() async {
    if (_isDisposed) return;

    await hostApi.dispose(modelId);
    _isDisposed = true;
  }

  /// Check if this model has been disposed
  bool get isDisposed => _isDisposed;
}
