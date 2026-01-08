/// Native platform implementation of ExecuTorchModel
///
/// This implementation uses Pigeon for platform communication with
/// Android, iOS, and macOS native ExecuTorch libraries.
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

import 'executorch_errors.dart';
import 'executorch_model.dart';
import 'generated/executorch_api.dart';

/// Native platform implementation of ExecuTorchModel
///
/// Uses Pigeon-generated host API for communication with native platforms
/// (Android, iOS, macOS).
class ExecuTorchModelNative implements ExecuTorchModel {
  ExecuTorchModelNative._({
    required this.modelId,
    required this.hostApi,
    this.tempFile,
  });

  @override
  final String modelId;

  /// Reference to the host API for platform communication
  final ExecutorchHostApi hostApi;

  /// Temporary file used for byte-based loading (if any)
  final File? tempFile;

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
  static Future<ExecuTorchModelNative> load(String filePath) async {
    final hostApi = ExecutorchHostApi();

    try {
      final loadResult = await hostApi.load(filePath);

      return ExecuTorchModelNative._(
        modelId: loadResult.modelId,
        hostApi: hostApi,
      );
    } catch (e) {
      throw ExecuTorchException(
        'Failed to load model from $filePath: $e',
      );
    }
  }

  /// Load an ExecuTorch model from bytes (works on all platforms)
  ///
  /// This method works on all platforms:
  /// - **Native (Android, iOS, macOS)**: Writes bytes to temp file, then loads
  /// - **Web**: Loads bytes directly into WebAssembly virtual filesystem
  ///
  /// ### Loading from Assets:
  /// ```dart
  /// import 'package:flutter/services.dart';
  ///
  /// final byteData = await rootBundle.load('assets/models/model.pte');
  /// final model = await ExecuTorchModel.loadFromBytes(
  ///   byteData.buffer.asUint8List(),
  /// );
  /// ```
  ///
  /// ### Parameters:
  /// - [modelBytes]: Model file bytes in .pte format
  ///
  /// ### Returns:
  /// A loaded model instance ready for inference
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if model format is invalid or loading fails
  static Future<ExecuTorchModelNative> loadFromBytes(
    Uint8List modelBytes,
  ) async {
    final hostApi = ExecutorchHostApi();

    try {
      // Write bytes to temp file, then load via file path
      // Note: The temp file is kept until dispose() is called because
      // native implementations memory-map the file
      final tempDir = await getTemporaryDirectory();
      final tempFile = File(
        '${tempDir.path}/model_${DateTime.now().millisecondsSinceEpoch}.pte',
      );
      await tempFile.writeAsBytes(modelBytes);

      final loadResult = await hostApi.load(tempFile.path);

      return ExecuTorchModelNative._(
        modelId: loadResult.modelId,
        hostApi: hostApi,
        tempFile: tempFile, // Keep temp file for memory mapping
      );
    } catch (e) {
      throw ExecuTorchException(
        'Failed to load model from bytes: $e',
      );
    }
  }

  /// Load an ExecuTorch model from Flutter asset bundle
  ///
  /// Works on all platforms including web.
  ///
  /// This is a convenience method that loads model bytes from the asset bundle
  /// and then calls [loadFromBytes].
  ///
  /// ### Example:
  /// ```dart
  /// // Load from assets (requires asset declaration in pubspec.yaml)
  /// final model = await ExecuTorchModel.loadFromAsset(
  ///   'assets/models/model.pte',
  /// );
  /// ```
  ///
  /// ### Parameters:
  /// - [assetPath]: Path to the model in the asset bundle
  ///
  /// ### Returns:
  /// A loaded model instance ready for inference
  ///
  /// ### Throws:
  /// - [ExecuTorchException] if asset is not found or loading fails
  static Future<ExecuTorchModelNative> loadFromAsset(
    String assetPath,
  ) async {
    try {
      final byteData = await rootBundle.load(assetPath);
      return loadFromBytes(byteData.buffer.asUint8List());
    } catch (e) {
      throw ExecuTorchException(
        'Failed to load model from asset $assetPath: $e',
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
  @override
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
  /// This also cleans up any temporary files created during byte-based loading.
  /// The user has full control over memory management.
  @override
  Future<void> dispose() async {
    if (_isDisposed) return;

    await hostApi.dispose(modelId);
    _isDisposed = true;

    // Clean up temporary file if it was created for byte-based loading
    if (tempFile != null) {
      try {
        await tempFile!.delete();
      } catch (_) {
        // Ignore deletion errors (file may already be deleted)
      }
    }
  }

  /// Check if this model has been disposed
  @override
  bool get isDisposed => _isDisposed;
}
