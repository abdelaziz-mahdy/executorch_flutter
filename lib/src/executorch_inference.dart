/// Main ExecutorchManager API class providing high-level inference management
library;

import 'dart:async';
import 'dart:typed_data';

import 'executorch_manager_unsupported_stub.dart'
    if (dart.library.io) 'executorch_manager_native_stub.dart'
    if (dart.library.js_interop) 'executorch_manager_web_stub.dart'
    if (dart.library.js) 'executorch_manager_web_stub.dart' as stub;
import 'executorch_model.dart';
import 'types.dart';

/// High-level manager for ExecuTorch inference operations
///
/// This class provides the main API for ExecuTorch Flutter integration,
/// managing model lifecycle, inference execution, and resource management.
/// It acts as a facade over the lower-level platform-specific implementations.
///
/// Platform-specific implementations:
/// - **Native (Android, iOS, macOS)**: Uses Pigeon for platform communication
/// - **Web**: Uses WebAssembly via JavaScript interop
abstract class ExecutorchManager {
  /// Get the singleton instance of ExecutorchManager
  ///
  /// The actual implementation is platform-specific:
  /// - Native platforms use ExecutorchManagerNative
  /// - Web platform uses ExecutorchManagerWeb
  static ExecutorchManager get instance => stub.instance;

  /// Initialize the ExecutorchManager
  ///
  /// This should be called once before using any other methods.
  /// It sets up the platform communication channels and verifies
  /// that the ExecuTorch libraries are available.
  Future<void> initialize();

  /// Enable or disable ExecuTorch debug logging
  ///
  /// When enabled, detailed ExecuTorch internal logs will be printed.
  /// This is useful for debugging model loading and inference issues.
  ///
  /// Note: Debug logging only works in debug builds of the native libraries.
  /// In release builds, this method has no effect.
  /// On web platform, logs will be printed to the browser console.
  ///
  /// [enabled] - true to enable logging, false to disable
  Future<void> setDebugLogging(bool enabled);

  /// Load an ExecuTorch model from a file path
  ///
  /// [filePath] must point to a valid ExecuTorch .pte model file.
  /// Returns the loaded model instance that can be used for inference.
  ///
  /// The model will be cached and accessed later via [getLoadedModel].
  ///
  /// Note: On web platform, use [loadModelFromAssets] or
  /// [loadModelFromBytes] instead.
  Future<ExecuTorchModel> loadModel(String filePath);

  /// Load an ExecuTorch model from asset bundle
  ///
  /// [assetPath] should be the path to the model in the Flutter assets bundle.
  /// This is a convenience method for loading models packaged with the app.
  /// Works on all platforms including web.
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath);

  /// Load an ExecuTorch model from bytes
  ///
  /// [modelBytes] should contain the raw .pte model data.
  /// This is useful for loading models downloaded from network or
  /// generated dynamically. Works on all platforms including web.
  Future<ExecuTorchModel> loadModelFromBytes(Uint8List modelBytes);

  /// Get a loaded model by its ID
  ///
  /// Returns null if no model with the given ID is loaded.
  ExecuTorchModel? getLoadedModel(String modelId);

  /// Get all currently loaded models
  ///
  /// Returns a list of all ExecuTorchModel instances that are currently loaded
  /// and available for inference via [ExecuTorchModel.forward].
  List<ExecuTorchModel> getLoadedModels();

  /// Get the IDs of all loaded models
  ///
  /// This queries the platform side for the most up-to-date list on
  /// native platforms. On web, returns the cached list of model IDs.
  Future<List<String>> getLoadedModelIds();

  /// Dispose a loaded model and free its resources
  ///
  /// After calling this method, the model cannot be used for inference.
  /// The model is removed from the loaded models cache.
  Future<void> disposeModel(String modelId);

  /// Dispose all loaded models and free their resources
  ///
  /// This is useful for cleanup when the app is shutting down or
  /// when you want to free all model memory at once.
  Future<void> disposeAllModels();

  /// Get detailed information about system memory usage
  ///
  /// Returns a map with memory statistics, if available on the platform.
  /// This is useful for monitoring memory usage and detecting leaks.
  Future<Map<String, Object>> getMemoryInfo();

  /// Check if ExecuTorch is properly initialized and available
  ///
  /// Returns true if the manager is initialized and can communicate
  /// with the native ExecuTorch libraries or web Wasm module.
  Future<bool> isAvailable();

  /// Create tensor data with validation
  ///
  /// This is a convenience factory method for creating properly validated
  /// TensorData instances. It performs shape and data size validation.
  TensorData createTensorData({
    required List<int> shape,
    required TensorType dataType,
    required List<num> data,
    String? name,
  });

  /// Cleanup resources when the manager is no longer needed
  ///
  /// This should be called when the app is shutting down to ensure
  /// proper cleanup of all loaded models and platform resources.
  Future<void> shutdown();
}

/// Utility class for working with ExecuTorch tensors
class TensorUtils {
  TensorUtils._();

  /// Create a float32 tensor from a 2D list (commonly used for images)
  static TensorData createFloat32Tensor2D({
    required List<List<double>> data,
    String? name,
  }) {
    final height = data.length;
    final width = height > 0 ? data[0].length : 0;
    final flatData = data.expand((row) => row).toList();

    return ExecutorchManager.instance.createTensorData(
      shape: [height, width],
      dataType: TensorType.float32,
      data: flatData,
      name: name,
    );
  }

  /// Create a float32 tensor from a 3D list (commonly used for RGB images)
  static TensorData createFloat32Tensor3D({
    required List<List<List<double>>> data,
    String? name,
  }) {
    final depth = data.length;
    final height = depth > 0 ? data[0].length : 0;
    final width = height > 0 ? data[0][0].length : 0;
    final flatData =
        data.expand((plane) => plane.expand((row) => row)).toList();

    return ExecutorchManager.instance.createTensorData(
      shape: [depth, height, width],
      dataType: TensorType.float32,
      data: flatData,
      name: name,
    );
  }

  /// Create a float32 tensor from a 4D list (commonly used for batched images)
  static TensorData createFloat32Tensor4D({
    required List<List<List<List<double>>>> data,
    String? name,
  }) {
    final batch = data.length;
    final depth = batch > 0 ? data[0].length : 0;
    final height = depth > 0 ? data[0][0].length : 0;
    final width = height > 0 ? data[0][0][0].length : 0;
    final flatData = data
        .expand((batchItem) =>
            batchItem.expand((plane) => plane.expand((row) => row)))
        .toList();

    return ExecutorchManager.instance.createTensorData(
      shape: [batch, depth, height, width],
      dataType: TensorType.float32,
      data: flatData,
      name: name,
    );
  }

  /// Extract numeric data from a tensor
  static List<double> extractFloat32Data(TensorData tensor) {
    if (tensor.dataType != TensorType.float32) {
      throw ArgumentError('Tensor is not float32 type');
    }

    final float32List = Float32List.view(tensor.data.buffer);
    return float32List.toList();
  }

  /// Extract integer data from a tensor
  static List<int> extractInt32Data(TensorData tensor) {
    if (tensor.dataType != TensorType.int32) {
      throw ArgumentError('Tensor is not int32 type');
    }

    final int32List = Int32List.view(tensor.data.buffer);
    return int32List.toList();
  }

  /// Calculate the total number of elements in a tensor shape
  static int calculateElementCount(List<int> shape) =>
      shape.fold(1, (total, dim) => total * dim.abs());

  /// Format tensor shape as a human-readable string
  static String formatShape(List<int> shape) => '[${shape.join(', ')}]';
}
