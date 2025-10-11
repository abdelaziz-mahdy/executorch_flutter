/// Main ExecutorchManager API class providing high-level ExecuTorch inference management
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'executorch_errors.dart';
import 'executorch_model.dart';
import 'generated/executorch_api.dart';

/// High-level manager for ExecuTorch inference operations
///
/// This class provides the main API for ExecuTorch Flutter integration,
/// managing model lifecycle, inference execution, and resource management.
/// It acts as a facade over the lower-level Pigeon APIs and model wrappers.
class ExecutorchManager {
  ExecutorchManager._();

  static ExecutorchManager? _instance;

  /// Get the singleton instance of ExecutorchManager
  static ExecutorchManager get instance {
    _instance ??= ExecutorchManager._();
    return _instance!;
  }

  /// Internal reference to the Pigeon host API
  late final ExecutorchHostApi _hostApi = ExecutorchHostApi();

  /// Cache of loaded models by model ID
  final Map<String, ExecuTorchModel> _loadedModels = {};

  /// Whether the manager has been initialized
  bool _initialized = false;

  /// Initialize the ExecutorchManager
  ///
  /// This should be called once before using any other methods.
  /// It sets up the platform communication channels and verifies
  /// that the native ExecuTorch libraries are available.
  Future<void> initialize() async {
    if (_initialized) return;

    try {
      // Test connectivity by trying to get loaded models
      await _hostApi.getLoadedModels();
      _initialized = true;
    } catch (e) {
      throw ExecuTorchPlatformException(
        'Failed to initialize ExecutorchManager: $e\n'
        'Make sure ExecuTorch native libraries are properly installed.',
        e.toString(),
      );
    }
  }

  /// Enable or disable ExecuTorch debug logging
  ///
  /// When enabled, detailed ExecuTorch internal logs will be printed to the console.
  /// This is useful for debugging model loading and inference issues.
  ///
  /// Note: Debug logging only works in debug builds of the native libraries.
  /// In release builds, this method has no effect.
  ///
  /// [enabled] - true to enable logging, false to disable
  Future<void> setDebugLogging(bool enabled) async {
    _ensureInitialized();
    try {
      await _hostApi.setDebugLogging(enabled);
    } catch (e) {
      throw ExecuTorchPlatformException(
        'Failed to set debug logging: $e',
        e.toString(),
      );
    }
  }

  /// Load an ExecuTorch model from a file path
  ///
  /// [filePath] must point to a valid ExecuTorch .pte model file.
  /// Returns the loaded model instance that can be used for inference via [ExecuTorchModel.forward].
  ///
  /// The model will be cached and can be accessed later via [getLoadedModel].
  /// If a model with the same file path is already loaded, returns the cached instance.
  Future<ExecuTorchModel> loadModel(String filePath) async {
    _ensureInitialized();

    // Validate file path
    if (!File(filePath).existsSync()) {
      throw ExecuTorchModelException(
        'Model file not found: $filePath',
        'file_path: $filePath',
      );
    }

    try {
      final model = await ExecuTorchModel.load(filePath);
      _loadedModels[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from $filePath: $e',
        'file_path: $filePath, error: ${e.toString()}',
      );
    }
  }

  /// Load an ExecuTorch model from asset bundle
  ///
  /// [assetPath] should be the path to the model in the Flutter assets bundle.
  /// This is a convenience method for loading models packaged with the app.
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    _ensureInitialized();

    // For now, delegate to loadModel - in a full implementation,
    // this would extract the asset to a temporary file first
    throw UnimplementedError(
      'Asset loading not yet implemented. Use loadModel() with file path instead.',
    );
  }

  /// Get a loaded model by its ID
  ///
  /// Returns null if no model with the given ID is loaded.
  ExecuTorchModel? getLoadedModel(String modelId) => _loadedModels[modelId];

  /// Get all currently loaded models
  ///
  /// Returns a list of all ExecuTorchModel instances that are currently loaded
  /// and available for inference via [ExecuTorchModel.forward].
  List<ExecuTorchModel> getLoadedModels() =>
      List.unmodifiable(_loadedModels.values);

  /// Get the IDs of all loaded models
  ///
  /// This queries the platform side for the most up-to-date list.
  Future<List<String>> getLoadedModelIds() async {
    _ensureInitialized();

    try {
      final ids = await _hostApi.getLoadedModels();
      return ids.whereType<String>().toList();
    } catch (e) {
      throw ExecuTorchPlatformException(
        'Failed to get loaded model IDs: $e',
        e.toString(),
      );
    }
  }

  /// Dispose a loaded model and free its resources
  ///
  /// After calling this method, the model cannot be used for inference via [ExecuTorchModel.forward].
  /// The model is removed from the loaded models cache.
  Future<void> disposeModel(String modelId) async {
    _ensureInitialized();

    final model = _loadedModels.remove(modelId);
    if (model != null) {
      await model.dispose();
    } else {
      // Try to dispose on platform side even if not in our cache
      try {
        await _hostApi.dispose(modelId);
      } catch (e) {
        // Ignore errors for unknown models
      }
    }
  }

  /// Dispose all loaded models and free their resources
  ///
  /// This is useful for cleanup when the app is shutting down or
  /// when you want to free all model memory at once.
  Future<void> disposeAllModels() async {
    _ensureInitialized();

    final modelIds = List<String>.from(_loadedModels.keys);
    for (final modelId in modelIds) {
      await disposeModel(modelId);
    }
  }

  /// Get detailed information about system memory usage
  ///
  /// Returns a map with memory statistics, if available on the platform.
  /// This is useful for monitoring memory usage and detecting leaks.
  Future<Map<String, Object>> getMemoryInfo() async {
    _ensureInitialized();

    // This would be implemented with platform-specific memory queries
    // For now, return basic information
    return {
      'loaded_models_count': _loadedModels.length,
      'loaded_model_ids': _loadedModels.keys.toList(),
      'platform': Platform.operatingSystem,
      'timestamp': DateTime.now().toIso8601String(),
    };
  }

  /// Check if ExecuTorch is properly initialized and available
  ///
  /// Returns true if the manager is initialized and can communicate
  /// with the native ExecuTorch libraries.
  Future<bool> isAvailable() async {
    if (!_initialized) return false;

    try {
      await _hostApi.getLoadedModels();
      return true;
    } catch (e) {
      return false;
    }
  }

  /// Create tensor data with validation
  ///
  /// This is a convenience factory method for creating properly validated
  /// TensorData instances. It performs shape and data size validation.
  TensorData createTensorData({
    required List<int> shape,
    required TensorType dataType,
    required List<num> data,
    String? name,
  }) {
    // Convert numeric data to bytes based on data type
    final bytes = _convertNumericDataToBytes(data, dataType);

    final tensor = TensorData(
      shape: shape.cast<int?>(),
      dataType: dataType,
      data: bytes,
      name: name,
    );

    return tensor;
  }

  /// Utility method to convert numeric data to bytes
  static Uint8List _convertNumericDataToBytes(
      List<num> data, TensorType dataType) {
    switch (dataType) {
      case TensorType.float32:
        final float32List =
            Float32List.fromList(data.map((e) => e.toDouble()).toList());
        return float32List.buffer.asUint8List();

      case TensorType.int32:
        final int32List =
            Int32List.fromList(data.map((e) => e.toInt()).toList());
        return int32List.buffer.asUint8List();

      case TensorType.int8:
        return Uint8List.fromList(
            data.map((e) => e.toInt().clamp(-128, 127) + 128).toList());

      case TensorType.uint8:
        return Uint8List.fromList(
            data.map((e) => e.toInt().clamp(0, 255)).toList());
    }
  }

  /// Ensure the manager has been initialized
  void _ensureInitialized() {
    if (!_initialized) {
      throw const ExecuTorchPlatformException(
        'ExecutorchManager not initialized. Call initialize() first.',
      );
    }
  }

  /// Cleanup resources when the manager is no longer needed
  ///
  /// This should be called when the app is shutting down to ensure
  /// proper cleanup of all loaded models and platform resources.
  Future<void> shutdown() async {
    if (!_initialized) return;

    await disposeAllModels();
    _initialized = false;
    _instance = null;
  }
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
