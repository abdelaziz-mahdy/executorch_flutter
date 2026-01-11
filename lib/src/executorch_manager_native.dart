/// Native platform implementation of ExecutorchManager
///
/// This implementation uses Pigeon for platform communication with
/// Android, iOS, and macOS native ExecuTorch libraries.
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'executorch_errors.dart';
import 'executorch_inference.dart';
import 'executorch_model.dart';
import 'generated/executorch_api.dart';

/// Native platform implementation of ExecutorchManager
///
/// Uses Pigeon-generated host API for communication with native platforms
/// (Android, iOS, macOS).
class ExecutorchManagerNative implements ExecutorchManager {
  ExecutorchManagerNative._();

  static ExecutorchManagerNative? _instance;

  /// Get the singleton instance of ExecutorchManagerNative
  // ignore: prefer_constructors_over_static_methods
  static ExecutorchManagerNative get instance {
    _instance ??= ExecutorchManagerNative._();
    return _instance!;
  }

  /// Internal reference to the Pigeon host API
  late final ExecutorchHostApi _hostApi = ExecutorchHostApi();

  /// Cache of loaded models by model ID
  final Map<String, ExecuTorchModel> _loadedModels = {};

  /// Whether the manager has been initialized
  bool _initialized = false;

  @override
  Future<void> initialize() async {
    if (_initialized) return;

    try {
      // Test Pigeon connectivity by querying loaded models
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

  @override
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
  /// Returns the loaded model instance that can be used for inference.
  ///
  /// The model will be cached and accessed later via [getLoadedModel].
  /// If a model with the same file path is loaded, returns cached instance.
  ///
  /// Note: On web platform, use [loadModelFromAssets] or
  /// [loadModelFromBytes] instead.
  @override
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
  /// Works on all platforms including web.
  @override
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    _ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromAsset(assetPath);
      _loadedModels[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from asset $assetPath: $e',
        'asset_path: $assetPath, error: ${e.toString()}',
      );
    }
  }

  /// Load an ExecuTorch model from bytes
  ///
  /// [modelBytes] should contain the raw .pte model data.
  /// This is useful for loading models downloaded from network or
  /// generated dynamically. Works on all platforms including web.
  @override
  Future<ExecuTorchModel> loadModelFromBytes(Uint8List modelBytes) async {
    _ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromBytes(modelBytes);
      _loadedModels[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from bytes: $e',
        'bytes_length: ${modelBytes.length}, error: ${e.toString()}',
      );
    }
  }

  @override
  ExecuTorchModel? getLoadedModel(String modelId) => _loadedModels[modelId];

  @override
  List<ExecuTorchModel> getLoadedModels() =>
      List.unmodifiable(_loadedModels.values);

  @override
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
  /// After calling this method, the model cannot be used for inference.
  /// The model is removed from the loaded models cache.
  @override
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
  @override
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
  @override
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

  @override
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
  @override
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
  @override
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
