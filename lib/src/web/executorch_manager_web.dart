/// Web platform implementation of ExecutorchManager
///
/// This implementation uses JavaScript interop for communication with
/// the ExecuTorch WebAssembly module.
library;

import 'dart:async';
import 'dart:typed_data';

import '../executorch_errors.dart';
import '../executorch_inference.dart';
import '../executorch_model.dart';
import '../generated/executorch_api.dart';
import 'js_interop.dart' as js;
import 'wasm_module_loader.dart';

/// Web platform implementation of ExecutorchManager
///
/// Uses JavaScript interop for communication with the ExecuTorch Wasm module.
class ExecutorchManagerWeb implements ExecutorchManager {
  ExecutorchManagerWeb._();

  static ExecutorchManagerWeb? _instance;

  /// Get the singleton instance of ExecutorchManagerWeb
  // ignore: prefer_constructors_over_static_methods
  static ExecutorchManagerWeb get instance {
    _instance ??= ExecutorchManagerWeb._();
    return _instance!;
  }

  /// Cache of loaded models by model ID
  final Map<String, ExecuTorchModel> _loadedModels = {};

  /// Whether the manager has been initialized
  bool _initialized = false;

  @override
  Future<void> initialize() async {
    if (_initialized) return;

    try {
      // Initialize Wasm module
      await WasmModuleLoader.ensureInitialized();
      _initialized = true;
    } catch (e) {
      throw ExecuTorchPlatformException(
        'Failed to initialize ExecutorchManager: $e\n'
        'Make sure ExecuTorch Wasm module is available.',
        e.toString(),
      );
    }
  }

  @override
  Future<void> setDebugLogging(bool enabled) async {
    _ensureInitialized();

    try {
      final runner = js.execuTorchRunner;
      runner.setDebugLogging(enabled);
    } catch (e) {
      throw ExecuTorchPlatformException(
        'Failed to set debug logging: $e',
        e.toString(),
      );
    }
  }

  @override
  Future<ExecuTorchModel> loadModel(String filePath) async {
    throw UnsupportedError(
      'loadModel() from file path is not supported on web. '
      'Use loadModelFromAssets() or loadModelFromBytes() instead.',
    );
  }

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

    // Web: Return cached model IDs
    return _loadedModels.keys.toList();
  }

  @override
  Future<void> disposeModel(String modelId) async {
    _ensureInitialized();

    final model = _loadedModels.remove(modelId);
    if (model != null) {
      await model.dispose();
    }
  }

  @override
  Future<void> disposeAllModels() async {
    _ensureInitialized();

    final modelIds = List<String>.from(_loadedModels.keys);
    for (final modelId in modelIds) {
      await disposeModel(modelId);
    }
  }

  @override
  Future<Map<String, Object>> getMemoryInfo() async {
    _ensureInitialized();

    return {
      'loaded_models_count': _loadedModels.length,
      'loaded_model_ids': _loadedModels.keys.toList(),
      'platform': 'web',
      'timestamp': DateTime.now().toIso8601String(),
    };
  }

  @override
  Future<bool> isAvailable() async {
    if (!_initialized) return false;

    // Web: Always available if initialized
    return true;
  }

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

  @override
  Future<void> shutdown() async {
    if (!_initialized) return;

    await disposeAllModels();
    _initialized = false;
    _instance = null;
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
}
