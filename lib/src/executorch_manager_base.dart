/// Base implementation of ExecutorchManager with shared logic
///
/// This class contains the common implementation for both native and web
/// platforms, reducing code duplication. Platform-specific managers
/// extend this class and override only the methods that differ.
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:meta/meta.dart';

import 'executorch_errors.dart';
import 'executorch_inference.dart';
import 'executorch_model.dart';
import 'types.dart';

/// Base implementation of ExecutorchManager with shared logic
///
/// Contains common implementations for model caching, tensor creation,
/// and resource management. Platform-specific managers extend this class.
abstract class ExecutorchManagerBase implements ExecutorchManager {
  /// Cache of loaded models by model ID (accessible to subclasses)
  @protected
  final Map<String, ExecuTorchModel> loadedModelsMap = {};

  /// Whether the manager has been initialized (accessible to subclasses)
  @protected
  bool initialized = false;

  @override
  ExecuTorchModel? getLoadedModel(String modelId) => loadedModelsMap[modelId];

  @override
  List<ExecuTorchModel> getLoadedModels() =>
      List.unmodifiable(loadedModelsMap.values);

  @override
  Future<List<String>> getLoadedModelIds() async {
    ensureInitialized();
    return loadedModelsMap.keys.toList();
  }

  @override
  Future<void> disposeModel(String modelId) async {
    ensureInitialized();

    final model = loadedModelsMap.remove(modelId);
    if (model != null) {
      await model.dispose();
    }
  }

  @override
  Future<void> disposeAllModels() async {
    ensureInitialized();

    final modelIds = List<String>.from(loadedModelsMap.keys);
    for (final modelId in modelIds) {
      await disposeModel(modelId);
    }
  }

  @override
  TensorData createTensorData({
    required List<int> shape,
    required TensorType dataType,
    required List<num> data,
    String? name,
  }) {
    // Convert numeric data to bytes based on data type
    final bytes = convertNumericDataToBytes(data, dataType);

    final tensor = TensorData(
      shape: shape.cast<int?>(),
      dataType: dataType,
      data: bytes,
      name: name,
    );

    return tensor;
  }

  @override
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromAsset(assetPath);
      loadedModelsMap[model.modelId] = model;
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
    ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromBytes(modelBytes);
      loadedModelsMap[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from bytes: $e',
        'bytes_length: ${modelBytes.length}, error: ${e.toString()}',
      );
    }
  }

  /// Ensure the manager has been initialized
  ///
  /// Throws [ExecuTorchPlatformException] if not initialized.
  void ensureInitialized() {
    if (!initialized) {
      throw const ExecuTorchPlatformException(
        'ExecutorchManager not initialized. Call initialize() first.',
      );
    }
  }

  /// Utility method to convert numeric data to bytes
  ///
  /// Converts a list of numeric values to the appropriate byte representation
  /// based on the specified [dataType].
  static Uint8List convertNumericDataToBytes(
    List<num> data,
    TensorType dataType,
  ) {
    switch (dataType) {
      case TensorType.float32:
        final float32List = Float32List.fromList(
          data.map((e) => e.toDouble()).toList(),
        );
        return float32List.buffer.asUint8List();

      case TensorType.int32:
        final int32List = Int32List.fromList(
          data.map((e) => e.toInt()).toList(),
        );
        return int32List.buffer.asUint8List();

      case TensorType.int8:
        return Uint8List.fromList(
          data.map((e) => e.toInt().clamp(-128, 127) + 128).toList(),
        );

      case TensorType.uint8:
        return Uint8List.fromList(
          data.map((e) => e.toInt().clamp(0, 255)).toList(),
        );
    }
  }
}
