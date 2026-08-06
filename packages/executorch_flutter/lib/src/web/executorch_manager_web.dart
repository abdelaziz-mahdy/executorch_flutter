/// Web platform implementation of ExecutorchManager
///
/// This implementation uses JavaScript interop for communication with
/// the ExecuTorch WebAssembly module.
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:executorch_dart/executorch_dart.dart' as core;

import 'executorch_model_web.dart';
import 'js_interop.dart' as js;
import 'wasm_module_loader.dart';

/// Web implementation of the ExecuTorch manager, backed by WebAssembly.
///
/// Uses JavaScript interop for communication with the ExecuTorch Wasm module.
class ExecutorchManager extends core.ExecutorchManagerBase {
  ExecutorchManager._();

  static ExecutorchManager? _instance;

  /// Get the singleton instance of ExecutorchManager
  // ignore: prefer_constructors_over_static_methods
  static ExecutorchManager get instance {
    _instance ??= ExecutorchManager._();
    return _instance!;
  }

  @override
  Future<void> initialize() async {
    if (initialized) return;

    try {
      // Initialize Wasm module
      await WasmModuleLoader.ensureInitialized();
      initialized = true;
    } catch (e) {
      throw core.ExecuTorchPlatformException(
        'Failed to initialize ExecutorchManager: $e\n'
        'Make sure ExecuTorch Wasm module is available.',
        e.toString(),
      );
    }
  }

  @override
  Future<void> setDebugLogging(bool enabled) async {
    ensureInitialized();

    try {
      js.execuTorchRunner.setDebugLogging(enabled);
    } catch (e) {
      throw core.ExecuTorchPlatformException(
        'Failed to set debug logging: $e',
        e.toString(),
      );
    }
  }

  @override
  Future<core.ExecuTorchModel> loadModel(String filePath) async {
    throw UnsupportedError(
      'loadModel() from file path is not supported on web. '
      'Use loadModelFromAssets() or loadModelFromBytes() instead.',
    );
  }

  /// Load a model from bytes and cache it.
  ///
  /// Overrides the base implementation, which delegates to the pure-Dart
  /// core's dart:ffi loader and is therefore unavailable on web. This routes
  /// through the WebAssembly model implementation instead.
  @override
  Future<core.ExecuTorchModel> loadModelFromBytes(Uint8List modelBytes) async {
    ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromBytes(modelBytes);
      loadedModelsMap[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is core.ExecuTorchException) rethrow;
      throw core.ExecuTorchModelException(
        'Failed to load model from bytes: $e',
        'bytes_length: ${modelBytes.length}, error: $e',
      );
    }
  }

  @override
  Future<Map<String, Object>> getMemoryInfo() async {
    ensureInitialized();

    return {
      'loaded_models_count': loadedModelsMap.length,
      'loaded_model_ids': loadedModelsMap.keys.toList(),
      'platform': 'web',
      'timestamp': DateTime.now().toIso8601String(),
    };
  }

  @override
  Future<bool> isAvailable() async {
    if (!initialized) return false;

    // Web: Always available if initialized
    return true;
  }

  @override
  Future<void> shutdown() async {
    if (!initialized) return;

    await disposeAllModels();
    initialized = false;
    _instance = null;
  }
}
