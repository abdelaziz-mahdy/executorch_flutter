/// Web platform implementation of ExecutorchManager
///
/// This implementation uses JavaScript interop for communication with
/// the ExecuTorch WebAssembly module.
library;

import 'dart:async';

import '../executorch_errors.dart';
import '../executorch_manager_base.dart';
import '../executorch_model.dart';
import 'js_interop.dart' as js;
import 'wasm_module_loader.dart';

/// Web platform implementation of ExecutorchManager
///
/// Uses JavaScript interop for communication with the ExecuTorch Wasm module.
class ExecutorchManagerWeb extends ExecutorchManagerBase {
  ExecutorchManagerWeb._();

  static ExecutorchManagerWeb? _instance;

  /// Get the singleton instance of ExecutorchManagerWeb
  // ignore: prefer_constructors_over_static_methods
  static ExecutorchManagerWeb get instance {
    _instance ??= ExecutorchManagerWeb._();
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
      throw ExecuTorchPlatformException(
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
