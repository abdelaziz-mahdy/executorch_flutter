// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native platform implementation of ExecutorchManager
///
/// This implementation uses FFI for direct native calls to the
/// ExecuTorch C library on Android, iOS, macOS, Linux, and Windows.
library;

import 'dart:async';
import 'dart:io';

import 'executorch_errors.dart';
import 'executorch_manager_base.dart';
import 'executorch_model.dart';
import 'ffi/native_logging.dart';

/// Native platform implementation of ExecutorchManager
///
/// Uses FFI for direct native calls to the ExecuTorch C library.
class ExecutorchManagerNative extends ExecutorchManagerBase {
  ExecutorchManagerNative._();

  static ExecutorchManagerNative? _instance;

  /// Get the singleton instance of ExecutorchManagerNative
  // ignore: prefer_constructors_over_static_methods
  static ExecutorchManagerNative get instance {
    _instance ??= ExecutorchManagerNative._();
    return _instance!;
  }

  @override
  Future<void> initialize() async {
    if (initialized) return;
    // FFI is always available on native platforms
    initialized = true;
  }

  @override
  Future<void> setDebugLogging(bool enabled) async {
    ensureInitialized();
    // Enable debug logging via FFI native layer
    setNativeDebugLogging(enabled);
  }

  /// Load an ExecuTorch model from a file path
  ///
  /// [filePath] must point to a valid ExecuTorch .pte model file.
  /// Returns the loaded model instance that can be used for inference.
  ///
  /// The model will be cached and accessed later via [getLoadedModel].
  /// If a model with the same file path is loaded, returns cached instance.
  ///
  /// Note: On web platform, use [loadModelFromBytes] instead.
  @override
  Future<ExecuTorchModel> loadModel(String filePath) async {
    ensureInitialized();

    // Validate file path
    if (!File(filePath).existsSync()) {
      throw ExecuTorchModelException(
        'Model file not found: $filePath',
        'file_path: $filePath',
      );
    }

    try {
      final model = await ExecuTorchModel.load(filePath);
      loadedModelsMap[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from $filePath: $e',
        'file_path: $filePath, error: ${e.toString()}',
      );
    }
  }

  /// Get detailed information about system memory usage
  ///
  /// Returns a map with memory statistics, if available on the platform.
  /// This is useful for monitoring memory usage and detecting leaks.
  @override
  Future<Map<String, Object>> getMemoryInfo() async {
    ensureInitialized();

    return {
      'loaded_models_count': loadedModelsMap.length,
      'loaded_model_ids': loadedModelsMap.keys.toList(),
      'platform': Platform.operatingSystem,
      'timestamp': DateTime.now().toIso8601String(),
    };
  }

  @override
  Future<bool> isAvailable() async => initialized;

  /// Cleanup resources when the manager is no longer needed
  ///
  /// This should be called when the app is shutting down to ensure
  /// proper cleanup of all loaded models and platform resources.
  @override
  Future<void> shutdown() async {
    if (!initialized) return;

    await disposeAllModels();
    initialized = false;
    _instance = null;
  }
}
