// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// FFI implementation of ExecuTorchModel.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import 'executorch_model.dart';
import 'ffi/native_logging.dart';
import 'ffi/native_module.dart';
import 'types.dart';

/// FFI-based implementation of ExecuTorchModel.
///
/// Uses dart:ffi for direct native calls to the ExecuTorch C library.
class ExecuTorchModelFfiImpl extends ExecuTorchModel {
  ExecuTorchModelFfiImpl._(this._module, this._modelId);

  final NativeModule _module;
  final String _modelId;

  @override
  String get modelId => _modelId;

  @override
  bool get isDisposed => _module.isDisposed;

  @override
  Future<List<TensorData>> forward(List<TensorData> inputs) =>
      _module.forwardAsync(inputs);

  @override
  Future<void> dispose() async {
    // Await any in-flight async forward passes before freeing the native
    // module, so inference is never running when the model is destroyed.
    await _module.disposeAsync();
  }
}

/// Load an ExecuTorchModel from a file path using FFI.
Future<ExecuTorchModel> load(String filePath) async {
  // Ensure debug logging is enabled before model loading
  ensureNativeLoggingInitialized();
  final module = await NativeModule.loadFileAsync(filePath);
  final modelId = _generateModelId(filePath);
  return ExecuTorchModelFfiImpl._(module, modelId);
}

/// Load an ExecuTorchModel from bytes using FFI.
Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes) async {
  // Ensure debug logging is enabled before model loading
  ensureNativeLoggingInitialized();
  final module = await NativeModule.loadAsync(modelBytes);
  final modelId = _generateModelId('bytes_${modelBytes.length}');
  return ExecuTorchModelFfiImpl._(module, modelId);
}

/// Load an ExecuTorchModel from asset bundle using FFI.
Future<ExecuTorchModel> loadFromAsset(String assetPath) async {
  // Ensure debug logging is enabled before model loading
  ensureNativeLoggingInitialized();
  final byteData = await rootBundle.load(assetPath);
  final bytes = byteData.buffer.asUint8List();
  final module = await NativeModule.loadAsync(bytes);
  final modelId = _generateModelId(assetPath);
  return ExecuTorchModelFfiImpl._(module, modelId);
}

/// Generate a unique model ID.
String _generateModelId(String source) {
  final timestamp = DateTime.now().microsecondsSinceEpoch;
  final sourceHash = source.hashCode.abs();
  return 'model_${sourceHash}_$timestamp';
}
