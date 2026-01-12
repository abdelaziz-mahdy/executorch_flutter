// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// FFI implementation of ExecuTorchModel.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import 'executorch_model.dart';
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
  Future<List<TensorData>> forward(List<TensorData> inputs) async =>
      _module.forward(inputs);

  @override
  Future<void> dispose() async {
    _module.dispose();
  }
}

/// Load an ExecuTorchModel from a file path using FFI.
Future<ExecuTorchModel> load(String filePath) async {
  final module = NativeModule.loadFile(filePath);
  final modelId = _generateModelId(filePath);
  return ExecuTorchModelFfiImpl._(module, modelId);
}

/// Load an ExecuTorchModel from bytes using FFI.
Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes) async {
  final module = NativeModule.load(modelBytes);
  final modelId = _generateModelId('bytes_${modelBytes.length}');
  return ExecuTorchModelFfiImpl._(module, modelId);
}

/// Load an ExecuTorchModel from asset bundle using FFI.
Future<ExecuTorchModel> loadFromAsset(String assetPath) async {
  final byteData = await rootBundle.load(assetPath);
  final bytes = byteData.buffer.asUint8List();
  final module = NativeModule.load(bytes);
  final modelId = _generateModelId(assetPath);
  return ExecuTorchModelFfiImpl._(module, modelId);
}

/// Generate a unique model ID.
String _generateModelId(String source) {
  final timestamp = DateTime.now().microsecondsSinceEpoch;
  final sourceHash = source.hashCode.abs();
  return 'model_${sourceHash}_$timestamp';
}
