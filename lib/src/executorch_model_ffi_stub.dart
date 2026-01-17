/// FFI platform stub for ExecuTorchModel
///
/// Provides factory methods that use the FFI implementation.
/// Used internally by ExecuTorchModel static methods when
/// dart:ffi is available.
library;

import 'dart:typed_data';

import 'executorch_model.dart';
import 'executorch_model_ffi_impl.dart' as ffi_impl;

/// Load model from file path (FFI implementation)
Future<ExecuTorchModel> load(String filePath) => ffi_impl.load(filePath);

/// Load model from bytes (FFI implementation)
Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes) =>
    ffi_impl.loadFromBytes(modelBytes);

/// Load model from asset bundle (FFI implementation)
Future<ExecuTorchModel> loadFromAsset(String assetPath) =>
    ffi_impl.loadFromAsset(assetPath);
