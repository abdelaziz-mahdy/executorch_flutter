/// Native platform stub for ExecuTorchModel
///
/// Provides platform-specific factory methods for native platforms.
/// Used internally by ExecuTorchModel static methods.
library;

import 'dart:typed_data';

import 'executorch_model_native_impl.dart';

/// Load model from file path (native platforms only)
Future<ExecuTorchModelNative> load(String filePath) =>
    ExecuTorchModelNative.load(filePath);

/// Load model from bytes
Future<ExecuTorchModelNative> loadFromBytes(Uint8List modelBytes) =>
    ExecuTorchModelNative.loadFromBytes(modelBytes);

/// Load model from asset bundle
Future<ExecuTorchModelNative> loadFromAsset(String assetPath) =>
    ExecuTorchModelNative.loadFromAsset(assetPath);
