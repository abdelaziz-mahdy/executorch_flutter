/// Web platform stub for ExecuTorchModel
///
/// Provides platform-specific factory methods for web platform.
/// Used internally by ExecuTorchModel static methods.
library;

import 'dart:typed_data';

import 'web/executorch_model_web.dart';

/// Load model from file path (not supported on web)
Future<ExecuTorchModelWeb> load(String filePath) {
  throw UnsupportedError(
    'ExecuTorchModel.load() from file path is not supported on web. '
    'Use loadFromAsset() or loadFromBytes() instead.',
  );
}

/// Load model from bytes
Future<ExecuTorchModelWeb> loadFromBytes(Uint8List modelBytes) =>
    ExecuTorchModelWeb.loadFromBytes(modelBytes);

/// Load model from asset bundle
Future<ExecuTorchModelWeb> loadFromAsset(String assetPath) =>
    ExecuTorchModelWeb.loadFromAsset(assetPath);
