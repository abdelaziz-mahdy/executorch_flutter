/// Unsupported platform stub for ExecuTorchModel
///
/// This file should never be used at runtime - it's a fallback for
/// the conditional import pattern. If you see errors from this file,
/// the platform detection is not working correctly.
library;

import 'dart:typed_data';

import 'executorch_model.dart';

/// Load model from file path - throws on unsupported platforms
Future<ExecuTorchModel> load(String filePath) => throw UnsupportedError(
  'ExecuTorchModel is not supported on this platform. '
  'Supported platforms: Android, iOS, macOS, Linux, Windows. '
  'For web, use package:executorch_flutter.',
);

/// Load model from bytes - throws on unsupported platforms
Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes) =>
    throw UnsupportedError(
      'ExecuTorchModel is not supported on this platform. '
      'Supported platforms: Android, iOS, macOS, Linux, Windows. '
      'For web, use package:executorch_flutter.',
    );
