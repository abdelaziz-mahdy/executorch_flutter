/// Flutter asset-bundle loading for ExecuTorch models.
library;

import 'package:executorch_dart/executorch_dart.dart';
// Routed the same way the public library routes `ExecuTorchModel`: on native
// this is the core's class, on web it is this package's Wasm-backed one.
import 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'web/executorch_model_web.dart'
    if (dart.library.js) 'web/executorch_model_web.dart' as impl;
import 'package:flutter/services.dart' show rootBundle;

/// Loads an ExecuTorch model from the Flutter asset bundle.
///
/// Works on every platform, including web. The asset must be declared under
/// `flutter: assets:` in the application's `pubspec.yaml`.
///
/// ```dart
/// final model = await loadModelFromAsset('assets/models/model.pte');
/// ```
///
/// Throws [ExecuTorchException] if the asset is missing or the model fails
/// to load.
Future<ExecuTorchModel> loadModelFromAsset(String assetPath) async {
  final byteData = await rootBundle.load(assetPath);
  return impl.ExecuTorchModel.loadFromBytes(byteData.buffer.asUint8List());
}

/// Asset-bundle loading for [ExecutorchManager].
extension ExecutorchManagerAssets on ExecutorchManager {
  /// Loads a model from the Flutter asset bundle and caches it.
  ///
  /// Equivalent to reading the asset yourself and calling
  /// [ExecutorchManager.loadModelFromBytes].
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    return loadModelFromBytes(byteData.buffer.asUint8List());
  }
}
