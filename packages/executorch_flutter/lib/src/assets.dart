/// Flutter asset-bundle loading for ExecuTorch models.
library;

import 'package:executorch_dart/executorch_dart_shared.dart';
// Routed the same way the public library routes `ExecuTorchModel`: on native
// this is the core's class, on web it is this package's Wasm-backed one.
import 'package:executorch_dart/executorch_dart_shared.dart'
    if (dart.library.js_interop) 'web/executorch_model_web.dart'
    if (dart.library.js) 'web/executorch_model_web.dart'
    as impl;
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
///
/// The return type is routed, not the bare core interface: on web this
/// package exports its own Wasm-backed class under the name `ExecuTorchModel`,
/// and that class only *implements* the core interface. Returning the
/// interface here would make `ExecuTorchModel m = await loadModelFromAsset(…)`
/// a compile error on web — and a baffling one, since both types print as
/// `ExecuTorchModel`.
Future<impl.ExecuTorchModel> loadModelFromAsset(String assetPath) async {
  final byteData = await rootBundle.load(assetPath);
  return impl.ExecuTorchModel.loadFromBytes(byteData.buffer.asUint8List());
}

/// Asset-bundle loading for [ExecutorchManager].
extension ExecutorchManagerAssets on ExecutorchManager {
  /// Loads a model from the Flutter asset bundle and caches it.
  ///
  /// Equivalent to reading the asset yourself and calling
  /// [ExecutorchManager.loadModelFromBytes].
  ///
  /// Like [loadModelFromAsset], the result is typed as the class this package
  /// exports under the name `ExecuTorchModel` — the core interface on native,
  /// the Wasm-backed class on web — so assigning it to an `ExecuTorchModel`
  /// variable compiles on every platform. The narrowing cast is what makes
  /// that possible: [ExecutorchManager.loadModelFromBytes] is declared in the
  /// core and can only promise the core interface, but on web every manager
  /// this package can hand you returns the web class.
  Future<impl.ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final model = await loadModelFromBytes(byteData.buffer.asUint8List());
    // Redundant on native, where `impl.ExecuTorchModel` *is* the core
    // interface — which is the only branch the analyzer ever resolves. On web
    // it is a real downcast to this package's Wasm-backed class, and without
    // it this method does not compile there at all.
    // ignore: unnecessary_cast
    return model as impl.ExecuTorchModel;
  }
}
