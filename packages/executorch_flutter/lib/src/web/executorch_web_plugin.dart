/// Web platform registration for ExecuTorch Flutter.
///
/// This class does no work at runtime — model loading and inference on web go
/// through `ExecuTorchModel`, which the package's conditional exports route to
/// the WebAssembly implementation in `lib/src/web/`.
///
/// It exists because the `web:` entry in `pubspec.yaml`'s `flutter: plugin:
/// platforms:` map needs a `pluginClass`, and that entry is what declares Web
/// support to pub tooling. Without it, pub.dev reports "Package does not
/// support platform Web" and drops the Web platform tag, even though the
/// WebAssembly build works. It was removed in 0.6.0 as dead code on the
/// grounds that `registerWith` is empty, and restored in 0.6.2 once that
/// consequence showed up on the package page.
library;

import 'package:flutter_web_plugins/flutter_web_plugins.dart';

/// Registers the web implementation of ExecuTorch.
class ExecutorchFlutterWebPlugin {
  /// Called by the Flutter web plugin registrant.
  ///
  /// Intentionally empty — see the library doc comment above.
  static void registerWith(Registrar registrar) {}
}
