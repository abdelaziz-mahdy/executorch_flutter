/// Web platform plugin for ExecuTorch Flutter
///
/// This plugin registers the web implementation of ExecuTorch
/// when running on the web platform.
library;

import 'package:flutter_web_plugins/flutter_web_plugins.dart';

/// Web implementation of ExecuTorch plugin
///
/// This class handles plugin registration for the web platform.
/// The actual model loading and inference is handled by ExecuTorchModelWeb.
class ExecutorchFlutterWebPlugin {
  /// Registers the web plugin
  static void registerWith(Registrar registrar) {
    // Plugin registration for web platform
    // The actual implementation is in ExecuTorchModelWeb
  }
}
