// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Web platform implementation of version query functions.
library;

import 'package:executorch_dart/executorch_dart_shared.dart' as core;

/// Library version information.
abstract final class ExecuTorchVersion {
  /// Get the library version string.
  ///
  /// On web, returns the package version with a "-web" suffix since the WASM
  /// module may not expose version information directly.
  static String get version => '${core.executorchVersion}-web';

  /// Get the linked ExecuTorch version string.
  ///
  /// On web, returns the version constant from version.dart.
  static String get executorchVersion => core.executorchVersion;
}
