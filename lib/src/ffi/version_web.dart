// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Web platform implementation of version query functions.
library;

/// Library version information.
abstract final class ExecuTorchVersion {
  /// Get the library version string.
  ///
  /// On web, returns a placeholder version since the WASM module
  /// may not expose version information directly.
  static String get version => '1.0.0-web';

  /// Get the linked ExecuTorch version string.
  ///
  /// On web, returns a placeholder version.
  static String get executorchVersion => '1.0.0';
}
