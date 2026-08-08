// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Web platform implementation of backend query functions.
library;

import 'package:executorch_dart/executorch_dart_shared.dart' show Backend;

/// Query functions for hardware acceleration backend availability.
///
/// On web, only XNNPACK is available via WebAssembly.
abstract final class BackendQuery {
  /// Check if a backend is available (compiled in).
  ///
  /// On web, only XNNPACK is available via WebAssembly.
  static bool isAvailable(Backend backend) => backend == Backend.xnnpack;

  /// Get all available backends.
  ///
  /// On web, only XNNPACK is available.
  static List<Backend> get available => [Backend.xnnpack];
}
