// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Version query functions for FFI layer.
library;

import 'package:ffi/ffi.dart';

import '../generated/executorch_ffi.g.dart';

/// Library version information.
abstract final class ExecuTorchVersion {
  /// Get the library version string.
  static String get version {
    final ptr = et_version();
    return ptr.cast<Utf8>().toDartString();
  }

  /// Get the linked ExecuTorch version string.
  static String get executorchVersion {
    final ptr = et_executorch_version();
    return ptr.cast<Utf8>().toDartString();
  }
}
