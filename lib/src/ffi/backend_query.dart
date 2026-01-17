// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Backend query functions for FFI layer.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../generated/executorch_ffi.g.dart';
import '../types.dart';

/// Query functions for hardware acceleration backend availability.
///
/// Use this class to check which backends are available on the current
/// platform.
abstract final class BackendQuery {
  /// Check if a backend is available (compiled in).
  static bool isAvailable(Backend backend) =>
      et_backend_available(_toNative(backend)) == 1;

  /// Get all available backends.
  static List<Backend> get available {
    // Allocate array for backend list
    final outPtr = calloc<ffi.UnsignedInt>(16);
    try {
      final count = et_backend_list(outPtr, 16);
      final backends = <Backend>[];

      for (var i = 0; i < count; i++) {
        final native = ETBackend.fromValue(outPtr[i]);
        backends.add(_fromNative(native));
      }

      return backends;
    } finally {
      calloc.free(outPtr);
    }
  }

  /// Convert Backend to native ETBackend enum.
  static ETBackend _toNative(Backend backend) => switch (backend) {
        Backend.xnnpack => ETBackend.ET_BACKEND_XNNPACK,
        Backend.coreml => ETBackend.ET_BACKEND_COREML,
        Backend.mps => ETBackend.ET_BACKEND_MPS,
        Backend.vulkan => ETBackend.ET_BACKEND_VULKAN,
        Backend.qnn => ETBackend.ET_BACKEND_QNN,
      };

  /// Convert from native ETBackend enum.
  static Backend _fromNative(ETBackend native) => switch (native) {
        ETBackend.ET_BACKEND_XNNPACK => Backend.xnnpack,
        ETBackend.ET_BACKEND_COREML => Backend.coreml,
        ETBackend.ET_BACKEND_MPS => Backend.mps,
        ETBackend.ET_BACKEND_VULKAN => Backend.vulkan,
        ETBackend.ET_BACKEND_QNN => Backend.qnn,
      };
}
