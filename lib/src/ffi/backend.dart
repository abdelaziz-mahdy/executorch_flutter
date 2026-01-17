// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Backend enumeration and query functions for FFI layer.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../generated/executorch_ffi.g.dart';

/// Hardware acceleration backends supported by ExecuTorch.
enum Backend {
  /// XNNPACK CPU optimization library (available on all platforms).
  xnnpack,

  /// Apple CoreML backend (iOS/macOS only).
  coreml,

  /// Apple Metal Performance Shaders backend (macOS only).
  mps,

  /// Vulkan GPU compute backend (Android/Linux/Windows).
  vulkan,

  /// Qualcomm Neural Processing Unit backend (Android only).
  qnn;

  /// Check if this backend is available (compiled in).
  bool get isAvailable => et_backend_available(_toNative()) == 1;

  /// Human-readable display name for this backend.
  String get displayName => switch (this) {
        Backend.xnnpack => 'XNNPACK',
        Backend.coreml => 'CoreML',
        Backend.mps => 'Metal Performance Shaders',
        Backend.vulkan => 'Vulkan',
        Backend.qnn => 'Qualcomm QNN',
      };

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

  /// Convert to native ETBackend enum.
  ETBackend _toNative() => switch (this) {
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
