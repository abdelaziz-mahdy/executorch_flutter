// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Backend query functions for FFI layer.
///
/// This library provides runtime introspection of available hardware
/// acceleration backends compiled into the ExecuTorch native library.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../generated/executorch_ffi.g.dart';
import '../types.dart';

/// Query functions for hardware acceleration backend availability.
///
/// Use this class to check which backends are available on the current
/// platform at runtime. This is useful for:
/// - Selecting the optimal model variant for the current device
/// - Filtering model lists to only show compatible models
/// - Providing fallback logic when preferred backends are unavailable
///
/// ## Example
///
/// ```dart
/// import 'package:executorch_flutter/executorch_flutter.dart';
///
/// // Check if a specific backend is available
/// if (BackendQuery.isAvailable(Backend.vulkan)) {
///   print('Vulkan GPU acceleration is available!');
///   model = await ExecuTorchModel.loadFromAsset('assets/model_vulkan.pte');
/// } else {
///   print('Falling back to XNNPACK CPU backend');
///   model = await ExecuTorchModel.loadFromAsset('assets/model_xnnpack.pte');
/// }
///
/// // Get all available backends
/// final backends = BackendQuery.available;
/// print('Available: ${backends.map((b) => b.displayName).join(", ")}');
/// ```
///
/// ## Backend Availability
///
/// Backend availability depends on:
/// 1. **Build configuration**: Which backends were enabled at compile time
/// 2. **Platform support**: Some backends only work on certain platforms
///
/// | Backend | Platforms |
/// |---------|-----------|
/// | XNNPACK | All (including Web) |
/// | CoreML | iOS, macOS |
/// | MPS | macOS |
/// | Vulkan | Android, iOS, macOS, Windows, Linux |
///
/// XNNPACK is always available as a fallback on all platforms.
abstract final class BackendQuery {
  /// Check if a specific backend is available (compiled in).
  ///
  /// Returns `true` if the backend was compiled into the native library
  /// and is supported on the current platform.
  ///
  /// ```dart
  /// if (BackendQuery.isAvailable(Backend.coreml)) {
  ///   // Use CoreML-optimized model on Apple devices
  /// }
  /// ```
  static bool isAvailable(Backend backend) =>
      et_backend_available(_toNative(backend)) == 1;

  /// Get a list of all available backends.
  ///
  /// Returns a list of [Backend] values that are compiled into the
  /// native library and available on the current platform.
  ///
  /// ```dart
  /// final backends = BackendQuery.available;
  /// for (final backend in backends) {
  ///   print('${backend.displayName} is available');
  /// }
  /// ```
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
        Backend.metal => ETBackend.ET_BACKEND_METAL,
        Backend.vulkan => ETBackend.ET_BACKEND_VULKAN,
        Backend.qnn => ETBackend.ET_BACKEND_QNN,
      };

  /// Convert from native ETBackend enum.
  static Backend _fromNative(ETBackend native) => switch (native) {
        ETBackend.ET_BACKEND_XNNPACK => Backend.xnnpack,
        ETBackend.ET_BACKEND_COREML => Backend.coreml,
        ETBackend.ET_BACKEND_MPS => Backend.mps,
        ETBackend.ET_BACKEND_METAL => Backend.metal,
        ETBackend.ET_BACKEND_VULKAN => Backend.vulkan,
        ETBackend.ET_BACKEND_QNN => Backend.qnn,
      };
}
