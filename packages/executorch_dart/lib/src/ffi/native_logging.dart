// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native logging utilities for FFI layer.
library;

import '../generated/executorch_ffi.g.dart';

/// Global debug logging state - enabled by default for easier debugging.
bool _debugLoggingEnabled = true;

/// Track if we've initialized native logging.
bool _nativeLoggingInitialized = false;

/// Check if debug logging is enabled.
bool get isDebugLoggingEnabled => _debugLoggingEnabled;

/// Initialize native debug logging.
///
/// Called automatically before model operations.
void ensureNativeLoggingInitialized() {
  if (!_nativeLoggingInitialized) {
    et_set_debug_enabled(_debugLoggingEnabled ? 1 : 0);
    _nativeLoggingInitialized = true;
  }
}

/// Set debug logging on or off.
///
/// When enabled, detailed debug output is printed to the console.
/// Also sets the native FFI debug mode accordingly.
///
/// Debug logging is enabled by default to help diagnose model loading issues.
void setNativeDebugLogging(bool enabled) {
  _debugLoggingEnabled = enabled;
  // Pass 1 for enabled, 0 for disabled
  et_set_debug_enabled(enabled ? 1 : 0);
  _nativeLoggingInitialized = true;
}

/// Log a debug message if debug logging is enabled.
void logDebug(String message) {
  if (_debugLoggingEnabled) {
    // ignore: avoid_print
    print('[ExecuTorch DEBUG] $message');
  }
}

/// Log an error message (always logged).
void logError(String message) {
  // ignore: avoid_print
  print('[ExecuTorch ERROR] $message');
}
