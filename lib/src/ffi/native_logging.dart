// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native logging utilities for FFI layer.
library;

import '../generated/executorch_ffi.g.dart';

/// Global debug logging state.
bool _debugLoggingEnabled = false;

/// Check if debug logging is enabled.
bool get isDebugLoggingEnabled => _debugLoggingEnabled;

/// Set debug logging on or off.
///
/// When enabled, detailed debug output is printed to the console.
/// Also sets the native FFI debug mode accordingly.
void setNativeDebugLogging(bool enabled) {
  _debugLoggingEnabled = enabled;
  // Pass 1 for enabled, 0 for disabled
  et_set_debug_enabled(enabled ? 1 : 0);
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
