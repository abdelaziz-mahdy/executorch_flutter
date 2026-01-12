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
/// Also sets the native FFI log level accordingly.
void setNativeDebugLogging(bool enabled) {
  _debugLoggingEnabled = enabled;
  // Native level: 4=debug, 1=error only
  et_set_log_level(enabled ? 4 : 1);
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
