// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native logging stub for web platform.
library;

/// Set debug logging on or off.
///
/// On web, this is a no-op since there's no native FFI layer.
void setNativeDebugLogging(bool enabled) {
  // Web doesn't have native logging, no-op
}
