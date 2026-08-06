/// Unsupported platform stub for ExecutorchManager
///
/// This file should never be used at runtime - it's a fallback for
/// the conditional import pattern. If you see errors from this file,
/// the platform detection is not working correctly.
library;

import 'executorch_inference.dart';

/// Platform-specific instance getter - throws on unsupported platforms
ExecutorchManager get instance => throw UnsupportedError(
      'ExecutorchManager is not supported on this platform. '
      'Supported platforms: Android, iOS, macOS, Linux, Windows. '
      'For web, use package:executorch_flutter.',
    );
