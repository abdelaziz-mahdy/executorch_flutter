/// Platform-specific model loading
///
/// This file provides conditional exports based on the target platform:
/// - Native (Android, iOS, macOS, Linux, Windows): FFI-based implementation
/// - Anywhere without dart:ffi: a stub that throws [UnsupportedError]
library;

export 'executorch_model_unsupported_stub.dart'
    if (dart.library.ffi) 'executorch_model_ffi_stub.dart';
