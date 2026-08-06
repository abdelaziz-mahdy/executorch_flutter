/// Platform-specific model loading
///
/// This file provides conditional exports based on the target platform:
/// - Web: ExecuTorchModelWeb with byte-based loading
/// - Native (Android, iOS, macOS, Linux, Windows): FFI-based implementation
library;

export 'executorch_model_unsupported_stub.dart'
    if (dart.library.ffi) 'executorch_model_ffi_stub.dart';
