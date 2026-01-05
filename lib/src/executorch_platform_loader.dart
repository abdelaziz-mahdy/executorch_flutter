/// Platform-specific model loading
///
/// This file provides conditional exports based on the target platform:
/// - Web: ExecuTorchModelWeb with byte-based loading
/// - Native (Android, iOS, macOS): ExecuTorchModel with file-based loading
library;

export 'executorch_model_unsupported_stub.dart'
    if (dart.library.io) 'executorch_model_native_stub.dart'
    if (dart.library.js_interop) 'executorch_model_web_stub.dart'
    if (dart.library.js) 'executorch_model_web_stub.dart';
