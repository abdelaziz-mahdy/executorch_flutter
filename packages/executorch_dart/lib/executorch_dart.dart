/// ExecuTorch on-device ML inference for pure Dart.
///
/// Runs ExecuTorch models on Android, iOS, macOS, Linux, and Windows through
/// dart:ffi and native assets. Works in any Dart program, including servers
/// and command-line tools — no Flutter required.
///
/// ```dart
/// import 'package:executorch_dart/executorch_dart.dart';
///
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
/// final outputs = await model.forward([inputTensor]);
/// await model.dispose();
/// ```
///
/// Flutter applications should depend on `executorch_flutter` instead, which
/// adds asset-bundle loading and web support on top of this package.
library;

// Everything that does not need dart:ffi. Add new ffi-free exports there, not
// here, so the web branch of `package:executorch_flutter` keeps seeing them.
export 'executorch_dart_shared.dart';
// The native-only surface. These four reach dart:ffi, which is why they are
// kept out of `executorch_dart_shared.dart`.
export 'src/executorch_llm.dart' show ExecuTorchLLM, GenConfig;
export 'src/ffi/backend_query.dart' show BackendQuery;
export 'src/ffi/native_logging.dart' show setNativeDebugLogging;
export 'src/ffi/version.dart' show ExecuTorchVersion;
export 'src/tokenizer.dart' show Tokenizer, TokenizerFormat;
