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

export 'src/executorch_errors.dart';
export 'src/executorch_inference.dart';
export 'src/executorch_llm.dart' show ExecuTorchLLM, GenConfig;
export 'src/executorch_manager_base.dart' show ExecutorchManagerBase;
export 'src/executorch_model.dart';
export 'src/ffi/backend_query.dart' show BackendQuery;
export 'src/ffi/native_logging.dart' show setNativeDebugLogging;
export 'src/ffi/version.dart' show ExecuTorchVersion;
export 'src/processors/processors.dart';
export 'src/types.dart'
    show
        Backend,
        ExtendedTensorType,
        ModelLoadResult,
        TensorData,
        TensorType,
        TensorTypeExtension;
export 'src/version.dart' show executorchVersion;
