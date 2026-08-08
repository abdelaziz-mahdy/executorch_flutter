/// The `dart:ffi`-free subset of `package:executorch_dart`.
///
/// **Application code should not import this library — import
/// `package:executorch_dart/executorch_dart.dart` instead.** This one exists
/// for a single consumer: the web branch of `package:executorch_flutter`.
///
/// The reason it has to exist is that `export '...' hide Name` still *compiles*
/// the library it hides names from. `executorch_flutter` re-exports this
/// package wholesale and routes the handful of native-only names to web
/// implementations of its own, but hiding those names is not enough — on web,
/// the blanket re-export would still drag `executorch_dart.dart`, and with it
/// `dart:ffi`, into the dart2js compile and fail the build. Pointing the web
/// branch of that re-export here keeps `dart:ffi` out of the graph entirely.
///
/// So this library carries everything in the package that does not need
/// `dart:ffi`: the tensor and model types, the error hierarchy, the processor
/// base classes, the manager interface and its shared base implementation, and
/// the version constant. The four native-only pieces — `ExecuTorchLLM`,
/// `BackendQuery`, `ExecuTorchVersion`, and `setNativeDebugLogging` — are
/// exported only from `executorch_dart.dart`.
///
/// When adding a new export to this package, add it here unless it needs
/// `dart:ffi`; `executorch_dart.dart` re-exports this library in full.
library;

export 'src/executorch_errors.dart';
export 'src/executorch_inference.dart';
export 'src/executorch_manager_base.dart' show ExecutorchManagerBase;
export 'src/executorch_model.dart';
export 'src/llm_types.dart' show GenConfig;
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
