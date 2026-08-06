/// ExecuTorch Flutter Plugin - On-device ML inference with ExecuTorch
///
/// This package provides Flutter developers with the ability to run
/// ExecuTorch machine learning models on Android, iOS, macOS, Linux,
/// and Windows with high performance and low latency.
///
/// ## Key Features
///
/// - **High Performance**: Native FFI bindings for minimal latency
/// - **Cross Platform**: Identical APIs across all supported platforms
/// - **User-Controlled Resources**: Explicit model lifecycle with load/dispose
/// - **Easy Integration**: Simple API for loading models and running inference
/// - **Backend Query**: Check available hardware acceleration backends
///
/// ## Quick Start
///
/// ```dart
/// import 'package:executorch_flutter/executorch_flutter.dart';
///
/// // Load a model from asset bundle
/// final model = await loadModelFromAsset('assets/models/model.pte');
///
/// // Prepare input data
/// final inputTensor = TensorData(
///   shape: [1, 3, 224, 224],
///   dataType: TensorType.float32,
///   data: imageBytes,
///   name: 'input',
/// );
///
/// // Run inference
/// final outputs = await model.forward([inputTensor]);
///
/// // Process outputs (List<TensorData>)
/// for (var output in outputs) {
///   print('Output shape: ${output.shape}');
/// }
///
/// // Clean up
/// await model.dispose();
/// ```
///
/// ## Main Classes
///
/// - `ExecuTorchModel`: Main API for loading and running inference
/// - `TensorData`: Tensor data representation
/// - `Backend`: Hardware acceleration backend enumeration
/// - `ExecuTorchVersion`: Library version information
///
/// ## Processors
///
/// - `ExecuTorchPreprocessor`: Base class for input preprocessing
/// - `ExecuTorchPostprocessor`: Base class for output postprocessing
/// - `ExecuTorchProcessor`: Combined preprocessing and postprocessing
///
/// ## Platform Support
///
/// - **Android**: API 23+ (Android 6.0+), arm64-v8a architecture
/// - **iOS**: iOS 13.0+, arm64 (device only)
/// - **macOS**: macOS 11.0+, arm64 (Apple Silicon)
/// - **Linux**: x64 architecture
/// - **Windows**: x64 architecture
///
/// For detailed documentation and examples, see the class documentation.
library;

// Everything from the pure-Dart core except the names routed below. The web
// branch resolves to the core's ffi-free library: `hide` removes a name but
// still compiles the library it came from, so re-exporting the full core here
// would drag dart:ffi into web builds and fail dart2js.
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'package:executorch_dart/executorch_dart_shared.dart'
    if (dart.library.js) 'package:executorch_dart/executorch_dart_shared.dart'
    hide
        BackendQuery,
        ExecuTorchLLM,
        ExecuTorchModel,
        ExecuTorchVersion,
        ExecutorchManager,
        GenConfig,
        setNativeDebugLogging;

// Platform routing. On native each line re-exports the core declaration
// unchanged; on web it resolves to this package's implementation. Exactly one
// declaration of each name survives any given compile.
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/executorch_llm_web.dart'
    if (dart.library.js) 'src/executorch_llm_web.dart'
    show ExecuTorchLLM, GenConfig;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/ffi/backend_query_web.dart'
    if (dart.library.js) 'src/ffi/backend_query_web.dart' show BackendQuery;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/ffi/native_logging_web.dart'
    if (dart.library.js) 'src/ffi/native_logging_web.dart'
    show setNativeDebugLogging;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/ffi/version_web.dart'
    if (dart.library.js) 'src/ffi/version_web.dart' show ExecuTorchVersion;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/web/executorch_model_web.dart'
    if (dart.library.js) 'src/web/executorch_model_web.dart'
    show ExecuTorchModel;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/web/executorch_manager_web.dart'
    if (dart.library.js) 'src/web/executorch_manager_web.dart'
    show ExecutorchManager;

// Flutter-only asset helpers.
export 'src/assets.dart';
