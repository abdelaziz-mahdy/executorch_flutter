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
/// final model = await ExecuTorchModel.loadFromAsset('assets/models/model.pte');
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

// Core API exports
export 'src/executorch_errors.dart';
export 'src/executorch_inference.dart';
// On-device LLM (text generation) API — native only; web stub throws
// UnsupportedError on every call.
export 'src/executorch_llm.dart'
    if (dart.library.js_interop) 'src/executorch_llm_web.dart'
    if (dart.library.js) 'src/executorch_llm_web.dart'
    show ExecuTorchLLM, GenConfig;
export 'src/executorch_model.dart';
// Backend query (conditional for web)
export 'src/ffi/backend_query.dart'
    if (dart.library.js_interop) 'src/ffi/backend_query_web.dart'
    if (dart.library.js) 'src/ffi/backend_query_web.dart' show BackendQuery;
// Debug logging (native platforms only)
export 'src/ffi/native_logging.dart'
    if (dart.library.js_interop) 'src/ffi/native_logging_web.dart'
    if (dart.library.js) 'src/ffi/native_logging_web.dart'
    show setNativeDebugLogging;
// Version query (conditional for web)
export 'src/ffi/version.dart'
    if (dart.library.js_interop) 'src/ffi/version_web.dart'
    if (dart.library.js) 'src/ffi/version_web.dart' show ExecuTorchVersion;
// Preprocessing and postprocessing utilities
export 'src/processors/processors.dart';
// Core types - tensor data, model result, Backend enum
export 'src/types.dart'
    show Backend, ModelLoadResult, TensorData, TensorType;
// Plugin version constant (ExecuTorch version this plugin is built against)
export 'src/version.dart' show executorchVersion;
