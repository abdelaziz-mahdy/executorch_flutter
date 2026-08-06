/// Web platform implementation of ExecuTorchModel
///
/// Uses dart:js_interop to communicate with the JavaScript wrapper around
/// the ExecuTorch WebAssembly module.
library;

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:executorch_dart/executorch_dart_shared.dart' as core;

import 'js_interop.dart' as js;
import 'wasm_module_loader.dart';

/// Web implementation of [core.ExecuTorchModel], backed by WebAssembly.
///
/// This implementation:
/// - Loads models into the Wasm virtual filesystem
/// - Runs inference via JavaScript interop
/// - Manages model lifecycle using JavaScript wrapper
class ExecuTorchModel implements core.ExecuTorchModel {
  ExecuTorchModel._({
    required this.modelId,
    required this.inputShapes,
    required this.outputShapes,
  });

  @override
  final String modelId;

  /// Expected input tensor shapes
  final List<List<int>> inputShapes;

  /// Expected output tensor shapes
  final List<List<int>> outputShapes;

  bool _isDisposed = false;

  /// Loading from a file path is not supported on web.
  ///
  /// Use `loadModelFromAsset` or [loadFromBytes] instead.
  static Future<ExecuTorchModel> load(String filePath) =>
      throw UnsupportedError(
        'ExecuTorchModel.load() from a file path is not supported on web. '
        'Use loadModelFromAsset() or loadFromBytes() instead.',
      );

  /// Load a model from bytes
  ///
  /// [modelBytes] - Model file bytes (.pte format)
  ///
  /// Returns an [ExecuTorchModel] instance ready for inference
  ///
  /// Throws [core.ExecuTorchModelException] if loading fails
  static Future<ExecuTorchModel> loadFromBytes(Uint8List modelBytes) async {
    try {
      // Ensure Wasm module is initialized
      await WasmModuleLoader.ensureInitialized();

      // Get JavaScript runner instance
      final runner = js.execuTorchRunner;

      // Convert model bytes to JavaScript Uint8Array
      final jsModelBytes = modelBytes.toJSUint8Array();

      // Load model via JavaScript
      final jsResult = await runner.loadModel(jsModelBytes).toDart;

      // Extract results from JavaScript object
      final modelId = jsResult.modelId.toString();
      final inputShapes = jsResult.inputShapes.toDartList2DInt();
      final outputShapes = jsResult.outputShapes.toDartList2DInt();

      return ExecuTorchModel._(
        modelId: modelId,
        inputShapes: inputShapes,
        outputShapes: outputShapes,
      );
    } catch (e) {
      throw core.ExecuTorchModelException('Failed to load model: $e');
    }
  }

  /// Execute inference on the model
  ///
  /// [inputs] - List of input tensors matching model's input specification
  ///
  /// Returns list of output tensors from the model
  ///
  /// Throws [core.ExecuTorchInferenceException] if inference fails
  @override
  Future<List<core.TensorData>> forward(List<core.TensorData> inputs) async {
    if (_isDisposed) {
      throw const core.ExecuTorchModelException(
        'Cannot run inference on disposed model',
      );
    }

    try {
      // Get JavaScript runner instance
      final runner = js.execuTorchRunner;

      // Convert Dart tensors to JavaScript format
      final jsTensors = _convertTensorsToJS(inputs);

      // Run inference via JavaScript
      final jsOutputs = await runner
          .forward(
            int.parse(modelId),
            jsTensors.jsify() as JSArray<js.TensorData>,
          )
          .toDart;

      // Convert JavaScript outputs back to Dart
      return _convertTensorsFromJS(jsOutputs);
    } catch (e) {
      throw core.ExecuTorchInferenceException(
        'Inference failed: $e',
        e.toString(),
      );
    }
  }

  /// Dispose this model and free its resources
  ///
  /// Call this when you're done with the model to free platform resources
  @override
  Future<void> dispose() async {
    if (_isDisposed) {
      return;
    }

    try {
      // Get JavaScript runner instance
      final runner = js.execuTorchRunner;

      // Dispose model via JavaScript
      await runner.dispose(int.parse(modelId)).toDart;

      _isDisposed = true;
    } catch (e) {
      throw core.ExecuTorchModelException('Failed to dispose model: $e');
    }
  }

  /// Check if this model has been disposed
  @override
  bool get isDisposed => _isDisposed;

  /// Convert Dart TensorData list to JavaScript TensorData array
  List<js.TensorData> _convertTensorsToJS(List<core.TensorData> tensors) =>
      tensors
          .map(
            (tensor) => js.TensorData(
              shape: tensor.shape.map((dim) => (dim ?? 0).toJS).toList().jsify()
                  as JSArray<JSNumber>,
              dataType: _tensorTypeToString(tensor.dataType),
              data: tensor.data.toJSUint8Array(),
              name: tensor.name,
            ),
          )
          .toList();

  /// Convert JavaScript TensorData array to Dart TensorData list
  List<core.TensorData> _convertTensorsFromJS(
    JSArray<js.TensorData> jsTensors,
  ) {
    final dartList = jsTensors.toDart;
    return List.generate(dartList.length, (i) {
      final jsTensor = dartList[i];
      return core.TensorData(
        shape: jsTensor.shape.toDartListInt().cast<int?>(),
        dataType: _stringToTensorType(jsTensor.dataType),
        data: jsTensor.data.toUint8List(),
        name: jsTensor.name,
      );
    });
  }

  /// Convert TensorType enum to string for JavaScript.
  String _tensorTypeToString(core.TensorType type) => switch (type) {
        core.TensorType.float32 => 'float32',
        core.TensorType.float64 => 'float64',
        core.TensorType.int64 => 'int64',
        core.TensorType.int32 => 'int32',
        core.TensorType.int16 => 'int16',
        core.TensorType.int8 => 'int8',
        core.TensorType.uint8 => 'uint8',
        core.TensorType.bool_ => 'bool',
        core.TensorType.uint16 => 'uint16',
        core.TensorType.uint32 => 'uint32',
        core.TensorType.uint64 => 'uint64',
        core.TensorType.float16 => 'float16',
        core.TensorType.bfloat16 => 'bfloat16',
      };

  /// Convert string from JavaScript to TensorType enum.
  ///
  /// Throws [core.ExecuTorchValidationException] if the type string is
  /// unrecognized.
  core.TensorType _stringToTensorType(String type) => switch (type) {
        'float32' => core.TensorType.float32,
        'float64' => core.TensorType.float64,
        'int64' => core.TensorType.int64,
        'int32' => core.TensorType.int32,
        'int16' => core.TensorType.int16,
        'int8' => core.TensorType.int8,
        'uint8' => core.TensorType.uint8,
        'bool' => core.TensorType.bool_,
        'uint16' => core.TensorType.uint16,
        'uint32' => core.TensorType.uint32,
        'uint64' => core.TensorType.uint64,
        'float16' => core.TensorType.float16,
        'bfloat16' => core.TensorType.bfloat16,
        _ => throw core.ExecuTorchValidationException(
            'Unknown tensor type: $type',
          ),
      };
}
