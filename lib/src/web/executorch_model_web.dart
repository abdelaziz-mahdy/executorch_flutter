/// Web platform implementation of ExecuTorchModel
///
/// Uses dart:js_interop to communicate with the JavaScript wrapper around
/// the ExecuTorch WebAssembly module.
library;

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import '../executorch_errors.dart';
import '../executorch_model.dart';
import '../generated/executorch_api.dart';
import 'js_interop.dart' as js;
import 'wasm_module_loader.dart';

/// Web platform implementation of ExecuTorchModel
///
/// This implementation:
/// - Loads models into the Wasm virtual filesystem
/// - Runs inference via JavaScript interop
/// - Manages model lifecycle using JavaScript wrapper
class ExecuTorchModelWeb implements ExecuTorchModel {
  ExecuTorchModelWeb._({
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

  /// Load a model from bytes
  ///
  /// [modelBytes] - Model file bytes (.pte format)
  ///
  /// Returns an [ExecuTorchModelWeb] instance ready for inference
  ///
  /// Throws [ExecuTorchModelException] if loading fails
  static Future<ExecuTorchModelWeb> load(Uint8List modelBytes) async =>
      loadFromBytes(modelBytes);

  /// Load a model from bytes (alias for [load])
  ///
  /// [modelBytes] - Model file bytes (.pte format)
  ///
  /// Returns an [ExecuTorchModelWeb] instance ready for inference
  ///
  /// Throws [ExecuTorchModelException] if loading fails
  static Future<ExecuTorchModelWeb> loadFromBytes(Uint8List modelBytes) async {
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

      return ExecuTorchModelWeb._(
        modelId: modelId,
        inputShapes: inputShapes,
        outputShapes: outputShapes,
      );
    } catch (e) {
      throw ExecuTorchModelException(
        'Failed to load model: $e',
      );
    }
  }

  /// Load a model from Flutter asset bundle
  ///
  /// [assetPath] - Path to the model in the asset bundle
  ///
  /// Returns an [ExecuTorchModelWeb] instance ready for inference
  ///
  /// Throws [ExecuTorchModelException] if asset is not found or loading fails
  static Future<ExecuTorchModelWeb> loadFromAsset(String assetPath) async {
    try {
      final byteData = await rootBundle.load(assetPath);
      return loadFromBytes(byteData.buffer.asUint8List());
    } catch (e) {
      throw ExecuTorchModelException(
        'Failed to load model from asset $assetPath: $e',
      );
    }
  }

  /// Execute inference on the model
  ///
  /// [inputs] - List of input tensors matching model's input specification
  ///
  /// Returns list of output tensors from the model
  ///
  /// Throws [ExecuTorchInferenceException] if inference fails
@override
  Future<List<TensorData>> forward(List<TensorData> inputs) async {
    if (_isDisposed) {
      throw const ExecuTorchModelException(
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
      throw ExecuTorchInferenceException(
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
      throw ExecuTorchModelException(
        'Failed to dispose model: $e',
      );
    }
  }

  /// Check if this model has been disposed
@override
  bool get isDisposed => _isDisposed;

  /// Convert Dart TensorData list to JavaScript TensorData array
  List<js.TensorData> _convertTensorsToJS(List<TensorData> tensors) =>
      tensors.map((tensor) => js.TensorData(
        shape: tensor.shape
            .map((dim) => (dim ?? 0).toJS)
            .toList()
            .jsify() as JSArray<JSNumber>,
        dataType: _tensorTypeToString(tensor.dataType),
        data: tensor.data.toJSUint8Array(),
        name: tensor.name,
      )).toList();

  /// Convert JavaScript TensorData array to Dart TensorData list
  List<TensorData> _convertTensorsFromJS(JSArray<js.TensorData> jsTensors) {
    final dartList = jsTensors.toDart;
    return List.generate(dartList.length, (i) {
      final jsTensor = dartList[i];
      return TensorData(
        shape: jsTensor.shape.toDartListInt().cast<int?>(),
        dataType: _stringToTensorType(jsTensor.dataType),
        data: jsTensor.data.toUint8List(),
        name: jsTensor.name,
      );
    });
  }

  /// Convert TensorType enum to string for JavaScript
  String _tensorTypeToString(TensorType type) {
    switch (type) {
      case TensorType.float32:
        return 'float32';
      case TensorType.int32:
        return 'int32';
      case TensorType.int8:
        return 'int8';
      case TensorType.uint8:
        return 'uint8';
    }
  }

  /// Convert string from JavaScript to TensorType enum
  TensorType _stringToTensorType(String type) {
    switch (type) {
      case 'float32':
        return TensorType.float32;
      case 'int32':
        return TensorType.int32;
      case 'int8':
        return TensorType.int8;
      case 'uint8':
        return TensorType.uint8;
      default:
        throw ExecuTorchValidationException(
          'Unknown tensor type: $type',
        );
    }
  }
}
