/// JavaScript interop bindings for ExecuTorch Wasm
///
/// This file provides Dart bindings to the JavaScript ExecuTorchRunner wrapper
/// using dart:js_interop for type-safe communication with the Wasm module.
library;

import 'dart:js_interop';
import 'dart:typed_data';

/// JavaScript ExecuTorchRunner class binding
///
/// Maps to window.ExecuTorchRunner instance created by executorch_wrapper.js
@JS('window.ExecuTorchRunner')
extension type ExecuTorchRunner._(JSObject _) implements JSObject {
  /// Get the global ExecuTorchRunner instance
  external static ExecuTorchRunner get instance;

  /// Initialize the Wasm module (call once before using)
  external JSPromise<JSAny?> initialize();

  /// Enable or disable debug logging
  ///
  /// When enabled, ExecuTorch operations will log to browser console
  external void setDebugLogging(bool enabled);

  /// Load a model from bytes
  ///
  /// Returns a Promise that resolves to a ModelLoadResult
  external JSPromise<ModelLoadResult> loadModel(JSUint8Array modelBytes);

  /// Run inference on a loaded model
  ///
  /// Returns a Promise that resolves to an array of TensorData objects
  external JSPromise<JSArray<TensorData>> forward(
    int modelId,
    JSArray<TensorData> inputs,
  );

  /// Dispose a loaded model and free resources
  external JSPromise<JSAny?> dispose(int modelId);

  /// Check if a model is loaded
  external bool isModelLoaded(int modelId);

  /// Get metadata for a loaded model
  external ModelMetadata getModelMetadata(int modelId);

  /// Get list of loaded model IDs
  external JSArray<JSNumber> getLoadedModelIds();
}

/// JavaScript ModelLoadResult type
///
/// Result from loadModel() call
@JS()
@anonymous
extension type ModelLoadResult._(JSObject _) implements JSObject {
  external factory ModelLoadResult({
    required int modelId,
    required JSArray<JSArray<JSNumber>> inputShapes,
    required JSArray<JSArray<JSNumber>> outputShapes,
  });

  external int get modelId;
  external JSArray<JSArray<JSNumber>> get inputShapes;
  external JSArray<JSArray<JSNumber>> get outputShapes;
}

/// JavaScript ModelMetadata type
///
/// Metadata for a loaded model
@JS()
@anonymous
extension type ModelMetadata._(JSObject _) implements JSObject {
  external JSArray<JSArray<JSNumber>> get inputShapes;
  external JSArray<JSArray<JSNumber>> get outputShapes;
}

/// JavaScript TensorData type
///
/// Represents an input or output tensor
@JS()
@anonymous
extension type TensorData._(JSObject _) implements JSObject {
  external factory TensorData({
    required JSArray<JSNumber> shape,
    required String dataType,
    required JSUint8Array data,
    String? name,
  });

  external JSArray<JSNumber> get shape;
  external String get dataType;
  external JSUint8Array get data;
  external String? get name;
}

/// Utility extensions for type conversions

/// Extension for converting JS number arrays to Dart int lists
extension JSArrayExtensions on JSArray<JSNumber> {
  /// Convert `JSArray<JSNumber>` to Dart `List<int>`
  List<int> toDartListInt() {
    final length = toDart.length;
    final result = <int>[];
    for (var i = 0; i < length; i++) {
      result.add(toDart[i].toDartInt);
    }
    return result;
  }
}

/// Extension for converting 2D JS number arrays to Dart 2D int lists
extension JSArray2DExtensions on JSArray<JSArray<JSNumber>> {
  /// Convert `JSArray<JSArray<JSNumber>>` to Dart `List<List<int>>`
  List<List<int>> toDartList2DInt() {
    final length = toDart.length;
    final result = <List<int>>[];
    for (var i = 0; i < length; i++) {
      result.add(toDart[i].toDartListInt());
    }
    return result;
  }
}

/// Extension for converting Dart Uint8List to JS Uint8Array
extension Uint8ListToJSExtension on Uint8List {
  /// Convert Dart Uint8List to JSUint8Array
  JSUint8Array toJSUint8Array() => toJS;
}

/// Extension for converting JS Uint8Array to Dart Uint8List
extension JSUint8ArrayToDartExtension on JSUint8Array {
  /// Convert JSUint8Array to Dart Uint8List
  Uint8List toUint8List() => toDart;
}

