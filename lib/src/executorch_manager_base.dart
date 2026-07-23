/// Base implementation of ExecutorchManager with shared logic
///
/// This class contains the common implementation for both native and web
/// platforms, reducing code duplication. Platform-specific managers
/// extend this class and override only the methods that differ.
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:meta/meta.dart';

import 'executorch_errors.dart';
import 'executorch_inference.dart';
import 'executorch_model.dart';
import 'types.dart';

/// Base implementation of ExecutorchManager with shared logic
///
/// Contains common implementations for model caching, tensor creation,
/// and resource management. Platform-specific managers extend this class.
abstract class ExecutorchManagerBase implements ExecutorchManager {
  /// Cache of loaded models by model ID (accessible to subclasses)
  @protected
  final Map<String, ExecuTorchModel> loadedModelsMap = {};

  /// Whether the manager has been initialized (accessible to subclasses)
  @protected
  bool initialized = false;

  @override
  ExecuTorchModel? getLoadedModel(String modelId) => loadedModelsMap[modelId];

  @override
  List<ExecuTorchModel> getLoadedModels() =>
      List.unmodifiable(loadedModelsMap.values);

  @override
  Future<List<String>> getLoadedModelIds() async {
    ensureInitialized();
    return loadedModelsMap.keys.toList();
  }

  @override
  Future<void> disposeModel(String modelId) async {
    ensureInitialized();

    final model = loadedModelsMap.remove(modelId);
    if (model != null) {
      await model.dispose();
    }
  }

  @override
  Future<void> disposeAllModels() async {
    ensureInitialized();

    final modelIds = List<String>.from(loadedModelsMap.keys);
    for (final modelId in modelIds) {
      await disposeModel(modelId);
    }
  }

  @override
  TensorData createTensorData({
    required List<int> shape,
    required TensorType dataType,
    required List<num> data,
    String? name,
  }) {
    // Convert numeric data to bytes based on data type
    final bytes = convertNumericDataToBytes(data, dataType);

    final tensor = TensorData(
      shape: shape.cast<int?>(),
      dataType: dataType,
      data: bytes,
      name: name,
    );

    return tensor;
  }

  @override
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromAsset(assetPath);
      loadedModelsMap[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from asset $assetPath: $e',
        'asset_path: $assetPath, error: ${e.toString()}',
      );
    }
  }

  @override
  Future<ExecuTorchModel> loadModelFromBytes(Uint8List modelBytes) async {
    ensureInitialized();

    try {
      final model = await ExecuTorchModel.loadFromBytes(modelBytes);
      loadedModelsMap[model.modelId] = model;
      return model;
    } catch (e) {
      if (e is ExecuTorchException) rethrow;
      throw ExecuTorchModelException(
        'Failed to load model from bytes: $e',
        'bytes_length: ${modelBytes.length}, error: ${e.toString()}',
      );
    }
  }

  /// Ensure the manager has been initialized
  ///
  /// Throws [ExecuTorchPlatformException] if not initialized.
  void ensureInitialized() {
    if (!initialized) {
      throw const ExecuTorchPlatformException(
        'ExecutorchManager not initialized. Call initialize() first.',
      );
    }
  }

  /// Utility method to convert numeric data to bytes
  ///
  /// Converts a list of numeric values to the appropriate byte representation
  /// based on the specified [dataType].
  static Uint8List convertNumericDataToBytes(
    List<num> data,
    TensorType dataType,
  ) {
    switch (dataType) {
      case TensorType.float32:
        final float32List = Float32List.fromList(
          data.map((e) => e.toDouble()).toList(),
        );
        return float32List.buffer.asUint8List();

      case TensorType.float64:
        final float64List = Float64List.fromList(
          data.map((e) => e.toDouble()).toList(),
        );
        return float64List.buffer.asUint8List();

      case TensorType.int64:
        final int64List = Int64List.fromList(
          data.map((e) => e.toInt()).toList(),
        );
        return int64List.buffer.asUint8List();

      case TensorType.int32:
        final int32List = Int32List.fromList(
          data.map((e) => e.toInt()).toList(),
        );
        return int32List.buffer.asUint8List();

      case TensorType.int16:
        final int16List = Int16List.fromList(
          data.map((e) => e.toInt()).toList(),
        );
        return int16List.buffer.asUint8List();

      case TensorType.int8:
        return Uint8List.fromList(
          data.map((e) => e.toInt().clamp(-128, 127) + 128).toList(),
        );

      case TensorType.uint8:
        return Uint8List.fromList(
          data.map((e) => e.toInt().clamp(0, 255)).toList(),
        );

      case TensorType.bool_:
        return Uint8List.fromList(
          data.map((e) => (e != 0 ? 1 : 0)).toList(),
        );

      case TensorType.uint16:
        final uint16List = Uint16List.fromList(
          data.map((e) => e.toInt().clamp(0, 65535)).toList(),
        );
        return uint16List.buffer.asUint8List();

      case TensorType.uint32:
        final uint32List = Uint32List.fromList(
          data.map((e) => e.toInt().clamp(0, 0xFFFFFFFF)).toList(),
        );
        return uint32List.buffer.asUint8List();

      case TensorType.uint64:
        final uint64List = Uint64List.fromList(
          data.map((e) => e.toInt().clamp(0, 0xFFFFFFFFFFFFFFFF)).toList(),
        );
        return uint64List.buffer.asUint8List();

      case TensorType.float16:
        // Convert float32 → float16 via IEEE 754 bit processing
        return _encodeFloat16(data);

      case TensorType.bfloat16:
        // Convert float32 → bfloat16 by truncating to upper 16 bits
        return _encodeBfloat16(data);
    }
  }

  /// Encode a list of numeric values as bfloat16 bytes.
  ///
  /// bfloat16 is simply the upper 16 bits of an IEEE 754 float32:
  /// - Sign bit: 1 bit
  /// - Exponent: 8 bits (same bias as float32)
  /// - Mantissa: 7 bits (truncated from float32's 23)
  static Uint8List _encodeBfloat16(List<num> data) {
    final float32List = Float32List.fromList(
      data.map((e) => e.toDouble()).toList(),
    );
    final view = float32List.buffer.asUint32List();
    final result = Uint8List(data.length * 2);
    for (int i = 0; i < data.length; i++) {
      // Shift right by 16 to keep upper 16 bits
      final bf16 = view[i] >>> 16;
      // Store in little-endian byte order
      result[i * 2] = bf16 & 0xFF;
      result[i * 2 + 1] = (bf16 >> 8) & 0xFF;
    }
    return result;
  }

  /// Encode a list of numeric values as float16 (IEEE 754 half precision) bytes.
  ///
  /// Converts float32 → float16 using bitwise operations:
  /// - Sign bit: 1 bit
  /// - Exponent: 5 bits (bias 15, vs float32 bias 127)
  /// - Mantissa: 10 bits (truncated/rounded from float32's 23)
  static Uint8List _encodeFloat16(List<num> data) {
    final float32List = Float32List.fromList(
      data.map((e) => e.toDouble()).toList(),
    );
    final view = float32List.buffer.asUint32List();
    final result = Uint8List(data.length * 2);
    for (int i = 0; i < data.length; i++) {
      final f32 = view[i];
      final f16 = _float32ToFloat16(f32);
      // Store in little-endian byte order
      result[i * 2] = f16 & 0xFF;
      result[i * 2 + 1] = (f16 >> 8) & 0xFF;
    }
    return result;
  }

  /// Convert a single IEEE 754 float32 bit pattern to float16 bit pattern.
  static int _float32ToFloat16(int f32) {
    // Extract float32 components
    final sign32 = (f32 >>> 31) & 0x1;
    final exp32 = (f32 >>> 23) & 0xFF;
    final mant32 = f32 & 0x7FFFFF;

    final sign16 = sign32 << 15;

    if (exp32 == 0) {
      // Zero or denormal → zero in float16
      return sign16;
    }

    if (exp32 == 0xFF) {
      // Infinity or NaN → infinity (or NaN) in float16
      return sign16 | 0x7C00 | (mant32 != 0 ? 0x0200 : 0);
    }

    // Convert exponent bias: float32 uses bias 127, float16 uses bias 15
    final exp16 = exp32 - 127 + 15;

    if (exp16 >= 31) {
      // Overflow → infinity
      return sign16 | 0x7C00;
    }

    if (exp16 <= 0) {
      if (exp16 < -10) {
        // Underflow → zero
        return sign16;
      }
      // Denormal: shift mantissa to fit denormal representation
      final mantShift = mant32 | 0x7FFFFF; // Add implicit leading 1
      final shiftedMant = mantShift >> (14 - exp16);
      return sign16 | shiftedMant & 0x3FF;
    }

    // Normal: truncate mantissa from 23 to 10 bits with rounding
    final mant16 = mant32 >> 13;

    return sign16 | (exp16 << 10) | mant16;
  }
}
