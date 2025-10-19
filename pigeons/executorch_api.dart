// Pigeon API specification for Flutter ExecuTorch package
// This file now only defines shared types (TensorData, TensorType)
// All communication happens via FFI, not platform channels
// Updated: 2025-10-18 - Migrated to FFI architecture

import 'package:pigeon/pigeon.dart';

@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  dartOptions: DartOptions(),
  dartPackageName: 'executorch_flutter',
))

/// Tensor data type enumeration
/// Maps to ETFlutterDataType in C wrapper
enum TensorType {
  float32,  // 32-bit IEEE 754 floating point
  int8,     // 8-bit signed integer
  int32,    // 32-bit signed integer
  uint8,    // 8-bit unsigned integer
}

/// Tensor data for input/output
/// This is the primary data structure shared between Dart and C (via FFI)
/// It matches ETFlutterTensorData in the C wrapper
class TensorData {
  TensorData({
    required this.shape,
    required this.dataType,
    required this.data,
    this.name,
  });

  /// Tensor shape (e.g., [1, 3, 224, 224] for NCHW image)
  /// Max 8 dimensions (ET_FLUTTER_MAX_TENSOR_DIMS)
  List<int?> shape; // Pigeon requires nullable generics

  /// Element data type
  TensorType dataType;

  /// Raw tensor data as bytes
  /// Length must match shape * dtype size
  Uint8List data;

  /// Optional tensor name (max 64 chars)
  String? name;
}

// Note: ExecutorchHostApi and other platform channel interfaces removed
// All communication now happens via:
// - C wrapper: src/c_wrapper/executorch_flutter_wrapper.h
// - FFI bridge: lib/src/ffi/executorch_ffi_bridge.dart
// - Dart API: lib/src/executorch_model.dart
