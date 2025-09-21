/// ExecuTorch Flutter package type definitions and utilities
///
/// This file provides high-level wrapper classes and utilities for the
/// ExecuTorch Flutter package, built on top of the generated Pigeon types.
library executorch_types;

export '../src/generated/executorch_api.dart';

import 'dart:typed_data';
import '../src/generated/executorch_api.dart' as pigeon;

/// High-level wrapper for TensorData with validation and utilities
class TensorDataWrapper {
  const TensorDataWrapper({
    required this.shape,
    required this.dataType,
    required this.data,
    this.name,
  });

  final List<int> shape;
  final pigeon.TensorType dataType;
  final Uint8List data;
  final String? name;

  /// Create TensorDataWrapper from Pigeon TensorData
  factory TensorDataWrapper.fromPigeon(pigeon.TensorData pigeonData) {
    return TensorDataWrapper(
      shape: pigeonData.shape.cast<int>(),
      dataType: pigeonData.dataType,
      data: pigeonData.data,
      name: pigeonData.name,
    );
  }

  /// Convert to Pigeon TensorData for platform communication
  pigeon.TensorData toPigeon() {
    return pigeon.TensorData(
      shape: shape.cast<int?>(),
      dataType: dataType,
      data: data,
      name: name,
    );
  }

  /// Calculate the expected data size in bytes based on shape and data type
  int get expectedSizeBytes {
    final elementCount = shape.fold<int>(1, (total, dim) => total * dim.abs());
    return elementCount * _getBytesPerElement(dataType);
  }

  /// Validate that the data size matches the shape and data type
  bool get isValid {
    return data.length == expectedSizeBytes;
  }

  /// Get human-readable description of the tensor
  String get description {
    final shapeStr = shape.join('×');
    final sizeStr = '${data.length} bytes';
    final nameStr = name != null ? '$name: ' : '';
    return '${nameStr}TensorData($shapeStr, ${dataType.name}, $sizeStr)';
  }

  /// Create a copy with modified properties
  TensorDataWrapper copyWith({
    List<int>? shape,
    pigeon.TensorType? dataType,
    Uint8List? data,
    String? name,
  }) {
    return TensorDataWrapper(
      shape: shape ?? this.shape,
      dataType: dataType ?? this.dataType,
      data: data ?? this.data,
      name: name ?? this.name,
    );
  }

  static int _getBytesPerElement(pigeon.TensorType type) {
    switch (type) {
      case pigeon.TensorType.float32:
      case pigeon.TensorType.int32:
        return 4;
      case pigeon.TensorType.int8:
      case pigeon.TensorType.uint8:
        return 1;
    }
  }

  @override
  String toString() => description;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TensorDataWrapper &&
        other.shape.length == shape.length &&
        _listEquals(other.shape, shape) &&
        other.dataType == dataType &&
        _uint8ListEquals(other.data, data) &&
        other.name == name;
  }

  @override
  int get hashCode => Object.hash(
        Object.hashAll(shape),
        dataType,
        Object.hashAll(data),
        name,
      );

  static bool _listEquals<T>(List<T> a, List<T> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static bool _uint8ListEquals(Uint8List a, Uint8List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// High-level wrapper for ModelMetadata with validation and utilities
class ModelMetadataWrapper {
  const ModelMetadataWrapper({
    required this.modelName,
    required this.version,
    required this.inputSpecs,
    required this.outputSpecs,
    required this.estimatedMemoryMB,
    this.properties,
  });

  final String modelName;
  final String version;
  final List<pigeon.TensorSpec> inputSpecs;
  final List<pigeon.TensorSpec> outputSpecs;
  final int estimatedMemoryMB;
  final Map<String, Object>? properties;

  /// Create ModelMetadataWrapper from Pigeon ModelMetadata
  factory ModelMetadataWrapper.fromPigeon(pigeon.ModelMetadata pigeonData) {
    return ModelMetadataWrapper(
      modelName: pigeonData.modelName,
      version: pigeonData.version,
      inputSpecs: pigeonData.inputSpecs.whereType<pigeon.TensorSpec>().toList(),
      outputSpecs: pigeonData.outputSpecs.whereType<pigeon.TensorSpec>().toList(),
      estimatedMemoryMB: pigeonData.estimatedMemoryMB,
      properties: pigeonData.properties?.cast<String, Object>(),
    );
  }

  /// Convert to Pigeon ModelMetadata for platform communication
  pigeon.ModelMetadata toPigeon() {
    return pigeon.ModelMetadata(
      modelName: modelName,
      version: version,
      inputSpecs: inputSpecs.cast<pigeon.TensorSpec?>(),
      outputSpecs: outputSpecs.cast<pigeon.TensorSpec?>(),
      estimatedMemoryMB: estimatedMemoryMB,
      properties: properties?.cast<String?, Object?>(),
    );
  }

  /// Get human-readable description of the model
  String get description {
    return 'ModelMetadata($modelName v$version, '
        '${inputSpecs.length} inputs, ${outputSpecs.length} outputs, '
        '${estimatedMemoryMB}MB)';
  }

  /// Check if model supports dynamic batch size
  bool get supportsDynamicBatch {
    return inputSpecs.any((spec) => spec.shape.contains(-1));
  }

  /// Get the first input specification (most common case)
  pigeon.TensorSpec? get primaryInput {
    return inputSpecs.isNotEmpty ? inputSpecs.first : null;
  }

  /// Get the first output specification (most common case)
  pigeon.TensorSpec? get primaryOutput {
    return outputSpecs.isNotEmpty ? outputSpecs.first : null;
  }

  @override
  String toString() => description;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ModelMetadataWrapper &&
        other.modelName == modelName &&
        other.version == version &&
        _listEquals(other.inputSpecs, inputSpecs) &&
        _listEquals(other.outputSpecs, outputSpecs) &&
        other.estimatedMemoryMB == estimatedMemoryMB &&
        _mapEquals(other.properties, properties);
  }

  @override
  int get hashCode => Object.hash(
        modelName,
        version,
        Object.hashAll(inputSpecs),
        Object.hashAll(outputSpecs),
        estimatedMemoryMB,
        properties,
      );

  static bool _listEquals<T>(List<T> a, List<T> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static bool _mapEquals<K, V>(Map<K, V>? a, Map<K, V>? b) {
    if (a == null && b == null) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (final key in a.keys) {
      if (!b.containsKey(key) || a[key] != b[key]) return false;
    }
    return true;
  }
}

/// High-level wrapper for InferenceRequest with validation and utilities
class InferenceRequestWrapper {
  const InferenceRequestWrapper({
    required this.modelId,
    required this.inputs,
    this.options,
    this.timeoutMs,
    this.requestId,
  });

  final String modelId;
  final List<TensorDataWrapper> inputs;
  final Map<String, Object>? options;
  final int? timeoutMs;
  final String? requestId;

  /// Create InferenceRequestWrapper from Pigeon InferenceRequest
  factory InferenceRequestWrapper.fromPigeon(pigeon.InferenceRequest pigeonData) {
    return InferenceRequestWrapper(
      modelId: pigeonData.modelId,
      inputs: pigeonData.inputs
          .whereType<pigeon.TensorData>()
          .map(TensorDataWrapper.fromPigeon)
          .toList(),
      options: pigeonData.options?.cast<String, Object>(),
      timeoutMs: pigeonData.timeoutMs,
      requestId: pigeonData.requestId,
    );
  }

  /// Convert to Pigeon InferenceRequest for platform communication
  pigeon.InferenceRequest toPigeon() {
    return pigeon.InferenceRequest(
      modelId: modelId,
      inputs: inputs.map((input) => input.toPigeon()).toList().cast<pigeon.TensorData?>(),
      options: options?.cast<String?, Object?>(),
      timeoutMs: timeoutMs,
      requestId: requestId,
    );
  }

  /// Validate that all inputs are valid
  bool get isValid {
    return inputs.every((input) => input.isValid);
  }

  /// Get human-readable description of the request
  String get description {
    final reqIdStr = requestId != null ? ' ($requestId)' : '';
    final timeoutStr = timeoutMs != null ? ', ${timeoutMs}ms timeout' : '';
    return 'InferenceRequest$reqIdStr: ${inputs.length} inputs for $modelId$timeoutStr';
  }

  @override
  String toString() => description;
}

/// High-level wrapper for InferenceResult with validation and utilities
class InferenceResultWrapper {
  const InferenceResultWrapper({
    required this.status,
    required this.executionTimeMs,
    this.requestId,
    this.outputs,
    this.errorMessage,
    this.metadata,
  });

  final pigeon.InferenceStatus status;
  final double executionTimeMs;
  final String? requestId;
  final List<TensorDataWrapper>? outputs;
  final String? errorMessage;
  final Map<String, Object>? metadata;

  /// Create InferenceResultWrapper from Pigeon InferenceResult
  factory InferenceResultWrapper.fromPigeon(pigeon.InferenceResult pigeonData) {
    return InferenceResultWrapper(
      status: pigeonData.status,
      executionTimeMs: pigeonData.executionTimeMs,
      requestId: pigeonData.requestId,
      outputs: pigeonData.outputs
          ?.whereType<pigeon.TensorData>()
          .map(TensorDataWrapper.fromPigeon)
          .toList(),
      errorMessage: pigeonData.errorMessage,
      metadata: pigeonData.metadata?.cast<String, Object>(),
    );
  }

  /// Convert to Pigeon InferenceResult for platform communication
  pigeon.InferenceResult toPigeon() {
    return pigeon.InferenceResult(
      status: status,
      executionTimeMs: executionTimeMs,
      requestId: requestId,
      outputs: outputs?.map((output) => output.toPigeon()).toList().cast<pigeon.TensorData?>(),
      errorMessage: errorMessage,
      metadata: metadata?.cast<String?, Object?>(),
    );
  }

  /// Check if the inference was successful
  bool get isSuccess => status == pigeon.InferenceStatus.success;

  /// Check if the inference failed
  bool get isError => status == pigeon.InferenceStatus.error;

  /// Check if the inference timed out
  bool get isTimeout => status == pigeon.InferenceStatus.timeout;

  /// Check if the inference was cancelled
  bool get isCancelled => status == pigeon.InferenceStatus.cancelled;

  /// Get human-readable description of the result
  String get description {
    final reqIdStr = requestId != null ? ' ($requestId)' : '';
    final timeStr = '${executionTimeMs.toStringAsFixed(1)}ms';

    switch (status) {
      case pigeon.InferenceStatus.success:
        final outputCount = outputs?.length ?? 0;
        return 'InferenceResult$reqIdStr: SUCCESS in $timeStr, $outputCount outputs';
      case pigeon.InferenceStatus.error:
        return 'InferenceResult$reqIdStr: ERROR in $timeStr - ${errorMessage ?? "Unknown error"}';
      case pigeon.InferenceStatus.timeout:
        return 'InferenceResult$reqIdStr: TIMEOUT after $timeStr';
      case pigeon.InferenceStatus.cancelled:
        return 'InferenceResult$reqIdStr: CANCELLED after $timeStr';
    }
  }

  @override
  String toString() => description;
}