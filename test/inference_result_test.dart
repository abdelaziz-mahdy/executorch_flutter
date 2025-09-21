import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';

void main() {
  group('T016: InferenceResult validation tests', () {
    test('should create valid InferenceResult with successful execution', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 42.5;
      const requestId = 'test-request-789';
      final outputs = [
        TensorData(
          shape: [1, 1000],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(1000 * 4, 0)),
          name: 'probabilities',
        )
      ];
      final metadata = <String, Object>{'model_version': '1.2.0', 'backend': 'cpu'};

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        requestId: requestId,
        outputs: outputs,
        metadata: metadata,
      );

      // Assert
      expect(result.status, equals(status));
      expect(result.executionTimeMs, equals(executionTimeMs));
      expect(result.requestId, equals(requestId));
      expect(result.outputs, equals(outputs));
      expect(result.metadata, equals(metadata));
      expect(result.errorMessage, isNull);
    });

    test('should create valid InferenceResult with error status', () {
      // Arrange
      const status = InferenceStatus.error;
      const executionTimeMs = 0.0;
      const requestId = 'failed-request-123';
      const errorMessage = 'Input tensor shape mismatch';

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        requestId: requestId,
        errorMessage: errorMessage,
      );

      // Assert
      expect(result.status, equals(status));
      expect(result.executionTimeMs, equals(executionTimeMs));
      expect(result.requestId, equals(requestId));
      expect(result.errorMessage, equals(errorMessage));
      expect(result.outputs, isNull);
      expect(result.metadata, isNull);
    });

    test('should create valid InferenceResult with timeout status', () {
      // Arrange
      const status = InferenceStatus.timeout;
      const executionTimeMs = 5000.0;
      const requestId = 'timeout-request-456';
      const errorMessage = 'Inference execution exceeded timeout of 5000ms';

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        requestId: requestId,
        errorMessage: errorMessage,
      );

      // Assert
      expect(result.status, equals(InferenceStatus.timeout));
      expect(result.executionTimeMs, equals(5000.0));
      expect(result.errorMessage, contains('timeout'));
    });

    test('should create valid InferenceResult with cancelled status', () {
      // Arrange
      const status = InferenceStatus.cancelled;
      const executionTimeMs = 1250.5;
      const requestId = 'cancelled-request-789';
      const errorMessage = 'Inference was cancelled by user';

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        requestId: requestId,
        errorMessage: errorMessage,
      );

      // Assert
      expect(result.status, equals(InferenceStatus.cancelled));
      expect(result.executionTimeMs, equals(1250.5));
      expect(result.errorMessage, contains('cancelled'));
    });

    test('should handle multiple output tensors', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 75.2;
      final outputs = [
        TensorData(
          shape: [1, 1000],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(1000 * 4, 0)),
          name: 'classification',
        ),
        TensorData(
          shape: [1, 512],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(512 * 4, 0)),
          name: 'features',
        ),
        TensorData(
          shape: [1, 4],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(4 * 4, 0)),
          name: 'bounding_box',
        ),
      ];

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        outputs: outputs,
      );

      // Assert
      expect(result.outputs!.length, equals(3));
      expect(result.outputs![0].name, equals('classification'));
      expect(result.outputs![1].name, equals('features'));
      expect(result.outputs![2].name, equals('bounding_box'));
      expect(result.outputs![0].shape, equals([1, 1000]));
      expect(result.outputs![1].shape, equals([1, 512]));
      expect(result.outputs![2].shape, equals([1, 4]));
    });

    test('should handle empty outputs list', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 15.0;
      final outputs = <TensorData>[];

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        outputs: outputs,
      );

      // Assert
      expect(result.outputs, isEmpty);
      expect(result.status, equals(InferenceStatus.success));
    });

    test('should handle various metadata types', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 33.7;
      final metadata = <String, Object>{
        'model_name': 'efficientnet_b0',
        'backend': 'gpu',
        'precision': 'fp16',
        'memory_used_mb': 128,
        'peak_memory_mb': 156,
        'optimization_enabled': true,
        'inference_latency_ms': 33.7,
        'preprocessing_time_ms': 2.1,
      };

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        metadata: metadata,
      );

      // Assert
      expect(result.metadata?['model_name'], equals('efficientnet_b0'));
      expect(result.metadata?['backend'], equals('gpu'));
      expect(result.metadata?['precision'], equals('fp16'));
      expect(result.metadata?['memory_used_mb'], equals(128));
      expect(result.metadata?['peak_memory_mb'], equals(156));
      expect(result.metadata?['optimization_enabled'], equals(true));
      expect(result.metadata?['inference_latency_ms'], equals(33.7));
      expect(result.metadata?['preprocessing_time_ms'], equals(2.1));
    });

    test('should handle very fast execution times', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 0.1;
      final outputs = [
        TensorData(
          shape: [1, 1],
          dataType: TensorType.float32,
          data: Uint8List.fromList([0, 0, 0, 0]),
        )
      ];

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        outputs: outputs,
      );

      // Assert
      expect(result.executionTimeMs, equals(0.1));
      expect(result.status, equals(InferenceStatus.success));
    });

    test('should handle very slow execution times', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 30000.0; // 30 seconds
      const requestId = 'slow-model-request';

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        requestId: requestId,
      );

      // Assert
      expect(result.executionTimeMs, equals(30000.0));
      expect(result.requestId, equals(requestId));
    });

    test('should handle result without optional fields', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 25.0;

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
      );

      // Assert
      expect(result.status, equals(status));
      expect(result.executionTimeMs, equals(executionTimeMs));
      expect(result.requestId, isNull);
      expect(result.outputs, isNull);
      expect(result.errorMessage, isNull);
      expect(result.metadata, isNull);
    });

    test('should handle different tensor data types in outputs', () {
      // Arrange
      const status = InferenceStatus.success;
      const executionTimeMs = 55.3;
      final outputs = [
        TensorData(
          shape: [1, 10],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(10 * 4, 0)),
          name: 'float_output',
        ),
        TensorData(
          shape: [1, 100],
          dataType: TensorType.int32,
          data: Uint8List.fromList(List.filled(100 * 4, 0)),
          name: 'int_output',
        ),
        TensorData(
          shape: [1, 50],
          dataType: TensorType.uint8,
          data: Uint8List.fromList(List.filled(50, 255)),
          name: 'byte_output',
        ),
        TensorData(
          shape: [1, 25],
          dataType: TensorType.int8,
          data: Uint8List.fromList(List.filled(25, 127)),
          name: 'signed_byte_output',
        ),
      ];

      // Act
      final result = InferenceResult(
        status: status,
        executionTimeMs: executionTimeMs,
        outputs: outputs,
      );

      // Assert
      expect(result.outputs!.length, equals(4));
      expect(result.outputs![0].dataType, equals(TensorType.float32));
      expect(result.outputs![1].dataType, equals(TensorType.int32));
      expect(result.outputs![2].dataType, equals(TensorType.uint8));
      expect(result.outputs![3].dataType, equals(TensorType.int8));
    });
  });
}
