import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';

void main() {
  group('T015: InferenceRequest validation tests', () {
    test('should create valid InferenceRequest with all fields', () {
      // Arrange
      const modelId = 'test-model-123';
      final inputs = [
        TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(1 * 3 * 224 * 224 * 4, 0)),
          name: 'input',
        )
      ];
      final options = <String, Object>{'use_gpu': true, 'precision': 'fp16'};
      const timeoutMs = 5000;
      const requestId = 'request-456';

      // Act
      final request = InferenceRequest(
        modelId: modelId,
        inputs: inputs,
        options: options,
        timeoutMs: timeoutMs,
        requestId: requestId,
      );

      // Assert
      expect(request.modelId, equals(modelId));
      expect(request.inputs, equals(inputs));
      expect(request.options, equals(options));
      expect(request.timeoutMs, equals(timeoutMs));
      expect(request.requestId, equals(requestId));
    });

    test('should create valid InferenceRequest with only required fields', () {
      // Arrange
      const modelId = 'minimal-model';
      final inputs = [
        TensorData(
          shape: [1, 10],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(10 * 4, 0)),
        )
      ];

      // Act
      final request = InferenceRequest(
        modelId: modelId,
        inputs: inputs,
      );

      // Assert
      expect(request.modelId, equals(modelId));
      expect(request.inputs, equals(inputs));
      expect(request.options, isNull);
      expect(request.timeoutMs, isNull);
      expect(request.requestId, isNull);
    });

    test('should handle multiple input tensors', () {
      // Arrange
      const modelId = 'multi-input-model';
      final inputs = [
        TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(1 * 3 * 224 * 224 * 4, 0)),
          name: 'image_input',
        ),
        TensorData(
          shape: [1, 512],
          dataType: TensorType.int32,
          data: Uint8List.fromList(List.filled(512 * 4, 0)),
          name: 'text_input',
        ),
        TensorData(
          shape: [1],
          dataType: TensorType.int8,
          data: Uint8List.fromList([1]),
          name: 'metadata_input',
        ),
      ];

      // Act
      final request = InferenceRequest(
        modelId: modelId,
        inputs: inputs,
        requestId: 'multi-input-request',
      );

      // Assert
      expect(request.inputs.length, equals(3));
      expect(request.inputs[0].name, equals('image_input'));
      expect(request.inputs[1].name, equals('text_input'));
      expect(request.inputs[2].name, equals('metadata_input'));
      expect(request.inputs[0].dataType, equals(TensorType.float32));
      expect(request.inputs[1].dataType, equals(TensorType.int32));
      expect(request.inputs[2].dataType, equals(TensorType.int8));
    });

    test('should handle empty inputs list', () {
      // Arrange
      const modelId = 'no-input-model';
      final inputs = <TensorData>[];

      // Act
      final request = InferenceRequest(
        modelId: modelId,
        inputs: inputs,
      );

      // Assert
      expect(request.modelId, equals(modelId));
      expect(request.inputs, isEmpty);
    });

    test('should handle various option types', () {
      // Arrange
      const modelId = 'configurable-model';
      final inputs = [
        TensorData(
          shape: [1, 100],
          dataType: TensorType.float32,
          data: Uint8List.fromList(List.filled(100 * 4, 0)),
        )
      ];
      final options = <String, Object>{
        'use_gpu': true,
        'batch_size': 8,
        'precision': 'fp16',
        'temperature': 0.7,
        'optimization_level': 2,
        'enable_profiling': false,
      };

      // Act
      final request = InferenceRequest(
        modelId: modelId,
        inputs: inputs,
        options: options,
      );

      // Assert
      expect(request.options?['use_gpu'], equals(true));
      expect(request.options?['batch_size'], equals(8));
      expect(request.options?['precision'], equals('fp16'));
      expect(request.options?['temperature'], equals(0.7));
      expect(request.options?['optimization_level'], equals(2));
      expect(request.options?['enable_profiling'], equals(false));
    });

    test('should handle different timeout values', () {
      // Arrange
      const modelId = 'timeout-test-model';
      final inputs = [
        TensorData(
          shape: [1, 1],
          dataType: TensorType.float32,
          data: Uint8List.fromList([0, 0, 0, 0]),
        )
      ];
      
      final testTimeouts = [100, 1000, 5000, 30000, 60000];

      for (final timeoutMs in testTimeouts) {
        // Act
        final request = InferenceRequest(
          modelId: modelId,
          inputs: inputs,
          timeoutMs: timeoutMs,
          requestId: 'timeout-test-$timeoutMs',
        );

        // Assert
        expect(request.timeoutMs, equals(timeoutMs));
        expect(request.requestId, equals('timeout-test-$timeoutMs'));
      }
    });

    test('should handle long request IDs and model IDs', () {
      // Arrange
      final longModelId = 'model-' + 'a' * 100;
      final longRequestId = 'request-' + 'b' * 200;
      final inputs = [
        TensorData(
          shape: [1, 1],
          dataType: TensorType.uint8,
          data: Uint8List.fromList([255]),
        )
      ];

      // Act
      final request = InferenceRequest(
        modelId: longModelId,
        inputs: inputs,
        requestId: longRequestId,
      );

      // Assert
      expect(request.modelId.length, equals(106)); // 'model-' + 100 'a's
      expect(request.requestId!.length, equals(208)); // 'request-' + 200 'b's
      expect(request.modelId, startsWith('model-'));
      expect(request.requestId, startsWith('request-'));
    });

    test('should handle zero timeout', () {
      // Arrange
      const modelId = 'zero-timeout-model';
      final inputs = [
        TensorData(
          shape: [1, 1],
          dataType: TensorType.int32,
          data: Uint8List.fromList([0, 0, 0, 1]),
        )
      ];
      const timeoutMs = 0;

      // Act
      final request = InferenceRequest(
        modelId: modelId,
        inputs: inputs,
        timeoutMs: timeoutMs,
      );

      // Assert
      expect(request.timeoutMs, equals(0));
      expect(request.modelId, equals(modelId));
    });
  });
}
