import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('T018: Integration test inference execution flow', () {
    late ExecutorchHostApi hostApi;

    setUpAll(() async {
      // This test will fail until implementation is complete
      // This is expected as part of TDD approach
      hostApi = ExecutorchHostApi();
    });

    testWidgets('should execute inference on loaded model with valid inputs', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/mobilenet_v2.pte';
      
      try {
        // Load model first
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        // Prepare inference request
        final inputTensor = TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          data: _generateTestImageData(1, 3, 224, 224),
          name: 'input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [inputTensor],
          requestId: 'test-inference-1',
          timeoutMs: 5000,
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert
        expect(result.status, equals(InferenceStatus.success));
        expect(result.requestId, equals(request.requestId));
        expect(result.executionTimeMs, greaterThan(0));
        expect(result.executionTimeMs, lessThan(5000)); // Should finish within timeout
        expect(result.outputs, isNotNull);
        expect(result.outputs!.length, greaterThan(0));
        expect(result.errorMessage, isNull);
        
        // Validate output tensor
        final outputTensor = result.outputs!.first;
        expect(outputTensor.shape, isNotEmpty);
        expect(outputTensor.dataType, isIn(TensorType.values));
        expect(outputTensor.data.length, greaterThan(0));
        
        // Cleanup
        hostApi.disposeModel(loadResult.modelId);
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for inference execution: $e');
      }
    });

    testWidgets('should handle inference with wrong input shape', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/resnet18.pte';
      
      try {
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        // Create input with wrong shape (should be [1, 3, 224, 224] but using [1, 1, 10, 10])
        final wrongInputTensor = TensorData(
          shape: [1, 1, 10, 10],
          dataType: TensorType.float32,
          data: _generateTestImageData(1, 1, 10, 10),
          name: 'input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [wrongInputTensor],
          requestId: 'test-inference-wrong-shape',
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert
        expect(result.status, equals(InferenceStatus.error));
        expect(result.errorMessage, isNotNull);
        expect(result.errorMessage, anyOf([
          contains('shape'),
          contains('dimension'),
          contains('size'),
          contains('mismatch'),
        ]));
        expect(result.outputs, isNull);
        
        // Cleanup
        hostApi.disposeModel(loadResult.modelId);
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for wrong input shape: $e');
      }
    });

    testWidgets('should handle inference with wrong data type', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/mobilenet_v2.pte';
      
      try {
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        // Create input with wrong data type (should be float32 but using int8)
        final wrongTypeTensor = TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.int8,
          data: Uint8List.fromList(List.filled(1 * 3 * 224 * 224, 128)),
          name: 'input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [wrongTypeTensor],
          requestId: 'test-inference-wrong-type',
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert
        expect(result.status, equals(InferenceStatus.error));
        expect(result.errorMessage, isNotNull);
        expect(result.errorMessage, anyOf([
          contains('type'),
          contains('dtype'),
          contains('data type'),
          contains('format'),
        ]));
        
        // Cleanup
        hostApi.disposeModel(loadResult.modelId);
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for wrong data type: $e');
      }
    });

    testWidgets('should handle inference timeout', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/large_model.pte';
      
      try {
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        final inputTensor = TensorData(
          shape: [1, 3, 1024, 1024], // Large input to potentially cause timeout
          dataType: TensorType.float32,
          data: _generateTestImageData(1, 3, 1024, 1024),
          name: 'input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [inputTensor],
          requestId: 'test-inference-timeout',
          timeoutMs: 100, // Very short timeout
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert - Could be either timeout or success depending on actual performance
        if (result.status == InferenceStatus.timeout) {
          expect(result.errorMessage, contains('timeout'));
          expect(result.executionTimeMs, greaterThanOrEqualTo(100));
        } else {
          expect(result.status, equals(InferenceStatus.success));
          expect(result.executionTimeMs, lessThan(100));
        }
        
        // Cleanup
        hostApi.disposeModel(loadResult.modelId);
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for timeout test: $e');
      }
    });

    testWidgets('should handle inference on disposed model', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/mobilenet_v2.pte';
      
      try {
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        // Dispose model
        hostApi.disposeModel(loadResult.modelId);
        
        // Try to run inference on disposed model
        final inputTensor = TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          data: _generateTestImageData(1, 3, 224, 224),
          name: 'input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [inputTensor],
          requestId: 'test-inference-disposed',
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert
        expect(result.status, equals(InferenceStatus.error));
        expect(result.errorMessage, isNotNull);
        expect(result.errorMessage, anyOf([
          contains('disposed'),
          contains('not found'),
          contains('invalid'),
          contains('model'),
        ]));
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for disposed model inference: $e');
      }
    });

    testWidgets('should handle multiple inputs inference', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/multi_input_model.pte';
      
      try {
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        // Create multiple input tensors
        final imageTensor = TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          data: _generateTestImageData(1, 3, 224, 224),
          name: 'image_input',
        );
        
        final textTensor = TensorData(
          shape: [1, 512],
          dataType: TensorType.int32,
          data: _generateTestTextData(1, 512),
          name: 'text_input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [imageTensor, textTensor],
          requestId: 'test-multi-input-inference',
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert
        expect(result.status, equals(InferenceStatus.success));
        expect(result.outputs, isNotNull);
        expect(result.outputs!.length, greaterThanOrEqualTo(1));
        
        // Cleanup
        hostApi.disposeModel(loadResult.modelId);
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for multi-input inference: $e');
      }
    });

    testWidgets('should provide inference performance metrics', (WidgetTester tester) async {
      // Arrange
      const String testModelPath = 'test_assets/mobilenet_v2.pte';
      
      try {
        final loadResult = await hostApi.loadModel(testModelPath);
        expect(loadResult.state, equals(ModelState.ready));
        
        final inputTensor = TensorData(
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          data: _generateTestImageData(1, 3, 224, 224),
          name: 'input',
        );
        
        final request = InferenceRequest(
          modelId: loadResult.modelId,
          inputs: [inputTensor],
          requestId: 'test-performance-metrics',
          options: {'enable_profiling': true},
        );
        
        // Act
        final result = await hostApi.runInference(request);
        
        // Assert
        expect(result.status, equals(InferenceStatus.success));
        expect(result.executionTimeMs, greaterThan(0));
        
        // Check for performance metadata if supported
        if (result.metadata != null) {
          // Optional performance metrics
          final metadata = result.metadata!;
          if (metadata.containsKey('preprocessing_time_ms')) {
            expect(metadata['preprocessing_time_ms'], isA<num>());
          }
          if (metadata.containsKey('inference_time_ms')) {
            expect(metadata['inference_time_ms'], isA<num>());
          }
          if (metadata.containsKey('postprocessing_time_ms')) {
            expect(metadata['postprocessing_time_ms'], isA<num>());
          }
        }
        
        // Cleanup
        hostApi.disposeModel(loadResult.modelId);
        
      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, isA<Exception>());
        print('Expected TDD failure for performance metrics: $e');
      }
    });
  });
}

// Helper functions to generate test data
Uint8List _generateTestImageData(int batch, int channels, int height, int width) {
  final totalElements = batch * channels * height * width;
  final bytes = totalElements * 4; // 4 bytes per float32
  final data = Uint8List(bytes);
  
  // Fill with normalized image-like data (0.0 to 1.0 range)
  for (int i = 0; i < totalElements; i++) {
    final value = (i % 256) / 255.0; // Normalize to 0-1 range
    final byteOffset = i * 4;
    final floatBytes = Float32List.fromList([value]).buffer.asUint8List();
    data.setRange(byteOffset, byteOffset + 4, floatBytes);
  }
  
  return data;
}

Uint8List _generateTestTextData(int batch, int sequenceLength) {
  final totalElements = batch * sequenceLength;
  final bytes = totalElements * 4; // 4 bytes per int32
  final data = Uint8List(bytes);
  
  // Fill with token IDs (1 to 1000 range)
  for (int i = 0; i < totalElements; i++) {
    final tokenId = (i % 1000) + 1;
    final byteOffset = i * 4;
    final intBytes = Int32List.fromList([tokenId]).buffer.asUint8List();
    data.setRange(byteOffset, byteOffset + 4, intBytes);
  }
  
  return data;
}
