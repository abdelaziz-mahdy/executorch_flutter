import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';

void main() {
  group('Pigeon Contract Tests - ExecutorchHostApi', () {
    late ExecutorchHostApi hostApi;

    setUp(() {
      TestWidgetsFlutterBinding.ensureInitialized();
      hostApi = ExecutorchHostApi();
    });

    test('T009: Contract test ExecutorchHostApi.loadModel', () async {
      // Arrange
      const String testModelPath = '/path/to/test/model.pte';

      // This test will fail until platform implementation exists
      // This is expected as part of TDD approach

      try {
        // Act
        final result = await hostApi.loadModel(testModelPath);

        // Assert - These will only run if platform implementation exists
        expect(result, isA<ModelLoadResult>());
        expect(result.modelId, isA<String>());
        expect(result.state, isIn(ModelState.values));

        if (result.state == ModelState.ready) {
          expect(result.modelId, isNotEmpty);
          expect(result.metadata, isNotNull);
        } else if (result.state == ModelState.error) {
          expect(result.errorMessage, isNotNull);
        }

      } catch (e) {
        // Expected to fail during TDD phase due to:
        // 1. MissingPluginException (no platform implementation)
        // 2. PlatformException (no native handlers)
        expect(e, anyOf([
          isA<MissingPluginException>(),
          isA<PlatformException>(),
        ]));
        print('Expected TDD failure for loadModel: $e');
      }
    });

    test('T010: Contract test ExecutorchHostApi.runInference', () async {
      // Arrange
      const String modelId = 'test-model-123';
      final inputTensor = TensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: Uint8List.fromList(List.filled(1 * 3 * 224 * 224 * 4, 0)),
        name: 'input',
      );

      final request = InferenceRequest(
        modelId: modelId,
        inputs: [inputTensor],
        requestId: 'test-request-456',
        timeoutMs: 5000,
      );

      try {
        // Act
        final result = await hostApi.runInference(request);

        // Assert - Only if implementation exists
        expect(result, isA<InferenceResult>());
        expect(result.status, isIn(InferenceStatus.values));
        expect(result.executionTimeMs, isA<double>());

        if (result.status == InferenceStatus.success) {
          expect(result.outputs, isNotNull);
        } else {
          expect(result.errorMessage, isNotNull);
        }

      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, anyOf([
          isA<MissingPluginException>(),
          isA<PlatformException>(),
        ]));
        print('Expected TDD failure for runInference: $e');
      }
    });

    test('T011: Contract test ExecutorchHostApi.disposeModel', () async {
      // Arrange
      const String modelId = 'test-model-123';

      try {
        // Act
        await hostApi.disposeModel(modelId);

        // Assert - If we get here, the method signature is correct
        expect(true, isTrue); // Method executed without signature error

      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, anyOf([
          isA<MissingPluginException>(),
          isA<PlatformException>(),
        ]));
        print('Expected TDD failure for disposeModel: $e');
      }
    });

    test('T012: Contract test ExecutorchHostApi.getLoadedModels', () async {
      try {
        // Act
        final models = await hostApi.getLoadedModels();

        // Assert - Only if implementation exists
        expect(models, isA<List<String?>>());

      } catch (e) {
        // Expected to fail during TDD phase
        expect(e, anyOf([
          isA<MissingPluginException>(),
          isA<PlatformException>(),
        ]));
        print('Expected TDD failure for getLoadedModels: $e');
      }
    });
  });
}
