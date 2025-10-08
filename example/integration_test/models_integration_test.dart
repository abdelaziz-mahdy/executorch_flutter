import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('ExecuTorch Models Integration Tests', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};

    /// Load an asset model to the cache directory
    Future<String> _loadAssetModel(String assetPath) async {
      final byteData = await rootBundle.load(assetPath);
      final directory = await getApplicationCacheDirectory();
      final fileName = assetPath.split('/').last;
      final file = File('${directory.path}/$fileName');

      if (!file.existsSync()) {
        await file.writeAsBytes(
          byteData.buffer.asUint8List(
            byteData.offsetInBytes,
            byteData.lengthInBytes,
          ),
        );
      }

      return file.path;
    }

    setUpAll(() async {
      manager = ExecutorchManager.instance;
      await manager.initialize();

      // Load all required models from assets to cache
      final models = {
        'mobilenet': 'assets/models/mobilenet_v3_small_xnnpack.pte',
        'yolo11n': 'assets/models/yolo11n_xnnpack.pte',
        'yolov5n': 'assets/models/yolov5n_xnnpack.pte',
        'yolov8n': 'assets/models/yolov8n_xnnpack.pte',
      };

      for (final entry in models.entries) {
        modelPaths[entry.key] = await _loadAssetModel(entry.value);
        print('📦 Loaded ${entry.key}: ${modelPaths[entry.key]}');
      }
    });

    tearDownAll(() async {
      await manager.disposeAllModels();
    });

    testWidgets('ExecutorchManager should initialize successfully',
        (WidgetTester tester) async {
      final isAvailable = await manager.isAvailable();
      expect(isAvailable, true,
          reason: 'ExecutorchManager should be available after initialization');
    });

    testWidgets('Should load MobileNet V3 model successfully',
        (WidgetTester tester) async {
      final modelPath = modelPaths['mobilenet']!;

      // Load the model
      final model = await manager.loadModel(modelPath);

      expect(model.modelId, isNotEmpty,
          reason: 'Loaded model should have a valid ID');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load YOLO11n model successfully',
        (WidgetTester tester) async {
      final modelPath = modelPaths['yolo11n']!;

      // Load the model
      final model = await manager.loadModel(modelPath);

      expect(model.modelId, isNotEmpty,
          reason: 'Loaded YOLO11n model should have a valid ID');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load YOLOv5n model successfully',
        (WidgetTester tester) async {
      final modelPath = modelPaths['yolov5n']!;

      // Load the model
      final model = await manager.loadModel(modelPath);

      expect(model.modelId, isNotEmpty,
          reason: 'Loaded YOLOv5n model should have a valid ID');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load YOLOv8n model successfully',
        (WidgetTester tester) async {
      final modelPath = modelPaths['yolov8n']!;

      // Load the model
      final model = await manager.loadModel(modelPath);

      expect(model.modelId, isNotEmpty,
          reason: 'Loaded YOLOv8n model should have a valid ID');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on MobileNet V3 model',
        (WidgetTester tester) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await manager.loadModel(modelPath);

      // Create dummy input tensor for MobileNet (1, 3, 224, 224)
      final inputData = List.filled(1 * 3 * 224 * 224, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final result = await model.runInference(inputs: [inputTensor]);

      expect(result.outputs, isNotNull,
          reason: 'Inference should return output tensors');
      expect(result.outputs!.isNotEmpty, true,
          reason: 'Output tensors should not be empty');
      expect(result.executionTimeMs, greaterThan(0),
          reason: 'Execution time should be recorded');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on YOLO11n model',
        (WidgetTester tester) async {
      final modelPath = modelPaths['yolo11n']!;
      final model = await manager.loadModel(modelPath);

      // Create dummy input tensor for YOLO (1, 3, 640, 640)
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final result = await model.runInference(inputs: [inputTensor]);

      expect(result.outputs, isNotNull,
          reason: 'Inference should return output tensors');
      expect(result.executionTimeMs, greaterThan(0),
          reason: 'Execution time should be recorded');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on YOLOv5n model',
        (WidgetTester tester) async {
      final modelPath = modelPaths['yolov5n']!;
      final model = await manager.loadModel(modelPath);

      // Create dummy input tensor for YOLO (1, 3, 640, 640)
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final result = await model.runInference(inputs: [inputTensor]);

      expect(result.outputs, isNotNull,
          reason: 'Inference should return output tensors');
      expect(result.executionTimeMs, greaterThan(0),
          reason: 'Execution time should be recorded');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on YOLOv8n model',
        (WidgetTester tester) async {
      final modelPath = modelPaths['yolov8n']!;
      final model = await manager.loadModel(modelPath);

      // Create dummy input tensor for YOLO (1, 3, 640, 640)
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final result = await model.runInference(inputs: [inputTensor]);

      expect(result.outputs, isNotNull,
          reason: 'Inference should return output tensors');
      expect(result.executionTimeMs, greaterThan(0),
          reason: 'Execution time should be recorded');

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should handle multiple models concurrently',
        (WidgetTester tester) async {
      final mobilenetPath = modelPaths['mobilenet']!;
      final yoloPath = modelPaths['yolo11n']!;

      // Load both models
      final mobilenet = await manager.loadModel(mobilenetPath);
      final yolo = await manager.loadModel(yoloPath);

      // Verify both models are loaded
      final loadedModelIds = await manager.getLoadedModelIds();
      expect(loadedModelIds.length, greaterThanOrEqualTo(2),
          reason: 'Should have at least 2 models loaded');
      expect(loadedModelIds.contains(mobilenet.modelId), true,
          reason: 'MobileNet should be in loaded models');
      expect(loadedModelIds.contains(yolo.modelId), true,
          reason: 'YOLO should be in loaded models');

      // Cleanup
      await mobilenet.dispose();
      await yolo.dispose();
    });

    testWidgets('Should properly dispose models and free resources',
        (WidgetTester tester) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await manager.loadModel(modelPath);
      final modelId = model.modelId;

      // Verify model is loaded
      var loadedIds = await manager.getLoadedModelIds();
      expect(loadedIds.contains(modelId), true,
          reason: 'Model should be in loaded models list');

      // Dispose the model
      await model.dispose();

      // Verify model is no longer loaded
      loadedIds = await manager.getLoadedModelIds();
      expect(loadedIds.contains(modelId), false,
          reason: 'Model should not be in loaded models list after disposal');
    });

    testWidgets('Should handle model reload correctly',
        (WidgetTester tester) async {
      final modelPath = modelPaths['mobilenet']!;

      // Load model first time
      final model1 = await manager.loadModel(modelPath);
      final modelId1 = model1.modelId;

      // Dispose it
      await model1.dispose();

      // Load the same model again
      final model2 = await manager.loadModel(modelPath);
      final modelId2 = model2.modelId;

      // Model IDs should be different (new instance)
      expect(modelId1 != modelId2, true,
          reason: 'Reloaded model should have a new ID');

      // Cleanup
      await model2.dispose();
    });

    // Error handling tests
    testWidgets('Should throw exception when loading non-existent model',
        (WidgetTester tester) async {
      final invalidPath = '/non/existent/model.pte';

      // Attempt to load non-existent model
      expect(
        () async => await manager.loadModel(invalidPath),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Loading non-existent model should throw exception',
      );
    });

    testWidgets('Should throw exception when running inference on disposed model',
        (WidgetTester tester) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await manager.loadModel(modelPath);

      // Dispose the model
      await model.dispose();

      // Create dummy input
      final inputData = List.filled(1 * 3 * 224 * 224, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Attempt to run inference on disposed model
      expect(
        () async => await model.runInference(inputs: [inputTensor]),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Running inference on disposed model should throw exception',
      );
    });

    testWidgets('Should handle invalid model file format',
        (WidgetTester tester) async {
      // Create a temporary invalid file
      final directory = await getApplicationCacheDirectory();
      final invalidFile = File('${directory.path}/invalid_model.pte');
      await invalidFile.writeAsString('This is not a valid model file');

      // Attempt to load invalid model
      expect(
        () async => await manager.loadModel(invalidFile.path),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Loading invalid model file should throw exception',
      );

      // Cleanup
      await invalidFile.delete();
    });

    testWidgets('Should handle multiple dispose calls gracefully',
        (WidgetTester tester) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await manager.loadModel(modelPath);

      // First dispose should succeed
      await model.dispose();

      // Second dispose should not throw (idempotent)
      await model.dispose();

      expect(model.isDisposed, true,
          reason: 'Model should be marked as disposed');
    });

    // Tensor shape and data type tests
    group('Tensor Shape Tests', () {
      testWidgets('Should handle 1D tensor shapes', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        // Test various 1D shapes
        final shapes = [
          [10],
          [100],
          [1000],
        ];

        for (final shape in shapes) {
          final size = shape.reduce((a, b) => a * b);
          final inputData = List.filled(size, 0.5);
          final inputTensor = manager.createTensorData(
            shape: shape,
            dataType: TensorType.float32,
            data: inputData,
          );

          // Verify tensor was created with correct shape
          expect(inputTensor.shape, equals(shape),
              reason: 'Tensor shape should match input shape');
        }

        await model.dispose();
      });

      testWidgets('Should handle 2D tensor shapes', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        // Test various 2D shapes
        final shapes = [
          [1, 10],
          [10, 10],
          [100, 100],
          [224, 224],
        ];

        for (final shape in shapes) {
          final size = shape.reduce((a, b) => a * b);
          final inputData = List.filled(size, 0.5);
          final inputTensor = manager.createTensorData(
            shape: shape,
            dataType: TensorType.float32,
            data: inputData,
          );

          expect(inputTensor.shape, equals(shape),
              reason: 'Tensor shape should match input shape');
        }

        await model.dispose();
      });

      testWidgets('Should handle 3D tensor shapes', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        // Test various 3D shapes
        final shapes = [
          [1, 3, 10],
          [3, 224, 224],
          [4, 128, 128],
        ];

        for (final shape in shapes) {
          final size = shape.reduce((a, b) => a * b);
          final inputData = List.filled(size, 0.5);
          final inputTensor = manager.createTensorData(
            shape: shape,
            dataType: TensorType.float32,
            data: inputData,
          );

          expect(inputTensor.shape, equals(shape),
              reason: 'Tensor shape should match input shape');
        }

        await model.dispose();
      });

      testWidgets('Should handle 4D tensor shapes', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        // Test various 4D shapes (typical image tensor shapes)
        final shapes = [
          [1, 3, 224, 224], // MobileNet input
          [1, 3, 640, 640], // YOLO input
          [2, 3, 224, 224], // Batch size 2
          [1, 1, 128, 128], // Grayscale
        ];

        for (final shape in shapes) {
          final size = shape.reduce((a, b) => a * b);
          final inputData = List.filled(size, 0.5);
          final inputTensor = manager.createTensorData(
            shape: shape,
            dataType: TensorType.float32,
            data: inputData,
          );

          expect(inputTensor.shape, equals(shape),
              reason: 'Tensor shape should match input shape');
        }

        await model.dispose();
      });

      testWidgets('Should handle float32 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        final shape = [1, 3, 224, 224];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 0.5);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        expect(inputTensor.dataType, equals(TensorType.float32),
            reason: 'Tensor data type should be float32');

        await model.dispose();
      });

      testWidgets('Should handle int32 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        final shape = [1, 10];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 42);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.int32,
          data: inputData,
        );

        expect(inputTensor.dataType, equals(TensorType.int32),
            reason: 'Tensor data type should be int32');

        await model.dispose();
      });

      testWidgets('Should handle uint8 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        final shape = [1, 224, 224];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 128);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.uint8,
          data: inputData,
        );

        expect(inputTensor.dataType, equals(TensorType.uint8),
            reason: 'Tensor data type should be uint8');

        await model.dispose();
      });

      testWidgets('Should handle int8 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        final shape = [1, 100];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, -50);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.int8,
          data: inputData,
        );

        expect(inputTensor.dataType, equals(TensorType.int8),
            reason: 'Tensor data type should be int8');

        await model.dispose();
      });

      testWidgets('Should handle single element tensor', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        final shape = [1];
        final inputData = [1.0];
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        expect(inputTensor.shape, equals(shape),
            reason: 'Single element tensor should have correct shape');

        await model.dispose();
      });

      testWidgets('Should handle large tensor shapes', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        // Test with a reasonably large tensor (not too large to cause OOM)
        final shape = [1, 3, 512, 512];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 0.5);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        expect(inputTensor.shape, equals(shape),
            reason: 'Large tensor should have correct shape');

        await model.dispose();
      });

      testWidgets('Should handle different batch sizes', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        // Test various batch sizes
        final batchSizes = [1, 2, 4];

        for (final batchSize in batchSizes) {
          final shape = [batchSize, 3, 224, 224];
          final size = shape.reduce((a, b) => a * b);
          final inputData = List.filled(size, 0.5);
          final inputTensor = manager.createTensorData(
            shape: shape,
            dataType: TensorType.float32,
            data: inputData,
          );

          expect(inputTensor.shape[0], equals(batchSize),
              reason: 'Batch size should be preserved in tensor shape');
        }

        await model.dispose();
      });

      testWidgets('Should verify tensor data size matches shape',
          (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await manager.loadModel(modelPath);

        final shape = [2, 3, 4, 5];
        final expectedSize = shape.reduce((a, b) => a * b); // 120
        final inputData = List.filled(expectedSize, 0.5);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        // Verify the tensor shape produces the correct total size
        final actualSize = inputTensor.shape
            .whereType<int>()
            .reduce((a, b) => a * b);
        expect(actualSize, equals(expectedSize),
            reason: 'Tensor data size should match product of shape dimensions');

        await model.dispose();
      });
    });
  });
}
