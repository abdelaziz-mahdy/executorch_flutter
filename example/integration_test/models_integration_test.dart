// ignore_for_file: avoid_print

import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:path_provider/path_provider.dart';

// Import the example app services for model downloading
import 'package:executorch_flutter_example/services/model_index_service.dart';
import 'package:executorch_flutter_example/services/model_download_service.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('ExecuTorch Models Integration Tests', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};
    final Map<String, ModelIndexEntry> modelEntries = {};

    /// Download a model from GitHub and save to cache directory
    Future<String> downloadAndCacheModel(ModelIndexEntry entry) async {
      print('📥 Downloading ${entry.displayName}...');

      final downloadService = ModelDownloadService.instance;
      final info = await downloadService.downloadModel(
        modelName: entry.name,
        remoteUrl: entry.remoteUrl,
        onProgress: (progress, received, total) {
          if (progress == 0.0 || progress == 0.5 || progress == 1.0) {
            print('   ${(progress * 100).toStringAsFixed(0)}% downloaded');
          }
        },
      );

      if (info.state != ModelDownloadState.downloaded) {
        throw Exception('Failed to download ${entry.name}: ${info.errorMessage}');
      }

      // For native platforms, return the local path
      if (info.localPath != null) {
        print('✅ Downloaded ${entry.displayName} to ${info.localPath}');
        return info.localPath!;
      }

      // For web or bytes-only, write to cache directory
      if (info.bytes != null) {
        final directory = await getApplicationCacheDirectory();
        final file = File('${directory.path}/${entry.name}');
        await file.writeAsBytes(info.bytes!);
        print('✅ Downloaded ${entry.displayName} to ${file.path}');
        return file.path;
      }

      throw Exception('Download succeeded but no path or bytes available');
    }

    setUpAll(() async {
      manager = ExecutorchManager.instance;
      await manager.initialize();

      // Fetch model index from GitHub
      print('📦 Fetching model index from GitHub...');
      final index = await ModelIndexService.fetchIndex(forceRefresh: true);
      print('✅ Found ${index.models.length} models in index');

      // Find required models (XNNPACK backend for cross-platform support)
      final requiredModels = {
        'mobilenet': 'mobilenet_v3_small_xnnpack.pte',
        'yolo11n': 'yolo11n_xnnpack.pte',
        'yolov5n': 'yolov5n_xnnpack.pte',
        'yolov8n': 'yolov8n_xnnpack.pte',
      };

      for (final entry in requiredModels.entries) {
        final modelEntry = index.models.firstWhere(
          (m) => m.name == entry.value,
          orElse: () => throw Exception('Model ${entry.value} not found in index'),
        );
        modelEntries[entry.key] = modelEntry;
        modelPaths[entry.key] = await downloadAndCacheModel(modelEntry);
        print('📦 Ready: ${entry.key} -> ${modelPaths[entry.key]}');
      }

      print('');
      print('✅ All models downloaded and ready for testing');
      print('');
    });

    tearDownAll(() async {
      await manager.disposeAllModels();
    });

    testWidgets('ExecutorchManager should initialize successfully', (
      WidgetTester tester,
    ) async {
      final isAvailable = await manager.isAvailable();
      expect(
        isAvailable,
        true,
        reason: 'ExecutorchManager should be available after initialization',
      );
    });

    testWidgets('Should load MobileNet V3 model successfully', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;

      // Load the model
      final model = await ExecuTorchModel.load(modelPath);

      expect(
        model.modelId,
        isNotEmpty,
        reason: 'Loaded model should have a valid ID',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load YOLO11n model successfully', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['yolo11n']!;

      // Load the model
      final model = await ExecuTorchModel.load(modelPath);

      expect(
        model.modelId,
        isNotEmpty,
        reason: 'Loaded YOLO11n model should have a valid ID',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load YOLOv5n model successfully', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['yolov5n']!;

      // Load the model
      final model = await ExecuTorchModel.load(modelPath);

      expect(
        model.modelId,
        isNotEmpty,
        reason: 'Loaded YOLOv5n model should have a valid ID',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load YOLOv8n model successfully', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['yolov8n']!;

      // Load the model
      final model = await ExecuTorchModel.load(modelPath);

      expect(
        model.modelId,
        isNotEmpty,
        reason: 'Loaded YOLOv8n model should have a valid ID',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on MobileNet V3 model', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Create dummy input tensor for MobileNet (1, 3, 224, 224)
      final inputData = List.filled(1 * 3 * 224 * 224, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotNull,
        reason: 'Inference should return output tensors',
      );
      expect(
        outputs.isNotEmpty,
        true,
        reason: 'Output tensors should not be empty',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on YOLO11n model', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['yolo11n']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Create dummy input tensor for YOLO (1, 3, 640, 640)
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotNull,
        reason: 'Inference should return output tensors',
      );
      expect(
        outputs.isNotEmpty,
        true,
        reason: 'Output tensors should not be empty',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on YOLOv5n model', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['yolov5n']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Create dummy input tensor for YOLO (1, 3, 640, 640)
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotNull,
        reason: 'Inference should return output tensors',
      );
      expect(
        outputs.isNotEmpty,
        true,
        reason: 'Output tensors should not be empty',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should run inference on YOLOv8n model', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['yolov8n']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Create dummy input tensor for YOLO (1, 3, 640, 640)
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotNull,
        reason: 'Inference should return output tensors',
      );
      expect(
        outputs.isNotEmpty,
        true,
        reason: 'Output tensors should not be empty',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should handle multiple models concurrently', (
      WidgetTester tester,
    ) async {
      final mobilenetPath = modelPaths['mobilenet']!;
      final yoloPath = modelPaths['yolo11n']!;

      // Load both models
      final mobilenet = await ExecuTorchModel.load(mobilenetPath);
      final yolo = await ExecuTorchModel.load(yoloPath);

      // Verify both models are loaded (both have valid model IDs)
      expect(
        mobilenet.modelId,
        isNotEmpty,
        reason: 'MobileNet should have a valid model ID',
      );
      expect(
        yolo.modelId,
        isNotEmpty,
        reason: 'YOLO should have a valid model ID',
      );

      // Cleanup
      await mobilenet.dispose();
      await yolo.dispose();
    });

    testWidgets('Should properly dispose models and free resources', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Verify model is not disposed initially
      expect(
        model.isDisposed,
        false,
        reason: 'Model should not be disposed initially',
      );

      // Dispose the model
      await model.dispose();

      // Verify model is disposed
      expect(
        model.isDisposed,
        true,
        reason: 'Model should be disposed after calling dispose',
      );
    });

    testWidgets('Should handle model reload correctly', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;

      // Load model first time
      final model1 = await ExecuTorchModel.load(modelPath);
      final modelId1 = model1.modelId;

      // Dispose it
      await model1.dispose();

      // Load the same model again
      final model2 = await ExecuTorchModel.load(modelPath);
      final modelId2 = model2.modelId;

      // Model IDs should be different (new instance)
      expect(
        modelId1 != modelId2,
        true,
        reason: 'Reloaded model should have a new ID',
      );

      // Cleanup
      await model2.dispose();
    });

    // Error handling tests
    testWidgets('Should throw exception when loading non-existent model', (
      WidgetTester tester,
    ) async {
      final invalidPath = '/non/existent/model.pte';

      // Attempt to load non-existent model
      expect(
        () async => await ExecuTorchModel.load(invalidPath),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Loading non-existent model should throw exception',
      );
    });

    testWidgets(
      'Should throw exception when running inference on disposed model',
      (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

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
          () async => await model.forward([inputTensor]),
          throwsA(isA<ExecuTorchException>()),
          reason: 'Running inference on disposed model should throw exception',
        );
      },
    );

    testWidgets('Should handle invalid model file format', (
      WidgetTester tester,
    ) async {
      // Create a temporary invalid file
      final directory = await getApplicationCacheDirectory();
      final invalidFile = File('${directory.path}/invalid_model.pte');
      await invalidFile.writeAsString('This is not a valid model file');

      // Attempt to load invalid model
      expect(
        () async => await ExecuTorchModel.load(invalidFile.path),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Loading invalid model file should throw exception',
      );

      // Cleanup
      await invalidFile.delete();
    });

    testWidgets('Should handle multiple dispose calls gracefully', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await ExecuTorchModel.load(modelPath);

      // First dispose should succeed
      await model.dispose();

      // Second dispose should not throw (idempotent)
      await model.dispose();

      expect(
        model.isDisposed,
        true,
        reason: 'Model should be marked as disposed',
      );
    });

    // Backend query tests
    group('Backend Query Tests', () {
      testWidgets('BackendQuery.available should return non-empty list', (
        WidgetTester tester,
      ) async {
        final availableBackends = BackendQuery.available;

        expect(
          availableBackends,
          isNotEmpty,
          reason: 'At least one backend should be available',
        );

        print('Available backends: ${availableBackends.map((b) => b.name).join(", ")}');
      });

      testWidgets('XNNPACK backend should be available on all platforms', (
        WidgetTester tester,
      ) async {
        final isXnnpackAvailable = BackendQuery.isAvailable(Backend.xnnpack);

        expect(
          isXnnpackAvailable,
          true,
          reason: 'XNNPACK backend should be available on all platforms',
        );
      });

      testWidgets('BackendQuery.available should include XNNPACK', (
        WidgetTester tester,
      ) async {
        final availableBackends = BackendQuery.available;

        expect(
          availableBackends.contains(Backend.xnnpack),
          true,
          reason: 'Available backends list should include XNNPACK',
        );
      });

      testWidgets('Backend.displayName should return human-readable names', (
        WidgetTester tester,
      ) async {
        expect(Backend.xnnpack.displayName, equals('XNNPACK'));
        expect(Backend.coreml.displayName, equals('CoreML'));
        expect(Backend.mps.displayName, equals('Metal Performance Shaders'));
        expect(Backend.vulkan.displayName, equals('Vulkan'));
        expect(Backend.qnn.displayName, equals('Qualcomm QNN'));
      });

      testWidgets('BackendQuery should handle all Backend enum values', (
        WidgetTester tester,
      ) async {
        // Test that isAvailable doesn't throw for any backend
        for (final backend in Backend.values) {
          expect(
            () => BackendQuery.isAvailable(backend),
            returnsNormally,
            reason: 'isAvailable should not throw for ${backend.name}',
          );
        }
      });

      testWidgets('Vulkan backend availability should be queryable', (
        WidgetTester tester,
      ) async {
        // Test that we can query Vulkan availability without error
        final isVulkanAvailable = BackendQuery.isAvailable(Backend.vulkan);

        // Print result for debugging (Vulkan may or may not be available)
        print('Vulkan backend available: $isVulkanAvailable');

        // Just verify the call succeeds and returns a boolean
        expect(isVulkanAvailable, isA<bool>());
      });

      testWidgets('Available backends list should be consistent', (
        WidgetTester tester,
      ) async {
        // Get available backends twice and verify consistency
        final list1 = BackendQuery.available;
        final list2 = BackendQuery.available;

        expect(
          list1.length,
          equals(list2.length),
          reason: 'Available backends should be consistent across calls',
        );

        for (final backend in list1) {
          expect(
            list2.contains(backend),
            true,
            reason: '${backend.name} should be in both lists',
          );
        }
      });

      testWidgets(
        'isAvailable should match available list for all backends',
        (WidgetTester tester) async {
          final availableBackends = BackendQuery.available;

          for (final backend in Backend.values) {
            final isAvailable = BackendQuery.isAvailable(backend);
            final isInList = availableBackends.contains(backend);

            expect(
              isAvailable,
              equals(isInList),
              reason:
                  '${backend.name}: isAvailable($isAvailable) should match '
                  'presence in available list($isInList)',
            );
          }
        },
      );
    });

    // Tensor shape and data type tests
    group('Tensor Shape Tests', () {
      testWidgets('Should handle 1D tensor shapes', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

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
          expect(
            inputTensor.shape,
            equals(shape),
            reason: 'Tensor shape should match input shape',
          );
        }

        await model.dispose();
      });

      testWidgets('Should handle 2D tensor shapes', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

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

          expect(
            inputTensor.shape,
            equals(shape),
            reason: 'Tensor shape should match input shape',
          );
        }

        await model.dispose();
      });

      testWidgets('Should handle 3D tensor shapes', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

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

          expect(
            inputTensor.shape,
            equals(shape),
            reason: 'Tensor shape should match input shape',
          );
        }

        await model.dispose();
      });

      testWidgets('Should handle 4D tensor shapes', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

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

          expect(
            inputTensor.shape,
            equals(shape),
            reason: 'Tensor shape should match input shape',
          );
        }

        await model.dispose();
      });

      testWidgets('Should handle float32 data type', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        final shape = [1, 3, 224, 224];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 0.5);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        expect(
          inputTensor.dataType,
          equals(TensorType.float32),
          reason: 'Tensor data type should be float32',
        );

        await model.dispose();
      });

      testWidgets('Should handle int32 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        final shape = [1, 10];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 42);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.int32,
          data: inputData,
        );

        expect(
          inputTensor.dataType,
          equals(TensorType.int32),
          reason: 'Tensor data type should be int32',
        );

        await model.dispose();
      });

      testWidgets('Should handle uint8 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        final shape = [1, 224, 224];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 128);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.uint8,
          data: inputData,
        );

        expect(
          inputTensor.dataType,
          equals(TensorType.uint8),
          reason: 'Tensor data type should be uint8',
        );

        await model.dispose();
      });

      testWidgets('Should handle int8 data type', (WidgetTester tester) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        final shape = [1, 100];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, -50);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.int8,
          data: inputData,
        );

        expect(
          inputTensor.dataType,
          equals(TensorType.int8),
          reason: 'Tensor data type should be int8',
        );

        await model.dispose();
      });

      testWidgets('Should handle single element tensor', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        final shape = [1];
        final inputData = [1.0];
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        expect(
          inputTensor.shape,
          equals(shape),
          reason: 'Single element tensor should have correct shape',
        );

        await model.dispose();
      });

      testWidgets('Should handle large tensor shapes', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        // Test with a reasonably large tensor (not too large to cause OOM)
        final shape = [1, 3, 512, 512];
        final size = shape.reduce((a, b) => a * b);
        final inputData = List.filled(size, 0.5);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        expect(
          inputTensor.shape,
          equals(shape),
          reason: 'Large tensor should have correct shape',
        );

        await model.dispose();
      });

      testWidgets('Should handle different batch sizes', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

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

          expect(
            inputTensor.shape[0],
            equals(batchSize),
            reason: 'Batch size should be preserved in tensor shape',
          );
        }

        await model.dispose();
      });

      testWidgets('Should verify tensor data size matches shape', (
        WidgetTester tester,
      ) async {
        final modelPath = modelPaths['mobilenet']!;
        final model = await ExecuTorchModel.load(modelPath);

        final shape = [2, 3, 4, 5];
        final expectedSize = shape.reduce((a, b) => a * b); // 120
        final inputData = List.filled(expectedSize, 0.5);
        final inputTensor = manager.createTensorData(
          shape: shape,
          dataType: TensorType.float32,
          data: inputData,
        );

        // Verify the tensor shape produces the correct total size
        final actualSize = inputTensor.shape.whereType<int>().reduce(
          (a, b) => a * b,
        );
        expect(
          actualSize,
          equals(expectedSize),
          reason: 'Tensor data size should match product of shape dimensions',
        );

        await model.dispose();
      });
    });
  });
}
