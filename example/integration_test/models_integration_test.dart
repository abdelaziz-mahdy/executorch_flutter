import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('ExecuTorch Models Integration Tests', () {
    final Map<String, String> modelPaths = {};
    final Map<String, ExecuTorchModel> loadedModels = {};

    /// Load an asset model to the cache directory
    Future<String> loadAssetModel(String assetPath) async {
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

    /// Create a TensorData object
    TensorData createTensorData({
      required List<int> shape,
      required TensorType dataType,
      required List<double> data,
      String? name,
    }) {
      // Convert double list to bytes based on data type
      Uint8List bytes;
      if (dataType == TensorType.float32) {
        final float32List = Float32List.fromList(data);
        bytes = float32List.buffer.asUint8List();
      } else if (dataType == TensorType.int32) {
        final int32List = Int32List.fromList(data.map((e) => e.toInt()).toList());
        bytes = int32List.buffer.asUint8List();
      } else {
        throw UnsupportedError('Unsupported data type: $dataType');
      }

      return TensorData(
        shape: shape,
        dataType: dataType,
        data: bytes,
        name: name,
      );
    }

    setUpAll(() async {
      // Load all required models from assets to cache
      final models = {
        'mobilenet': 'assets/models/mobilenet_v3_small_xnnpack.pte',
        'yolo11n': 'assets/models/yolo11n_xnnpack.pte',
        'yolov5n': 'assets/models/yolov5n_xnnpack.pte',
        'yolov8n': 'assets/models/yolov8n_xnnpack.pte',
      };

      for (final entry in models.entries) {
        modelPaths[entry.key] = await loadAssetModel(entry.value);
        print('📦 Loaded ${entry.key}: ${modelPaths[entry.key]}');
      }
    });

    tearDownAll(() async {
      // Dispose all loaded models
      for (final model in loadedModels.values) {
        await model.dispose();
      }
      loadedModels.clear();
    });

    testWidgets('Should load MobileNet V3 model successfully', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;

      // Load the model
      final model = await ExecuTorchModel.load(modelPath);

      expect(
        model.modelHandle,
        isNotEmpty,
        reason: 'Loaded model should have a valid handle',
      );

      expect(
        model.filePath,
        modelPath,
        reason: 'Model file path should match',
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
        model.modelHandle,
        isNotEmpty,
        reason: 'Loaded YOLO11n model should have a valid handle',
      );

      expect(
        model.filePath,
        modelPath,
        reason: 'Model file path should match',
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
        model.modelHandle,
        isNotEmpty,
        reason: 'Loaded YOLOv5n model should have a valid handle',
      );

      expect(
        model.filePath,
        modelPath,
        reason: 'Model file path should match',
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
        model.modelHandle,
        isNotEmpty,
        reason: 'Loaded YOLOv8n model should have a valid handle',
      );

      expect(
        model.filePath,
        modelPath,
        reason: 'Model file path should match',
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
      final inputTensor = createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
        name: 'input',
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotEmpty,
        reason: 'Model should return at least one output tensor',
      );

      expect(
        outputs[0].shape,
        isNotEmpty,
        reason: 'Output tensor should have a shape',
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
      final inputTensor = createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
        name: 'images',
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotEmpty,
        reason: 'YOLO model should return at least one output tensor',
      );

      expect(
        outputs[0].shape,
        isNotEmpty,
        reason: 'Output tensor should have a shape',
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
      final inputTensor = createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
        name: 'images',
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotEmpty,
        reason: 'YOLOv5n model should return at least one output tensor',
      );

      expect(
        outputs[0].shape,
        isNotEmpty,
        reason: 'Output tensor should have a shape',
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
      final inputTensor = createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
        name: 'images',
      );

      // Run inference
      final outputs = await model.forward([inputTensor]);

      expect(
        outputs,
        isNotEmpty,
        reason: 'YOLOv8n model should return at least one output tensor',
      );

      expect(
        outputs[0].shape,
        isNotEmpty,
        reason: 'Output tensor should have a shape',
      );

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should handle multiple sequential inferences', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Create input tensor
      final inputData = List.filled(1 * 3 * 224 * 224, 0.5);
      final inputTensor = createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
        name: 'input',
      );

      // Run inference 5 times
      for (int i = 0; i < 5; i++) {
        final outputs = await model.forward([inputTensor]);
        expect(
          outputs,
          isNotEmpty,
          reason: 'Inference #${i + 1} should succeed',
        );
      }

      // Cleanup
      await model.dispose();
    });

    testWidgets('Should load multiple models simultaneously', (
      WidgetTester tester,
    ) async {
      final mobilenetPath = modelPaths['mobilenet']!;
      final yoloPath = modelPaths['yolo11n']!;

      // Load both models
      final mobilenet = await ExecuTorchModel.load(mobilenetPath);
      final yolo = await ExecuTorchModel.load(yoloPath);

      expect(
        mobilenet.modelHandle,
        isNotEmpty,
        reason: 'MobileNet should have a valid handle',
      );

      expect(
        yolo.modelHandle,
        isNotEmpty,
        reason: 'YOLO should have a valid handle',
      );

      expect(
        mobilenet.modelHandle,
        isNot(equals(yolo.modelHandle)),
        reason: 'Each model should have a unique handle',
      );

      // Cleanup
      await mobilenet.dispose();
      await yolo.dispose();
    });

    testWidgets('Should handle model disposal correctly', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;

      // Load model
      final model1 = await ExecuTorchModel.load(modelPath);
      final handle1 = model1.modelHandle;

      // Dispose
      await model1.dispose();

      // Load same model again
      final model2 = await ExecuTorchModel.load(modelPath);
      final handle2 = model2.modelHandle;

      expect(
        handle1,
        isNot(equals(handle2)),
        reason: 'New model instance should have a different handle',
      );

      // Cleanup
      await model2.dispose();
    });

    testWidgets('Should throw error when loading invalid model', (
      WidgetTester tester,
    ) async {
      final invalidPath = '/path/to/nonexistent/model.pte';

      expect(
        () => ExecuTorchModel.load(invalidPath),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Loading invalid model should throw ExecuTorchException',
      );
    });

    testWidgets('Should throw error when running inference with wrong input shape', (
      WidgetTester tester,
    ) async {
      final modelPath = modelPaths['mobilenet']!;
      final model = await ExecuTorchModel.load(modelPath);

      // Create input with wrong shape (should be 1x3x224x224)
      final inputData = List.filled(1 * 3 * 100 * 100, 0.5);
      final inputTensor = createTensorData(
        shape: [1, 3, 100, 100],
        dataType: TensorType.float32,
        data: inputData,
        name: 'input',
      );

      expect(
        () => model.forward([inputTensor]),
        throwsA(isA<ExecuTorchException>()),
        reason: 'Inference with wrong input shape should throw error',
      );

      // Cleanup
      await model.dispose();
    });
  });
}
