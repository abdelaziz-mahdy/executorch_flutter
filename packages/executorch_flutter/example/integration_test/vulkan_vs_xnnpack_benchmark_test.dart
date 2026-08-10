// ignore_for_file: avoid_print

@Timeout(Duration(minutes: 30))
library;

import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:path_provider/path_provider.dart';

import 'package:executorch_flutter_example/services/model_index_service.dart';
import 'package:executorch_flutter_example/services/model_download_service.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan vs XNNPACK Benchmark', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};

    Future<String> downloadModel(ModelIndexEntry entry) async {
      print('  Downloading ${entry.name} (${entry.sizeMB} MB)...');
      final downloadService = ModelDownloadService.instance;
      final info = await downloadService.downloadModel(
        modelName: entry.name,
        remoteUrl: entry.remoteUrl,
        expectedHash: entry.hash,
        onProgress: (progress, received, total) {
          if (progress == 1.0) {
            print('  100% downloaded');
          }
        },
      );

      if (info.state != ModelDownloadState.downloaded) {
        throw Exception(
          'Failed to download ${entry.name}: ${info.errorMessage}',
        );
      }

      if (info.localPath != null) return info.localPath!;

      if (info.bytes != null) {
        final directory = await getApplicationCacheDirectory();
        final file = File('${directory.path}/${entry.name}');
        await file.writeAsBytes(info.bytes!);
        return file.path;
      }

      throw Exception('Download succeeded but no path or bytes available');
    }

    setUpAll(() async {
      manager = ExecutorchManager.instance;
      await manager.initialize();

      print('');
      print('========================================');
      print('  Vulkan vs XNNPACK Benchmark');
      print('========================================');
      print('');

      // Check backend availability
      final vulkanAvailable = BackendQuery.isAvailable(Backend.vulkan);
      final xnnpackAvailable = BackendQuery.isAvailable(Backend.xnnpack);
      print('Backends: XNNPACK=$xnnpackAvailable, Vulkan=$vulkanAvailable');
      print('');

      // Fetch model index
      print('Fetching model index...');
      final index = await ModelIndexService.fetchIndex(forceRefresh: true);

      // Models to benchmark
      final modelsToDownload = [
        'yolo11n_xnnpack.pte',
        'yolo11n_vulkan.pte',
        'yolov8n_xnnpack.pte',
        'yolov8n_vulkan.pte',
        'mobilenet_v3_small_xnnpack.pte',
        'mobilenet_v3_small_vulkan.pte',
      ];

      for (final modelName in modelsToDownload) {
        try {
          final entry = index.models.firstWhere((m) => m.name == modelName);
          modelPaths[modelName] = await downloadModel(entry);
        } catch (e) {
          print('  SKIP: $modelName not found in index');
        }
      }

      print('');
      print('Downloaded ${modelPaths.length} models');
      print('');
    });

    tearDownAll(() async {
      await manager.disposeAllModels();
    });

    /// Benchmark model loading time (3 runs, report all + average)
    Future<List<double>> benchmarkLoad(String modelName, int runs) async {
      final path = modelPaths[modelName];
      if (path == null) {
        print('  SKIP: $modelName not available');
        return [];
      }

      final times = <double>[];
      for (var i = 0; i < runs; i++) {
        final sw = Stopwatch()..start();
        final model = await ExecuTorchModel.load(path);
        sw.stop();
        times.add(sw.elapsedMilliseconds.toDouble());
        await model.dispose();
      }
      return times;
    }

    /// Benchmark inference time (multiple runs, report all + average)
    Future<List<double>> benchmarkInference(
      String modelName,
      List<int> inputShape,
      int runs,
    ) async {
      final path = modelPaths[modelName];
      if (path == null) {
        print('  SKIP: $modelName not available');
        return [];
      }

      final model = await ExecuTorchModel.load(path);

      final size = inputShape.reduce((a, b) => a * b);
      final inputData = List.filled(size, 0.5);
      final inputTensor = manager.createTensorData(
        shape: inputShape,
        dataType: TensorType.float32,
        data: inputData,
      );

      final times = <double>[];

      // Warmup run
      try {
        await model.forward([inputTensor]);
      } catch (e) {
        print('  Warmup failed for $modelName: $e');
        await model.dispose();
        return [];
      }

      for (var i = 0; i < runs; i++) {
        final sw = Stopwatch()..start();
        await model.forward([inputTensor]);
        sw.stop();
        times.add(sw.elapsedMilliseconds.toDouble());
      }

      await model.dispose();
      return times;
    }

    void printBenchmarkResult(String label, List<double> times) {
      if (times.isEmpty) {
        print('  $label: SKIPPED');
        return;
      }
      final avg = times.reduce((a, b) => a + b) / times.length;
      final min = times.reduce((a, b) => a < b ? a : b);
      final max = times.reduce((a, b) => a > b ? a : b);
      print(
        '  $label: avg=${avg.toStringAsFixed(1)}ms '
        'min=${min.toStringAsFixed(1)}ms '
        'max=${max.toStringAsFixed(1)}ms '
        'runs=$times',
      );
    }

    testWidgets('YOLO11n: Load time Vulkan vs XNNPACK', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- YOLO11n Load Time (3 runs) ---');

      final xnnpackTimes = await benchmarkLoad('yolo11n_xnnpack.pte', 3);
      printBenchmarkResult('XNNPACK', xnnpackTimes);

      final vulkanTimes = await benchmarkLoad('yolo11n_vulkan.pte', 3);
      printBenchmarkResult('Vulkan ', vulkanTimes);

      if (xnnpackTimes.isNotEmpty && vulkanTimes.isNotEmpty) {
        final xAvg = xnnpackTimes.reduce((a, b) => a + b) / xnnpackTimes.length;
        final vAvg = vulkanTimes.reduce((a, b) => a + b) / vulkanTimes.length;
        print(
          '  Ratio: Vulkan is ${(vAvg / xAvg).toStringAsFixed(1)}x XNNPACK load time',
        );
      }

      print('');
    });

    testWidgets('YOLOv8n: Load time Vulkan vs XNNPACK', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- YOLOv8n Load Time (3 runs) ---');

      final xnnpackTimes = await benchmarkLoad('yolov8n_xnnpack.pte', 3);
      printBenchmarkResult('XNNPACK', xnnpackTimes);

      final vulkanTimes = await benchmarkLoad('yolov8n_vulkan.pte', 3);
      printBenchmarkResult('Vulkan ', vulkanTimes);

      if (xnnpackTimes.isNotEmpty && vulkanTimes.isNotEmpty) {
        final xAvg = xnnpackTimes.reduce((a, b) => a + b) / xnnpackTimes.length;
        final vAvg = vulkanTimes.reduce((a, b) => a + b) / vulkanTimes.length;
        print(
          '  Ratio: Vulkan is ${(vAvg / xAvg).toStringAsFixed(1)}x XNNPACK load time',
        );
      }

      print('');
    });

    testWidgets('MobileNet V3: Load time Vulkan vs XNNPACK', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- MobileNet V3 Load Time (3 runs) ---');

      final xnnpackTimes = await benchmarkLoad(
        'mobilenet_v3_small_xnnpack.pte',
        3,
      );
      printBenchmarkResult('XNNPACK', xnnpackTimes);

      final vulkanTimes = await benchmarkLoad(
        'mobilenet_v3_small_vulkan.pte',
        3,
      );
      printBenchmarkResult('Vulkan ', vulkanTimes);

      if (xnnpackTimes.isNotEmpty && vulkanTimes.isNotEmpty) {
        final xAvg = xnnpackTimes.reduce((a, b) => a + b) / xnnpackTimes.length;
        final vAvg = vulkanTimes.reduce((a, b) => a + b) / vulkanTimes.length;
        print(
          '  Ratio: Vulkan is ${(vAvg / xAvg).toStringAsFixed(1)}x XNNPACK load time',
        );
      }

      print('');
    });

    testWidgets('YOLO11n: Inference time Vulkan vs XNNPACK', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- YOLO11n Inference Time (5 runs after warmup) ---');

      final xnnpackTimes = await benchmarkInference('yolo11n_xnnpack.pte', [
        1,
        3,
        640,
        640,
      ], 5);
      printBenchmarkResult('XNNPACK', xnnpackTimes);

      final vulkanTimes = await benchmarkInference('yolo11n_vulkan.pte', [
        1,
        3,
        640,
        640,
      ], 5);
      printBenchmarkResult('Vulkan ', vulkanTimes);

      if (xnnpackTimes.isNotEmpty && vulkanTimes.isNotEmpty) {
        final xAvg = xnnpackTimes.reduce((a, b) => a + b) / xnnpackTimes.length;
        final vAvg = vulkanTimes.reduce((a, b) => a + b) / vulkanTimes.length;
        final ratio = vAvg / xAvg;
        print(
          '  Ratio: Vulkan is ${ratio.toStringAsFixed(1)}x XNNPACK inference time '
          '(${ratio < 1 ? "faster" : "slower"})',
        );
      }

      print('');
    });

    testWidgets('YOLOv8n: Inference time Vulkan vs XNNPACK', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- YOLOv8n Inference Time (5 runs after warmup) ---');

      final xnnpackTimes = await benchmarkInference('yolov8n_xnnpack.pte', [
        1,
        3,
        640,
        640,
      ], 5);
      printBenchmarkResult('XNNPACK', xnnpackTimes);

      final vulkanTimes = await benchmarkInference('yolov8n_vulkan.pte', [
        1,
        3,
        640,
        640,
      ], 5);
      printBenchmarkResult('Vulkan ', vulkanTimes);

      if (xnnpackTimes.isNotEmpty && vulkanTimes.isNotEmpty) {
        final xAvg = xnnpackTimes.reduce((a, b) => a + b) / xnnpackTimes.length;
        final vAvg = vulkanTimes.reduce((a, b) => a + b) / vulkanTimes.length;
        final ratio = vAvg / xAvg;
        print(
          '  Ratio: Vulkan is ${ratio.toStringAsFixed(1)}x XNNPACK inference time '
          '(${ratio < 1 ? "faster" : "slower"})',
        );
      }

      print('');
    });

    testWidgets('MobileNet V3: Inference time Vulkan vs XNNPACK', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- MobileNet V3 Inference Time (5 runs after warmup) ---');

      final xnnpackTimes = await benchmarkInference(
        'mobilenet_v3_small_xnnpack.pte',
        [1, 3, 224, 224],
        5,
      );
      printBenchmarkResult('XNNPACK', xnnpackTimes);

      final vulkanTimes = await benchmarkInference(
        'mobilenet_v3_small_vulkan.pte',
        [1, 3, 224, 224],
        5,
      );
      printBenchmarkResult('Vulkan ', vulkanTimes);

      if (xnnpackTimes.isNotEmpty && vulkanTimes.isNotEmpty) {
        final xAvg = xnnpackTimes.reduce((a, b) => a + b) / xnnpackTimes.length;
        final vAvg = vulkanTimes.reduce((a, b) => a + b) / vulkanTimes.length;
        final ratio = vAvg / xAvg;
        print(
          '  Ratio: Vulkan is ${ratio.toStringAsFixed(1)}x XNNPACK inference time '
          '(${ratio < 1 ? "faster" : "slower"})',
        );
      }

      print('');
    });

    testWidgets('Summary', (WidgetTester tester) async {
      print('');
      print('========================================');
      print('  Benchmark Complete');
      print('========================================');
      print('');
      print(
        'Available backends: ${BackendQuery.available.map((b) => b.name).join(", ")}',
      );
      print('Models tested: ${modelPaths.keys.join(", ")}');
      print('');
      print('NOTE: Vulkan load time includes shader compilation');
      print('and GPU resource allocation (prepacking weights to');
      print('textures). This is a one-time cost per model load.');
      print('');
    });
  });
}
