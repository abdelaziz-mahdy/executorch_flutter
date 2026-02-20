// ignore_for_file: avoid_print

/// Vulkan correctness + load time test.
/// Compares Vulkan output against XNNPACK reference and reports load times.

@Timeout(Duration(minutes: 10))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:path_provider/path_provider.dart';

import 'package:executorch_flutter_example/services/model_index_service.dart';
import 'package:executorch_flutter_example/services/model_download_service.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan Correctness & Load Time', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};

    Future<String> downloadModel(ModelIndexEntry entry) async {
      final dir = await getApplicationCacheDirectory();
      final cached = File('${dir.path}/test_models/${entry.name}');

      // Use cached model if it exists
      if (await cached.exists()) {
        print('  Using cached ${entry.name}');
        return cached.path;
      }

      print('  Downloading ${entry.name}...');
      final info = await ModelDownloadService.instance.downloadModel(
        modelName: entry.name,
        remoteUrl: entry.remoteUrl,
        expectedHash: entry.hash,
      );
      if (info.state != ModelDownloadState.downloaded) {
        throw Exception('Failed: ${info.errorMessage}');
      }

      // Cache the model
      await cached.parent.create(recursive: true);
      if (info.localPath != null) {
        await File(info.localPath!).copy(cached.path);
        return cached.path;
      }
      if (info.bytes != null) {
        await cached.writeAsBytes(info.bytes!);
        return cached.path;
      }
      throw Exception('No path or bytes');
    }

    setUpAll(() async {
      manager = ExecutorchManager.instance;
      await manager.initialize();

      final index = await ModelIndexService.fetchIndex(forceRefresh: true);
      for (final name in [
        'mobilenet_v3_small_xnnpack.pte',
        'mobilenet_v3_small_vulkan.pte',
        'yolo11n_xnnpack.pte',
        'yolo11n_vulkan.pte',
      ]) {
        try {
          final entry = index.models.firstWhere((m) => m.name == name);
          modelPaths[name] = await downloadModel(entry);
        } catch (e) {
          print('  SKIP: $name ($e)');
        }
      }
    });

    tearDownAll(() async {
      await manager.disposeAllModels();
    });

    /// Load model and return (model, loadTimeMs)
    Future<(ExecuTorchModel, double)?> loadModelTimed(String modelName) async {
      final path = modelPaths[modelName];
      if (path == null) return null;

      final sw = Stopwatch()..start();
      final model = await ExecuTorchModel.load(path);
      sw.stop();
      return (model, sw.elapsedMilliseconds.toDouble());
    }

    testWidgets('MobileNet: Vulkan correctness & load time', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- MobileNet V3 Small ---');

      // Load both models with timing
      final xnnpackResult = await loadModelTimed(
        'mobilenet_v3_small_xnnpack.pte',
      );
      final vulkanResult = await loadModelTimed(
        'mobilenet_v3_small_vulkan.pte',
      );

      if (xnnpackResult == null || vulkanResult == null) {
        print('  SKIP: models not available');
        return;
      }

      final (xnnpackModel, xnnpackLoadMs) = xnnpackResult;
      final (vulkanModel, vulkanLoadMs) = vulkanResult;

      print(
        '  Load time: XNNPACK=${xnnpackLoadMs.toStringAsFixed(0)}ms, '
        'Vulkan=${vulkanLoadMs.toStringAsFixed(0)}ms '
        '(${(vulkanLoadMs / xnnpackLoadMs).toStringAsFixed(0)}x)',
      );

      // Run inference with consistent input
      final inputData = List.filled(1 * 3 * 224 * 224, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
      );

      final xnnpackOutputs = await xnnpackModel.forward([inputTensor]);
      final vulkanOutputs = await vulkanModel.forward([inputTensor]);

      await xnnpackModel.dispose();
      await vulkanModel.dispose();

      final xOut = xnnpackOutputs[0].data.buffer.asFloat32List();
      final vOut = vulkanOutputs[0].data.buffer.asFloat32List();

      expect(xOut.length, equals(vOut.length));

      // Top-5 comparison
      final xTop5 = _topK(xOut, 5);
      final vTop5 = _topK(vOut, 5);

      print(
        '  XNNPACK top-5: ${xTop5.map((e) => "class ${e.$1} (${e.$2.toStringAsFixed(4)})").join(", ")}',
      );
      print(
        '  Vulkan  top-5: ${vTop5.map((e) => "class ${e.$1} (${e.$2.toStringAsFixed(4)})").join(", ")}',
      );

      // Check NaN/Inf
      final hasNaN = vOut.any((v) => v.isNaN);
      final hasInf = vOut.any((v) => v.isInfinite);
      expect(hasNaN, isFalse, reason: 'Vulkan output should not contain NaN');
      expect(hasInf, isFalse, reason: 'Vulkan output should not contain Inf');

      // Check top-1 match
      expect(vTop5[0].$1, equals(xTop5[0].$1),
          reason: 'Vulkan top-1 class should match XNNPACK');

      // Max/avg diff
      double maxDiff = 0, sumDiff = 0;
      for (int i = 0; i < xOut.length; i++) {
        final diff = (xOut[i] - vOut[i]).abs();
        maxDiff = math.max(maxDiff, diff);
        sumDiff += diff;
      }
      print(
        '  Max diff: ${maxDiff.toStringAsFixed(6)}, '
        'Avg diff: ${(sumDiff / xOut.length).toStringAsFixed(6)}',
      );
      print('  Top-1 match: PASS');
      print('');
    });

    testWidgets('YOLO11n: Vulkan correctness & load time', (
      WidgetTester tester,
    ) async {
      print('');
      print('--- YOLO11n ---');

      // Load both models with timing
      final xnnpackResult = await loadModelTimed('yolo11n_xnnpack.pte');
      final vulkanResult = await loadModelTimed('yolo11n_vulkan.pte');

      if (xnnpackResult == null || vulkanResult == null) {
        print('  SKIP: models not available');
        return;
      }

      final (xnnpackModel, xnnpackLoadMs) = xnnpackResult;
      final (vulkanModel, vulkanLoadMs) = vulkanResult;

      print(
        '  Load time: XNNPACK=${xnnpackLoadMs.toStringAsFixed(0)}ms, '
        'Vulkan=${vulkanLoadMs.toStringAsFixed(0)}ms '
        '(${(vulkanLoadMs / xnnpackLoadMs).toStringAsFixed(0)}x)',
      );

      // Run inference
      final inputData = List.filled(1 * 3 * 640 * 640, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: inputData,
      );

      final xnnpackOutputs = await xnnpackModel.forward([inputTensor]);
      final vulkanOutputs = await vulkanModel.forward([inputTensor]);

      await xnnpackModel.dispose();
      await vulkanModel.dispose();

      expect(vulkanOutputs.length, equals(xnnpackOutputs.length));

      bool allPassed = true;
      for (int out = 0; out < xnnpackOutputs.length; out++) {
        final xData = xnnpackOutputs[out].data.buffer.asFloat32List();
        final vData = vulkanOutputs[out].data.buffer.asFloat32List();

        final hasNaN = vData.any((v) => v.isNaN);
        final hasInf = vData.any((v) => v.isInfinite);
        if (hasNaN || hasInf) allPassed = false;

        double maxDiff = 0;
        for (int i = 0; i < xData.length && i < vData.length; i++) {
          if (!xData[i].isNaN && !vData[i].isNaN) {
            maxDiff = math.max(maxDiff, (xData[i] - vData[i]).abs());
          }
        }

        final status = (!hasNaN && !hasInf) ? 'OK' : 'FAIL';
        print(
          '  Output[$out] shape=${xnnpackOutputs[out].shape}: '
          'maxDiff=${maxDiff.toStringAsFixed(4)} '
          'NaN=$hasNaN Inf=$hasInf [$status]',
        );
      }

      expect(allPassed, isTrue,
          reason: 'All Vulkan YOLO outputs should be free of NaN/Inf');
      print('');
    });
  });
}

List<(int, double)> _topK(Float32List values, int k) {
  final indexed = List.generate(values.length, (i) => (i, values[i]));
  indexed.sort((a, b) => b.$2.compareTo(a.$2));
  return indexed.take(k).toList();
}
