// ignore_for_file: avoid_print

/// Vulkan PowerVR Barrier Mode Test
///
/// Tests 6 different barrier strategies between prepack dispatches on PowerVR:
///   Mode 0: No barrier (baseline, known broken)
///   Mode 1: Execution-only barrier (stage dependency, no memory visibility)
///   Mode 2: Global memory barrier (compute→compute with access flags)
///   Mode 3: Full submit+wait (current fix, slowest)
///   Mode 4: Submit without wait (fire-and-forget, new cmd buffer, no CPU stall)
///   Mode 5: Submit without wait + flush (fire-and-forget + recycle staging)
///
/// The C++ code auto-cycles through modes on each Vulkan model load.
/// This test loads the same Vulkan model 6 times and compares each
/// against XNNPACK reference output.

@Timeout(Duration(minutes: 15))
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

const _numModes = 11;

const _modeNames = [
  'NO_BARRIER (baseline)',
  'EXECUTION_BARRIER (stage-only)',
  'MEMORY_BARRIER (compute→compute)',
  'FULL_SUBMIT_WAIT (current fix)',
  'SUBMIT_NO_WAIT (fire-and-forget)',
  'SUBMIT_NO_WAIT_FLUSH (fire-and-forget + flush)',
  'BATCH_2 (submit every 2 nodes)',
  'BATCH_4 (submit every 4 nodes)',
  'BATCH_8 (submit every 8 nodes)',
  'BATCH_16 (submit every 16 nodes)',
  'HYBRID_UBO (serialize push-const only)',
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('PowerVR Barrier Mode Test', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};

    Future<String> downloadModel(ModelIndexEntry entry) async {
      final dir = await getApplicationCacheDirectory();
      final cached = File('${dir.path}/test_models/${entry.name}');

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

    testWidgets('MobileNet: test all $_numModes barrier modes', (
      WidgetTester tester,
    ) async {
      final xnnpackPath = modelPaths['mobilenet_v3_small_xnnpack.pte'];
      final vulkanPath = modelPaths['mobilenet_v3_small_vulkan.pte'];

      if (xnnpackPath == null || vulkanPath == null) {
        print('SKIP: models not available');
        return;
      }

      // --- Load XNNPACK reference ---
      print('');
      print('=' * 70);
      print('  XNNPACK REFERENCE');
      print('=' * 70);

      final xnnpackModel = await ExecuTorchModel.load(xnnpackPath);

      final inputData = List.filled(1 * 3 * 224 * 224, 0.5);
      final inputTensor = manager.createTensorData(
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        data: inputData,
      );

      final xnnpackOutputs = await xnnpackModel.forward([inputTensor]);
      final xOut = xnnpackOutputs[0].data.buffer.asFloat32List();
      final xTop5 = _topK(xOut, 5);

      print(
        '  Top-5: ${xTop5.map((e) => "class ${e.$1} (${e.$2.toStringAsFixed(4)})").join(", ")}',
      );
      await xnnpackModel.dispose();

      // --- Test each barrier mode ---
      // The C++ static counter auto-cycles: first vulkan load = mode 0,
      // second = mode 1, ..., sixth = mode 5.

      final results = <int, _ModeResult>{};

      for (int mode = 0; mode < _numModes; mode++) {
        print('');
        print('=' * 70);
        print('  MODE $mode: ${_modeNames[mode]}');
        print('=' * 70);

        try {
          final sw = Stopwatch()..start();
          final vulkanModel = await ExecuTorchModel.load(vulkanPath);
          sw.stop();
          final loadTimeMs = sw.elapsedMilliseconds;

          print('  Load time: ${loadTimeMs}ms');

          final vulkanOutputs = await vulkanModel.forward([inputTensor]);
          final vOut = vulkanOutputs[0].data.buffer.asFloat32List();

          await vulkanModel.dispose();

          // Check for NaN/Inf
          final hasNaN = vOut.any((v) => v.isNaN);
          final hasInf = vOut.any((v) => v.isInfinite);

          // Compute diffs
          double maxDiff = 0, sumDiff = 0;
          int nanCount = 0, infCount = 0;
          for (int i = 0; i < xOut.length && i < vOut.length; i++) {
            if (vOut[i].isNaN) {
              nanCount++;
              continue;
            }
            if (vOut[i].isInfinite) {
              infCount++;
              continue;
            }
            final diff = (xOut[i] - vOut[i]).abs();
            maxDiff = math.max(maxDiff, diff);
            sumDiff += diff;
          }
          final avgDiff =
              (xOut.length - nanCount - infCount) > 0
                  ? sumDiff / (xOut.length - nanCount - infCount)
                  : double.nan;

          // Top-5
          final vTop5 = _topK(vOut, 5);
          final top1Match = vTop5[0].$1 == xTop5[0].$1;

          print(
            '  Top-5: ${vTop5.map((e) => "class ${e.$1} (${e.$2.toStringAsFixed(4)})").join(", ")}',
          );
          print(
            '  Max diff: ${maxDiff.toStringAsFixed(6)}, '
            'Avg diff: ${avgDiff.toStringAsFixed(6)}',
          );
          print('  NaN count: $nanCount, Inf count: $infCount');
          print('  Top-1 match: ${top1Match ? "PASS" : "FAIL"}');

          final correct = !hasNaN && !hasInf && top1Match && maxDiff < 1.0;
          print(
            '  >>> MODE $mode RESULT: ${correct ? "CORRECT" : "INCORRECT"} <<<',
          );

          results[mode] = _ModeResult(
            mode: mode,
            name: _modeNames[mode],
            loadTimeMs: loadTimeMs,
            maxDiff: maxDiff,
            avgDiff: avgDiff,
            nanCount: nanCount,
            infCount: infCount,
            top1Match: top1Match,
            correct: correct,
          );
        } catch (e, stack) {
          print('  >>> MODE $mode CRASHED: $e <<<');
          print('  $stack');
          results[mode] = _ModeResult(
            mode: mode,
            name: _modeNames[mode],
            loadTimeMs: -1,
            maxDiff: double.nan,
            avgDiff: double.nan,
            nanCount: -1,
            infCount: -1,
            top1Match: false,
            correct: false,
            error: e.toString(),
          );
        }
      }

      // --- Summary ---
      print('');
      print('=' * 70);
      print('  SUMMARY');
      print('=' * 70);
      print('');
      print(
        '  ${'Mode'.padRight(6)} '
        '${'Name'.padRight(48)} '
        '${'Load'.padRight(9)} '
        '${'MaxDiff'.padRight(10)} '
        '${'NaN'.padRight(6)} '
        '${'Top1'.padRight(6)} '
        'Result',
      );
      print('  ${'-' * 100}');

      for (int mode = 0; mode < _numModes; mode++) {
        final r = results[mode];
        if (r == null) continue;

        final loadStr =
            r.loadTimeMs >= 0 ? '${r.loadTimeMs}ms' : 'CRASH';
        final diffStr =
            r.maxDiff.isNaN ? 'N/A' : r.maxDiff.toStringAsFixed(4);
        final nanStr = r.nanCount >= 0 ? '${r.nanCount}' : 'N/A';
        final top1Str = r.top1Match ? 'YES' : 'NO';
        final resultStr = r.error != null
            ? 'CRASH'
            : (r.correct ? 'PASS' : 'FAIL');

        print(
          '  ${mode.toString().padRight(6)} '
          '${r.name.padRight(48)} '
          '${loadStr.padRight(9)} '
          '${diffStr.padRight(10)} '
          '${nanStr.padRight(6)} '
          '${top1Str.padRight(6)} '
          '$resultStr',
        );
      }
      print('');

      // At least mode 3 (full submit+wait) should pass
      final mode3 = results[3];
      expect(
        mode3?.correct,
        isTrue,
        reason: 'Mode 3 (full submit+wait) should always produce correct results',
      );
    });
  });
}

class _ModeResult {
  final int mode;
  final String name;
  final int loadTimeMs;
  final double maxDiff;
  final double avgDiff;
  final int nanCount;
  final int infCount;
  final bool top1Match;
  final bool correct;
  final String? error;

  _ModeResult({
    required this.mode,
    required this.name,
    required this.loadTimeMs,
    required this.maxDiff,
    required this.avgDiff,
    required this.nanCount,
    required this.infCount,
    required this.top1Match,
    required this.correct,
    this.error,
  });
}

List<(int, double)> _topK(Float32List values, int k) {
  final indexed = List.generate(values.length, (i) => (i, values[i]));
  indexed.sort((a, b) => b.$2.compareTo(a.$2));
  return indexed.take(k).toList();
}
