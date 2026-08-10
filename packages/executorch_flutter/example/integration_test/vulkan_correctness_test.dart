// Vulkan output correctness against XNNPACK, on the same inputs.
//
// The benchmark test next door only times things: it would pass just as happily
// on a backend that returned all-NaN. This one feeds byte-identical input to the
// XNNPACK and Vulkan builds of the same model and compares what comes back.
//
// Context: PowerVR GPUs (Pixel 10 Pro) corrupt prepacked weights when several
// prepack dispatches share one Vulkan command buffer, which shows up as NaN or
// wildly wrong outputs while every timing test still passes.
// See https://github.com/abdelaziz-mahdy/executorch_flutter/issues/26 and
// https://github.com/pytorch/executorch/issues/17299.
//
// Usage: flutter test integration_test/vulkan_correctness_test.dart -d <device>

// ignore_for_file: avoid_print

@Timeout(Duration(minutes: 30))
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

/// Deterministic input, so both backends see bit-identical bytes and any
/// difference in the output is the backend's doing. A fixed ramp would leave
/// large parts of the tensor identical and could hide per-texel corruption, so
/// this is a cheap LCG instead.
Float32List _deterministicInput(int count, {int seed = 0x2545F491}) {
  final out = Float32List(count);
  var state = seed;
  for (var i = 0; i < count; i++) {
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
    out[i] = (state % 10000) / 10000.0; // [0, 1)
  }
  return out;
}

Float32List _asFloat32(TensorData tensor) {
  final bytes = tensor.data;
  return bytes.buffer.asFloat32List(
    bytes.offsetInBytes,
    bytes.lengthInBytes ~/ 4,
  );
}

class _Comparison {
  _Comparison({
    required this.nanCount,
    required this.infCount,
    required this.maxAbsDiff,
    required this.meanAbsDiff,
    required this.cosine,
    required this.refArgmax,
    required this.testArgmax,
    required this.refAbsMax,
    required this.worstRefValue,
    required this.worstTestValue,
    required this.refValueAtRefArgmax,
    required this.refValueAtTestArgmax,
    required this.length,
  });

  final int nanCount;
  final int infCount;
  final double maxAbsDiff;
  final double meanAbsDiff;
  final double cosine;
  final int refArgmax;
  final int testArgmax;

  /// Largest magnitude in the reference tensor — the scale `maxAbsDiff` has to
  /// be read against. A diff of 13.6 is corruption in a tensor that peaks at 20
  /// and rounding noise in one that peaks at 5000.
  final double refAbsMax;
  final double worstRefValue;
  final double worstTestValue;
  final int length;

  /// Reference values at the two peak positions.
  ///
  /// Index equality is the wrong test on YOLO's raw `[1, 84, 8400]` output: its
  /// global maximum is a box coordinate near the image size, and many of the
  /// 8400 anchors carry near-identical maxima, so FP16 noise reorders a tie and
  /// moves the index while the tensor is unchanged. What must hold is that the
  /// peak Vulkan picked is *also* a peak in the reference.
  final double refValueAtRefArgmax;
  final double refValueAtTestArgmax;

  bool get argmaxMatches => refArgmax == testArgmax;

  /// How far below the reference peak the Vulkan-chosen position sits, as a
  /// fraction of the tensor scale. Zero when the indices agree, near zero for a
  /// tie, large when Vulkan peaked somewhere the reference considers ordinary.
  double get argmaxValueGap => refAbsMax > 0
      ? (refValueAtRefArgmax - refValueAtTestArgmax).abs() / refAbsMax
      : 0.0;

  /// Worst elementwise difference as a fraction of the tensor's own scale.
  double get relativeDiff => refAbsMax > 0 ? maxAbsDiff / refAbsMax : 0.0;

  @override
  String toString() =>
      'len=$length NaN=$nanCount Inf=$infCount '
      'maxDiff=${maxAbsDiff.toStringAsExponential(3)} '
      '(${worstRefValue.toStringAsExponential(3)} vs '
      '${worstTestValue.toStringAsExponential(3)}, '
      'scale=${refAbsMax.toStringAsExponential(3)}, '
      'rel=${relativeDiff.toStringAsExponential(3)}) '
      'meanDiff=${meanAbsDiff.toStringAsExponential(3)} '
      'cosine=${cosine.toStringAsFixed(6)} '
      'argmax=$testArgmax (ref $refArgmax, '
      'refVal ${refValueAtTestArgmax.toStringAsExponential(3)} vs peak '
      '${refValueAtRefArgmax.toStringAsExponential(3)}, '
      'gap=${argmaxValueGap.toStringAsExponential(3)})';
}

_Comparison _compare(Float32List reference, Float32List test) {
  final n = math.min(reference.length, test.length);

  var nanCount = 0;
  var infCount = 0;
  var maxAbsDiff = 0.0;
  var sumAbsDiff = 0.0;
  var dot = 0.0;
  var refNorm = 0.0;
  var testNorm = 0.0;
  var refArgmax = 0;
  var testArgmax = 0;
  var refAbsMax = 0.0;
  var worstRefValue = 0.0;
  var worstTestValue = 0.0;

  for (var i = 0; i < n; i++) {
    final r = reference[i];
    final t = test[i];

    final refAbs = r.abs();
    if (refAbs > refAbsMax) refAbsMax = refAbs;

    if (t.isNaN) {
      nanCount++;
      continue; // NaN poisons every other statistic
    }
    if (t.isInfinite) {
      infCount++;
      continue;
    }

    final diff = (r - t).abs();
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      worstRefValue = r;
      worstTestValue = t;
    }
    sumAbsDiff += diff;

    dot += r * t;
    refNorm += r * r;
    testNorm += t * t;

    if (r > reference[refArgmax]) refArgmax = i;
    if (t > test[testArgmax] || test[testArgmax].isNaN) testArgmax = i;
  }

  final finite = n - nanCount - infCount;
  final denom = math.sqrt(refNorm) * math.sqrt(testNorm);

  return _Comparison(
    nanCount: nanCount,
    infCount: infCount,
    maxAbsDiff: maxAbsDiff,
    meanAbsDiff: finite > 0 ? sumAbsDiff / finite : double.nan,
    cosine: denom > 0 ? dot / denom : double.nan,
    refArgmax: refArgmax,
    testArgmax: testArgmax,
    refAbsMax: refAbsMax,
    worstRefValue: worstRefValue,
    worstTestValue: worstTestValue,
    refValueAtRefArgmax: reference[refArgmax],
    refValueAtTestArgmax: reference[testArgmax],
    length: n,
  );
}

/// Assert one Vulkan run against the XNNPACK reference, tensor by tensor.
///
/// Thresholds are set against measured behaviour on both sides:
///
///  - The Vulkan `.pte` files are FP16 (half the size of the XNNPACK ones), so
///    an absolute tolerance on raw activations is meaningless — a diff of 13.6
///    is rounding noise in a tensor that peaks near 640. Everything below is
///    therefore scale-relative or scale-free.
///  - The prepack corruption this guards against was never subtle: it produced
///    NaN across whole tensors, or wrong argmax with diffs of 4.27 and 50.12
///    (pytorch/executorch#17299). Cosine similarity and argmax agreement both
///    collapse under it while staying pinned near 1.0 under FP16 rounding.
void _assertOutputsMatch(
  String label,
  int load,
  List<Float32List> reference,
  List<Float32List> actual,
) {
  for (var i = 0; i < reference.length; i++) {
    final cmp = _compare(reference[i], actual[i]);
    print('  load $load output[$i]: $cmp');

    expect(
      cmp.nanCount + cmp.infCount,
      0,
      reason: '$label load $load output[$i]: Vulkan produced NaN/Inf values',
    );
    // Not index equality — see the doc on `argmaxValueGap`. The position Vulkan
    // peaked at must also be a peak in the reference; a tie reordered by FP16
    // rounding is fine, peaking somewhere the reference considers ordinary is
    // not.
    expect(
      cmp.argmaxValueGap,
      lessThan(0.02),
      reason:
          '$label load $load output[$i]: Vulkan peaked at index '
          '${cmp.testArgmax}, where the reference holds '
          '${cmp.refValueAtTestArgmax} — far below its own peak '
          '${cmp.refValueAtRefArgmax} at index ${cmp.refArgmax}',
    );
    expect(
      cmp.cosine,
      greaterThan(0.999),
      reason:
          '$label load $load output[$i]: cosine similarity ${cmp.cosine} — '
          'the tensors disagree in shape, not just precision',
    );
    expect(
      cmp.relativeDiff,
      lessThan(0.05),
      reason:
          '$label load $load output[$i]: worst element differs by '
          '${(cmp.relativeDiff * 100).toStringAsFixed(1)}% of the tensor scale '
          '(${cmp.worstRefValue} vs ${cmp.worstTestValue})',
    );
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan output correctness vs XNNPACK', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};

    Future<String> downloadModel(ModelIndexEntry entry) async {
      final info = await ModelDownloadService.instance.downloadModel(
        modelName: entry.name,
        remoteUrl: entry.remoteUrl,
        expectedHash: entry.hash,
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
      print('  Vulkan correctness vs XNNPACK');
      print('========================================');
      print('Vulkan available : ${BackendQuery.isAvailable(Backend.vulkan)}');
      print('XNNPACK available: ${BackendQuery.isAvailable(Backend.xnnpack)}');
      print('');

      const wanted = [
        'mobilenet_v3_small_xnnpack.pte',
        'mobilenet_v3_small_vulkan.pte',
        'yolo11n_xnnpack.pte',
        'yolo11n_vulkan.pte',
        'yolov8n_xnnpack.pte',
        'yolov8n_vulkan.pte',
        'yolov5n_xnnpack.pte',
        'yolov5n_vulkan.pte',
      ];

      // `flutter test` reinstalls the app every run, wiping its storage, so all
      // ~70 MB comes down the wire each time. Pushing the files in with adb
      // does not help: writes into Android/data/<package>/ land in the shell's
      // storage view and the app cannot see them.
      //
      // The device radio drops connections often enough to fail a run outright,
      // so retry rather than let a flaky download decide whether a GPU
      // correctness test gets to run at all.
      final index = await ModelIndexService.fetchIndex(forceRefresh: true);

      for (final name in wanted) {
        final matches = index.models.where((m) => m.name == name);
        if (matches.isEmpty) {
          print('  UNAVAILABLE: $name is not listed in the model index');
          continue;
        }

        const attempts = 4;
        for (var attempt = 1; attempt <= attempts; attempt++) {
          try {
            modelPaths[name] = await downloadModel(matches.first);
            break;
          } catch (e) {
            final last = attempt == attempts;
            print(
              '  ${last ? "UNAVAILABLE" : "retry"}: $name '
              'attempt $attempt/$attempts failed: $e',
            );
            if (last) break;
            // A dropped connection tends to take the next few DNS lookups with
            // it, so back off rather than retrying straight into the failure.
            await Future<void>.delayed(Duration(seconds: 3 * attempt));
          }
        }
      }
      print('Models ready: ${modelPaths.length}/${wanted.length}');
      print('');
    });

    tearDownAll(() async {
      await manager.disposeAllModels();
    });

    /// Run one model on one input tensor and hand back its outputs as floats.
    Future<List<Float32List>> runModel(
      String modelName,
      List<int> shape,
      Float32List input,
    ) async {
      final path = modelPaths[modelName];
      if (path == null) throw StateError('$modelName not downloaded');

      final model = await ExecuTorchModel.load(path);
      try {
        final outputs = await model.forward([
          TensorData(
            shape: shape,
            dataType: TensorType.float32,
            data: input.buffer.asUint8List(),
            name: 'input',
          ),
        ]);
        // Copy out before dispose frees the native buffers.
        return outputs.map((o) => Float32List.fromList(_asFloat32(o))).toList();
      } finally {
        await model.dispose();
      }
    }

    Future<void> checkModel({
      required String label,
      required String xnnpackModel,
      required String vulkanModel,
      required List<int> shape,
      // The corruption this guards against happened during prepack, at load
      // time, and depended on dispatch ordering within a command buffer — so a
      // single load is a thin sample. Each repeat is a fresh load of the model.
      int loads = 4,
    }) async {
      // Never skip quietly. A comparison test that reports success because it
      // had nothing to compare is worse than no test at all — it is exactly how
      // a backend regression would slip through unnoticed.
      if (!modelPaths.containsKey(xnnpackModel) ||
          !modelPaths.containsKey(vulkanModel)) {
        fail(
          '$label: required models unavailable ($xnnpackModel / $vulkanModel). '
          'See the setUpAll output above for the download error.',
        );
      }

      final elementCount = shape.reduce((a, b) => a * b);
      final input = _deterministicInput(elementCount);

      print('--- $label ---');
      final reference = await runModel(xnnpackModel, shape, input);

      List<Float32List>? firstVulkanRun;
      for (var load = 1; load <= loads; load++) {
        final actual = await runModel(vulkanModel, shape, input);
        expect(
          actual.length,
          reference.length,
          reason: '$label: backends returned different output counts',
        );

        // Same weights, same input, same device: two loads must agree exactly.
        // Any drift here is intermittent prepack corruption, which an
        // against-reference check alone can miss when both loads are wrong the
        // same way.
        if (firstVulkanRun == null) {
          firstVulkanRun = actual;
        } else {
          var crossRunDiff = 0.0;
          for (var i = 0; i < actual.length; i++) {
            for (var j = 0; j < actual[i].length; j++) {
              final d = (firstVulkanRun[i][j] - actual[i][j]).abs();
              if (d > crossRunDiff) crossRunDiff = d;
            }
          }
          print(
            '  load $load vs load 1: maxDiff='
            '${crossRunDiff.toStringAsExponential(3)}',
          );
          expect(
            crossRunDiff,
            0.0,
            reason:
                '$label load $load: Vulkan output differs between loads of the '
                'same model — prepack is not deterministic',
          );
        }

        _assertOutputsMatch(label, load, reference, actual);
      }
      print('');
    }

    testWidgets('MobileNet V3 Small: Vulkan matches XNNPACK', (_) async {
      await checkModel(
        label: 'MobileNet V3 Small',
        xnnpackModel: 'mobilenet_v3_small_xnnpack.pte',
        vulkanModel: 'mobilenet_v3_small_vulkan.pte',
        shape: [1, 3, 224, 224],
      );
    });

    testWidgets('YOLO11n: Vulkan matches XNNPACK', (_) async {
      // More prepack nodes than MobileNet, so a better shot at tripping the
      // batched-dispatch corruption if it is still present.
      await checkModel(
        label: 'YOLO11n',
        xnnpackModel: 'yolo11n_xnnpack.pte',
        vulkanModel: 'yolo11n_vulkan.pte',
        shape: [1, 3, 640, 640],
      );
    });

    testWidgets('YOLOv8n: Vulkan matches XNNPACK', (_) async {
      await checkModel(
        label: 'YOLOv8n',
        xnnpackModel: 'yolov8n_xnnpack.pte',
        vulkanModel: 'yolov8n_vulkan.pte',
        shape: [1, 3, 640, 640],
      );
    });

    testWidgets('YOLOv5n: Vulkan matches XNNPACK', (_) async {
      await checkModel(
        label: 'YOLOv5n',
        xnnpackModel: 'yolov5n_xnnpack.pte',
        vulkanModel: 'yolov5n_vulkan.pte',
        shape: [1, 3, 640, 640],
      );
    });
  });
}
