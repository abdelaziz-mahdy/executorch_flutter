// ignore_for_file: avoid_print

/// V2 Progressive Vulkan diagnostic test.
///
/// Tests isolated conv operations and progressive MobileNet slices
/// to pinpoint exactly where CPU prepack produces incorrect results.
///
/// Run:
///   cd example
///   flutter test integration_test/vulkan_v2_progressive_test.dart -d 56241FDCH00AA2

@Timeout(Duration(minutes: 10))
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Test case loaded from v2_manifest.json.
class V2TestCase {
  final String name;
  final String dtype;
  final String description;
  final List<int> inputShape;
  final Map<String, dynamic> cpuRef;
  final String vulkanAsset;
  final String? xnnpackAsset;

  V2TestCase({
    required this.name,
    required this.dtype,
    required this.description,
    required this.inputShape,
    required this.cpuRef,
    required this.vulkanAsset,
    this.xnnpackAsset,
  });

  factory V2TestCase.fromJson(Map<String, dynamic> json) {
    return V2TestCase(
      name: json['name'] as String,
      dtype: json['dtype'] as String,
      description: json['description'] as String,
      inputShape:
          (json['input_shape'] as List).map((e) => (e as num).toInt()).toList(),
      cpuRef: json['cpu_ref'] as Map<String, dynamic>,
      vulkanAsset: json['vulkan_asset'] as String,
      xnnpackAsset: json['xnnpack_asset'] as String?,
    );
  }

  List<int> get outputShape =>
      (cpuRef['shape'] as List).map((e) => (e as num).toInt()).toList();
  double get expectedMean => (cpuRef['mean'] as num).toDouble();
  double get expectedMin => (cpuRef['min'] as num).toDouble();
  double get expectedMax => (cpuRef['max'] as num).toDouble();
  List<double> get expectedFirst8 =>
      (cpuRef['first8'] as List).map((e) => (e as num).toDouble()).toList();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('V2 Progressive Vulkan Tests', () {
    late List<V2TestCase> allTests;

    setUpAll(() async {
      final manifestStr =
          await rootBundle.loadString('assets/debug_models/v2_manifest.json');
      final manifest = jsonDecode(manifestStr) as Map<String, dynamic>;
      allTests = (manifest['tests'] as List)
          .map((e) => V2TestCase.fromJson(e as Map<String, dynamic>))
          .toList();
      print('Loaded ${allTests.length} test cases from v2_manifest.json');
    });

    // ── Phase 1: Isolated ops (FP32 only, with XNNPACK reference) ──
    for (final testName in [
      'add_bias_only',
      'conv3x3_nobias',
      'conv3x3_bias',
      'conv3x3_bias32ch',
      'pw_nobias',
      'pw_bias',
      'dw_nobias',
      'dw_bias',
    ]) {
      testWidgets('FP32 $testName', (tester) async {
        // Find the FP32 test case
        final tc = allTests.firstWhere(
          (t) => t.name == testName && t.dtype == 'FP32',
        );
        await _runTestCase(tc);
      });
    }

    // ── Phase 2: MobileNet slices (FP32 only) ──
    for (final sliceNum in [1, 2, 3, 4]) {
      testWidgets('FP32 mobilenet_slice_$sliceNum', (tester) async {
        final tc = allTests.firstWhere(
          (t) => t.name == 'mobilenet_slice_$sliceNum' && t.dtype == 'FP32',
        );
        await _runTestCase(tc);
      });
    }

    // ── Phase 3: FP16 isolated ops ──
    for (final testName in [
      'add_bias_only',
      'conv3x3_nobias',
      'conv3x3_bias',
      'pw_nobias',
      'dw_nobias',
    ]) {
      testWidgets('FP16 $testName', (tester) async {
        final tc = allTests.firstWhere(
          (t) => t.name == testName && t.dtype == 'FP16',
        );
        await _runTestCase(tc);
      });
    }

    // ── Phase 4: FP16 MobileNet slices ──
    for (final sliceNum in [1, 2]) {
      testWidgets('FP16 mobilenet_slice_$sliceNum', (tester) async {
        final tc = allTests.firstWhere(
          (t) => t.name == 'mobilenet_slice_$sliceNum' && t.dtype == 'FP16',
        );
        await _runTestCase(tc);
      });
    }
  });
}

/// Create all-ones input tensor matching the test case's input shape.
Float32List _createOnesInput(List<int> shape) {
  int numel = 1;
  for (final s in shape) {
    numel *= s;
  }
  return Float32List(numel)..fillRange(0, numel, 1.0);
}

/// Run a single test case: load Vulkan model (and XNNPACK if available),
/// compare against expected CPU reference values.
Future<void> _runTestCase(V2TestCase tc) async {
  print('\n${"=" * 70}');
  print('TEST: ${tc.dtype} ${tc.name}');
  print('  ${tc.description}');
  print('  Input: ${tc.inputShape}, Output: ${tc.outputShape}');
  print(
    '  Expected: mean=${tc.expectedMean.toStringAsFixed(4)}, '
    'range=[${tc.expectedMin.toStringAsFixed(4)}, ${tc.expectedMax.toStringAsFixed(4)}]',
  );
  print(
    '  Expected first8: ${tc.expectedFirst8.map((v) => v.toStringAsFixed(4)).toList()}',
  );

  // Create input tensor (all ones)
  final inputData = _createOnesInput(tc.inputShape);
  final inputTensor = TensorData(
    shape: tc.inputShape,
    dataType: TensorType.float32,
    data: inputData.buffer.asUint8List(),
  );

  // Run XNNPACK reference first (if available)
  Float32List? xnnpackOutput;
  if (tc.xnnpackAsset != null) {
    xnnpackOutput = await _runModel(
      'assets/debug_models/${tc.xnnpackAsset}',
      [inputTensor],
      'XNNPACK',
    );
  }

  // Run Vulkan
  final vulkanOutput = await _runModel(
    'assets/debug_models/${tc.vulkanAsset}',
    [inputTensor],
    'Vulkan',
  );

  // Compare Vulkan vs CPU reference
  if (vulkanOutput != null) {
    _compareAgainstRef(vulkanOutput, tc, 'Vulkan');
  }

  // Compare Vulkan vs XNNPACK
  if (vulkanOutput != null && xnnpackOutput != null) {
    _compareOutputs(xnnpackOutput, vulkanOutput, 'XNNPACK vs Vulkan');
  }

  print('${"=" * 70}\n');
}

/// Run a model and return its float output (or null on failure).
Future<Float32List?> _runModel(
  String asset,
  List<TensorData> input,
  String label,
) async {
  ExecuTorchModel? model;
  try {
    try {
      model = await ExecuTorchModel.loadFromAsset(asset);
    } catch (e) {
      print('  [$label] LOAD FAILED: $e');
      return null;
    }

    final sw = Stopwatch()..start();
    final outputs = await model.forward(input);
    sw.stop();

    final output = outputs.first;
    final floats = output.data.buffer.asFloat32List();

    // Stats
    int nanCount = 0, infCount = 0, zeroCount = 0;
    double sum = 0, minVal = double.infinity, maxVal = double.negativeInfinity;

    for (final v in floats) {
      if (v.isNaN) {
        nanCount++;
        continue;
      }
      if (v.isInfinite) {
        infCount++;
        continue;
      }
      if (v == 0.0) zeroCount++;
      sum += v;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    final validCount = floats.length - nanCount - infCount;
    final mean = validCount > 0 ? sum / validCount : 0.0;
    final first8 = floats
        .take(8)
        .map((v) => v.toStringAsFixed(4))
        .toList();

    String status;
    if (nanCount > 0) {
      status = 'NaN=$nanCount/${floats.length}';
    } else if (infCount > 0) {
      status = 'Inf=$infCount/${floats.length}';
    } else if (zeroCount == floats.length) {
      status = 'ALL-ZERO';
    } else {
      status = 'OK';
    }

    print(
      '  [$label] ${sw.elapsedMilliseconds}ms | '
      'shape=${output.shape} | $status | '
      'mean=${mean.toStringAsFixed(4)} | '
      'range=[${minVal.toStringAsFixed(4)}, ${maxVal.toStringAsFixed(4)}] | '
      'zero=$zeroCount/${floats.length}',
    );
    print('  [$label] first8=$first8');

    return floats;
  } finally {
    if (model != null) await model.dispose();
  }
}

/// Compare model output against CPU reference from manifest.
void _compareAgainstRef(Float32List output, V2TestCase tc, String label) {
  final expectedFirst8 = tc.expectedFirst8;

  // Compare first 8 values
  double maxFirst8Diff = 0;
  for (int i = 0; i < math.min(8, output.length); i++) {
    if (output[i].isNaN || output[i].isInfinite) continue;
    final diff = (output[i] - expectedFirst8[i]).abs();
    if (diff > maxFirst8Diff) maxFirst8Diff = diff;
  }

  // Compare mean
  double sum = 0;
  int validCount = 0;
  for (final v in output) {
    if (!v.isNaN && !v.isInfinite) {
      sum += v;
      validCount++;
    }
  }
  final actualMean = validCount > 0 ? sum / validCount : 0.0;
  final meanDiff = (actualMean - tc.expectedMean).abs();

  final pass = maxFirst8Diff < 0.1 && meanDiff < 0.5;
  print(
    '  [$label vs CPU_REF] ${pass ? "PASS" : "FAIL"} | '
    'meanDiff=${meanDiff.toStringAsFixed(4)} | '
    'maxFirst8Diff=${maxFirst8Diff.toStringAsFixed(4)}',
  );
}

/// Compare two model outputs element-wise.
void _compareOutputs(Float32List ref, Float32List test, String label) {
  double maxDiff = 0;
  int mismatchCount = 0;
  int nanInTest = 0;

  for (int i = 0; i < math.min(ref.length, test.length); i++) {
    if (test[i].isNaN) {
      nanInTest++;
      continue;
    }
    if (ref[i].isNaN) continue;
    final diff = (ref[i] - test[i]).abs();
    if (diff > 0.01) mismatchCount++;
    if (diff > maxDiff) maxDiff = diff;
  }

  final pass = nanInTest == 0 && maxDiff < 0.1;
  print(
    '  [$label] ${pass ? "PASS" : "FAIL"} | '
    'maxDiff=${maxDiff.toStringAsFixed(4)} | '
    'mismatches(>0.01)=$mismatchCount | '
    'nanInTest=$nanInTest',
  );
}
