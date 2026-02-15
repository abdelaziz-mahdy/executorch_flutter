// ignore_for_file: avoid_print

/// V7 Multi-op chain isolation test.
///
/// Tests various combinations to isolate multi-op chain failures.
///
/// Run:
///   cd example
///   flutter test integration_test/vulkan_v7_multi_op_test.dart -d 56241FDCH00AA2

@Timeout(Duration(minutes: 10))
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('V7 Multi-Op', () {
    late List<Map<String, dynamic>> tests;

    setUpAll(() async {
      final s = await rootBundle.loadString(
        'assets/debug_models/v7_manifest.json',
      );
      final manifest = jsonDecode(s) as Map<String, dynamic>;
      tests = (manifest['tests'] as List).cast<Map<String, dynamic>>();
      print('Loaded ${tests.length} v7 test cases');
    });

    for (final testName in [
      'add_f_div_f',
      'add_i_div_f',
      'add_f_div_i',
      'two_adds_int',
      'two_adds_float',
      'add_then_mul',
      'mul_then_add',
      'three_ops',
      'add_self',
      'mul_self',
      'add_self_div_f',
      'add_self_div_i',
    ]) {
      testWidgets('$testName', (tester) async {
        final tc = tests.firstWhere((t) => t['name'] == testName);
        final cpuRef = tc['cpu_ref'] as Map<String, dynamic>;
        final inputShape = (tc['input_shape'] as List).cast<int>();
        final expectedFirst8 = (cpuRef['first8'] as List)
            .map((e) => (e as num).toDouble())
            .toList();
        final expectedMean = (cpuRef['mean'] as num).toDouble();

        print('\n${"=" * 70}');
        print('TEST: $testName');
        print('  ${tc["description"]}');
        print(
          '  Expected: mean=${expectedMean.toStringAsFixed(4)}, '
          'first8=${expectedFirst8.map((v) => v.toStringAsFixed(4)).toList()}',
        );

        int numel = 1;
        for (final s in inputShape) {
          numel *= s;
        }
        final inputData = Float32List(numel)..fillRange(0, numel, 1.0);
        final input = TensorData(
          shape: inputShape,
          dataType: TensorType.float32,
          data: inputData.buffer.asUint8List(),
        );

        final xnAsset = tc['xnnpack_asset'] as String?;
        Float32List? xnOutput;
        if (xnAsset != null) {
          xnOutput = await _runModel(
            'assets/debug_models/$xnAsset',
            [input],
            'XNNPACK',
          );
        }

        final vkAsset = tc['vulkan_asset'] as String?;
        Float32List? vkOutput;
        if (vkAsset != null) {
          vkOutput = await _runModel(
            'assets/debug_models/$vkAsset',
            [input],
            'Vulkan',
          );
        }

        if (vkOutput != null) {
          _compareRef(vkOutput, expectedFirst8, expectedMean);
        }

        if (xnOutput != null && vkOutput != null) {
          _compare(xnOutput, vkOutput, 'XN vs VK');
        }

        print('${"=" * 70}\n');
      });
    }
  });
}

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
    final first8 = floats.take(8).map((v) => v.toStringAsFixed(6)).toList();

    String status;
    if (nanCount > 0) {
      status = 'NaN=$nanCount/${floats.length}';
    } else if (zeroCount == floats.length) {
      status = 'ALL-ZERO';
    } else {
      status = 'OK';
    }

    print(
      '  [$label] ${sw.elapsedMilliseconds}ms | shape=${output.shape} | '
      '$status | mean=${mean.toStringAsFixed(6)} | '
      'range=[${minVal.toStringAsFixed(6)}, ${maxVal.toStringAsFixed(6)}] | '
      'zero=$zeroCount/${floats.length} | nan=$nanCount | inf=$infCount',
    );
    print('  [$label] first8=$first8');

    return floats;
  } finally {
    if (model != null) await model.dispose();
  }
}

void _compareRef(
  Float32List output,
  List<double> expectedFirst8,
  double expectedMean,
) {
  double maxDiff = 0;
  for (int i = 0; i < math.min(8, output.length); i++) {
    if (output[i].isNaN || output[i].isInfinite) continue;
    final diff = (output[i] - expectedFirst8[i]).abs();
    if (diff > maxDiff) maxDiff = diff;
  }

  double sum = 0;
  int valid = 0;
  for (final v in output) {
    if (!v.isNaN && !v.isInfinite) {
      sum += v;
      valid++;
    }
  }
  final meanDiff = valid > 0 ? (sum / valid - expectedMean).abs() : 999.0;

  final pass = maxDiff < 0.01 && meanDiff < 0.01;
  print(
    '  [VK vs REF] ${pass ? "PASS" : "FAIL"} | '
    'meanDiff=${meanDiff.toStringAsFixed(6)} | '
    'maxFirst8Diff=${maxDiff.toStringAsFixed(6)}',
  );
}

void _compare(Float32List ref, Float32List test, String label) {
  double maxDiff = 0;
  int mismatch = 0;
  int nanInTest = 0;

  for (int i = 0; i < math.min(ref.length, test.length); i++) {
    if (test[i].isNaN) {
      nanInTest++;
      continue;
    }
    if (ref[i].isNaN) continue;
    final diff = (ref[i] - test[i]).abs();
    if (diff > 0.01) mismatch++;
    if (diff > maxDiff) maxDiff = diff;
  }

  final pass = nanInTest == 0 && maxDiff < 0.01;
  print(
    '  [$label] ${pass ? "PASS" : "FAIL"} | '
    'maxDiff=${maxDiff.toStringAsFixed(6)} | '
    'mismatches=$mismatch | nanInTest=$nanInTest',
  );
}
