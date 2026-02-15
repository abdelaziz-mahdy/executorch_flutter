// ignore_for_file: avoid_print

/// V3 Large Conv diagnostic test.
///
/// Tests whether spatial size or Hardswish causes the failure.
///
/// Run:
///   cd example
///   flutter test integration_test/vulkan_v3_large_conv_test.dart -d 56241FDCH00AA2

@Timeout(Duration(minutes: 10))
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

class TestCase {
  final String name;
  final String description;
  final List<int> inputShape;
  final Map<String, dynamic> cpuRef;
  final String? vulkanAsset;
  final String? xnnpackAsset;

  TestCase.fromJson(Map<String, dynamic> json)
      : name = json['name'] as String,
        description = json['description'] as String,
        inputShape = (json['input_shape'] as List)
            .map((e) => (e as num).toInt())
            .toList(),
        cpuRef = json['cpu_ref'] as Map<String, dynamic>,
        vulkanAsset = json['vulkan_asset'] as String?,
        xnnpackAsset = json['xnnpack_asset'] as String?;

  List<double> get expectedFirst8 =>
      (cpuRef['first8'] as List).map((e) => (e as num).toDouble()).toList();
  double get expectedMean => (cpuRef['mean'] as num).toDouble();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('V3 Large Conv Diagnostic', () {
    late List<TestCase> tests;

    setUpAll(() async {
      final s = await rootBundle.loadString(
        'assets/debug_models/v3_manifest.json',
      );
      final manifest = jsonDecode(s) as Map<String, dynamic>;
      tests = (manifest['tests'] as List)
          .map((e) => TestCase.fromJson(e as Map<String, dynamic>))
          .toList();
      print('Loaded ${tests.length} v3 test cases');
    });

    // Test each case dynamically
    for (final testName in [
      'large_conv_nobias',
      'large_conv_bias',
      'conv_hardswish',
      'conv_bn_hs',
      'mobilenet_stem',
      'medium_conv',
    ]) {
      testWidgets('$testName', (tester) async {
        final tc = tests.firstWhere((t) => t.name == testName);
        print('\n${"=" * 70}');
        print('TEST: ${tc.name}');
        print('  ${tc.description}');
        print('  Input: ${tc.inputShape}');
        print(
          '  Expected: mean=${tc.expectedMean.toStringAsFixed(4)}, '
          'first8=${tc.expectedFirst8.map((v) => v.toStringAsFixed(4)).toList()}',
        );

        // Create all-ones input
        int numel = 1;
        for (final s in tc.inputShape) {
          numel *= s;
        }
        final inputData = Float32List(numel)..fillRange(0, numel, 1.0);
        final input = TensorData(
          shape: tc.inputShape,
          dataType: TensorType.float32,
          data: inputData.buffer.asUint8List(),
        );

        // Run XNNPACK
        Float32List? xnOutput;
        if (tc.xnnpackAsset != null) {
          xnOutput = await _runModel(
            'assets/debug_models/${tc.xnnpackAsset}',
            [input],
            'XNNPACK',
          );
        }

        // Run Vulkan
        final vkOutput = await _runModel(
          'assets/debug_models/${tc.vulkanAsset}',
          [input],
          'Vulkan',
        );

        // Compare Vulkan vs ref
        if (vkOutput != null) {
          _compareRef(vkOutput, tc);
        }

        // Compare XNNPACK vs Vulkan
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
      'zero=$zeroCount/${floats.length}',
    );
    print('  [$label] first8=$first8');

    return floats;
  } finally {
    if (model != null) await model.dispose();
  }
}

void _compareRef(Float32List output, TestCase tc) {
  final expected = tc.expectedFirst8;
  double maxDiff = 0;
  for (int i = 0; i < math.min(8, output.length); i++) {
    if (output[i].isNaN || output[i].isInfinite) continue;
    final diff = (output[i] - expected[i]).abs();
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
  final meanDiff = valid > 0 ? (sum / valid - tc.expectedMean).abs() : 999.0;

  final pass = maxDiff < 0.1 && meanDiff < 0.5;
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

  final pass = nanInTest == 0 && maxDiff < 0.1;
  print(
    '  [$label] ${pass ? "PASS" : "FAIL"} | '
    'maxDiff=${maxDiff.toStringAsFixed(6)} | '
    'mismatches=$mismatch | nanInTest=$nanInTest',
  );
}
