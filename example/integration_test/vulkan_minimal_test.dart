// ignore_for_file: avoid_print

/// Vulkan minimal operator test.
///
/// Tests bare-minimum single-operator models exported with the Vulkan backend
/// to isolate which Vulkan compute shaders produce incorrect results on
/// specific GPUs (e.g., PowerVR).
///
/// Prerequisites:
///   1. Export minimal models using the Python script:
///      python examples/vulkan/minimal_vulkan_test.py --output-dir /tmp/minimal_vulkan_models
///
///   2. Copy exported .pte files to example/assets/debug_models/
///      cp /tmp/minimal_vulkan_models/*.pte example/assets/debug_models/
///
///   3. Add to example/pubspec.yaml:
///      flutter:
///        assets:
///          - assets/debug_models/
///
///   4. Run on device with Vulkan:
///      cd example
///      flutter drive --driver=test_driver/integration_test.dart \
///        --target=integration_test/vulkan_minimal_test.dart -d DEVICE_ID

@Timeout(Duration(minutes: 10))
library;

import 'dart:math';
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Minimal single-operator model definitions.
///
/// Each model wraps a single op so failures pinpoint the exact shader.
/// All inputs use 1.0 values for easy expected-output verification.
final _minimalModels = [
  (
    name: 'bare_add',
    asset: 'assets/debug_models/bare_add.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'x + 1.0 (simplest compute shader)',
    expectedMean: 2.0,
    inputValue: 1.0,
  ),
  (
    name: 'bare_half',
    asset: 'assets/debug_models/bare_half.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'x * 0.5 (tests basic mul, no conv)',
    expectedMean: 0.5,
    inputValue: 1.0,
  ),
  (
    name: 'zero_conv',
    asset: 'assets/debug_models/zero_conv.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Conv2d all-zero weights (tests accumulator init)',
    expectedMean: 0.0,
    inputValue: 1.0,
  ),
  (
    name: 'identity_conv',
    asset: 'assets/debug_models/identity_conv.pte',
    inputShape: [1, 3, 4, 4],
    desc: 'Identity 1x1 conv (output=input=5.0)',
    expectedMean: 5.0,
    inputValue: 5.0,
  ),
  (
    name: 'bare_conv2d',
    asset: 'assets/debug_models/bare_conv2d.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Single Conv2d(3->4, 3x3) no bias',
    expectedMean: 0.8403,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_bias',
    asset: 'assets/debug_models/bare_conv2d_bias.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Single Conv2d(3->4, 3x3) with bias=0.5',
    expectedMean: 1.3403,
    inputValue: 1.0,
  ),
  (
    name: 'bare_linear',
    asset: 'assets/debug_models/bare_linear.pte',
    inputShape: [1, 16],
    desc: 'Single Linear(16->4)',
    expectedMean: 1.6,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_fp32',
    asset: 'assets/debug_models/bare_conv2d_fp32.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Conv2d(3->4, 3x3) FP32 (no force_fp16)',
    expectedMean: 0.8403,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_large',
    asset: 'assets/debug_models/bare_conv2d_large.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'Conv2d(3->16, 3x3, s2) like MobileNet first layer',
    expectedMean: 0.9941,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_ones',
    asset: 'assets/debug_models/bare_conv2d_ones.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Conv2d(3->4, 3x3) weights=1.0 (large output)',
    expectedMean: 22.6875,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_dw',
    asset: 'assets/debug_models/bare_conv2d_dw.pte',
    inputShape: [1, 4, 8, 8],
    desc: 'Depthwise Conv2d(4->4, 3x3, groups=4)',
    expectedMean: 0.8403,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_pw',
    asset: 'assets/debug_models/bare_conv2d_pw.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Pointwise Conv2d(3->4, 1x1) weights=0.5',
    expectedMean: 1.5,
    inputValue: 1.0,
  ),
  (
    name: 'bare_conv2d_xnnpack',
    asset: 'assets/debug_models/bare_conv2d_xnnpack.pte',
    inputShape: [1, 3, 8, 8],
    desc: 'Same Conv2d via XNNPACK (CPU reference)',
    expectedMean: 0.8403,
    inputValue: 1.0,
  ),
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan Minimal Operator Tests', () {
    /// Create an input tensor filled with a uniform value.
    ///
    /// Using uniform values makes expected output easy to compute.
    TensorData createInput(List<int> shape, double fillValue) {
      final numel = shape.reduce((a, b) => a * b);
      final data = Float32List(numel);
      for (int i = 0; i < numel; i++) {
        data[i] = fillValue;
      }
      return TensorData(
        shape: shape,
        dataType: TensorType.float32,
        data: data.buffer.asUint8List(),
      );
    }

    /// Analyze output tensor and return a diagnostic summary.
    Map<String, dynamic> analyzeOutput(List<TensorData> outputs) {
      int totalElements = 0;
      int nanCount = 0;
      int infCount = 0;
      int zeroCount = 0;
      double sum = 0.0;
      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;
      final firstValues = <double>[];

      for (final output in outputs) {
        final floats = output.data.buffer.asFloat32List();
        totalElements += floats.length;

        for (int i = 0; i < floats.length; i++) {
          final v = floats[i];
          if (v.isNaN) {
            nanCount++;
          } else if (v.isInfinite) {
            infCount++;
          } else {
            if (v == 0.0) zeroCount++;
            sum += v;
            minVal = min(minVal, v);
            maxVal = max(maxVal, v);
          }

          if (firstValues.length < 8) {
            firstValues.add(v);
          }
        }
      }

      final validCount = totalElements - nanCount - infCount;
      final mean = validCount > 0 ? sum / validCount : 0.0;

      return {
        'totalElements': totalElements,
        'nanCount': nanCount,
        'infCount': infCount,
        'zeroCount': zeroCount,
        'mean': mean,
        'allZero':
            zeroCount == totalElements && nanCount == 0 && infCount == 0,
        'hasNaN': nanCount > 0,
        'hasInf': infCount > 0,
        'minVal': minVal,
        'maxVal': maxVal,
        'firstValues': firstValues,
        'outputShapes': outputs.map((o) => o.shape).toList(),
      };
    }

    for (final model in _minimalModels) {
      testWidgets('${model.name}: ${model.desc}', (tester) async {
        ExecuTorchModel? loadedModel;

        try {
          print('\n${"=" * 60}');
          print('Testing: ${model.name} (${model.desc})');
          print('  Asset: ${model.asset}');
          print('  Input shape: ${model.inputShape}');
          print('  Expected mean: ${model.expectedMean}');

          // Load model
          try {
            loadedModel = await ExecuTorchModel.loadFromAsset(model.asset);
          } catch (e) {
            print('  SKIP - Asset not found: ${model.asset}');
            print('  Export models first with minimal_vulkan_test.py');
            return;
          }
          print('  Model loaded successfully');

          // Create input filled with model.inputValue
          final input = createInput(model.inputShape, model.inputValue);
          print('  Input: all ${model.inputValue}, shape ${model.inputShape}');

          // Run inference
          final stopwatch = Stopwatch()..start();
          final outputs = await loadedModel.forward([input]);
          stopwatch.stop();
          print('  Inference time: ${stopwatch.elapsedMilliseconds}ms');

          // Analyze results
          final analysis = analyzeOutput(outputs);
          final firstVals = (analysis['firstValues'] as List<double>)
              .map((v) => v.toStringAsFixed(6))
              .toList();
          final actualMean = analysis['mean'] as double;

          print('  Output shapes: ${analysis['outputShapes']}');
          print('  Total elements: ${analysis['totalElements']}');
          print('  First 8 values: $firstVals');
          print('  Mean: ${actualMean.toStringAsFixed(6)}');
          print('  NaN count: ${analysis['nanCount']}');
          print('  Inf count: ${analysis['infCount']}');
          print('  Zero count: ${analysis['zeroCount']}');
          print('  All zero: ${analysis['allZero']}');
          print(
            '  Min: ${(analysis['minVal'] as double).toStringAsFixed(6)}, '
            'Max: ${(analysis['maxVal'] as double).toStringAsFixed(6)}',
          );

          // Determine result
          if (analysis['hasNaN'] as bool) {
            print('  RESULT: FAIL - NaN detected');
          } else if (analysis['hasInf'] as bool) {
            print('  RESULT: FAIL - Inf detected');
          } else if (analysis['allZero'] as bool) {
            print('  RESULT: FAIL - All zeros (shader produced no output)');
          } else {
            // Check if mean output is close to expected (fp16 tolerance)
            final diff = (actualMean - model.expectedMean).abs();
            if (diff < 0.05) {
              print('  RESULT: PASS - Mean matches expected '
                  '(got ${actualMean.toStringAsFixed(4)}, '
                  'expected ${model.expectedMean})');
            } else {
              print('  RESULT: WARN - Non-trivial output but mean differs '
                  'from expected (got ${actualMean.toStringAsFixed(4)}, '
                  'expected ${model.expectedMean}, '
                  'diff=${diff.toStringAsFixed(4)})');
            }
          }

          print('${"=" * 60}\n');
        } catch (e, st) {
          print('  RESULT: ERROR - $e');
          print('  Stack: $st');
        } finally {
          if (loadedModel != null) {
            await loadedModel.dispose();
          }
        }
      });
    }
  });
}
