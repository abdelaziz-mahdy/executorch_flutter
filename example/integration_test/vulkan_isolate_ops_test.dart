// ignore_for_file: avoid_print

/// Isolate which operator produces NaN on PowerVR.
///
/// Tests:
/// 1. Conv2d only (real MobileNet fused-BN weights) - no activation
/// 2. Hardswish only (applied to conv-output-sized tensor)
/// 3. Conv2d + Hardswish (random weights)
/// 4. Conv2d + Hardswish (real MobileNet fused weights)

@Timeout(Duration(minutes: 5))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/processors/image_processor.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

final _tests = [
  (
    name: 'Conv2d only (real weights)',
    vulkan: 'isolate_conv_real_vulkan',
    xnnpack: 'isolate_conv_real_xnnpack',
    inputShape: [1, 3, 224, 224],
    useImage: true,
  ),
  (
    name: 'Hardswish only',
    vulkan: 'isolate_hardswish_vulkan',
    xnnpack: 'isolate_hardswish_xnnpack',
    inputShape: [1, 16, 112, 112],
    useImage: false,
  ),
  (
    name: 'Conv2d + Hardswish (random weights)',
    vulkan: 'isolate_conv_hs_random_vulkan',
    xnnpack: 'isolate_conv_hs_random_xnnpack',
    inputShape: [1, 3, 224, 224],
    useImage: true,
  ),
  (
    name: 'Conv2d + Hardswish (real weights)',
    vulkan: 'isolate_conv_hs_real_vulkan',
    xnnpack: 'isolate_conv_hs_real_xnnpack',
    inputShape: [1, 3, 224, 224],
    useImage: true,
  ),
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Isolate Op NaN Source', () {
    late List<TensorData> imageInput;

    setUpAll(() async {
      final byteData = await rootBundle.load('assets/images/cat.jpg');
      final catBytes = byteData.buffer.asUint8List();
      final preprocessor = ImageNetPreprocessor(
        config: const ImagePreprocessConfig(),
      );
      imageInput = await preprocessor.preprocess(catBytes);
      print('Image input: shape=${imageInput[0].shape}');
    });

    for (final test in _tests) {
      testWidgets(test.name, (tester) async {
        print('\n${"=" * 60}');
        print('TEST: ${test.name}');

        // Create input
        List<TensorData> input;
        if (test.useImage) {
          input = imageInput;
        } else {
          // Generate realistic input values (similar to conv2d output range)
          final numel = test.inputShape.reduce((a, b) => a * b);
          final rng = math.Random(42);
          final floats = Float32List(numel);
          for (int i = 0; i < numel; i++) {
            // Range roughly [-5, 50] to match real conv output
            floats[i] = (rng.nextDouble() * 55) - 5;
          }
          input = [
            TensorData(
              shape: test.inputShape,
              dataType: TensorType.float32,
              data: floats.buffer.asUint8List(),
            ),
          ];
        }

        // Run XNNPACK (reference)
        final xnResult = await _runModel(
          'assets/debug_models/${test.xnnpack}.pte',
          input,
          'XNNPACK',
        );

        // Run Vulkan
        final vkResult = await _runModel(
          'assets/debug_models/${test.vulkan}.pte',
          input,
          'Vulkan',
        );

        // Compare
        if (xnResult != null && vkResult != null) {
          final xnFloats = xnResult.data.buffer.asFloat32List();
          final vkFloats = vkResult.data.buffer.asFloat32List();

          double maxDiff = 0;
          int mismatchCount = 0;
          int vkNanCount = 0;
          for (int i = 0; i < math.min(xnFloats.length, vkFloats.length); i++) {
            if (vkFloats[i].isNaN) {
              vkNanCount++;
              continue;
            }
            final diff = (xnFloats[i] - vkFloats[i]).abs();
            if (diff > 0.01) mismatchCount++;
            if (diff > maxDiff) maxDiff = diff;
          }
          print(
            '  COMPARE: maxDiff=${maxDiff.toStringAsFixed(6)}, '
            'mismatches(>0.01)=$mismatchCount, '
            'vkNaN=$vkNanCount/${vkFloats.length}',
          );

          if (vkNanCount > 0) {
            print('  >>> NaN FOUND <<<');
          } else if (mismatchCount == 0) {
            print('  >>> MATCH <<<');
          } else {
            print('  >>> DIVERGED (no NaN but values differ) <<<');
          }
        }
        print('${"=" * 60}\n');
      });
    }
  });
}

Future<TensorData?> _runModel(
  String asset,
  List<TensorData> input,
  String label,
) async {
  ExecuTorchModel? model;
  try {
    try {
      model = await ExecuTorchModel.loadFromAsset(asset);
    } catch (e) {
      print('  [$label] SKIP - $e');
      return null;
    }

    final sw = Stopwatch()..start();
    final outputs = await model.forward(input);
    sw.stop();

    final output = outputs.first;
    final floats = output.data.buffer.asFloat32List();

    int nanCount = 0;
    int zeroCount = 0;
    double sum = 0;
    double minVal = double.infinity;
    double maxVal = double.negativeInfinity;

    for (final v in floats) {
      if (v.isNaN) {
        nanCount++;
        continue;
      }
      if (v == 0.0) zeroCount++;
      sum += v;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    final validCount = floats.length - nanCount;
    final mean = validCount > 0 ? sum / validCount : 0.0;
    final first4 = floats.take(4).map((v) => v.toStringAsFixed(4)).toList();

    String status;
    if (nanCount > 0) {
      status = 'FAIL(NaN=$nanCount/${floats.length})';
    } else if (zeroCount == floats.length) {
      status = 'FAIL(all-zero)';
    } else {
      status = 'OK';
    }

    print(
      '  [$label] ${sw.elapsedMilliseconds}ms | '
      'shape=${output.shape} | '
      '$status | '
      'mean=${mean.toStringAsFixed(4)} | '
      'range=[${minVal.toStringAsFixed(4)}, ${maxVal.toStringAsFixed(4)}] | '
      'first4=$first4',
    );

    return output;
  } finally {
    if (model != null) await model.dispose();
  }
}
