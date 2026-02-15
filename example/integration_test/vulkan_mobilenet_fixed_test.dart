// ignore_for_file: avoid_print

/// Test full MobileNet V3 Small with native hardswish fix on Vulkan.
///
/// The fix: hardswish and hardsigmoid are now in the Vulkan partitioner's
/// ops_not_to_decompose list, so they use native GLSL shaders instead of
/// being decomposed into mul/add/clamp/div (which produces NaN on PowerVR).

@Timeout(Duration(minutes: 5))
library;

import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/processors/image_processor.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('MobileNet Fixed (native hardswish)', () {
    late List<TensorData> imageInput;

    setUpAll(() async {
      final byteData = await rootBundle.load('assets/images/cat.jpg');
      final catBytes = byteData.buffer.asUint8List();
      final preprocessor = ImageNetPreprocessor(
        config: const ImagePreprocessConfig(),
      );
      imageInput = await preprocessor.preprocess(catBytes);
      print('Image input ready: shape=${imageInput[0].shape}');
    });

    for (final variant in [
      (name: 'Vulkan FP32', asset: 'mobilenet_fixed_vulkan'),
      (name: 'Vulkan FP16', asset: 'mobilenet_fixed_fp16_vulkan'),
    ]) {
      testWidgets('Full MobileNet - ${variant.name} vs XNNPACK',
          (tester) async {
        print('\n${"=" * 60}');
        print('TEST: MobileNet V3 Small - ${variant.name} vs XNNPACK');
        print('${"=" * 60}');

        // Run XNNPACK (reference)
        print('\n--- XNNPACK (reference) ---');
        final xnResult = await _runClassification(
          'assets/debug_models/mobilenet_fixed_xnnpack.pte',
          imageInput,
          'XNNPACK',
        );

        // Run Vulkan
        print('\n--- ${variant.name} ---');
        final vkResult = await _runClassification(
          'assets/debug_models/${variant.asset}.pte',
          imageInput,
          variant.name,
        );

        // Compare
        if (xnResult != null && vkResult != null) {
          final xnFloats = xnResult.data.buffer.asFloat32List();
          final vkFloats = vkResult.data.buffer.asFloat32List();

          double maxDiff = 0;
          int nanCount = 0;
          int mismatchCount = 0;

          for (int i = 0; i < xnFloats.length; i++) {
            if (vkFloats[i].isNaN) {
              nanCount++;
              continue;
            }
            final diff = (xnFloats[i] - vkFloats[i]).abs();
            if (diff > 0.1) mismatchCount++;
            if (diff > maxDiff) maxDiff = diff;
          }

          print('\n--- COMPARISON ---');
          print('  maxDiff: ${maxDiff.toStringAsFixed(4)}');
          print('  NaN count: $nanCount / ${vkFloats.length}');
          print('  Mismatches (>0.1): $mismatchCount / ${vkFloats.length}');

          if (nanCount > 0) {
            print('  >>> STILL HAS NaN <<<');
          } else if (maxDiff < 1.0) {
            print('  >>> SUCCESS - Vulkan matches XNNPACK closely <<<');
          } else {
            print('  >>> NO NaN but values diverge (maxDiff=$maxDiff) <<<');
          }
        }

        print('\n${"=" * 60}\n');
      });
    }
  });
}

Future<TensorData?> _runClassification(
  String asset,
  List<TensorData> input,
  String label,
) async {
  ExecuTorchModel? model;
  try {
    try {
      model = await ExecuTorchModel.loadFromAsset(asset);
    } catch (e) {
      print('  [$label] SKIP - load failed: $e');
      return null;
    }

    final sw = Stopwatch()..start();
    final outputs = await model.forward(input);
    sw.stop();

    final output = outputs.first;
    final floats = output.data.buffer.asFloat32List();

    // Basic stats
    int nanCount = 0;
    double sum = 0;
    for (final v in floats) {
      if (v.isNaN) {
        nanCount++;
      } else {
        sum += v;
      }
    }
    final validCount = floats.length - nanCount;
    final mean = validCount > 0 ? sum / validCount : 0.0;

    print('  [$label] ${sw.elapsedMilliseconds}ms | '
        'shape=${output.shape} | '
        'NaN=$nanCount/${floats.length} | '
        'mean=${mean.toStringAsFixed(4)}');

    // Top-5 predictions
    if (nanCount == 0) {
      final indexed = List.generate(floats.length, (i) => (i, floats[i]));
      indexed.sort((a, b) => b.$2.compareTo(a.$2));

      print('  [$label] Top-5 predictions:');
      for (int i = 0; i < 5 && i < indexed.length; i++) {
        final idx = indexed[i].$1;
        final score = indexed[i].$2;
        print('    #${i + 1}: class $idx (score=${score.toStringAsFixed(3)})');
      }
    }

    return output;
  } finally {
    if (model != null) await model.dispose();
  }
}
