// ignore_for_file: avoid_print

/// FP32 Progressive MobileNet slice test with real image.
///
/// Tests each progressive MobileNet slice on both Vulkan and XNNPACK
/// with a real cat.jpg to find exactly where NaN first appears.

@Timeout(Duration(minutes: 10))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/processors/image_processor.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

final _slices = [
  (name: 'features[0:1]', vulkan: 'mobilenet_fp32_slice_0_1_vulkan', xnnpack: 'mobilenet_fp32_slice_0_1_xnnpack'),
  (name: 'features[0:2]', vulkan: 'mobilenet_fp32_slice_0_2_vulkan', xnnpack: 'mobilenet_fp32_slice_0_2_xnnpack'),
  (name: 'features[0:4]', vulkan: 'mobilenet_fp32_slice_0_4_vulkan', xnnpack: 'mobilenet_fp32_slice_0_4_xnnpack'),
  (name: 'features[0:7]', vulkan: 'mobilenet_fp32_slice_0_7_vulkan', xnnpack: 'mobilenet_fp32_slice_0_7_xnnpack'),
  (name: 'features[0:9]', vulkan: 'mobilenet_fp32_slice_0_9_vulkan', xnnpack: 'mobilenet_fp32_slice_0_9_xnnpack'),
  (name: 'features_all', vulkan: 'mobilenet_fp32_features_all_vulkan', xnnpack: 'mobilenet_fp32_features_all_xnnpack'),
  (name: 'features+pool', vulkan: 'mobilenet_fp32_features_pool_vulkan', xnnpack: 'mobilenet_fp32_features_pool_xnnpack'),
  (name: 'full_model', vulkan: 'mobilenet_fp32_full_vulkan', xnnpack: 'mobilenet_fp32_full_xnnpack'),
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('FP32 Progressive MobileNet - Real Image', () {
    late List<TensorData> preprocessedInput;

    setUpAll(() async {
      final byteData = await rootBundle.load('assets/images/cat.jpg');
      final catBytes = byteData.buffer.asUint8List();
      print('Loaded cat.jpg: ${catBytes.length} bytes');

      final preprocessor = ImageNetPreprocessor(
        config: const ImagePreprocessConfig(),
      );
      preprocessedInput = await preprocessor.preprocess(catBytes);
      print('Preprocessed: shape=${preprocessedInput[0].shape}');
    });

    for (final slice in _slices) {
      testWidgets('${slice.name} - Vulkan vs XNNPACK', (tester) async {
        print('\n${"=" * 60}');
        print('SLICE: ${slice.name}');

        // Run XNNPACK first (reference)
        final xnResult = await _runModel(
          'assets/debug_models/${slice.xnnpack}.pte',
          preprocessedInput,
          'XNNPACK',
        );

        // Run Vulkan
        final vkResult = await _runModel(
          'assets/debug_models/${slice.vulkan}.pte',
          preprocessedInput,
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
          print('  COMPARE: maxDiff=$maxDiff, mismatches(>0.01)=$mismatchCount, vkNaN=$vkNanCount/${vkFloats.length}');

          if (vkNanCount > 0) {
            print('  >>> NaN FOUND at ${slice.name} <<<');
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
    int infCount = 0;
    int zeroCount = 0;
    double sum = 0;
    double minVal = double.infinity;
    double maxVal = double.negativeInfinity;

    for (final v in floats) {
      if (v.isNaN) { nanCount++; continue; }
      if (v.isInfinite) { infCount++; continue; }
      if (v == 0.0) zeroCount++;
      sum += v;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    final validCount = floats.length - nanCount - infCount;
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
