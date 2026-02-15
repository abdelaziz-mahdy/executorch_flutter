// ignore_for_file: avoid_print

/// Direct comparison: v3_mobilenet_stem vs v4_mobilenet_slice_1
/// Both should be identical (same architecture, same weights).

@Timeout(Duration(minutes: 5))
library;

import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Stem vs Slice Comparison', () {
    late TensorData input;

    setUpAll(() {
      const shape = [1, 3, 224, 224];
      int numel = 1;
      for (final s in shape) {
        numel *= s;
      }
      final data = Float32List(numel)..fillRange(0, numel, 1.0);
      input = TensorData(
        shape: shape,
        dataType: TensorType.float32,
        data: data.buffer.asUint8List(),
      );
    });

    // Test v3 mobilenet_stem (this PASSED earlier)
    testWidgets('v3_mobilenet_stem Vulkan', (tester) async {
      await _testModel(
        'assets/debug_models/v3_mobilenet_stem_fp32_vulkan.pte',
        input,
        'v3_stem_VK',
      );
    });

    // Test v3 mobilenet_stem XNNPACK reference
    testWidgets('v3_mobilenet_stem XNNPACK', (tester) async {
      await _testModel(
        'assets/debug_models/v3_mobilenet_stem_fp32_xnnpack.pte',
        input,
        'v3_stem_XN',
      );
    });

    // Test v4 mobilenet_slice_1 (this FAILED earlier)
    testWidgets('v4_mobilenet_slice_1 Vulkan', (tester) async {
      await _testModel(
        'assets/debug_models/v4_mobilenet_slice_1_fp32_vulkan.pte',
        input,
        'v4_slice1_VK',
      );
    });

    // Test v4 mobilenet_slice_1 XNNPACK reference
    testWidgets('v4_mobilenet_slice_1 XNNPACK', (tester) async {
      await _testModel(
        'assets/debug_models/v4_mobilenet_slice_1_fp32_xnnpack.pte',
        input,
        'v4_slice1_XN',
      );
    });

    // Test v2 mobilenet_slice_1 (original, also FAILED)
    testWidgets('v2_mobilenet_slice_1 Vulkan', (tester) async {
      await _testModel(
        'assets/debug_models/v2_mobilenet_slice_1_fp32_vulkan.pte',
        input,
        'v2_slice1_VK',
      );
    });

    // v2 XNNPACK reference
    testWidgets('v2_mobilenet_slice_1 XNNPACK', (tester) async {
      await _testModel(
        'assets/debug_models/v2_mobilenet_slice_1_fp32_xnnpack.pte',
        input,
        'v2_slice1_XN',
      );
    });
  });
}

Future<void> _testModel(
  String asset,
  TensorData input,
  String label,
) async {
  ExecuTorchModel? model;
  try {
    try {
      model = await ExecuTorchModel.loadFromAsset(asset);
    } catch (e) {
      print('[$label] LOAD FAILED: $e');
      return;
    }

    final sw = Stopwatch()..start();
    final outputs = await model.forward([input]);
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

    print(
      '[$label] ${sw.elapsedMilliseconds}ms | shape=${output.shape} | '
      'NaN=$nanCount Inf=$infCount Zero=$zeroCount/${floats.length} | '
      'mean=${mean.toStringAsFixed(6)} | '
      'range=[${minVal.toStringAsFixed(6)}, ${maxVal.toStringAsFixed(6)}]',
    );
    print('[$label] first8=$first8');
  } finally {
    if (model != null) await model.dispose();
  }
}
