// ignore_for_file: avoid_print

/// Vulkan MobileNet V3 Small progressive debug test.
///
/// This test loads partial MobileNet models exported with the Vulkan backend
/// and checks for NaN/zero outputs to narrow down which operator produces
/// incorrect results.
///
/// Prerequisites:
///   1. Export partial models using the Python script:
///      cd <executorch_source>/examples/vulkan
///      python mobilenet_debug.py --output-dir /tmp/mobilenet_debug_models
///
///   2. Copy exported .pte files to example/assets/debug_models/
///      cp /tmp/mobilenet_debug_models/*.pte example/assets/debug_models/
///
///   3. Add to example/pubspec.yaml:
///      flutter:
///        assets:
///          - assets/debug_models/
///
///   4. Run on Android device with Vulkan:
///      cd example
///      flutter test integration_test/vulkan_mobilenet_debug_test.dart -d <device>

@Timeout(Duration(minutes: 30))
library;

import 'dart:math';
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Partial model definitions in order of increasing complexity.
/// Each entry: (name, asset path, input shape, description)
final _partialModels = [
  (
    name: 'features_0_1',
    asset: 'assets/debug_models/mobilenet_debug_features_0_1_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'First conv block only',
  ),
  (
    name: 'features_0_2',
    asset: 'assets/debug_models/mobilenet_debug_features_0_2_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'First 2 blocks',
  ),
  (
    name: 'features_0_4',
    asset: 'assets/debug_models/mobilenet_debug_features_0_4_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'First 4 blocks',
  ),
  (
    name: 'features_0_7',
    asset: 'assets/debug_models/mobilenet_debug_features_0_7_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'First 7 blocks',
  ),
  (
    name: 'features_0_9',
    asset: 'assets/debug_models/mobilenet_debug_features_0_9_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'First 9 blocks',
  ),
  (
    name: 'features_all',
    asset: 'assets/debug_models/mobilenet_debug_features_all_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'All 13 feature blocks',
  ),
  (
    name: 'features_pool',
    asset: 'assets/debug_models/mobilenet_debug_features_pool_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'Features + avgpool + flatten',
  ),
  (
    name: 'classifier_only',
    asset: 'assets/debug_models/mobilenet_debug_classifier_only_vulkan.pte',
    inputShape: [1, 576],
    desc: 'Classifier only (576-d input)',
  ),
  (
    name: 'full_model',
    asset: 'assets/debug_models/mobilenet_debug_full_model_vulkan.pte',
    inputShape: [1, 3, 224, 224],
    desc: 'Complete MobileNet V3 Small',
  ),
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan MobileNet Progressive Debug', () {
    /// Create a deterministic input tensor with known values.
    TensorData createInput(List<int> shape) {
      final numel = shape.reduce((a, b) => a * b);
      final data = Float32List(numel);

      // Fill with deterministic values (0.0 - 1.0 range, like normalized image)
      for (int i = 0; i < numel; i++) {
        data[i] = (i % 256) / 255.0;
      }

      return TensorData(
        shape: shape,
        dataType: TensorType.float32,
        data: data.buffer.asUint8List(),
      );
    }

    /// Analyze output tensor for issues.
    Map<String, dynamic> analyzeOutput(List<TensorData> outputs) {
      int totalElements = 0;
      int nanCount = 0;
      int infCount = 0;
      int zeroCount = 0;
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
            minVal = min(minVal, v);
            maxVal = max(maxVal, v);
          }

          if (firstValues.length < 8) {
            firstValues.add(v);
          }
        }
      }

      return {
        'totalElements': totalElements,
        'nanCount': nanCount,
        'infCount': infCount,
        'zeroCount': zeroCount,
        'allZero': zeroCount == totalElements && nanCount == 0 && infCount == 0,
        'hasNaN': nanCount > 0,
        'hasInf': infCount > 0,
        'minVal': minVal,
        'maxVal': maxVal,
        'firstValues': firstValues,
        'outputShapes': outputs.map((o) => o.shape).toList(),
      };
    }

    for (final partial in _partialModels) {
      testWidgets('${partial.name}: ${partial.desc}', (tester) async {
        ExecuTorchModel? model;

        try {
          // Load model from assets
          print('\n${"=" * 60}');
          print('Testing: ${partial.name} (${partial.desc})');
          print('  Asset: ${partial.asset}');
          print('  Input shape: ${partial.inputShape}');

          // Load model from asset bundle
          try {
            model = await ExecuTorchModel.loadFromAsset(partial.asset);
          } catch (e) {
            print('  SKIP - Asset not found: ${partial.asset}');
            print('  Export models first with mobilenet_debug.py');
            return;
          }
          print('  Model loaded successfully');

          // Create input
          final input = createInput(partial.inputShape);
          print('  Input created: ${partial.inputShape}');

          // Run inference
          final stopwatch = Stopwatch()..start();
          final outputs = await model.forward([input]);
          stopwatch.stop();
          print('  Inference time: ${stopwatch.elapsedMilliseconds}ms');

          // Analyze results
          final analysis = analyzeOutput(outputs);
          print('  Output shapes: ${analysis['outputShapes']}');
          print('  Total elements: ${analysis['totalElements']}');
          print(
              '  First 8 values: ${(analysis['firstValues'] as List<double>).map((v) => v.toStringAsFixed(6)).toList()}');
          print('  NaN count: ${analysis['nanCount']}');
          print('  Inf count: ${analysis['infCount']}');
          print('  Zero count: ${analysis['zeroCount']}');
          print('  All zero: ${analysis['allZero']}');
          print('  Min: ${analysis['minVal']}, Max: ${analysis['maxVal']}');

          // Determine result
          if (analysis['hasNaN'] as bool) {
            print('  RESULT: FAIL - NaN detected');
          } else if (analysis['hasInf'] as bool) {
            print('  RESULT: FAIL - Inf detected');
          } else if (analysis['allZero'] as bool) {
            print('  RESULT: FAIL - All zeros');
          } else {
            print('  RESULT: PASS - Non-trivial output');
          }

          print('${"=" * 60}\n');
        } catch (e, st) {
          print('  RESULT: ERROR - $e');
          print('  Stack: $st');
        } finally {
          if (model != null) {
            await model.dispose();
          }
        }
      });
    }
  });
}
