// ignore_for_file: avoid_print

/// Vulkan MobileNet real-image test.
///
/// Tests FP32 vs FP16 Vulkan MobileNet on a real cat image using the
/// existing ImageNetPreprocessor/ImageNetPostprocessor.
///
/// Run:
///   cd example
///   flutter drive --driver=test_driver/integration_test.dart \
///     --target=integration_test/vulkan_mobilenet_image_test.dart -d DEVICE_ID

@Timeout(Duration(minutes: 5))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/processors/image_processor.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Models to test with real image input.
final _models = [
  (
    name: 'MobileNet XNNPACK (CPU ref)',
    asset: 'assets/debug_models/mobilenet_full_xnnpack.pte',
  ),
  (
    name: 'MobileNet FP32 Vulkan',
    asset: 'assets/debug_models/mobilenet_full_fp32_vulkan.pte',
  ),
  (
    name: 'MobileNet FP16 Vulkan',
    asset: 'assets/debug_models/mobilenet_debug_full_model_vulkan.pte',
  ),
];

/// Well-known ImageNet cat class indices (281-285 are various cat breeds).
const _catClassIndices = {
  281, // tabby
  282, // tiger cat
  283, // Persian cat
  284, // Siamese cat
  285, // Egyptian cat
};

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan MobileNet Real Image Tests', () {
    late Uint8List catImageBytes;
    late List<TensorData> preprocessedInput;

    setUpAll(() async {
      // Load cat.jpg from assets
      final byteData = await rootBundle.load('assets/images/cat.jpg');
      catImageBytes = byteData.buffer.asUint8List();
      print('Loaded cat.jpg: ${catImageBytes.length} bytes');

      // Preprocess with ImageNetPreprocessor (224x224, ImageNet normalization)
      final preprocessor = ImageNetPreprocessor(
        config: const ImagePreprocessConfig(),
      );
      preprocessedInput = await preprocessor.preprocess(catImageBytes);
      print(
        'Preprocessed: shape=${preprocessedInput[0].shape}, '
        'dtype=${preprocessedInput[0].dataType}, '
        'bytes=${preprocessedInput[0].data.length}',
      );
    });

    for (final model in _models) {
      testWidgets('${model.name} - cat.jpg classification', (tester) async {
        ExecuTorchModel? loadedModel;

        try {
          print('\n${"=" * 60}');
          print('Testing: ${model.name}');
          print('  Asset: ${model.asset}');

          // Load model
          try {
            loadedModel = await ExecuTorchModel.loadFromAsset(model.asset);
          } catch (e) {
            print('  SKIP - Asset not found: ${model.asset}');
            return;
          }
          print('  Model loaded successfully');

          // Run inference
          final stopwatch = Stopwatch()..start();
          final outputs = await loadedModel.forward(preprocessedInput);
          stopwatch.stop();
          print('  Inference time: ${stopwatch.elapsedMilliseconds}ms');

          // Analyze raw output
          final output = outputs.first;
          final floats = output.data.buffer.asFloat32List();
          print('  Output shape: ${output.shape}');
          print('  Output elements: ${floats.length}');

          // Check for NaN/Inf
          int nanCount = 0;
          int infCount = 0;
          int zeroCount = 0;
          for (final v in floats) {
            if (v.isNaN) nanCount++;
            if (v.isInfinite) infCount++;
            if (v == 0.0) zeroCount++;
          }
          print('  NaN: $nanCount, Inf: $infCount, Zero: $zeroCount');

          if (nanCount > 0) {
            print('  RESULT: FAIL - NaN in output');
            print('${"=" * 60}\n');
            return;
          }
          if (zeroCount == floats.length) {
            print('  RESULT: FAIL - All zeros');
            print('${"=" * 60}\n');
            return;
          }

          // Apply softmax and get top-5
          final logits = List<double>.from(floats);
          final maxLogit = logits.reduce((a, b) => a > b ? a : b);
          final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
          final sumExp = exps.reduce((a, b) => a + b);
          final probs = exps.map((e) => e / sumExp).toList();

          // Get top-5 indices
          final indexed = List.generate(probs.length, (i) => i);
          indexed.sort((a, b) => probs[b].compareTo(probs[a]));

          print('  Top-5 predictions:');
          for (int i = 0; i < 5 && i < indexed.length; i++) {
            final idx = indexed[i];
            final prob = probs[idx];
            final isCat = _catClassIndices.contains(idx);
            print(
              '    #${i + 1}: class $idx '
              '(${(prob * 100).toStringAsFixed(2)}%)'
              '${isCat ? " <- CAT" : ""}',
            );
          }

          // Check if top-1 is a cat class
          final top1Index = indexed[0];
          final top1IsCat = _catClassIndices.contains(top1Index);
          final top1Confidence = probs[top1Index];

          if (top1IsCat) {
            print(
              '  RESULT: PASS - Top-1 is cat class $top1Index '
              '(${(top1Confidence * 100).toStringAsFixed(1)}%)',
            );
          } else {
            // Check if any cat class is in top-5
            final catInTop5 = indexed
                .take(5)
                .any((idx) => _catClassIndices.contains(idx));
            if (catInTop5) {
              print(
                '  RESULT: WARN - Cat not top-1 but in top-5 '
                '(top-1: class $top1Index)',
              );
            } else {
              print(
                '  RESULT: FAIL - No cat class in top-5 '
                '(top-1: class $top1Index, '
                '${(top1Confidence * 100).toStringAsFixed(1)}%)',
              );
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
