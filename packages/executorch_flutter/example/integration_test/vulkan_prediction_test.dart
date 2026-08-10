// Do the Vulkan builds make the *right* predictions on real images?
//
// vulkan_correctness_test.dart proves Vulkan and XNNPACK agree numerically, but
// it feeds pseudo-random noise: it would pass just as happily if both backends
// were confidently wrong, and its "argmax" on noise is not a prediction of
// anything. This test runs actual photographs through the full app pipeline —
// the same preprocessors and postprocessors the example app uses — and checks
// the predictions mean what they should.
//
// Usage: flutter test integration_test/vulkan_prediction_test.dart -d <device>

// ignore_for_file: avoid_print

@Timeout(Duration(minutes: 30))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:path_provider/path_provider.dart';

import 'package:executorch_flutter_example/processors/image_processor.dart';
import 'package:executorch_flutter_example/processors/yolo_processor.dart';
import 'package:executorch_flutter_example/processors/imagelib/imagelib_mobilenet_preprocessor.dart';
import 'package:executorch_flutter_example/processors/imagelib/imagelib_yolo_preprocessor.dart';
import 'package:executorch_flutter_example/processors/yolo_output_processor.dart';
import 'package:executorch_flutter_example/services/model_index_service.dart';
import 'package:executorch_flutter_example/services/model_download_service.dart';

/// ImageNet indices for the classes we assert on.
///
/// MobileNet is free to disagree with itself about *which* cat, so a range is
/// the honest assertion — what would signal breakage is "cat photo classified
/// as an object".
const _catClasses = {281, 282, 283, 284, 285}; // tabby .. Egyptian cat

/// ImageNet dog breeds occupy one contiguous block, 151 (Chihuahua) through
/// 268 (Mexican hairless). Asserting on a hand-picked subset instead just
/// encodes which breeds happened to come to mind — dog.jpg classifies as 162
/// (beagle), which a sparse list misses while the model is perfectly right.
final _dogClasses = List<int>.generate(268 - 151 + 1, (i) => 151 + i).toSet();

/// COCO class index for `person`, the top detection expected in person.jpg.
const _cocoPerson = 0;

List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce(math.max);
  final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
  final sum = exps.reduce((a, b) => a + b);
  return exps.map((e) => e / sum).toList();
}

Float32List _asFloat32(TensorData tensor) => tensor.data.buffer.asFloat32List(
  tensor.data.offsetInBytes,
  tensor.data.lengthInBytes ~/ 4,
);

class _TopK {
  _TopK(this.indices, this.probabilities);
  final List<int> indices;
  final List<double> probabilities;

  int get top1 => indices.first;
  double get top1Probability => probabilities.first;

  String describe() {
    final parts = <String>[];
    for (var i = 0; i < math.min(3, indices.length); i++) {
      parts.add(
        '${indices[i]}@${(probabilities[i] * 100).toStringAsFixed(1)}%',
      );
    }
    return parts.join(', ');
  }
}

_TopK _topK(Float32List logits, int k) {
  final probs = _softmax(logits.toList());
  final order = List<int>.generate(probs.length, (i) => i)
    ..sort((a, b) => probs[b].compareTo(probs[a]));
  final top = order.take(k).toList();
  return _TopK(top, top.map((i) => probs[i]).toList());
}

double _iou(BoundingBox a, BoundingBox b) {
  final left = math.max(a.x, b.x);
  final top = math.max(a.y, b.y);
  final right = math.min(a.x + a.width, b.x + b.width);
  final bottom = math.min(a.y + a.height, b.y + b.height);
  if (right <= left || bottom <= top) return 0.0;
  final overlap = (right - left) * (bottom - top);
  return overlap / (a.width * a.height + b.width * b.height - overlap);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Vulkan predictions on real images', () {
    late ExecutorchManager manager;
    final Map<String, String> modelPaths = {};

    Future<String> downloadModel(ModelIndexEntry entry) async {
      final info = await ModelDownloadService.instance.downloadModel(
        modelName: entry.name,
        remoteUrl: entry.remoteUrl,
        expectedHash: entry.hash,
      );
      if (info.state != ModelDownloadState.downloaded) {
        throw Exception(
          'Failed to download ${entry.name}: ${info.errorMessage}',
        );
      }
      if (info.localPath != null) return info.localPath!;
      if (info.bytes != null) {
        final dir = await getApplicationCacheDirectory();
        final file = File('${dir.path}/${entry.name}');
        await file.writeAsBytes(info.bytes!);
        return file.path;
      }
      throw Exception('Download succeeded but no path or bytes available');
    }

    setUpAll(() async {
      manager = ExecutorchManager.instance;
      await manager.initialize();

      print('');
      print('========================================');
      print('  Vulkan predictions on real images');
      print('========================================');
      print('');

      const wanted = [
        'mobilenet_v3_small_xnnpack.pte',
        'mobilenet_v3_small_vulkan.pte',
        'yolo11n_xnnpack.pte',
        'yolo11n_vulkan.pte',
      ];

      final index = await ModelIndexService.fetchIndex(forceRefresh: true);
      for (final name in wanted) {
        final matches = index.models.where((m) => m.name == name);
        if (matches.isEmpty) {
          print('  UNAVAILABLE: $name not in index');
          continue;
        }
        const attempts = 4;
        for (var attempt = 1; attempt <= attempts; attempt++) {
          try {
            modelPaths[name] = await downloadModel(matches.first);
            break;
          } catch (e) {
            final last = attempt == attempts;
            print(
              '  ${last ? "UNAVAILABLE" : "retry"}: $name '
              'attempt $attempt/$attempts: $e',
            );
            if (last) break;
            await Future<void>.delayed(Duration(seconds: 3 * attempt));
          }
        }
      }
      print('Models ready: ${modelPaths.length}/${wanted.length}');
      print('');
    });

    tearDownAll(() async {
      await manager.disposeAllModels();
    });

    Future<List<TensorData>> runModel(
      String modelName,
      List<TensorData> inputs,
    ) async {
      final path = modelPaths[modelName];
      if (path == null) fail('$modelName unavailable — see setUpAll output');

      final model = await ExecuTorchModel.load(path);
      try {
        final outputs = await model.forward(inputs);
        return outputs
            .map(
              (o) => TensorData(
                shape: o.shape,
                dataType: o.dataType,
                data: Uint8List.fromList(o.data),
                name: o.name,
              ),
            )
            .toList();
      } finally {
        await model.dispose();
      }
    }

    testWidgets('MobileNet classifies real photos on both backends', (_) async {
      final preprocessor = ImageLibMobileNetPreprocessor(
        config: const ImagePreprocessConfig(),
      );

      for (final probe in [
        ('cat.jpg', _catClasses, 'cat'),
        ('dog.jpg', _dogClasses, 'dog'),
      ]) {
        final (asset, expectedClasses, label) = probe;
        final bytes = (await rootBundle.load(
          'assets/images/$asset',
        )).buffer.asUint8List();
        final inputs = await preprocessor.preprocess(bytes);

        final xnnpack = _topK(
          _asFloat32(
            (await runModel('mobilenet_v3_small_xnnpack.pte', inputs)).first,
          ),
          5,
        );
        final vulkan = _topK(
          _asFloat32(
            (await runModel('mobilenet_v3_small_vulkan.pte', inputs)).first,
          ),
          5,
        );

        print('--- $asset ---');
        print('  XNNPACK top-3: ${xnnpack.describe()}');
        print('  Vulkan  top-3: ${vulkan.describe()}');

        expect(
          expectedClasses.contains(xnnpack.top1),
          isTrue,
          reason:
              'XNNPACK reference is itself wrong on $asset — top-1 class '
              '${xnnpack.top1} is not a $label. The Vulkan comparison below '
              'would be meaningless.',
        );
        expect(
          expectedClasses.contains(vulkan.top1),
          isTrue,
          reason:
              'Vulkan classified $asset as class ${vulkan.top1}, not a '
              '$label',
        );
        expect(
          vulkan.top1,
          xnnpack.top1,
          reason:
              'Backends disagree on $asset: Vulkan says ${vulkan.top1}, '
              'XNNPACK says ${xnnpack.top1}',
        );
        // FP16 shifts confidence a little; a large gap would mean the class
        // won for different reasons on the two backends.
        expect(
          (vulkan.top1Probability - xnnpack.top1Probability).abs(),
          lessThan(0.05),
          reason:
              'Confidence differs sharply on $asset: Vulkan '
              '${vulkan.top1Probability} vs XNNPACK ${xnnpack.top1Probability}',
        );
        print('  PASS: both backends say class ${vulkan.top1} ($label)');
        print('');
      }
    });

    testWidgets('YOLO11n detects real objects on both backends', (_) async {
      final preprocessor = ImageLibYoloPreprocessor(
        config: const YoloPreprocessConfig(),
      );
      // Real COCO names are not bundled here, and the assertions below are on
      // class index and geometry, so positional placeholders suffice.
      final labels = List<String>.generate(80, (i) => 'class_$i');
      final postprocessor = YoloOutputProcessor(
        classLabels: labels,
        inputWidth: 640,
        inputHeight: 640,
        confidenceThreshold: 0.25,
        iouThreshold: 0.45,
      );

      final bytes = (await rootBundle.load(
        'assets/images/person.jpg',
      )).buffer.asUint8List();
      final inputs = await preprocessor.preprocess(bytes);

      final xnnpackResult = await postprocessor.process(
        await runModel('yolo11n_xnnpack.pte', inputs),
      );
      final vulkanResult = await postprocessor.process(
        await runModel('yolo11n_vulkan.pte', inputs),
      );

      final refDets = xnnpackResult.detectedObjects;
      final vkDets = vulkanResult.detectedObjects;

      print('--- person.jpg ---');
      print('  XNNPACK: ${refDets.length} detections');
      for (final d in refDets.take(3)) {
        print(
          '    class ${d.classIndex} @ '
          '${(d.confidence * 100).toStringAsFixed(1)}% '
          'box=(${d.boundingBox.x.toStringAsFixed(1)}, '
          '${d.boundingBox.y.toStringAsFixed(1)}, '
          '${d.boundingBox.width.toStringAsFixed(1)}, '
          '${d.boundingBox.height.toStringAsFixed(1)})',
        );
      }
      print('  Vulkan : ${vkDets.length} detections');
      for (final d in vkDets.take(3)) {
        print(
          '    class ${d.classIndex} @ '
          '${(d.confidence * 100).toStringAsFixed(1)}% '
          'box=(${d.boundingBox.x.toStringAsFixed(1)}, '
          '${d.boundingBox.y.toStringAsFixed(1)}, '
          '${d.boundingBox.width.toStringAsFixed(1)}, '
          '${d.boundingBox.height.toStringAsFixed(1)})',
        );
      }

      expect(
        refDets,
        isNotEmpty,
        reason:
            'XNNPACK reference found nothing in person.jpg — the Vulkan '
            'comparison below would be meaningless',
      );
      expect(
        vkDets,
        isNotEmpty,
        reason:
            'Vulkan found no objects in person.jpg while XNNPACK found '
            '${refDets.length}',
      );

      final refTop = refDets.first;
      final vkTop = vkDets.first;

      expect(
        refTop.classIndex,
        _cocoPerson,
        reason:
            'XNNPACK top detection is class ${refTop.classIndex}, not '
            'person — the reference itself is wrong',
      );
      expect(
        vkTop.classIndex,
        _cocoPerson,
        reason: 'Vulkan top detection is class ${vkTop.classIndex}, not person',
      );

      final overlap = _iou(refTop.boundingBox, vkTop.boundingBox);
      print('  top-detection IoU: ${overlap.toStringAsFixed(4)}');
      expect(
        overlap,
        greaterThan(0.9),
        reason:
            'Top boxes barely overlap (IoU $overlap) — the backends are '
            'localising the same object differently',
      );
      expect(
        (vkTop.confidence - refTop.confidence).abs(),
        lessThan(0.05),
        reason:
            'Top-detection confidence differs sharply: Vulkan '
            '${vkTop.confidence} vs XNNPACK ${refTop.confidence}',
      );

      // NMS is a thresholded process, so counts can legitimately differ by one
      // near the cutoff — but not wildly.
      expect(
        (vkDets.length - refDets.length).abs(),
        lessThanOrEqualTo(1),
        reason:
            'Detection counts diverge: Vulkan ${vkDets.length} vs XNNPACK '
            '${refDets.length}',
      );
      print(
        '  PASS: both backends detect person with IoU '
        '${overlap.toStringAsFixed(4)}',
      );
    });
  });
}
