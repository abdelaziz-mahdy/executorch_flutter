// ignore_for_file: avoid_print
//
// Regression test for the use-after-free crash where disposing a model while
// an async forward is still running (e.g. turning off the live camera
// mid-inference) freed the model's weights out from under the running kernel,
// crashing deep in convolution_out.
//
// Run with:
//   flutter test integration_test/dispose_race_test.dart -d macos

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

import 'package:executorch_flutter_example/services/model_index_service.dart';
import 'package:executorch_flutter_example/services/model_download_service.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final downloadService = ModelDownloadService.instance;

  Future<ExecuTorchModel> loadModel(ModelIndexEntry entry) async {
    final info = await downloadService.downloadModel(
      modelName: entry.name,
      remoteUrl: entry.remoteUrl,
      expectedHash: entry.hash,
    );
    if (info.localPath != null) {
      return ExecuTorchModel.load(info.localPath!);
    }
    return ExecuTorchModel.loadFromBytes(info.bytes!);
  }

  TensorData makeInput(int size) {
    final floats = Float32List(1 * 3 * size * size);
    for (var i = 0; i < floats.length; i++) {
      floats[i] = (i % 256) / 255.0;
    }
    return TensorData(
      shape: [1, 3, size, size],
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'input',
    );
  }

  testWidgets('dispose during in-flight forward does not crash', (tester) async {
    final index = await ModelIndexService.fetchIndex(forceRefresh: true);

    // Prefer the exact model from the crash report (partially-delegated MPS,
    // which runs a Conv2d on the portable convolution_out kernel). Fall back to
    // any available model so the test still runs on platforms without MPS.
    final entry = index.models.firstWhere(
      (m) => m.name == 'yolo11n_mps.pte',
      orElse: () => index.models.firstWhere(
        (m) => m.backend == 'xnnpack' && m.category == 'yolo',
        orElse: () => index.models.first,
      ),
    );
    print('Using model: ${entry.name} (${entry.backend})');

    final input = makeInput(entry.inputSize ?? 640);

    // Simulate the camera-teardown race a few times: fire several forwards
    // WITHOUT awaiting them, then dispose while they are still in flight.
    for (var round = 0; round < 5; round++) {
      final model = await loadModel(entry);

      final pending = <Future<void>>[];
      for (var i = 0; i < 4; i++) {
        pending.add(
          model.forward([input]).then((_) {}).catchError((_) {}),
        );
      }
      // Give the worker threads a moment to actually enter forward().
      await Future<void>.delayed(const Duration(milliseconds: 3));

      // Dispose mid-inference. Before the fix this freed the model while the
      // portable conv kernel was reading its weights -> SIGSEGV. With the fix
      // dispose waits for the in-flight forwards to finish.
      await model.dispose();
      expect(model.isDisposed, isTrue);

      // The fired-off forwards should all have settled (completed or thrown a
      // clean "disposed" error) - never crashed the process.
      await Future.wait(pending);
      print('  round $round: survived dispose-during-forward');
    }
  });
}
