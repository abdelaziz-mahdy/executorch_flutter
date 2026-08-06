// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:executorch_dart/executorch_dart.dart';

Future<void> main(List<String> args) async {
  if (args.length != 1) {
    print('usage: dart run bin/infer.dart <model.pte>');
    exit(64);
  }

  print('ExecuTorch ${ExecuTorchVersion.version}');

  try {
    final model = await ExecuTorchModel.load(args.single);
    print('loaded ${model.modelId}');

    final input = TensorData(
      shape: [1, 3, 224, 224],
      dataType: TensorType.float32,
      data: Float32List(1 * 3 * 224 * 224).buffer.asUint8List(),
      name: 'input',
    );

    final outputs = await model.forward([input]);
    print('outputs: ${outputs.length}');
    for (final output in outputs) {
      print('  shape ${output.shape} dtype ${output.dataType}');
    }

    await model.dispose();
    print('ok');
  } on ExecuTorchException catch (e) {
    stderr.writeln('Error: $e');
    exit(1);
  }
}
