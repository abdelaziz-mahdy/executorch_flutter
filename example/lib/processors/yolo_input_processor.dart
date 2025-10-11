import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_input.dart';
import 'base_processor.dart';
import 'yolo_processor.dart';
import 'opencv/opencv_yolo_preprocessor.dart';

/// YOLO input processor with settings baked in
class YoloInputProcessor extends InputProcessor<ModelInput> {
  const YoloInputProcessor({required this.config, required this.useOpenCV});

  final YoloPreprocessConfig config;
  final bool useOpenCV;

  @override
  Future<List<TensorData>> process(ModelInput input) async {
    // Extract bytes from input
    final Uint8List bytes;
    if (input is ImageFileInput) {
      bytes = await input.file.readAsBytes();
    } else if (input is LiveCameraInput) {
      bytes = input.frameBytes;
    } else {
      throw UnsupportedError('Unsupported input type: ${input.runtimeType}');
    }

    // Select preprocessor based on settings and preprocess
    if (useOpenCV) {
      final preprocessor = OpenCVYoloPreprocessor(config: config);
      return await preprocessor.preprocess(bytes);
    } else {
      final preprocessor = YoloPreprocessor(config: config);
      return await preprocessor.preprocess(bytes);
    }
  }
}
