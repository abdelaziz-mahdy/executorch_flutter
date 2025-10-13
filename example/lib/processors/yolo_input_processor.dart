import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_input.dart';
import '../models/model_settings.dart';
import 'base_processor.dart';
import 'yolo_processor.dart';
import 'opencv/opencv_yolo_preprocessor.dart';
import 'shaders/gpu_yolo_preprocessor.dart';

/// YOLO input processor with settings baked in
class YoloInputProcessor extends InputProcessor<ModelInput> {
  const YoloInputProcessor({
    required this.config,
    required this.preprocessingProvider,
  });

  final YoloPreprocessConfig config;
  final PreprocessingProvider preprocessingProvider;

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
    switch (preprocessingProvider) {
      case PreprocessingProvider.gpu:
        final preprocessor = GpuYoloPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.opencv:
        final preprocessor = OpenCVYoloPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.imageLib:
        final preprocessor = YoloPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
    }
  }
}
