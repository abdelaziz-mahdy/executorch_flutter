import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_input.dart';
import 'base_processor.dart';
import 'image_processor.dart';
import 'opencv/opencv_imagenet_preprocessor.dart';

/// MobileNet/ImageNet input processor with settings baked in
class MobileNetInputProcessor extends InputProcessor<ModelInput> {
  const MobileNetInputProcessor({
    required this.config,
    required this.useOpenCV,
  });

  final ImagePreprocessConfig config;
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

    // Select preprocessor based on settings
    if (useOpenCV) {
      final preprocessor = OpenCVImageNetPreprocessor(config: config);
      return await preprocessor.preprocess(bytes);
    } else {
      final preprocessor = ImageNetPreprocessor(config: config);
      return await preprocessor.preprocess(bytes);
    }
  }
}
