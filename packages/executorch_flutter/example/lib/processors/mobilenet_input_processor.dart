import 'package:executorch_flutter_example/processors/shaders/gpu_mobilenet_preprocessor.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_input.dart';
import '../models/model_settings.dart';
import 'base_processor.dart';
import 'image_processor.dart';
import 'imagelib/imagelib_mobilenet_preprocessor.dart';
import 'opencv/opencv_imagenet_preprocessor_stub.dart'
    if (dart.library.io) 'opencv/opencv_imagenet_preprocessor.dart';

/// MobileNet/ImageNet input processor with settings baked in
class MobileNetInputProcessor extends InputProcessor<ModelInput> {
  const MobileNetInputProcessor({
    required this.config,
    required this.preprocessingProvider,
  });

  final ImagePreprocessConfig config;
  final PreprocessingProvider preprocessingProvider;

  @override
  Future<List<TensorData>> process(ModelInput input) async {
    // Extract bytes from input
    final Uint8List bytes;
    if (input is ImageBytesInput) {
      bytes = input.imageBytes;
    } else if (input is LiveCameraInput) {
      bytes = input.frameBytes;
    } else {
      throw UnsupportedError('Unsupported input type: ${input.runtimeType}');
    }

    // Select preprocessor based on settings
    switch (preprocessingProvider) {
      case PreprocessingProvider.gpu:
        final preprocessor = GpuMobileNetPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.opencv:
        final preprocessor = OpenCVImageNetPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.imageLib:
        final preprocessor = ImageLibMobileNetPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
    }
  }
}
