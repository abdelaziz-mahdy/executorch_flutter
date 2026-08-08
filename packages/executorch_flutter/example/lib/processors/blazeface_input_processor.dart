import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';

import '../models/model_input.dart';
import '../models/model_settings.dart';
import 'base_processor.dart';
import 'imagelib/imagelib_blazeface_preprocessor.dart';
import 'opencv/opencv_blazeface_preprocessor_stub.dart'
    if (dart.library.io) 'opencv/opencv_blazeface_preprocessor.dart';
import 'shaders/gpu_blazeface_preprocessor.dart';

/// Preprocessing configuration for BlazeFace
class BlazeFacePreprocessConfig {
  const BlazeFacePreprocessConfig({this.targetSize = 128});

  /// Target input size (128x128 for BlazeFace)
  final int targetSize;
}

/// Input processor for BlazeFace face detection
/// Handles image preprocessing: resize to 128x128, normalize to [-1, 1]
class BlazeFaceInputProcessor extends InputProcessor<ModelInput> {
  BlazeFaceInputProcessor({
    required this.config,
    required this.preprocessingProvider,
  });

  final BlazeFacePreprocessConfig config;
  final PreprocessingProvider preprocessingProvider;

  @override
  Future<List<TensorData>> process(ModelInput input) async {
    // Get bytes from input
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
        final preprocessor = GpuBlazeFacePreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.opencv:
        final preprocessor = OpenCVBlazeFacePreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.imageLib:
        final preprocessor = ImageLibBlazeFacePreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
    }
  }
}
