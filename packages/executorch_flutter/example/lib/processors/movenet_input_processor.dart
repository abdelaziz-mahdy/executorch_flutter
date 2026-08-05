import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';

import '../models/model_input.dart';
import '../models/model_settings.dart';
import 'base_processor.dart';
import 'imagelib/imagelib_movenet_preprocessor.dart';
import 'opencv/opencv_movenet_preprocessor_stub.dart'
    if (dart.library.io) 'opencv/opencv_movenet_preprocessor.dart';
import 'shaders/gpu_movenet_preprocessor.dart';

/// Preprocessing configuration for MoveNet
class MoveNetPreprocessConfig {
  const MoveNetPreprocessConfig({
    this.targetSize = 192,
  });

  /// Target input size (192 for Lightning, 256 for Thunder)
  final int targetSize;
}

/// Input processor for MoveNet pose detection
/// Handles image preprocessing: resize to square, normalize to [0, 1]
class MoveNetInputProcessor extends InputProcessor<ModelInput> {
  MoveNetInputProcessor({
    required this.config,
    required this.preprocessingProvider,
  });

  final MoveNetPreprocessConfig config;
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
        final preprocessor = GpuMoveNetPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.opencv:
        final preprocessor = OpenCVMoveNetPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
      case PreprocessingProvider.imageLib:
        final preprocessor = ImageLibMoveNetPreprocessor(config: config);
        return await preprocessor.preprocess(bytes);
    }
  }
}
