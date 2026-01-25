import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../models/model_input.dart';
import '../models/model_settings.dart';
import 'base_processor.dart';

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

    // Use CPU preprocessing (GPU shader can be added later)
    return _preprocessCpu(bytes);
  }

  Future<List<TensorData>> _preprocessCpu(Uint8List imageBytes) async {
    // Decode image
    final decodedImage = img.decodeImage(imageBytes);
    if (decodedImage == null) {
      throw Exception('Failed to decode image');
    }

    // Convert to RGB
    final rgbImage = decodedImage.convert(numChannels: 3);

    // Resize to target size (simple resize, MoveNet doesn't use letterbox)
    final resized = img.copyResize(
      rgbImage,
      width: config.targetSize,
      height: config.targetSize,
    );

    // Convert to tensor in NHWC format with normalization to [0, 1]
    // MoveNet uses NHWC format (batch, height, width, channels)
    final floats = Float32List(
      1 * config.targetSize * config.targetSize * 3,
    );

    int index = 0;
    for (int y = 0; y < config.targetSize; y++) {
      for (int x = 0; x < config.targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        floats[index++] = pixel.r / 255.0;
        floats[index++] = pixel.g / 255.0;
        floats[index++] = pixel.b / 255.0;
      }
    }

    debugPrint(
      '📊 MoveNet Tensor shape: [1, ${config.targetSize}, ${config.targetSize}, 3] (NHWC)',
    );

    return [
      TensorData(
        shape: [1, config.targetSize, config.targetSize, 3].cast<int?>(),
        dataType: TensorType.float32,
        data: floats.buffer.asUint8List(),
        name: 'input',
      ),
    ];
  }
}
