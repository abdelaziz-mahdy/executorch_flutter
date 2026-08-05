import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../movenet_input_processor.dart';

/// ImageLib-based MoveNet preprocessor
///
/// Uses the `image` package for preprocessing:
/// - Decodes image from bytes
/// - Resizes to target size (192 for Lightning, 256 for Thunder)
/// - Converts to NHWC tensor format (height, width, channels)
/// - Normalizes to [0, 1] range
class ImageLibMoveNetPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  ImageLibMoveNetPreprocessor({required this.config});

  final MoveNetPreprocessConfig config;

  @override
  String get inputTypeName => 'Image (Uint8List) [ImageLib]';

  @override
  bool validateInput(Uint8List input) {
    if (input.isEmpty) return false;
    try {
      final image = img.decodeImage(input);
      return image != null && image.width > 0 && image.height > 0;
    } catch (e) {
      return false;
    }
  }

  @override
  Future<List<TensorData>> preprocess(Uint8List input) async {
    try {
      // Decode image
      final decodedImage = img.decodeImage(input);
      if (decodedImage == null) {
        throw PreprocessingException('Failed to decode image');
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
        'ImageLibMoveNetPreprocessor: Tensor shape [1, ${config.targetSize}, ${config.targetSize}, 3] (NHWC)',
      );

      return [
        TensorData(
          shape: [1, config.targetSize, config.targetSize, 3].cast<int?>(),
          dataType: TensorType.float32,
          data: floats.buffer.asUint8List(),
          name: 'input',
        ),
      ];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('MoveNet preprocessing failed: $e', e);
    }
  }
}
