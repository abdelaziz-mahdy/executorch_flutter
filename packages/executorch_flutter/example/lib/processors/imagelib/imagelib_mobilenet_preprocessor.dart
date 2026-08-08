import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../image_processor.dart';

/// ImageLib-based MobileNet/ImageNet preprocessor
///
/// Uses the `image` package for preprocessing:
/// - Decodes image from bytes
/// - Resizes to target size (224x224 for ImageNet)
/// - Converts to NCHW tensor format
/// - Applies ImageNet normalization (mean subtraction and std division)
class ImageLibMobileNetPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  ImageLibMobileNetPreprocessor({required this.config});

  final ImagePreprocessConfig config;

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

      // Resize to model input size (224x224 for most image models)
      final resized = img.copyResize(
        decodedImage,
        width: config.targetWidth,
        height: config.targetHeight,
      );

      // Convert to RGB if needed
      final rgbImage = resized.convert(numChannels: 3);

      // ImageNet normalization constants
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];

      // Create float32 tensor in NCHW format
      final floats = Float32List(
        1 * 3 * config.targetHeight * config.targetWidth,
      );

      // Fill tensor in NCHW format: [batch, channel, height, width]
      int index = 0;

      // Channel 0 (Red)
      for (int y = 0; y < config.targetHeight; y++) {
        for (int x = 0; x < config.targetWidth; x++) {
          final pixel = rgbImage.getPixel(x, y);
          final normalizedValue = (pixel.r / 255.0 - mean[0]) / std[0];
          floats[index++] = normalizedValue;
        }
      }

      // Channel 1 (Green)
      for (int y = 0; y < config.targetHeight; y++) {
        for (int x = 0; x < config.targetWidth; x++) {
          final pixel = rgbImage.getPixel(x, y);
          final normalizedValue = (pixel.g / 255.0 - mean[1]) / std[1];
          floats[index++] = normalizedValue;
        }
      }

      // Channel 2 (Blue)
      for (int y = 0; y < config.targetHeight; y++) {
        for (int x = 0; x < config.targetWidth; x++) {
          final pixel = rgbImage.getPixel(x, y);
          final normalizedValue = (pixel.b / 255.0 - mean[2]) / std[2];
          floats[index++] = normalizedValue;
        }
      }

      debugPrint(
        'ImageLibMobileNetPreprocessor: Tensor shape [1, 3, ${config.targetHeight}, ${config.targetWidth}]',
      );

      // Create tensor data
      final tensorData = TensorData(
        shape: [
          1,
          3,
          config.targetHeight,
          config.targetWidth,
        ].cast<int?>(), // NCHW format
        dataType: TensorType.float32,
        data: floats.buffer.asUint8List(),
        name: 'input',
      );

      return [tensorData];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('MobileNet preprocessing failed: $e', e);
    }
  }
}
