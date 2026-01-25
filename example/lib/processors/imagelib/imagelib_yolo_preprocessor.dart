import 'dart:math' as math;

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../yolo_processor.dart';

/// ImageLib-based YOLO preprocessor
///
/// Uses the `image` package for preprocessing:
/// - Decodes image from bytes
/// - Letterbox resize to target size (maintains aspect ratio with gray padding)
/// - Converts to NCHW tensor format
/// - Normalizes to [0, 1] range
class ImageLibYoloPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  ImageLibYoloPreprocessor({required this.config});

  final YoloPreprocessConfig config;

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

      // Letterbox resize (YOLO standard)
      final resized = _letterboxResize(rgbImage);

      // Convert to tensor in NCHW format with normalization to [0, 1]
      final tensorData = _imageToTensor(resized);

      return [tensorData];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('YOLO preprocessing failed: $e', e);
    }
  }

  img.Image _letterboxResize(img.Image image) {
    // Calculate scale to fit image within target size while maintaining aspect ratio
    final scaleW = config.targetWidth / image.width;
    final scaleH = config.targetHeight / image.height;
    final scale = math.min(scaleW, scaleH);

    // Calculate new dimensions
    final newWidth = (image.width * scale).round();
    final newHeight = (image.height * scale).round();

    // Resize image
    final resized = img.copyResize(image, width: newWidth, height: newHeight);

    // Create target image with gray padding
    final target = img.Image(
      width: config.targetWidth,
      height: config.targetHeight,
      numChannels: 3,
    );
    img.fill(target, color: img.ColorRgb8(114, 114, 114)); // Gray padding

    // Calculate offsets to center the resized image
    final offsetX = (config.targetWidth - newWidth) ~/ 2;
    final offsetY = (config.targetHeight - newHeight) ~/ 2;

    // Composite the resized image onto the target
    img.compositeImage(target, resized, dstX: offsetX, dstY: offsetY);

    return target;
  }

  TensorData _imageToTensor(img.Image image) {
    // Create float32 tensor in NCHW format
    // Modern YOLO models (v8, v11, etc.) expect [0, 1] normalized inputs
    final floats = Float32List(
      1 * 3 * config.targetHeight * config.targetWidth,
    );

    int index = 0;

    // Channel 0 (Red) - normalize to [0, 1]
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = image.getPixel(x, y);
        floats[index++] = pixel.r / 255.0;
      }
    }

    // Channel 1 (Green) - normalize to [0, 1]
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = image.getPixel(x, y);
        floats[index++] = pixel.g / 255.0;
      }
    }

    // Channel 2 (Blue) - normalize to [0, 1]
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = image.getPixel(x, y);
        floats[index++] = pixel.b / 255.0;
      }
    }

    debugPrint(
      'ImageLibYoloPreprocessor: Tensor shape [1, 3, ${config.targetHeight}, ${config.targetWidth}]',
    );

    return TensorData(
      shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(),
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'images',
    );
  }
}
