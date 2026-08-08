import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../blazeface_input_processor.dart';

/// ImageLib-based BlazeFace preprocessor
///
/// Uses the `image` package for preprocessing:
/// - Decodes image from bytes
/// - Resizes to 128x128 (BlazeFace input size)
/// - Converts to NHWC tensor format (height, width, channels)
/// - Normalizes to [-1, 1] range
class ImageLibBlazeFacePreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  ImageLibBlazeFacePreprocessor({required this.config});

  final BlazeFacePreprocessConfig config;

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

      // Resize to target size
      final resized = img.copyResize(
        rgbImage,
        width: config.targetSize,
        height: config.targetSize,
      );

      // Convert to tensor in NHWC format with normalization to [-1, 1]
      // BlazeFace uses NHWC format and expects values in [-1, 1] range
      final floats = Float32List(1 * config.targetSize * config.targetSize * 3);

      int index = 0;
      for (int y = 0; y < config.targetSize; y++) {
        for (int x = 0; x < config.targetSize; x++) {
          final pixel = resized.getPixel(x, y);
          // Normalize to [-1, 1]: (value / 127.5) - 1
          floats[index++] = (pixel.r / 127.5) - 1.0;
          floats[index++] = (pixel.g / 127.5) - 1.0;
          floats[index++] = (pixel.b / 127.5) - 1.0;
        }
      }

      debugPrint(
        'ImageLibBlazeFacePreprocessor: Tensor shape [1, ${config.targetSize}, ${config.targetSize}, 3] (NHWC)',
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
      throw PreprocessingException('BlazeFace preprocessing failed: $e', e);
    }
  }
}
