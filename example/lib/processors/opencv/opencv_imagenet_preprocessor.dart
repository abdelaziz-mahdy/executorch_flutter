import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/processors/image_processor.dart';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// OpenCV-accelerated ImageNet/MobileNet preprocessor
/// Uses native OpenCV async operations for maximum efficiency
/// ALL image manipulation is done natively - Dart only handles final byte conversion
class OpenCVImageNetPreprocessor {
  OpenCVImageNetPreprocessor({required this.config});

  final ImagePreprocessConfig config;

  Future<List<TensorData>> preprocess(Uint8List imageBytes) async {
    // Decode image from bytes using OpenCV async
    final mat = await cv.imdecodeAsync(imageBytes, cv.IMREAD_COLOR);

    // Convert BGR to RGB async
    final rgbMat = await cv.cvtColorAsync(mat, cv.COLOR_BGR2RGB);

    // Resize to target size using OpenCV async
    final resized = await cv.resizeAsync(rgbMat, (
      config.targetWidth,
      config.targetHeight,
    ), interpolation: cv.INTER_LINEAR);

    // Convert to float32 and normalize - all done natively in OpenCV
    final float32Mat = config.normalizeToFloat
        ? resized.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0)
        : resized.convertTo(cv.MatType.CV_32FC3);

    // Split into separate channels for CHW format
    final channels = cv.split(float32Mat);

    // Create output tensor in CHW format
    final channelSize = config.targetHeight * config.targetWidth;
    final tensorSize = 3 * channelSize;
    final floats = Float32List(tensorSize);

    // Copy each channel's data and apply ImageNet normalization if configured
    // Mat.data returns Uint8List (raw bytes), convert to Float32List via ByteData
    for (int c = 0; c < 3; c++) {
      final channel = channels[c];
      final bytes = channel.data;
      final byteData = ByteData.sublistView(bytes);
      final offset = c * channelSize;

      // If ImageNet normalization is configured, apply it per channel
      if (config.meanSubtraction.isNotEmpty &&
          config.standardDeviation.isNotEmpty &&
          c < config.meanSubtraction.length &&
          c < config.standardDeviation.length) {
        final mean = config.meanSubtraction[c];
        final std = config.standardDeviation[c];
        for (int i = 0; i < channelSize; i++) {
          final value = byteData.getFloat32(i * 4, Endian.host);
          floats[offset + i] = (value - mean) / std;
        }
      } else {
        for (int i = 0; i < channelSize; i++) {
          floats[offset + i] = byteData.getFloat32(i * 4, Endian.host);
        }
      }
    }

    // Clean up OpenCV resources
    mat.dispose();
    rgbMat.dispose();
    resized.dispose();
    float32Mat.dispose();
    channels.dispose();

    debugPrint(
      '📊 OpenCV ImageNet Tensor shape: [1, 3, ${config.targetHeight}, ${config.targetWidth}]',
    );
    debugPrint('📊 OpenCV async processed ${floats.length} floats');

    return [
      TensorData(
        data: floats.buffer.asUint8List(),
        shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(),
        dataType: TensorType.float32,
      ),
    ];
  }
}
