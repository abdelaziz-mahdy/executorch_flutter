import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/processors/yolo_processor.dart';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// OpenCV-accelerated YOLO preprocessor
/// Uses native OpenCV async operations for maximum efficiency
/// ALL image manipulation is done natively - Dart only handles final byte conversion
class OpenCVYoloPreprocessor {
  OpenCVYoloPreprocessor({required this.config});

  final YoloPreprocessConfig config;

  Future<List<TensorData>> preprocess(Uint8List imageBytes) async {
    // Decode image from bytes using OpenCV async
    final mat = await cv.imdecodeAsync(imageBytes, cv.IMREAD_COLOR);

    // Convert BGR to RGB async
    final rgbMat = await cv.cvtColorAsync(mat, cv.COLOR_BGR2RGB);

    // Letterbox resize (YOLO standard) - maintain aspect ratio with padding
    final resized = await _letterboxResize(rgbMat);

    // Convert to float32 and normalize to [0, 1] - all done natively in OpenCV
    final float32Mat = resized.convertTo(
      cv.MatType.CV_32FC3,
      alpha: 1.0 / 255.0,
    );

    // Split into separate channels for CHW format
    final channels = cv.split(float32Mat);

    // Directly access underlying data from each channel Mat
    // This is the most efficient way - OpenCV stores data contiguously
    final channelSize = config.targetHeight * config.targetWidth;
    final tensorSize = 3 * channelSize;
    final floats = Float32List(tensorSize);

    // Copy each channel's data directly (R, G, B in sequence for CHW format)
    // Mat.data returns Uint8List (raw bytes), convert to Float32List via ByteData
    for (int c = 0; c < 3; c++) {
      final channel = channels[c];
      final bytes = channel.data;
      final byteData = ByteData.sublistView(bytes);
      final offset = c * channelSize;

      // Read floats from bytes
      for (int i = 0; i < channelSize; i++) {
        floats[offset + i] = byteData.getFloat32(i * 4, Endian.host);
      }
    }

    // Clean up OpenCV resources
    mat.dispose();
    rgbMat.dispose();
    resized.dispose();
    float32Mat.dispose();
    channels.dispose();

    debugPrint(
      '📊 OpenCV YOLO Tensor shape: [1, 3, ${config.targetHeight}, ${config.targetWidth}]',
    );
    debugPrint(
      '📊 OpenCV async processed ${floats.length} floats, range [0, 1]',
    );

    return [
      TensorData(
        data: floats.buffer.asUint8List(),
        shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(),
        dataType: TensorType.float32,
      ),
    ];
  }

  /// Letterbox resize - maintains aspect ratio with gray padding (YOLO standard)
  Future<cv.Mat> _letterboxResize(cv.Mat image) async {
    // Calculate scale to fit image within target size while maintaining aspect ratio
    final scaleW = config.targetWidth / image.cols;
    final scaleH = config.targetHeight / image.rows;
    final scale = scaleW < scaleH ? scaleW : scaleH;

    // Calculate new dimensions
    final newWidth = (image.cols * scale).round();
    final newHeight = (image.rows * scale).round();

    // Resize image maintaining aspect ratio
    final resized = await cv.resizeAsync(image, (
      newWidth,
      newHeight,
    ), interpolation: cv.INTER_LINEAR);

    // Create target mat with gray padding (114, 114, 114)
    final target = cv.Mat.create(
      rows: config.targetHeight,
      cols: config.targetWidth,
      type: cv.MatType.CV_8UC3,
    );
    target.setTo(cv.Scalar(114, 114, 114, 0)); // Gray padding

    // Calculate offsets to center the resized image
    final offsetX = (config.targetWidth - newWidth) ~/ 2;
    final offsetY = (config.targetHeight - newHeight) ~/ 2;

    // Copy resized image to center of target
    final roi = target.region(cv.Rect(offsetX, offsetY, newWidth, newHeight));
    resized.copyTo(roi);

    resized.dispose();
    return target;
  }
}
