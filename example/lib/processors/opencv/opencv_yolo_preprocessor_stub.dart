/// Web stub for OpenCVYoloPreprocessor
/// OpenCV is not available on web platform
library;

import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../yolo_processor.dart';

/// Stub implementation that throws UnsupportedError on web
class OpenCVYoloPreprocessor {
  OpenCVYoloPreprocessor({required this.config});

  final YoloPreprocessConfig config;

  Future<List<TensorData>> preprocess(Uint8List imageBytes) {
    throw UnsupportedError(
      'OpenCV preprocessing is not available on web. '
      'Use GPU or ImageLib preprocessing instead.',
    );
  }
}
