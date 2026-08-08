/// Web stub for OpenCVImageNetPreprocessor
/// OpenCV is not available on web platform
library;

import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../image_processor.dart';

/// Stub implementation that throws UnsupportedError on web
class OpenCVImageNetPreprocessor {
  OpenCVImageNetPreprocessor({required this.config});

  final ImagePreprocessConfig config;

  Future<List<TensorData>> preprocess(Uint8List imageBytes) {
    throw UnsupportedError(
      'OpenCV preprocessing is not available on web. '
      'Use GPU or ImageLib preprocessing instead.',
    );
  }
}
