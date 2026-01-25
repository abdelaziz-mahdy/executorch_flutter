import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../movenet_input_processor.dart';

/// OpenCV-accelerated MoveNet preprocessor
/// Uses native OpenCV async operations for maximum efficiency
/// ALL image manipulation is done natively - Dart only handles final byte conversion
class OpenCVMoveNetPreprocessor {
  OpenCVMoveNetPreprocessor({required this.config});

  final MoveNetPreprocessConfig config;

  Future<List<TensorData>> preprocess(Uint8List imageBytes) async {
    // Decode image from bytes using OpenCV async
    final mat = await cv.imdecodeAsync(imageBytes, cv.IMREAD_COLOR);

    // Convert BGR to RGB async
    final rgbMat = await cv.cvtColorAsync(mat, cv.COLOR_BGR2RGB);

    // Simple resize to target size (MoveNet doesn't use letterbox)
    final resized = await cv.resizeAsync(rgbMat, (
      config.targetSize,
      config.targetSize,
    ), interpolation: cv.INTER_LINEAR);

    // Convert to float32 and normalize to [0, 1] - all done natively in OpenCV
    final float32Mat = resized.convertTo(
      cv.MatType.CV_32FC3,
      alpha: 1.0 / 255.0,
    );

    // For NHWC format, we keep channels interleaved (HWC order)
    // MoveNet expects [1, targetSize, targetSize, 3] in NHWC format
    final totalPixels = config.targetSize * config.targetSize;
    final tensorSize = totalPixels * 3;
    final floats = Float32List(tensorSize);

    // Access raw float data from the Mat
    final bytes = float32Mat.data;
    final byteData = ByteData.sublistView(bytes);

    // Read floats directly - OpenCV stores in HWC format (interleaved RGB)
    // which is exactly what we need for NHWC output
    for (int i = 0; i < tensorSize; i++) {
      floats[i] = byteData.getFloat32(i * 4, Endian.host);
    }

    // Clean up OpenCV resources
    mat.dispose();
    rgbMat.dispose();
    resized.dispose();
    float32Mat.dispose();

    debugPrint(
      '📊 OpenCV MoveNet Tensor shape: [1, ${config.targetSize}, ${config.targetSize}, 3] (NHWC)',
    );
    debugPrint(
      '📊 OpenCV async processed ${floats.length} floats, range [0, 1]',
    );

    return [
      TensorData(
        data: floats.buffer.asUint8List(),
        shape: [1, config.targetSize, config.targetSize, 3].cast<int?>(),
        dataType: TensorType.float32,
        name: 'input',
      ),
    ];
  }
}
