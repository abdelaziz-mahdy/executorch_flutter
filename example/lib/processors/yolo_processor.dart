import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'object_detection_processor.dart';

/// YOLO-specific preprocessing configuration
@immutable
class YoloPreprocessConfig {
  const YoloPreprocessConfig({
    this.targetWidth = 640,
    this.targetHeight = 640,
    this.stride = 32,
  });

  /// Target width for YOLO (typically 640)
  final int targetWidth;

  /// Target height for YOLO (typically 640)
  final int targetHeight;

  /// Model stride (typically 32 for YOLO)
  final int stride;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is YoloPreprocessConfig &&
          runtimeType == other.runtimeType &&
          targetWidth == other.targetWidth &&
          targetHeight == other.targetHeight &&
          stride == other.stride;

  @override
  int get hashCode => targetWidth.hashCode ^ targetHeight.hashCode ^ stride.hashCode;
}

/// YOLO-specific preprocessor
class YoloPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  YoloPreprocessor({required this.config});

  final YoloPreprocessConfig config;

  @override
  String get inputTypeName => 'Image (Uint8List)';

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
  Future<List<TensorData>> preprocess(Uint8List input, {ModelMetadata? metadata}) async {
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
    // Create float32 tensor in NCHW format normalized to [0, 1]
    final floats = Float32List(1 * 3 * config.targetHeight * config.targetWidth);

    int index = 0;

    // Channel 0 (Red) - normalized to [0, 1]
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = image.getPixel(x, y);
        floats[index++] = pixel.r / 255.0;
      }
    }

    // Channel 1 (Green) - normalized to [0, 1]
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = image.getPixel(x, y);
        floats[index++] = pixel.g / 255.0;
      }
    }

    // Channel 2 (Blue) - normalized to [0, 1]
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = image.getPixel(x, y);
        floats[index++] = pixel.b / 255.0;
      }
    }

    print('📊 YOLO Tensor shape: [1, 3, ${config.targetHeight}, ${config.targetWidth}]');
    print('📊 YOLO Tensor data size: ${floats.length} floats');

    return TensorData(
      shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(),
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'images',
    );
  }
}

/// YOLO-specific postprocessor
///
/// Supports multiple YOLO versions:
/// - YOLOv5: Output [1, 85, 8400] - 4 bbox + 1 objectness + 80 classes
/// - YOLOv8: Output [1, 84, 8400] - 4 bbox + 80 classes (objectness integrated)
/// - YOLO11: Output [1, 84, 8400] - same format as YOLOv8
/// - YOLO12: Output [1, 84, 8400] - same format as YOLOv8
///
/// The processor automatically detects the format based on tensor shape.
class YoloPostprocessor extends ExecuTorchPostprocessor<ObjectDetectionResult> {
  YoloPostprocessor({
    required this.classLabels,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.45,
    this.maxDetections = 300,
  });

  final List<String> classLabels;
  final double confidenceThreshold;
  final double iouThreshold;
  final int maxDetections;

  @override
  String get outputTypeName => 'YOLO Detection Result';

  @override
  bool validateOutputs(List<TensorData> outputs) => outputs.isNotEmpty;

  @override
  Future<ObjectDetectionResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
    try {
      final stopwatch = Stopwatch()..start();

      if (outputs.isEmpty) {
        throw PostprocessingException('No output tensors provided');
      }

      // Parse YOLO output format
      final detections = _parseYoloOutput(outputs[0]);

      // Apply confidence threshold
      final filtered = detections.where((d) => d.confidence >= confidenceThreshold).toList();

      // Apply NMS
      final nmsDetections = _applyNMS(filtered);

      // Limit detections
      final limited = nmsDetections.take(maxDetections).toList();

      stopwatch.stop();

      return ObjectDetectionResult(
        detectedObjects: limited,
        processingTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException('YOLO postprocessing failed: $e', e);
    }
  }

  List<DetectedObject> _parseYoloOutput(TensorData output) {
    final detections = <DetectedObject>[];

    if (output.dataType != TensorType.float32) {
      return detections;
    }

    final byteData = ByteData.sublistView(output.data);
    final values = <double>[];

    for (int i = 0; i < output.data.length ~/ 4; i++) {
      values.add(byteData.getFloat32(i * 4, Endian.host));
    }

    // YOLO output format options:
    // YOLOv5:           [batch, 85, num_predictions] - 4 bbox + 1 objectness + 80 classes
    // YOLOv8/v11/v12:   [batch, 84, num_predictions] - 4 bbox + 80 classes (objectness integrated)
    final shape = output.shape?.where((dim) => dim != null).map((dim) => dim!).toList() ?? [];

    if (shape.length < 2) return detections;

    // Detect format based on tensor shape
    final isTransposed = shape.length >= 3 && shape[1] > shape[2];

    int numPredictions;
    int numFields;

    if (isTransposed) {
      // Format: [batch, features, predictions] - typical for YOLOv8/v11
      numFields = shape[1];
      numPredictions = shape[2];
    } else {
      // Format: [batch, predictions, features] - alternative format
      numPredictions = shape[1];
      numFields = shape.length > 2 ? shape[2] : (values.length ~/ numPredictions);
    }

    // Detect YOLO version by feature count
    // 85 = YOLOv5 (4 bbox + 1 objectness + 80 classes)
    // 84 = YOLOv8/v11 (4 bbox + 80 classes, objectness integrated)
    final isYolov5Format = numFields == 85;
    final numClasses = isYolov5Format ? 80 : (numFields - 4);

    for (int i = 0; i < numPredictions; i++) {
      // Calculate index based on format
      final baseIdx = isTransposed ? i : (i * numFields);

      if (baseIdx >= values.length) break;

      double xCenter, yCenter, width, height, objectness;

      if (isTransposed) {
        // Transposed format: [features, predictions]
        if (i + numPredictions * 4 >= values.length) break;

        xCenter = values[i];
        yCenter = values[numPredictions + i];
        width = values[numPredictions * 2 + i];
        height = values[numPredictions * 3 + i];
        objectness = isYolov5Format ? values[numPredictions * 4 + i] : 1.0;
      } else {
        // Normal format: [predictions, features]
        if (baseIdx + 4 + numClasses > values.length) break;

        xCenter = values[baseIdx];
        yCenter = values[baseIdx + 1];
        width = values[baseIdx + 2];
        height = values[baseIdx + 3];
        objectness = isYolov5Format ? values[baseIdx + 4] : 1.0;
      }

      // Find best class and confidence
      int bestClassIdx = 0;
      double bestClassConf = 0.0;

      final classStartOffset = isYolov5Format ? 5 : 4;

      for (int j = 0; j < numClasses; j++) {
        final classIdx = isTransposed
            ? (numPredictions * (classStartOffset + j) + i)
            : (baseIdx + classStartOffset + j);

        if (classIdx >= values.length) break;

        final classConf = values[classIdx];
        if (classConf > bestClassConf) {
          bestClassConf = classConf;
          bestClassIdx = j;
        }
      }

      // Calculate final confidence
      // YOLOv5: objectness * class_confidence
      // YOLOv8/v11: class_confidence already includes objectness
      final confidence = isYolov5Format ? (objectness * bestClassConf) : bestClassConf;

      // Only add if confidence is above minimum threshold
      if (confidence > 0.1) {
        // Convert center-based to corner-based coordinates
        final x = (xCenter - width / 2).clamp(0.0, 1.0);
        final y = (yCenter - height / 2).clamp(0.0, 1.0);
        final w = width.clamp(0.0, 1.0);
        final h = height.clamp(0.0, 1.0);

        final className = bestClassIdx < classLabels.length
            ? classLabels[bestClassIdx]
            : 'Class $bestClassIdx';

        detections.add(DetectedObject(
          className: className,
          confidence: confidence,
          classIndex: bestClassIdx,
          boundingBox: BoundingBox(x: x, y: y, width: w, height: h),
        ));
      }
    }

    return detections;
  }

  List<DetectedObject> _applyNMS(List<DetectedObject> detections) {
    if (detections.length <= 1) return detections;

    // Sort by confidence (highest first)
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    final keep = <DetectedObject>[];
    final suppressed = <bool>[]..length = detections.length;
    suppressed.fillRange(0, suppressed.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;

      keep.add(detections[i]);

      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;

        // Only suppress if same class
        if (detections[i].classIndex != detections[j].classIndex) continue;

        final iou = _calculateIoU(detections[i].boundingBox, detections[j].boundingBox);
        if (iou > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return keep;
  }

  double _calculateIoU(BoundingBox boxA, BoundingBox boxB) {
    final intersectionLeft = math.max(boxA.x, boxB.x);
    final intersectionTop = math.max(boxA.y, boxB.y);
    final intersectionRight = math.min(boxA.right, boxB.right);
    final intersectionBottom = math.min(boxA.bottom, boxB.bottom);

    if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
      return 0.0;
    }

    final intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
    final boxAArea = boxA.width * boxA.height;
    final boxBArea = boxB.width * boxB.height;
    final unionArea = boxAArea + boxBArea - intersectionArea;

    return unionArea > 0 ? intersectionArea / unionArea : 0.0;
  }
}

/// Complete YOLO processor for object detection
///
/// Supports all YOLO versions (v5, v8, v11, v12) with automatic format detection.
///
/// Example usage:
/// ```dart
/// final processor = YoloProcessor(
///   preprocessConfig: YoloPreprocessConfig(
///     targetWidth: 640,
///     targetHeight: 640,
///   ),
///   classLabels: cocoLabels,
///   confidenceThreshold: 0.25,
///   iouThreshold: 0.45,
/// );
///
/// final result = await processor.process(imageBytes, model);
/// ```
class YoloProcessor extends ExecuTorchProcessor<Uint8List, ObjectDetectionResult> {
  YoloProcessor({
    required this.preprocessConfig,
    required this.classLabels,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.45,
    this.maxDetections = 300,
  })  : _preprocessor = YoloPreprocessor(config: preprocessConfig),
        _postprocessor = YoloPostprocessor(
          classLabels: classLabels,
          confidenceThreshold: confidenceThreshold,
          iouThreshold: iouThreshold,
          maxDetections: maxDetections,
        );

  final YoloPreprocessConfig preprocessConfig;
  final List<String> classLabels;
  final double confidenceThreshold;
  final double iouThreshold;
  final int maxDetections;
  final YoloPreprocessor _preprocessor;
  final YoloPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<Uint8List> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<ObjectDetectionResult> get postprocessor => _postprocessor;
}
