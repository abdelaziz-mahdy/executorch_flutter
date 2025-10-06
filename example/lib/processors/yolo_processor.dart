import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

/// Bounding box coordinates
@immutable
class BoundingBox {
  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  final double x;
  final double y;
  final double width;
  final double height;

  /// Right edge coordinate
  double get right => x + width;

  /// Bottom edge coordinate
  double get bottom => y + height;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is BoundingBox &&
          x == other.x &&
          y == other.y &&
          width == other.width &&
          height == other.height;

  @override
  int get hashCode => x.hashCode ^ y.hashCode ^ width.hashCode ^ height.hashCode;
}

/// Detected object with bounding box
@immutable
class DetectedObject {
  const DetectedObject({
    required this.className,
    required this.confidence,
    required this.boundingBox,
    this.classIndex,
  });

  final String className;
  final double confidence;
  final BoundingBox boundingBox;
  final int? classIndex;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DetectedObject &&
          className == other.className &&
          confidence == other.confidence &&
          boundingBox == other.boundingBox &&
          classIndex == other.classIndex;

  @override
  int get hashCode =>
      className.hashCode ^
      confidence.hashCode ^
      boundingBox.hashCode ^
      classIndex.hashCode;
}

/// Object detection result
@immutable
class ObjectDetectionResult {
  const ObjectDetectionResult({
    required this.detectedObjects,
    required this.inferenceTimeMs,
    this.preprocessingTimeMs,
    this.postprocessingTimeMs,
  });

  final List<DetectedObject> detectedObjects;
  final double inferenceTimeMs;
  final double? preprocessingTimeMs;
  final double? postprocessingTimeMs;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ObjectDetectionResult &&
          detectedObjects == other.detectedObjects &&
          inferenceTimeMs == other.inferenceTimeMs;

  @override
  int get hashCode => detectedObjects.hashCode ^ inferenceTimeMs.hashCode;
}

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
    final floats = Float32List(1 * 3 * config.targetHeight * config.targetWidth);

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

    print('📊 YOLO Tensor shape: [1, 3, ${config.targetHeight}, ${config.targetWidth}]');
    print('📊 YOLO Tensor data size: ${floats.length} floats, range [0, 1]');

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
    this.inputWidth = 640,
    this.inputHeight = 640,
  });

  final List<String> classLabels;
  final double confidenceThreshold;
  final double iouThreshold;
  final int maxDetections;
  final int inputWidth;
  final int inputHeight;

  @override
  String get outputTypeName => 'YOLO Detection Result';

  @override
  bool validateOutputs(List<TensorData> outputs) => outputs.isNotEmpty;

  @override
  Future<ObjectDetectionResult> postprocess(List<TensorData> outputs) async {
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
        inferenceTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException('YOLO postprocessing failed: $e', e);
    }
  }

  List<DetectedObject> _parseYoloOutput(TensorData output) {
    if (output.dataType != TensorType.float32) {
      return [];
    }

    final byteData = ByteData.sublistView(output.data);
    final floatCount = output.data.length ~/ 4;
    final outputs = Float32List(floatCount);

    for (int i = 0; i < floatCount; i++) {
      outputs[i] = byteData.getFloat32(i * 4, Endian.host);
    }

    final shape = output.shape?.where((dim) => dim != null).map((dim) => dim!).toList() ?? [];
    if (shape.length < 2) return [];

    // Detect format:
    // Transposed: [batch, features, predictions] = [1, 84, 8400] - features is SMALL
    // Normal: [batch, predictions, features] = [1, 8400, 84] - predictions is LARGE
    final isTransposed = shape.length >= 3 && shape[1] < shape[2];

    int outputColumn; // number of features (84 or 85)
    int outputRow;    // number of predictions (8400, 25200, etc)

    if (isTransposed) {
      outputColumn = shape[1]; // features
      outputRow = shape[2];    // predictions
    } else {
      outputRow = shape.length >= 2 ? shape[1] : 0;  // predictions
      outputColumn = shape.length >= 3 ? shape[2] : (outputs.length ~/ outputRow); // features
    }

    print('🔍 Shape: $shape, isTransposed: $isTransposed, outputRow: $outputRow, outputColumn: $outputColumn');

    // Detect YOLO version: 85 = YOLOv5 (with objectness), 84 = YOLOv8+ (without)
    final isYolov5 = outputColumn == 85 || outputColumn == (classLabels.length + 5);
    final numClasses = isYolov5 ? (outputColumn - 5) : (outputColumn - 4);

    return isTransposed
        ? _parseYoloV8Transposed(outputs, outputRow, outputColumn, numClasses)
        : (isYolov5
            ? _parseYoloV5(outputs, outputRow, outputColumn, numClasses)
            : _parseYoloV8(outputs, outputRow, outputColumn, numClasses));
  }

  /// Parse YOLOv5 format: [predictions, features] where features = 4 bbox + 1 objectness + N classes
  List<DetectedObject> _parseYoloV5(Float32List outputs, int outputRow, int outputColumn, int numClasses) {
    final detections = <DetectedObject>[];
    print('📦 YOLOv5 parsing: outputRow=$outputRow, outputColumn=$outputColumn, numClasses=$numClasses');

    for (int i = 0; i < outputRow; i++) {
      final objectness = outputs[i * outputColumn + 4];

      if (objectness > confidenceThreshold) {
        final x = outputs[i * outputColumn];
        final y = outputs[i * outputColumn + 1];
        final w = outputs[i * outputColumn + 2];
        final h = outputs[i * outputColumn + 3];

        // Find best class
        double maxClassConf = outputs[i * outputColumn + 5];
        int classIdx = 0;

        for (int j = 0; j < numClasses; j++) {
          final classConf = outputs[i * outputColumn + 5 + j];
          if (classConf > maxClassConf) {
            maxClassConf = classConf;
            classIdx = j;
          }
        }

        final confidence = objectness * maxClassConf;

        if (confidence > confidenceThreshold) {
          if (detections.length < 3) {
            print('📍 YOLOv5 raw coords: x=$x, y=$y, w=$w, h=$h, conf=$confidence');
          }
          detections.add(_createDetection(x, y, w, h, confidence, classIdx));
        }
      }
    }

    return detections;
  }

  /// Parse YOLOv8+ format: [predictions, features] where features = 4 bbox + N classes (no objectness)
  List<DetectedObject> _parseYoloV8(Float32List outputs, int outputRow, int outputColumn, int numClasses) {
    final detections = <DetectedObject>[];
    print('📦 YOLOv8 parsing: outputRow=$outputRow, outputColumn=$outputColumn, numClasses=$numClasses');

    for (int i = 0; i < outputRow; i++) {
      final x = outputs[i * outputColumn];
      final y = outputs[i * outputColumn + 1];
      final w = outputs[i * outputColumn + 2];
      final h = outputs[i * outputColumn + 3];

      // Find best class
      double maxClassConf = outputs[i * outputColumn + 4];
      int classIdx = 0;

      for (int j = 0; j < numClasses; j++) {
        final classConf = outputs[i * outputColumn + 4 + j];
        if (classConf > maxClassConf) {
          maxClassConf = classConf;
          classIdx = j;
        }
      }

      if (maxClassConf > confidenceThreshold) {
        if (detections.length < 3) {
          print('📍 YOLOv8 raw coords: x=$x, y=$y, w=$w, h=$h, conf=$maxClassConf');
        }
        detections.add(_createDetection(x, y, w, h, maxClassConf, classIdx));
      }
    }

    return detections;
  }

  /// Parse YOLOv8+ transposed format: [features, predictions]
  List<DetectedObject> _parseYoloV8Transposed(Float32List outputs, int outputRow, int outputColumn, int numClasses) {
    final detections = <DetectedObject>[];
    print('📦 YOLOv8 Transposed parsing: outputRow=$outputRow, outputColumn=$outputColumn, numClasses=$numClasses');

    for (int i = 0; i < outputRow; i++) {
      final x = outputs[i];
      final y = outputs[outputRow + i];
      final w = outputs[2 * outputRow + i];
      final h = outputs[3 * outputRow + i];

      // Find best class
      double maxClassConf = outputs[4 * outputRow + i];
      int classIdx = 0;

      for (int j = 4; j < outputColumn; j++) {
        final classConf = outputs[j * outputRow + i];
        if (classConf > maxClassConf) {
          maxClassConf = classConf;
          classIdx = j - 4;
        }
      }

      if (maxClassConf > confidenceThreshold) {
        if (detections.length < 3) {
          print('📍 YOLOv8T raw coords: x=$x, y=$y, w=$w, h=$h, conf=$maxClassConf');
        }
        detections.add(_createDetection(x, y, w, h, maxClassConf, classIdx));
      }
    }

    return detections;
  }

  DetectedObject _createDetection(double xCenter, double yCenter, double w, double h, double confidence, int classIdx) {
    // YOLO models output coordinates in pixel space relative to input size
    // Convert center coords to corner coords and normalize to [0, 1]
    // This matches the working Java implementation exactly
    final left = (xCenter - w / 2) / inputWidth;
    final top = (yCenter - h / 2) / inputHeight;
    final width = w / inputWidth;
    final height = h / inputHeight;

    print('🔧 Raw: xC=$xCenter, yC=$yCenter, w=$w, h=$h | Input: ${inputWidth}x${inputHeight}');
    print('🔧 Normalized: left=$left, top=$top, width=$width, height=$height');

    final className = classIdx < classLabels.length
        ? classLabels[classIdx]
        : 'Class $classIdx';

    return DetectedObject(
      className: className,
      confidence: confidence,
      classIndex: classIdx,
      boundingBox: BoundingBox(
        x: left.clamp(0.0, 1.0),
        y: top.clamp(0.0, 1.0),
        width: width.clamp(0.0, 1.0),
        height: height.clamp(0.0, 1.0),
      ),
    );
  }

  List<DetectedObject> _applyNMS(List<DetectedObject> detections) {
    if (detections.length <= 1) return detections;

    // Sort by confidence (highest first)
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <DetectedObject>[];
    final active = List<bool>.filled(detections.length, true);
    int numActive = active.length;

    for (int i = 0; i < detections.length; i++) {
      if (!active[i]) continue;

      final boxA = detections[i];
      selected.add(boxA);

      if (selected.length >= maxDetections) break;

      for (int j = i + 1; j < detections.length; j++) {
        if (!active[j]) continue;

        final boxB = detections[j];
        if (_calculateIoU(boxA.boundingBox, boxB.boundingBox) > iouThreshold) {
          active[j] = false;
          numActive--;
          if (numActive <= 0) break;
        }
      }
    }

    return selected;
  }

  double _calculateIoU(BoundingBox a, BoundingBox b) {
    final areaA = (a.right - a.x) * (a.bottom - a.y);
    if (areaA <= 0.0) return 0.0;

    final areaB = (b.right - b.x) * (b.bottom - b.y);
    if (areaB <= 0.0) return 0.0;

    final intersectLeft = math.max(a.x, b.x);
    final intersectTop = math.max(a.y, b.y);
    final intersectRight = math.min(a.right, b.right);
    final intersectBottom = math.min(a.bottom, b.bottom);

    final intersectArea = math.max(0.0, intersectRight - intersectLeft) *
                          math.max(0.0, intersectBottom - intersectTop);

    return intersectArea / (areaA + areaB - intersectArea);
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
          inputWidth: preprocessConfig.targetWidth,
          inputHeight: preprocessConfig.targetHeight,
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
