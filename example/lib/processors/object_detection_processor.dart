import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';

import 'package:executorch_flutter/executorch_flutter.dart';

/// Configuration for object detection preprocessing
@immutable
class ObjectDetectionPreprocessConfig {
  const ObjectDetectionPreprocessConfig({
    this.targetWidth = 320,
    this.targetHeight = 320,
    this.normalizeToFloat = true,
    this.meanSubtraction = const [0.485, 0.456, 0.406],
    this.standardDeviation = const [0.229, 0.224, 0.225],
    this.cropMode = ObjectDetectionCropMode.centerCrop,
  });

  /// Target width for resizing (default: 320 for mobile object detection)
  final int targetWidth;

  /// Target height for resizing (default: 320 for mobile object detection)
  final int targetHeight;

  /// Whether to normalize pixel values to float range [0,1]
  final bool normalizeToFloat;

  /// Mean values for normalization (RGB channels)
  final List<double> meanSubtraction;

  /// Standard deviation values for normalization (RGB channels)
  final List<double> standardDeviation;

  /// How to crop/resize the image to target dimensions
  final ObjectDetectionCropMode cropMode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ObjectDetectionPreprocessConfig &&
          runtimeType == other.runtimeType &&
          targetWidth == other.targetWidth &&
          targetHeight == other.targetHeight &&
          normalizeToFloat == other.normalizeToFloat &&
          _listEquals(meanSubtraction, other.meanSubtraction) &&
          _listEquals(standardDeviation, other.standardDeviation) &&
          cropMode == other.cropMode;

  @override
  int get hashCode =>
      targetWidth.hashCode ^
      targetHeight.hashCode ^
      normalizeToFloat.hashCode ^
      meanSubtraction.hashCode ^
      standardDeviation.hashCode ^
      cropMode.hashCode;

  bool _listEquals(List<double> a, List<double> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Modes for cropping/resizing images for object detection
enum ObjectDetectionCropMode {
  /// Resize to exact dimensions (may distort aspect ratio)
  stretch,
  /// Crop from center to maintain aspect ratio
  centerCrop,
  /// Fit image within dimensions, padding with zeros
  letterbox,
}

/// Detected object with bounding box and confidence
@immutable
class DetectedObject {
  const DetectedObject({
    required this.className,
    required this.confidence,
    required this.classIndex,
    required this.boundingBox,
  });

  /// The detected object class name/label
  final String className;

  /// Confidence score for the detection (0.0 to 1.0)
  final double confidence;

  /// Index of the detected class
  final int classIndex;

  /// Bounding box coordinates (normalized 0.0 to 1.0)
  final BoundingBox boundingBox;

  @override
  String toString() =>
      'DetectedObject(class: $className, confidence: ${(confidence * 100).toStringAsFixed(1)}%, box: $boundingBox)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DetectedObject &&
          runtimeType == other.runtimeType &&
          className == other.className &&
          confidence == other.confidence &&
          classIndex == other.classIndex &&
          boundingBox == other.boundingBox;

  @override
  int get hashCode =>
      className.hashCode ^
      confidence.hashCode ^
      classIndex.hashCode ^
      boundingBox.hashCode;
}

/// Bounding box coordinates (normalized 0.0 to 1.0)
@immutable
class BoundingBox {
  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  /// Left coordinate (0.0 to 1.0)
  final double x;

  /// Top coordinate (0.0 to 1.0)
  final double y;

  /// Width (0.0 to 1.0)
  final double width;

  /// Height (0.0 to 1.0)
  final double height;

  /// Right coordinate
  double get right => x + width;

  /// Bottom coordinate
  double get bottom => y + height;

  /// Center X coordinate
  double get centerX => x + width / 2;

  /// Center Y coordinate
  double get centerY => y + height / 2;

  @override
  String toString() =>
      'BoundingBox(x: ${x.toStringAsFixed(3)}, y: ${y.toStringAsFixed(3)}, w: ${width.toStringAsFixed(3)}, h: ${height.toStringAsFixed(3)})';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is BoundingBox &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          width == other.width &&
          height == other.height;

  @override
  int get hashCode => x.hashCode ^ y.hashCode ^ width.hashCode ^ height.hashCode;
}

/// Result of object detection
@immutable
class ObjectDetectionResult {
  const ObjectDetectionResult({
    required this.detectedObjects,
    required this.processingTimeMs,
  });

  /// List of detected objects
  final List<DetectedObject> detectedObjects;

  /// Processing time in milliseconds
  final double processingTimeMs;

  @override
  String toString() =>
      'ObjectDetectionResult(objects: ${detectedObjects.length}, time: ${processingTimeMs.toStringAsFixed(1)}ms)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ObjectDetectionResult &&
          runtimeType == other.runtimeType &&
          _listEquals(detectedObjects, other.detectedObjects) &&
          processingTimeMs == other.processingTimeMs;

  @override
  int get hashCode => detectedObjects.hashCode ^ processingTimeMs.hashCode;

  bool _listEquals(List<DetectedObject> a, List<DetectedObject> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Preprocessor for object detection
class ObjectDetectionPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  ObjectDetectionPreprocessor({
    required this.config,
  });

  final ObjectDetectionPreprocessConfig config;

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

      // Resize to model input size
      final resized = _resizeImage(decodedImage);

      // Convert to RGB if needed
      final rgbImage = resized.convert(numChannels: 3);

      // Convert to tensor in NCHW format
      final tensorData = _imageToTensor(rgbImage);

      return [tensorData];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('Object detection preprocessing failed: $e', e);
    }
  }

  img.Image _resizeImage(img.Image image) {
    switch (config.cropMode) {
      case ObjectDetectionCropMode.stretch:
        return img.copyResize(
          image,
          width: config.targetWidth,
          height: config.targetHeight,
        );

      case ObjectDetectionCropMode.centerCrop:
        // Calculate crop dimensions to maintain aspect ratio
        final aspectRatio = config.targetWidth / config.targetHeight;
        final imageAspectRatio = image.width / image.height;

        late img.Image croppedImage;
        if (imageAspectRatio > aspectRatio) {
          // Image is wider, crop horizontally
          final newWidth = (image.height * aspectRatio).round();
          final cropX = (image.width - newWidth) ~/ 2;
          croppedImage = img.copyCrop(
            image,
            x: cropX,
            y: 0,
            width: newWidth,
            height: image.height,
          );
        } else {
          // Image is taller, crop vertically
          final newHeight = (image.width / aspectRatio).round();
          final cropY = (image.height - newHeight) ~/ 2;
          croppedImage = img.copyCrop(
            image,
            x: 0,
            y: cropY,
            width: image.width,
            height: newHeight,
          );
        }

        return img.copyResize(
          croppedImage,
          width: config.targetWidth,
          height: config.targetHeight,
        );

      case ObjectDetectionCropMode.letterbox:
        // Scale to fit within dimensions, then pad
        final scaleW = config.targetWidth / image.width;
        final scaleH = config.targetHeight / image.height;
        final scale = math.min(scaleW, scaleH);

        final newWidth = (image.width * scale).round();
        final newHeight = (image.height * scale).round();

        final resized = img.copyResize(image, width: newWidth, height: newHeight);

        // Create target image with padding
        final target = img.Image(
          width: config.targetWidth,
          height: config.targetHeight,
          numChannels: image.numChannels,
        );
        img.fill(target, color: img.ColorRgb8(0, 0, 0)); // Black padding

        final offsetX = (config.targetWidth - newWidth) ~/ 2;
        final offsetY = (config.targetHeight - newHeight) ~/ 2;

        img.compositeImage(target, resized, dstX: offsetX, dstY: offsetY);
        return target;
    }
  }

  TensorData _imageToTensor(img.Image image) {
    // Convert to RGB if needed - ensure we have exactly 3 channels
    final rgbImage = image.convert(numChannels: 3);

    // Create float32 tensor in NCHW format
    final floats = Float32List(1 * 3 * config.targetHeight * config.targetWidth);

    // Fill tensor in NCHW format: [batch, channel, height, width]
    int index = 0;

    // Channel 0 (Red)
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = rgbImage.getPixel(x, y);
        double normalizedValue = pixel.r.toDouble();

        // Normalize to [0, 1] if enabled
        if (config.normalizeToFloat) {
          normalizedValue /= 255.0;
        }

        // Apply mean subtraction and standard deviation for red channel
        if (config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty &&
            config.meanSubtraction.length > 0 &&
            config.standardDeviation.length > 0) {
          normalizedValue = (normalizedValue - config.meanSubtraction[0]) / config.standardDeviation[0];
        }

        floats[index++] = normalizedValue;
      }
    }

    // Channel 1 (Green)
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = rgbImage.getPixel(x, y);
        double normalizedValue = pixel.g.toDouble();

        // Normalize to [0, 1] if enabled
        if (config.normalizeToFloat) {
          normalizedValue /= 255.0;
        }

        // Apply mean subtraction and standard deviation for green channel
        if (config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty &&
            config.meanSubtraction.length > 1 &&
            config.standardDeviation.length > 1) {
          normalizedValue = (normalizedValue - config.meanSubtraction[1]) / config.standardDeviation[1];
        }

        floats[index++] = normalizedValue;
      }
    }

    // Channel 2 (Blue)
    for (int y = 0; y < config.targetHeight; y++) {
      for (int x = 0; x < config.targetWidth; x++) {
        final pixel = rgbImage.getPixel(x, y);
        double normalizedValue = pixel.b.toDouble();

        // Normalize to [0, 1] if enabled
        if (config.normalizeToFloat) {
          normalizedValue /= 255.0;
        }

        // Apply mean subtraction and standard deviation for blue channel
        if (config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty &&
            config.meanSubtraction.length > 2 &&
            config.standardDeviation.length > 2) {
          normalizedValue = (normalizedValue - config.meanSubtraction[2]) / config.standardDeviation[2];
        }

        floats[index++] = normalizedValue;
      }
    }

    // Create tensor data
    final tensorData = TensorData(
      shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(), // NCHW format
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'input',
    );

    return tensorData;
  }
}

/// Postprocessor for object detection results
class ObjectDetectionPostprocessor extends ExecuTorchPostprocessor<ObjectDetectionResult> {
  ObjectDetectionPostprocessor({
    required this.classLabels,
    this.confidenceThreshold = 0.5,
    this.nmsThreshold = 0.4,
    this.maxDetections = 10,
  });

  final List<String> classLabels;
  final double confidenceThreshold;
  final double nmsThreshold;
  final int maxDetections;

  @override
  String get outputTypeName => 'Object Detection Result';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    // Object detection models typically output multiple tensors:
    // - Bounding boxes
    // - Confidence scores
    // - Class predictions
    return outputs.isNotEmpty;
  }

  @override
  Future<ObjectDetectionResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
    try {
      final stopwatch = Stopwatch()..start();

      if (outputs.isEmpty) {
        throw PostprocessingException('No output tensors provided');
      }

      // Parse outputs based on model format
      final detections = _parseDetections(outputs);

      // Apply confidence threshold
      final filteredDetections = detections
          .where((detection) => detection.confidence >= confidenceThreshold)
          .toList();

      // Apply Non-Maximum Suppression (NMS)
      final finalDetections = _applyNMS(filteredDetections);

      // Limit number of detections
      final limitedDetections = finalDetections.take(maxDetections).toList();

      stopwatch.stop();

      return ObjectDetectionResult(
        detectedObjects: limitedDetections,
        processingTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException('Object detection postprocessing failed: $e', e);
    }
  }

  List<DetectedObject> _parseDetections(List<TensorData> outputs) {
    final detections = <DetectedObject>[];

    // Handle different output formats
    if (outputs.length == 1) {
      // Single output tensor format (e.g., YOLO-style)
      detections.addAll(_parseSingleTensorOutput(outputs[0]));
    } else if (outputs.length >= 3) {
      // Multiple output tensors (boxes, scores, classes)
      detections.addAll(_parseMultiTensorOutput(outputs));
    } else {
      // Fallback: treat as classification if format is unclear
      detections.addAll(_parseAsClassification(outputs[0]));
    }

    return detections;
  }

  List<DetectedObject> _parseSingleTensorOutput(TensorData output) {
    final detections = <DetectedObject>[];

    if (output.dataType == TensorType.float32) {
      final byteData = ByteData.sublistView(output.data);
      final values = <double>[];

      for (int i = 0; i < output.data.length ~/ 4; i++) {
        values.add(byteData.getFloat32(i * 4, Endian.host));
      }

      // Parse YOLO-style output: [batch, detections, (x, y, w, h, conf, class_scores...)]
      final shape = output.shape?.where((dim) => dim != null).map((dim) => dim!).toList() ?? [];
      if (shape.length >= 2) {
        final numDetections = shape[1];
        final numFields = shape.length > 2 ? shape[2] : values.length ~/ numDetections;

        for (int i = 0; i < numDetections; i++) {
          final startIdx = i * numFields;
          if (startIdx + 4 < values.length) {
            final x = values[startIdx];
            final y = values[startIdx + 1];
            final w = values[startIdx + 2];
            final h = values[startIdx + 3];
            final conf = startIdx + 4 < values.length ? values[startIdx + 4] : 0.0;

            // Find best class
            int bestClassIdx = 0;
            double bestClassScore = 0.0;

            for (int j = 5; j < numFields && startIdx + j < values.length; j++) {
              final classScore = values[startIdx + j];
              if (classScore > bestClassScore) {
                bestClassScore = classScore;
                bestClassIdx = j - 5;
              }
            }

            final finalConfidence = conf * bestClassScore;

            if (finalConfidence > 0.1) { // Low threshold for initial filtering
              final className = bestClassIdx < classLabels.length
                  ? classLabels[bestClassIdx]
                  : 'Unknown';

              detections.add(DetectedObject(
                className: className,
                confidence: finalConfidence,
                classIndex: bestClassIdx,
                boundingBox: BoundingBox(
                  x: x.clamp(0.0, 1.0),
                  y: y.clamp(0.0, 1.0),
                  width: w.clamp(0.0, 1.0),
                  height: h.clamp(0.0, 1.0),
                ),
              ));
            }
          }
        }
      }
    }

    return detections;
  }

  List<DetectedObject> _parseMultiTensorOutput(List<TensorData> outputs) {
    // Handle format with separate tensors for boxes, scores, classes
    final detections = <DetectedObject>[];

    if (outputs.length < 3) {
      // Not enough tensors for standard multi-output format
      return _parseAsClassification(outputs[0]);
    }

    // Common multi-tensor format: [boxes, scores, classes] or [boxes, classes, scores]
    TensorData? boxesTensor;
    TensorData? scoresTensor;
    TensorData? classesTensor;

    // Try to identify tensor types by shape
    for (final tensor in outputs) {
      final shape = tensor.shape?.where((dim) => dim != null).map((dim) => dim!).toList() ?? [];

      if (shape.length >= 2) {
        final lastDim = shape.last;

        // Boxes tensor typically has last dimension of 4 (x, y, w, h)
        if (lastDim == 4 && boxesTensor == null) {
          boxesTensor = tensor;
        }
        // Classes tensor typically has last dimension matching number of classes
        else if (lastDim == classLabels.length && classesTensor == null) {
          classesTensor = tensor;
        }
        // Scores tensor typically has last dimension of 1 or num_classes
        else if ((lastDim == 1 || lastDim == classLabels.length) && scoresTensor == null) {
          scoresTensor = tensor;
        }
      }
    }

    // Fallback: assume order [boxes, scores, classes]
    boxesTensor ??= outputs[0];
    scoresTensor ??= outputs.length > 1 ? outputs[1] : outputs[0];
    classesTensor ??= outputs.length > 2 ? outputs[2] : outputs[0];

    try {
      final boxes = _extractFloatValues(boxesTensor);
      final scores = _extractFloatValues(scoresTensor);
      final classes = _extractFloatValues(classesTensor);

      // Determine number of detections
      final boxesShape = boxesTensor.shape?.where((dim) => dim != null).map((dim) => dim!).toList() ?? [];
      final numDetections = boxesShape.length >= 2 ? boxesShape[boxesShape.length - 2] : 0;

      for (int i = 0; i < numDetections && i * 4 + 3 < boxes.length; i++) {
        // Extract box coordinates
        final x = boxes[i * 4];
        final y = boxes[i * 4 + 1];
        final w = boxes[i * 4 + 2];
        final h = boxes[i * 4 + 3];

        // Extract confidence score
        double confidence = 0.0;
        if (i < scores.length) {
          confidence = scores[i];
        }

        // Extract class
        int classIndex = 0;
        if (classesTensor == scoresTensor) {
          // Classes and scores are combined - find max score
          final startIdx = i * classLabels.length;
          for (int j = 0; j < classLabels.length && startIdx + j < scores.length; j++) {
            if (scores[startIdx + j] > confidence) {
              confidence = scores[startIdx + j];
              classIndex = j;
            }
          }
        } else if (i < classes.length) {
          classIndex = classes[i].round().clamp(0, classLabels.length - 1);
        }

        // Create detection if confidence is reasonable
        if (confidence > 0.1) {
          final className = classIndex < classLabels.length
              ? classLabels[classIndex]
              : 'Unknown';

          detections.add(DetectedObject(
            className: className,
            confidence: confidence,
            classIndex: classIndex,
            boundingBox: BoundingBox(
              x: x.clamp(0.0, 1.0),
              y: y.clamp(0.0, 1.0),
              width: w.clamp(0.0, 1.0),
              height: h.clamp(0.0, 1.0),
            ),
          ));
        }
      }
    } catch (e) {
      // If multi-tensor parsing fails, fallback to classification
      return _parseAsClassification(outputs[0]);
    }

    return detections;
  }

  List<double> _extractFloatValues(TensorData tensor) {
    final values = <double>[];

    if (tensor.dataType == TensorType.float32) {
      final byteData = ByteData.sublistView(tensor.data);
      for (int i = 0; i < tensor.data.length ~/ 4; i++) {
        values.add(byteData.getFloat32(i * 4, Endian.host));
      }
    } else if (tensor.dataType == TensorType.int32) {
      final byteData = ByteData.sublistView(tensor.data);
      for (int i = 0; i < tensor.data.length ~/ 4; i++) {
        values.add(byteData.getInt32(i * 4, Endian.host).toDouble());
      }
    } else if (tensor.dataType == TensorType.uint8) {
      for (int i = 0; i < tensor.data.length; i++) {
        values.add(tensor.data[i].toDouble());
      }
    }

    return values;
  }

  List<DetectedObject> _parseAsClassification(TensorData output) {
    // Fallback: treat as classification result
    final detections = <DetectedObject>[];

    if (output.dataType == TensorType.float32) {
      final byteData = ByteData.sublistView(output.data);
      final logits = <double>[];

      for (int i = 0; i < output.data.length ~/ 4; i++) {
        logits.add(byteData.getFloat32(i * 4, Endian.host));
      }

      // Apply softmax and find top detection
      final probabilities = _applySoftmax(logits);
      final indexed = List.generate(probabilities.length, (i) => MapEntry(i, probabilities[i]));
      indexed.sort((a, b) => b.value.compareTo(a.value));

      // Create detection for the top class covering the whole image
      if (indexed.isNotEmpty) {
        final topResult = indexed.first;
        final className = topResult.key < classLabels.length
            ? classLabels[topResult.key]
            : 'Unknown';

        detections.add(DetectedObject(
          className: className,
          confidence: topResult.value,
          classIndex: topResult.key,
          boundingBox: const BoundingBox(x: 0.0, y: 0.0, width: 1.0, height: 1.0),
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

        final iou = _calculateIoU(detections[i].boundingBox, detections[j].boundingBox);
        if (iou > nmsThreshold) {
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

  List<double> _applySoftmax(List<double> logits) {
    // Find max for numerical stability
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);

    // Compute exp(logit - max) for each logit
    final expValues = logits.map((logit) => math.exp(logit - maxLogit)).toList();

    // Compute sum of all exp values
    final sumExp = expValues.reduce((a, b) => a + b);

    // Normalize to get probabilities
    return expValues.map((exp) => exp / sumExp).toList();
  }
}

/// Complete object detection processor
class ObjectDetectionProcessor extends ExecuTorchProcessor<Uint8List, ObjectDetectionResult> {
  ObjectDetectionProcessor({
    required this.preprocessConfig,
    required this.classLabels,
    this.confidenceThreshold = 0.5,
    this.nmsThreshold = 0.4,
    this.maxDetections = 10,
  }) : _preprocessor = ObjectDetectionPreprocessor(config: preprocessConfig),
       _postprocessor = ObjectDetectionPostprocessor(
         classLabels: classLabels,
         confidenceThreshold: confidenceThreshold,
         nmsThreshold: nmsThreshold,
         maxDetections: maxDetections,
       );

  final ObjectDetectionPreprocessConfig preprocessConfig;
  final List<String> classLabels;
  final double confidenceThreshold;
  final double nmsThreshold;
  final int maxDetections;
  final ObjectDetectionPreprocessor _preprocessor;
  final ObjectDetectionPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<Uint8List> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<ObjectDetectionResult> get postprocessor => _postprocessor;
}