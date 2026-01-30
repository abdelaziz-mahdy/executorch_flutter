import 'dart:math' as math;

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';

import '../models/face_result.dart';
import 'yolo_processor.dart' show BoundingBox;
import 'base_processor.dart';

/// Output processor for YOLO-Face model
/// Parses YOLO output format with 5 landmarks: [1, 20, 8400]
/// 20 = 4 (bbox) + 1 (conf) + 1 (class) + 10 (5 landmarks * 2) + 4 (landmark conf optional)
class YoloFaceOutputProcessor extends OutputProcessor<FaceDetectionResult> {
  YoloFaceOutputProcessor({
    this.confidenceThreshold = 0.5,
    this.iouThreshold = 0.45,
    this.multiFaceMode = true,
    this.inputWidth = 640,
    this.inputHeight = 640,
    this.maxDetections = 100,
  });

  final double confidenceThreshold;
  final double iouThreshold;
  final bool multiFaceMode;
  final int inputWidth;
  final int inputHeight;
  final int maxDetections;

  @override
  Future<FaceDetectionResult> process(List<TensorData> tensorOutputs) async {
    if (tensorOutputs.isEmpty) {
      return const FaceDetectionResult(faces: []);
    }

    final output = tensorOutputs.first;

    if (output.dataType != TensorType.float32) {
      debugPrint('⚠️ Unexpected output type: ${output.dataType}');
      return const FaceDetectionResult(faces: []);
    }

    final byteData = ByteData.sublistView(output.data);
    final floatCount = output.data.length ~/ 4;
    final floatData = Float32List(floatCount);

    for (int i = 0; i < floatCount; i++) {
      floatData[i] = byteData.getFloat32(i * 4, Endian.host);
    }

    final shape = output.shape
        .where((dim) => dim != null)
        .map((dim) => dim!)
        .toList();

    debugPrint('🔍 YOLO-Face output shape: $shape');

    // Validate shape has at least 2 dimensions (excluding batch)
    if (shape.length < 2) {
      debugPrint('❌ Unexpected shape: $shape, expected at least [batch, features, predictions]');
      return const FaceDetectionResult(faces: []);
    }

    // Handle both 2D and 3D shapes
    // 3D: [1, 20, 8400] or [1, 8400, 20]
    // 2D: [20, 8400] or [8400, 20]
    final bool isTransposed;
    if (shape.length >= 3) {
      // Detect transposed format: [1, 20, 8400] vs [1, 8400, 20]
      isTransposed = shape[1] < shape[2];
    } else if (shape.length == 2) {
      // 2D shape: [20, 8400] vs [8400, 20]
      isTransposed = shape[0] < shape[1];
    } else {
      debugPrint('❌ Cannot determine format for shape: $shape');
      return const FaceDetectionResult(faces: []);
    }

    final faces = isTransposed
        ? _parseTransposedOutput(floatData, shape)
        : _parseNormalOutput(floatData, shape);

    // Apply NMS
    final nmsResults = _applyNMS(faces);

    // Sort by confidence
    nmsResults.sort((a, b) => b.confidence.compareTo(a.confidence));

    // Limit results
    final limited = nmsResults.take(maxDetections).toList();

    if (!multiFaceMode && limited.isNotEmpty) {
      return FaceDetectionResult(faces: [limited.first]);
    }

    return FaceDetectionResult(faces: limited);
  }

  /// Parse transposed format [1, N, 8400] or [N, 8400] where N is features
  List<DetectedFace> _parseTransposedOutput(
      Float32List floatData, List<int> shape) {
    // Handle both 3D [1, N, 8400] and 2D [N, 8400] shapes
    final numFeatures = shape.length >= 3 ? shape[1] : shape[0];
    final numPredictions = shape.length >= 3 ? shape[2] : shape[1];
    final faces = <DetectedFace>[];

    // Validate float data size
    final expectedSize = numFeatures * numPredictions;
    if (floatData.length < expectedSize) {
      debugPrint('❌ Float data too small: ${floatData.length} < $expectedSize');
      return faces;
    }

    debugPrint('📊 Parsing transposed face: features=$numFeatures, predictions=$numPredictions');

    // Detect format based on number of features
    // Common formats:
    // - 15: 4 bbox + 1 conf + 10 landmarks (5*2)
    // - 16: 4 bbox + 1 conf + 1 class + 10 landmarks
    // - 20: 4 bbox + 1 conf + 1 class + 10 landmarks + 4 (visibility)
    final hasClass = numFeatures >= 16;
    final landmarkOffset = hasClass ? 6 : 5;

    for (int i = 0; i < numPredictions; i++) {
      final cx = floatData[0 * numPredictions + i];
      final cy = floatData[1 * numPredictions + i];
      final w = floatData[2 * numPredictions + i];
      final h = floatData[3 * numPredictions + i];
      final conf = floatData[4 * numPredictions + i];

      if (conf >= confidenceThreshold) {
        final box = BoundingBox(
          x: ((cx - w / 2) / inputWidth).clamp(0.0, 1.0),
          y: ((cy - h / 2) / inputHeight).clamp(0.0, 1.0),
          width: (w / inputWidth).clamp(0.0, 1.0),
          height: (h / inputHeight).clamp(0.0, 1.0),
        );

        // Parse 5 landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        final landmarks = <FaceLandmark>[];
        for (int k = 0; k < 5; k++) {
          final lmX = floatData[(landmarkOffset + k * 2) * numPredictions + i] /
              inputWidth;
          final lmY =
              floatData[(landmarkOffset + k * 2 + 1) * numPredictions + i] /
                  inputHeight;

          landmarks.add(FaceLandmark.fromYoloFace(
            type: YoloFaceLandmarkType.values[k],
            x: lmX.clamp(0.0, 1.0),
            y: lmY.clamp(0.0, 1.0),
          ));
        }

        faces.add(DetectedFace(
          boundingBox: box,
          landmarks: landmarks,
          confidence: conf,
        ));
      }
    }

    return faces;
  }

  /// Parse normal format [1, 8400, N] or [8400, N]
  List<DetectedFace> _parseNormalOutput(
      Float32List floatData, List<int> shape) {
    // Handle both 3D [1, 8400, N] and 2D [8400, N] shapes
    final numPredictions = shape.length >= 3 ? shape[1] : shape[0];
    final numFeatures = shape.length >= 3 ? shape[2] : shape[1];
    final faces = <DetectedFace>[];

    // Validate float data size
    final expectedSize = numPredictions * numFeatures;
    if (floatData.length < expectedSize) {
      debugPrint('❌ Float data too small: ${floatData.length} < $expectedSize');
      return faces;
    }

    debugPrint('📊 Parsing normal face: predictions=$numPredictions, features=$numFeatures');

    final hasClass = numFeatures >= 16;
    final landmarkOffset = hasClass ? 6 : 5;

    for (int i = 0; i < numPredictions; i++) {
      final offset = i * numFeatures;

      final cx = floatData[offset + 0];
      final cy = floatData[offset + 1];
      final w = floatData[offset + 2];
      final h = floatData[offset + 3];
      final conf = floatData[offset + 4];

      if (conf >= confidenceThreshold) {
        final box = BoundingBox(
          x: ((cx - w / 2) / inputWidth).clamp(0.0, 1.0),
          y: ((cy - h / 2) / inputHeight).clamp(0.0, 1.0),
          width: (w / inputWidth).clamp(0.0, 1.0),
          height: (h / inputHeight).clamp(0.0, 1.0),
        );

        final landmarks = <FaceLandmark>[];
        for (int k = 0; k < 5; k++) {
          final lmOffset = offset + landmarkOffset + k * 2;
          final lmX = floatData[lmOffset] / inputWidth;
          final lmY = floatData[lmOffset + 1] / inputHeight;

          landmarks.add(FaceLandmark.fromYoloFace(
            type: YoloFaceLandmarkType.values[k],
            x: lmX.clamp(0.0, 1.0),
            y: lmY.clamp(0.0, 1.0),
          ));
        }

        faces.add(DetectedFace(
          boundingBox: box,
          landmarks: landmarks,
          confidence: conf,
        ));
      }
    }

    return faces;
  }

  List<DetectedFace> _applyNMS(List<DetectedFace> faces) {
    if (faces.length <= 1) return faces;

    faces.sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <DetectedFace>[];
    final active = List<bool>.filled(faces.length, true);

    for (int i = 0; i < faces.length; i++) {
      if (!active[i]) continue;

      selected.add(faces[i]);

      for (int j = i + 1; j < faces.length; j++) {
        if (!active[j]) continue;

        if (_calculateIoU(faces[i].boundingBox, faces[j].boundingBox) >
            iouThreshold) {
          active[j] = false;
        }
      }
    }

    return selected;
  }

  double _calculateIoU(BoundingBox a, BoundingBox b) {
    final areaA = a.width * a.height;
    if (areaA <= 0.0) return 0.0;

    final areaB = b.width * b.height;
    if (areaB <= 0.0) return 0.0;

    final intersectLeft = math.max(a.x, b.x);
    final intersectTop = math.max(a.y, b.y);
    final intersectRight = math.min(a.x + a.width, b.x + b.width);
    final intersectBottom = math.min(a.y + a.height, b.y + b.height);

    final intersectArea = math.max(0.0, intersectRight - intersectLeft) *
        math.max(0.0, intersectBottom - intersectTop);

    return intersectArea / (areaA + areaB - intersectArea);
  }
}
