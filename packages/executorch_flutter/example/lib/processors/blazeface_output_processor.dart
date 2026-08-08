import 'dart:math' as math;

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';

import '../models/face_result.dart';
import '../processors/yolo_processor.dart' show BoundingBox;
import 'base_processor.dart';

/// Output processor for BlazeFace face detection
/// Parses the model output tensors to extract face bounding boxes and landmarks
class BlazeFaceOutputProcessor extends OutputProcessor<FaceDetectionResult> {
  BlazeFaceOutputProcessor({
    this.confidenceThreshold = 0.5,
    this.multiFaceMode = true,
    this.iouThreshold = 0.3,
  });

  final double confidenceThreshold;
  final bool multiFaceMode;
  final double iouThreshold;

  @override
  Future<FaceDetectionResult> process(List<TensorData> tensorOutputs) async {
    if (tensorOutputs.isEmpty) {
      return const FaceDetectionResult(faces: []);
    }

    // BlazeFace typically outputs two tensors:
    // 1. Regressors: [1, 896, 16] - bounding box + landmarks
    // 2. Classificators: [1, 896, 1] - confidence scores
    // Or combined: [1, 896, 17]

    debugPrint('🔍 BlazeFace outputs: ${tensorOutputs.length} tensors');
    for (int i = 0; i < tensorOutputs.length; i++) {
      debugPrint('  Tensor $i shape: ${tensorOutputs[i].shape}');
    }

    final faces = <DetectedFace>[];

    if (tensorOutputs.length == 1) {
      // Combined output format
      faces.addAll(_parseCombinedOutput(tensorOutputs[0]));
    } else if (tensorOutputs.length >= 2) {
      // Separate regressors and classificators
      faces.addAll(_parseSeparateOutputs(tensorOutputs[0], tensorOutputs[1]));
    }

    // Apply NMS
    final nmsResults = _applyNMS(faces);

    // Sort by confidence
    nmsResults.sort((a, b) => b.confidence.compareTo(a.confidence));

    // Limit to single face if not in multi-face mode
    if (!multiFaceMode && nmsResults.isNotEmpty) {
      return FaceDetectionResult(faces: [nmsResults.first]);
    }

    return FaceDetectionResult(faces: nmsResults);
  }

  List<DetectedFace> _parseCombinedOutput(TensorData output) {
    if (output.dataType != TensorType.float32) {
      return [];
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

    if (shape.length < 2) return [];

    final numAnchors = shape.length >= 2 ? shape[1] : floatCount ~/ 17;
    final stride = shape.length >= 3 ? shape[2] : 17;

    final faces = <DetectedFace>[];

    for (int i = 0; i < numAnchors; i++) {
      final offset = i * stride;

      // Last value is confidence
      final confidence = _sigmoid(floatData[offset + stride - 1]);

      if (confidence >= confidenceThreshold) {
        // First 4 values are bounding box: [ymin, xmin, ymax, xmax] or [cx, cy, w, h]
        final x1 = floatData[offset + 0];
        final y1 = floatData[offset + 1];
        final x2 = floatData[offset + 2];
        final y2 = floatData[offset + 3];

        // Normalize and convert to [x, y, w, h] format
        final box = BoundingBox(
          x: math.min(x1, x2).clamp(0.0, 1.0),
          y: math.min(y1, y2).clamp(0.0, 1.0),
          width: (x2 - x1).abs().clamp(0.0, 1.0),
          height: (y2 - y1).abs().clamp(0.0, 1.0),
        );

        // Extract 6 landmarks (each has x, y)
        final landmarks = <FaceLandmark>[];
        for (int j = 0; j < 6; j++) {
          final lmOffset = offset + 4 + j * 2;
          if (lmOffset + 1 < floatData.length) {
            final lmX = floatData[lmOffset].clamp(0.0, 1.0);
            final lmY = floatData[lmOffset + 1].clamp(0.0, 1.0);

            landmarks.add(
              FaceLandmark.fromBlazeFace(
                type: BlazeFaceLandmarkType.values[j],
                x: lmX,
                y: lmY,
              ),
            );
          }
        }

        faces.add(
          DetectedFace(
            boundingBox: box,
            landmarks: landmarks,
            confidence: confidence,
          ),
        );
      }
    }

    return faces;
  }

  List<DetectedFace> _parseSeparateOutputs(
    TensorData regressors,
    TensorData classificators,
  ) {
    if (regressors.dataType != TensorType.float32 ||
        classificators.dataType != TensorType.float32) {
      return [];
    }

    final regByteData = ByteData.sublistView(regressors.data);
    final clsFloatCount = classificators.data.length ~/ 4;
    final clsByteData = ByteData.sublistView(classificators.data);

    final regFloatCount = regressors.data.length ~/ 4;
    final regFloatData = Float32List(regFloatCount);
    final clsFloatData = Float32List(clsFloatCount);

    for (int i = 0; i < regFloatCount; i++) {
      regFloatData[i] = regByteData.getFloat32(i * 4, Endian.host);
    }
    for (int i = 0; i < clsFloatCount; i++) {
      clsFloatData[i] = clsByteData.getFloat32(i * 4, Endian.host);
    }

    final regShape = regressors.shape
        .where((dim) => dim != null)
        .map((dim) => dim!)
        .toList();

    final numAnchors = regShape.length >= 2 ? regShape[1] : regFloatCount ~/ 16;
    final regStride = regShape.length >= 3 ? regShape[2] : 16;

    final faces = <DetectedFace>[];

    for (int i = 0; i < numAnchors; i++) {
      final confidence = _sigmoid(clsFloatData[i]);

      if (confidence >= confidenceThreshold) {
        final regOffset = i * regStride;

        // Bounding box
        final x1 = regFloatData[regOffset + 0];
        final y1 = regFloatData[regOffset + 1];
        final x2 = regFloatData[regOffset + 2];
        final y2 = regFloatData[regOffset + 3];

        final box = BoundingBox(
          x: math.min(x1, x2).clamp(0.0, 1.0),
          y: math.min(y1, y2).clamp(0.0, 1.0),
          width: (x2 - x1).abs().clamp(0.0, 1.0),
          height: (y2 - y1).abs().clamp(0.0, 1.0),
        );

        // 6 landmarks
        final landmarks = <FaceLandmark>[];
        for (int j = 0; j < 6; j++) {
          final lmOffset = regOffset + 4 + j * 2;
          if (lmOffset + 1 < regFloatData.length) {
            landmarks.add(
              FaceLandmark.fromBlazeFace(
                type: BlazeFaceLandmarkType.values[j],
                x: regFloatData[lmOffset].clamp(0.0, 1.0),
                y: regFloatData[lmOffset + 1].clamp(0.0, 1.0),
              ),
            );
          }
        }

        faces.add(
          DetectedFace(
            boundingBox: box,
            landmarks: landmarks,
            confidence: confidence,
          ),
        );
      }
    }

    return faces;
  }

  double _sigmoid(double x) {
    return 1.0 / (1.0 + math.exp(-x));
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

    final intersectArea =
        math.max(0.0, intersectRight - intersectLeft) *
        math.max(0.0, intersectBottom - intersectTop);

    return intersectArea / (areaA + areaB - intersectArea);
  }
}
