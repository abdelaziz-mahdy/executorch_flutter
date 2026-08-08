import 'dart:math' as math;

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';

import '../models/pose_result.dart';
import 'yolo_processor.dart' show BoundingBox;
import 'base_processor.dart';

/// Output processor for YOLO-Pose model
/// Parses YOLO output format with keypoints: [1, 56, 8400]
/// 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints * 3)
class YoloPoseOutputProcessor extends OutputProcessor<PoseDetectionResult> {
  YoloPoseOutputProcessor({
    this.confidenceThreshold = 0.3,
    this.iouThreshold = 0.45,
    this.multiPersonMode = true,
    this.inputWidth = 640,
    this.inputHeight = 640,
    this.maxDetections = 100,
  });

  final double confidenceThreshold;
  final double iouThreshold;
  final bool multiPersonMode;
  final int inputWidth;
  final int inputHeight;
  final int maxDetections;

  @override
  Future<PoseDetectionResult> process(List<TensorData> tensorOutputs) async {
    if (tensorOutputs.isEmpty) {
      return const PoseDetectionResult(poses: []);
    }

    final output = tensorOutputs.first;

    if (output.dataType != TensorType.float32) {
      debugPrint('⚠️ Unexpected output type: ${output.dataType}');
      return const PoseDetectionResult(poses: []);
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

    debugPrint('🔍 YOLO-Pose output shape: $shape');

    // Validate shape has at least 2 dimensions (excluding batch)
    if (shape.length < 2) {
      debugPrint(
        '❌ Unexpected shape: $shape, expected at least [batch, features, predictions]',
      );
      return const PoseDetectionResult(poses: []);
    }

    // Handle both 2D and 3D shapes
    // 3D: [1, 56, 8400] or [1, 8400, 56]
    // 2D: [56, 8400] or [8400, 56]
    final bool isTransposed;
    if (shape.length >= 3) {
      // Detect transposed format: [1, 56, 8400] vs [1, 8400, 56]
      isTransposed = shape[1] < shape[2];
    } else if (shape.length == 2) {
      // 2D shape: [56, 8400] vs [8400, 56]
      isTransposed = shape[0] < shape[1];
    } else {
      debugPrint('❌ Cannot determine format for shape: $shape');
      return const PoseDetectionResult(poses: []);
    }

    final poses = isTransposed
        ? _parseTransposedOutput(floatData, shape)
        : _parseNormalOutput(floatData, shape);

    // Apply NMS
    final nmsResults = _applyNMS(poses);

    // Sort by confidence
    nmsResults.sort((a, b) => b.confidence.compareTo(a.confidence));

    // Limit results
    final limited = nmsResults.take(maxDetections).toList();

    if (!multiPersonMode && limited.isNotEmpty) {
      return PoseDetectionResult(poses: [limited.first]);
    }

    return PoseDetectionResult(poses: limited);
  }

  /// Parse transposed format [1, 56, 8400] or [56, 8400]
  List<DetectedPose> _parseTransposedOutput(
    Float32List floatData,
    List<int> shape,
  ) {
    // Handle both 3D [1, 56, 8400] and 2D [56, 8400] shapes
    final numFeatures = shape.length >= 3 ? shape[1] : shape[0]; // 56
    final numPredictions = shape.length >= 3 ? shape[2] : shape[1]; // 8400
    final poses = <DetectedPose>[];

    // Validate float data size
    final expectedSize = numFeatures * numPredictions;
    if (floatData.length < expectedSize) {
      debugPrint('❌ Float data too small: ${floatData.length} < $expectedSize');
      return poses;
    }

    debugPrint(
      '📊 Parsing transposed pose: features=$numFeatures, predictions=$numPredictions',
    );

    for (int i = 0; i < numPredictions; i++) {
      // bbox: x, y, w, h
      final cx = floatData[0 * numPredictions + i];
      final cy = floatData[1 * numPredictions + i];
      final w = floatData[2 * numPredictions + i];
      final h = floatData[3 * numPredictions + i];

      // confidence
      final conf = floatData[4 * numPredictions + i];

      if (conf >= confidenceThreshold) {
        // Convert center coords to normalized corner coords
        final box = BoundingBox(
          x: ((cx - w / 2) / inputWidth).clamp(0.0, 1.0),
          y: ((cy - h / 2) / inputHeight).clamp(0.0, 1.0),
          width: (w / inputWidth).clamp(0.0, 1.0),
          height: (h / inputHeight).clamp(0.0, 1.0),
        );

        // Parse 17 keypoints (each has x, y, confidence)
        final keypoints = <PoseKeypoint>[];
        for (int k = 0; k < 17; k++) {
          final kpOffset = 5 + k * 3;
          final kpX = floatData[kpOffset * numPredictions + i] / inputWidth;
          final kpY =
              floatData[(kpOffset + 1) * numPredictions + i] / inputHeight;
          final kpConf = floatData[(kpOffset + 2) * numPredictions + i];

          keypoints.add(
            PoseKeypoint(
              type: PoseKeypointType.values[k],
              x: kpX.clamp(0.0, 1.0),
              y: kpY.clamp(0.0, 1.0),
              confidence: kpConf.clamp(0.0, 1.0),
            ),
          );
        }

        poses.add(
          DetectedPose(
            keypoints: keypoints,
            confidence: conf,
            boundingBox: box,
          ),
        );
      }
    }

    return poses;
  }

  /// Parse normal format [1, 8400, 56] or [8400, 56]
  List<DetectedPose> _parseNormalOutput(
    Float32List floatData,
    List<int> shape,
  ) {
    // Handle both 3D [1, 8400, 56] and 2D [8400, 56] shapes
    final numPredictions = shape.length >= 3 ? shape[1] : shape[0]; // 8400
    final stride = shape.length >= 3 ? shape[2] : shape[1]; // 56
    final poses = <DetectedPose>[];

    // Validate float data size
    final expectedSize = numPredictions * stride;
    if (floatData.length < expectedSize) {
      debugPrint('❌ Float data too small: ${floatData.length} < $expectedSize');
      return poses;
    }

    debugPrint(
      '📊 Parsing normal pose: predictions=$numPredictions, stride=$stride',
    );

    for (int i = 0; i < numPredictions; i++) {
      final offset = i * stride;

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

        final keypoints = <PoseKeypoint>[];
        for (int k = 0; k < 17; k++) {
          final kpOffset = offset + 5 + k * 3;
          final kpX = floatData[kpOffset] / inputWidth;
          final kpY = floatData[kpOffset + 1] / inputHeight;
          final kpConf = floatData[kpOffset + 2];

          keypoints.add(
            PoseKeypoint(
              type: PoseKeypointType.values[k],
              x: kpX.clamp(0.0, 1.0),
              y: kpY.clamp(0.0, 1.0),
              confidence: kpConf.clamp(0.0, 1.0),
            ),
          );
        }

        poses.add(
          DetectedPose(
            keypoints: keypoints,
            confidence: conf,
            boundingBox: box,
          ),
        );
      }
    }

    return poses;
  }

  List<DetectedPose> _applyNMS(List<DetectedPose> poses) {
    if (poses.length <= 1) return poses;

    poses.sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <DetectedPose>[];
    final active = List<bool>.filled(poses.length, true);

    for (int i = 0; i < poses.length; i++) {
      if (!active[i]) continue;
      if (poses[i].boundingBox == null) continue;

      selected.add(poses[i]);

      for (int j = i + 1; j < poses.length; j++) {
        if (!active[j]) continue;
        if (poses[j].boundingBox == null) continue;

        if (_calculateIoU(poses[i].boundingBox!, poses[j].boundingBox!) >
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
