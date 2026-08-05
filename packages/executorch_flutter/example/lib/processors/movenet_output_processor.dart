import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';

import '../models/pose_result.dart';
import 'base_processor.dart';

/// Output processor for MoveNet pose detection
/// Parses the model output tensor to extract keypoints
class MoveNetOutputProcessor extends OutputProcessor<PoseDetectionResult> {
  MoveNetOutputProcessor({
    this.confidenceThreshold = 0.3,
    this.multiPersonMode = false,
  });

  final double confidenceThreshold;
  final bool multiPersonMode;

  @override
  Future<PoseDetectionResult> process(List<TensorData> tensorOutputs) async {
    if (tensorOutputs.isEmpty) {
      return const PoseDetectionResult(poses: []);
    }

    final output = tensorOutputs.first;

    // MoveNet output shape: [1, 1, 17, 3] for single pose
    // or [1, 6, 56] for multi-pose (6 people, 17 keypoints * 3 + 5 detection values)
    // Each keypoint has: [y, x, confidence]

    final shape = output.shape
        .where((dim) => dim != null)
        .map((dim) => dim!)
        .toList();

    debugPrint('🔍 MoveNet output shape: $shape');

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

    // Detect format based on shape
    if (shape.length == 4 && shape[2] == 17 && shape[3] == 3) {
      // Single pose format: [1, 1, 17, 3]
      return _parseSinglePose(floatData);
    } else if (shape.length == 3) {
      // Multi-pose format: [1, N, 56] where N is max number of people
      return _parseMultiPose(floatData, shape);
    } else {
      debugPrint('⚠️ Unexpected MoveNet output shape: $shape');
      // Try to parse as single pose anyway
      if (floatCount >= 51) {
        // 17 * 3 = 51
        return _parseSinglePose(floatData);
      }
      return const PoseDetectionResult(poses: []);
    }
  }

  /// Parse single pose output format [1, 1, 17, 3]
  PoseDetectionResult _parseSinglePose(Float32List outputs) {
    final keypoints = <PoseKeypoint>[];
    double totalConfidence = 0;
    int validKeypoints = 0;

    for (int i = 0; i < 17; i++) {
      final y = outputs[i * 3]; // MoveNet outputs y first
      final x = outputs[i * 3 + 1];
      final confidence = outputs[i * 3 + 2];

      final type = PoseKeypointType.values[i];
      keypoints.add(PoseKeypoint(
        type: type,
        x: x.clamp(0.0, 1.0),
        y: y.clamp(0.0, 1.0),
        confidence: confidence.clamp(0.0, 1.0),
      ));

      if (confidence >= confidenceThreshold) {
        totalConfidence += confidence;
        validKeypoints++;
      }
    }

    final avgConfidence =
        validKeypoints > 0 ? totalConfidence / validKeypoints : 0.0;

    // Only return pose if enough keypoints are visible
    if (validKeypoints >= 5) {
      return PoseDetectionResult(
        poses: [
          DetectedPose(
            keypoints: keypoints,
            confidence: avgConfidence,
          ),
        ],
      );
    }

    return const PoseDetectionResult(poses: []);
  }

  /// Parse multi-pose output format [1, N, 56]
  /// Each person: 17 keypoints * 3 (y, x, conf) + 5 detection values
  PoseDetectionResult _parseMultiPose(Float32List outputs, List<int> shape) {
    final numPeople = shape[1];
    final poses = <DetectedPose>[];

    for (int p = 0; p < numPeople; p++) {
      final offset = p * 56;

      // Parse keypoints
      final keypoints = <PoseKeypoint>[];
      double totalConfidence = 0;
      int validKeypoints = 0;

      for (int i = 0; i < 17; i++) {
        final y = outputs[offset + i * 3];
        final x = outputs[offset + i * 3 + 1];
        final confidence = outputs[offset + i * 3 + 2];

        final type = PoseKeypointType.values[i];
        keypoints.add(PoseKeypoint(
          type: type,
          x: x.clamp(0.0, 1.0),
          y: y.clamp(0.0, 1.0),
          confidence: confidence.clamp(0.0, 1.0),
        ));

        if (confidence >= confidenceThreshold) {
          totalConfidence += confidence;
          validKeypoints++;
        }
      }

      // Detection score is at index 51 (after 17*3 keypoints)
      final detectionScore = outputs[offset + 51];

      final avgConfidence =
          validKeypoints > 0 ? totalConfidence / validKeypoints : 0.0;

      // Only add pose if detection score is high enough and enough keypoints visible
      if (detectionScore >= confidenceThreshold && validKeypoints >= 5) {
        poses.add(DetectedPose(
          keypoints: keypoints,
          confidence: avgConfidence * detectionScore,
        ));
      }
    }

    // Sort by confidence and optionally limit to single person
    poses.sort((a, b) => b.confidence.compareTo(a.confidence));

    if (!multiPersonMode && poses.isNotEmpty) {
      return PoseDetectionResult(poses: [poses.first]);
    }

    return PoseDetectionResult(poses: poses);
  }
}
