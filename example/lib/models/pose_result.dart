import 'package:flutter/material.dart';

import '../processors/yolo_processor.dart' show BoundingBox;

/// COCO 17-keypoint format used by MoveNet and YOLO-Pose
enum PoseKeypointType {
  nose, // 0
  leftEye, // 1
  rightEye, // 2
  leftEar, // 3
  rightEar, // 4
  leftShoulder, // 5
  rightShoulder, // 6
  leftElbow, // 7
  rightElbow, // 8
  leftWrist, // 9
  rightWrist, // 10
  leftHip, // 11
  rightHip, // 12
  leftKnee, // 13
  rightKnee, // 14
  leftAnkle, // 15
  rightAnkle, // 16
}

/// Display name for each keypoint type
extension PoseKeypointTypeExtension on PoseKeypointType {
  String get displayName {
    switch (this) {
      case PoseKeypointType.nose:
        return 'Nose';
      case PoseKeypointType.leftEye:
        return 'Left Eye';
      case PoseKeypointType.rightEye:
        return 'Right Eye';
      case PoseKeypointType.leftEar:
        return 'Left Ear';
      case PoseKeypointType.rightEar:
        return 'Right Ear';
      case PoseKeypointType.leftShoulder:
        return 'Left Shoulder';
      case PoseKeypointType.rightShoulder:
        return 'Right Shoulder';
      case PoseKeypointType.leftElbow:
        return 'Left Elbow';
      case PoseKeypointType.rightElbow:
        return 'Right Elbow';
      case PoseKeypointType.leftWrist:
        return 'Left Wrist';
      case PoseKeypointType.rightWrist:
        return 'Right Wrist';
      case PoseKeypointType.leftHip:
        return 'Left Hip';
      case PoseKeypointType.rightHip:
        return 'Right Hip';
      case PoseKeypointType.leftKnee:
        return 'Left Knee';
      case PoseKeypointType.rightKnee:
        return 'Right Knee';
      case PoseKeypointType.leftAnkle:
        return 'Left Ankle';
      case PoseKeypointType.rightAnkle:
        return 'Right Ankle';
    }
  }
}

/// Body part grouping for color-coded rendering
enum BodyPart {
  face, // nose, eyes, ears
  arms, // shoulders, elbows, wrists
  torso, // shoulder-hip connections
  legs, // hips, knees, ankles
}

/// Get body part for a keypoint type
BodyPart getBodyPart(PoseKeypointType keypoint) {
  switch (keypoint) {
    case PoseKeypointType.nose:
    case PoseKeypointType.leftEye:
    case PoseKeypointType.rightEye:
    case PoseKeypointType.leftEar:
    case PoseKeypointType.rightEar:
      return BodyPart.face;
    case PoseKeypointType.leftShoulder:
    case PoseKeypointType.rightShoulder:
    case PoseKeypointType.leftElbow:
    case PoseKeypointType.rightElbow:
    case PoseKeypointType.leftWrist:
    case PoseKeypointType.rightWrist:
      return BodyPart.arms;
    case PoseKeypointType.leftHip:
    case PoseKeypointType.rightHip:
      return BodyPart.torso;
    case PoseKeypointType.leftKnee:
    case PoseKeypointType.rightKnee:
    case PoseKeypointType.leftAnkle:
    case PoseKeypointType.rightAnkle:
      return BodyPart.legs;
  }
}

/// Color for each body part
Color getBodyPartColor(BodyPart part) {
  switch (part) {
    case BodyPart.face:
      return Colors.cyan;
    case BodyPart.arms:
      return Colors.green;
    case BodyPart.torso:
      return Colors.yellow;
    case BodyPart.legs:
      return Colors.pinkAccent;
  }
}

/// Skeleton connections for rendering
/// Each connection is a pair of keypoint indices
const List<(PoseKeypointType, PoseKeypointType)> skeletonConnections = [
  // Face connections
  (PoseKeypointType.leftEye, PoseKeypointType.nose),
  (PoseKeypointType.rightEye, PoseKeypointType.nose),
  (PoseKeypointType.leftEye, PoseKeypointType.leftEar),
  (PoseKeypointType.rightEye, PoseKeypointType.rightEar),

  // Arm connections
  (PoseKeypointType.leftShoulder, PoseKeypointType.leftElbow),
  (PoseKeypointType.leftElbow, PoseKeypointType.leftWrist),
  (PoseKeypointType.rightShoulder, PoseKeypointType.rightElbow),
  (PoseKeypointType.rightElbow, PoseKeypointType.rightWrist),

  // Torso connections
  (PoseKeypointType.leftShoulder, PoseKeypointType.rightShoulder),
  (PoseKeypointType.leftShoulder, PoseKeypointType.leftHip),
  (PoseKeypointType.rightShoulder, PoseKeypointType.rightHip),
  (PoseKeypointType.leftHip, PoseKeypointType.rightHip),

  // Leg connections
  (PoseKeypointType.leftHip, PoseKeypointType.leftKnee),
  (PoseKeypointType.leftKnee, PoseKeypointType.leftAnkle),
  (PoseKeypointType.rightHip, PoseKeypointType.rightKnee),
  (PoseKeypointType.rightKnee, PoseKeypointType.rightAnkle),
];

/// Get color for a skeleton connection based on body parts
Color getConnectionColor(PoseKeypointType from, PoseKeypointType to) {
  final fromPart = getBodyPart(from);
  final toPart = getBodyPart(to);

  // For torso connections (shoulder to hip), use torso color
  if ((fromPart == BodyPart.arms && toPart == BodyPart.torso) ||
      (fromPart == BodyPart.torso && toPart == BodyPart.arms)) {
    return getBodyPartColor(BodyPart.torso);
  }

  // Use the color of the "from" keypoint's body part
  return getBodyPartColor(fromPart);
}

/// A single keypoint in a pose
@immutable
class PoseKeypoint {
  const PoseKeypoint({
    required this.type,
    required this.x,
    required this.y,
    required this.confidence,
  });

  /// The type of keypoint
  final PoseKeypointType type;

  /// X coordinate normalized to [0, 1]
  final double x;

  /// Y coordinate normalized to [0, 1]
  final double y;

  /// Confidence score for this keypoint [0, 1]
  final double confidence;

  /// Display name for this keypoint
  String get name => type.displayName;

  /// Body part this keypoint belongs to
  BodyPart get bodyPart => getBodyPart(type);

  /// Color for rendering this keypoint
  Color get color => getBodyPartColor(bodyPart);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is PoseKeypoint &&
          type == other.type &&
          x == other.x &&
          y == other.y &&
          confidence == other.confidence;

  @override
  int get hashCode =>
      type.hashCode ^ x.hashCode ^ y.hashCode ^ confidence.hashCode;

  @override
  String toString() =>
      'PoseKeypoint($name: ($x, $y), conf: ${(confidence * 100).toStringAsFixed(1)}%)';
}

/// A detected pose with all keypoints
@immutable
class DetectedPose {
  const DetectedPose({
    required this.keypoints,
    required this.confidence,
    this.boundingBox,
  });

  /// All 17 keypoints for this pose
  final List<PoseKeypoint> keypoints;

  /// Overall confidence score for this pose [0, 1]
  final double confidence;

  /// Optional bounding box (provided by YOLO-Pose)
  final BoundingBox? boundingBox;

  /// Get a specific keypoint by type
  PoseKeypoint? getKeypoint(PoseKeypointType type) {
    final index = type.index;
    if (index >= 0 && index < keypoints.length) {
      return keypoints[index];
    }
    return null;
  }

  /// Get keypoints that are visible (above threshold)
  List<PoseKeypoint> getVisibleKeypoints({double threshold = 0.3}) {
    return keypoints.where((kp) => kp.confidence >= threshold).toList();
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DetectedPose &&
          keypoints == other.keypoints &&
          confidence == other.confidence &&
          boundingBox == other.boundingBox;

  @override
  int get hashCode =>
      keypoints.hashCode ^ confidence.hashCode ^ boundingBox.hashCode;

  @override
  String toString() =>
      'DetectedPose(conf: ${(confidence * 100).toStringAsFixed(1)}%, keypoints: ${keypoints.length})';
}

/// Result of pose detection
@immutable
class PoseDetectionResult {
  const PoseDetectionResult({
    required this.poses,
    this.inferenceTimeMs,
    this.preprocessingTimeMs,
    this.postprocessingTimeMs,
  });

  /// All detected poses
  final List<DetectedPose> poses;

  /// Inference time in milliseconds
  final double? inferenceTimeMs;

  /// Preprocessing time in milliseconds
  final double? preprocessingTimeMs;

  /// Postprocessing time in milliseconds
  final double? postprocessingTimeMs;

  /// Number of detected poses
  int get count => poses.length;

  /// Whether any poses were detected
  bool get hasPoses => poses.isNotEmpty;

  /// Get the pose with highest confidence
  DetectedPose? get bestPose {
    if (poses.isEmpty) return null;
    return poses.reduce((a, b) => a.confidence > b.confidence ? a : b);
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is PoseDetectionResult && poses == other.poses;

  @override
  int get hashCode => poses.hashCode;

  @override
  String toString() => 'PoseDetectionResult(poses: ${poses.length})';
}
