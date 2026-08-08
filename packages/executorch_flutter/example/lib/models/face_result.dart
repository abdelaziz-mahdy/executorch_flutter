import 'package:flutter/material.dart';

import '../processors/yolo_processor.dart' show BoundingBox;

/// BlazeFace 6-landmark format
enum BlazeFaceLandmarkType {
  rightEye, // 0
  leftEye, // 1
  noseTip, // 2
  mouthCenter, // 3
  rightEar, // 4
  leftEar, // 5
}

/// Display name for BlazeFace landmarks
extension BlazeFaceLandmarkTypeExtension on BlazeFaceLandmarkType {
  String get displayName {
    switch (this) {
      case BlazeFaceLandmarkType.rightEye:
        return 'Right Eye';
      case BlazeFaceLandmarkType.leftEye:
        return 'Left Eye';
      case BlazeFaceLandmarkType.noseTip:
        return 'Nose';
      case BlazeFaceLandmarkType.mouthCenter:
        return 'Mouth';
      case BlazeFaceLandmarkType.rightEar:
        return 'Right Ear';
      case BlazeFaceLandmarkType.leftEar:
        return 'Left Ear';
    }
  }

  /// Color for rendering this landmark
  Color get color {
    switch (this) {
      case BlazeFaceLandmarkType.rightEye:
      case BlazeFaceLandmarkType.leftEye:
        return Colors.blue;
      case BlazeFaceLandmarkType.noseTip:
        return Colors.green;
      case BlazeFaceLandmarkType.mouthCenter:
        return Colors.red;
      case BlazeFaceLandmarkType.rightEar:
      case BlazeFaceLandmarkType.leftEar:
        return Colors.yellow;
    }
  }
}

/// YOLO-Face 5-landmark format
enum YoloFaceLandmarkType {
  leftEye, // 0
  rightEye, // 1
  noseTip, // 2
  leftMouth, // 3
  rightMouth, // 4
}

/// Display name for YOLO-Face landmarks
extension YoloFaceLandmarkTypeExtension on YoloFaceLandmarkType {
  String get displayName {
    switch (this) {
      case YoloFaceLandmarkType.leftEye:
        return 'Left Eye';
      case YoloFaceLandmarkType.rightEye:
        return 'Right Eye';
      case YoloFaceLandmarkType.noseTip:
        return 'Nose';
      case YoloFaceLandmarkType.leftMouth:
        return 'Left Mouth';
      case YoloFaceLandmarkType.rightMouth:
        return 'Right Mouth';
    }
  }

  /// Color for rendering this landmark
  Color get color {
    switch (this) {
      case YoloFaceLandmarkType.leftEye:
      case YoloFaceLandmarkType.rightEye:
        return Colors.blue;
      case YoloFaceLandmarkType.noseTip:
        return Colors.green;
      case YoloFaceLandmarkType.leftMouth:
      case YoloFaceLandmarkType.rightMouth:
        return Colors.red;
    }
  }
}

/// Generic face landmark type for unified handling
enum FaceLandmarkCategory { eye, nose, mouth, ear }

/// Get color for a landmark category
Color getLandmarkCategoryColor(FaceLandmarkCategory category) {
  switch (category) {
    case FaceLandmarkCategory.eye:
      return Colors.blue;
    case FaceLandmarkCategory.nose:
      return Colors.green;
    case FaceLandmarkCategory.mouth:
      return Colors.red;
    case FaceLandmarkCategory.ear:
      return Colors.yellow;
  }
}

/// A single face landmark
@immutable
class FaceLandmark {
  const FaceLandmark({
    required this.name,
    required this.x,
    required this.y,
    required this.confidence,
    required this.category,
  });

  /// Name of the landmark (e.g., "Left Eye", "Nose")
  final String name;

  /// X coordinate normalized to [0, 1]
  final double x;

  /// Y coordinate normalized to [0, 1]
  final double y;

  /// Confidence score for this landmark [0, 1]
  final double confidence;

  /// Category for color-coded rendering
  final FaceLandmarkCategory category;

  /// Color for rendering this landmark
  Color get color => getLandmarkCategoryColor(category);

  /// Create from BlazeFace landmark type
  factory FaceLandmark.fromBlazeFace({
    required BlazeFaceLandmarkType type,
    required double x,
    required double y,
    double confidence = 1.0,
  }) {
    FaceLandmarkCategory category;
    switch (type) {
      case BlazeFaceLandmarkType.rightEye:
      case BlazeFaceLandmarkType.leftEye:
        category = FaceLandmarkCategory.eye;
      case BlazeFaceLandmarkType.noseTip:
        category = FaceLandmarkCategory.nose;
      case BlazeFaceLandmarkType.mouthCenter:
        category = FaceLandmarkCategory.mouth;
      case BlazeFaceLandmarkType.rightEar:
      case BlazeFaceLandmarkType.leftEar:
        category = FaceLandmarkCategory.ear;
    }

    return FaceLandmark(
      name: type.displayName,
      x: x,
      y: y,
      confidence: confidence,
      category: category,
    );
  }

  /// Create from YOLO-Face landmark type
  factory FaceLandmark.fromYoloFace({
    required YoloFaceLandmarkType type,
    required double x,
    required double y,
    double confidence = 1.0,
  }) {
    FaceLandmarkCategory category;
    switch (type) {
      case YoloFaceLandmarkType.leftEye:
      case YoloFaceLandmarkType.rightEye:
        category = FaceLandmarkCategory.eye;
      case YoloFaceLandmarkType.noseTip:
        category = FaceLandmarkCategory.nose;
      case YoloFaceLandmarkType.leftMouth:
      case YoloFaceLandmarkType.rightMouth:
        category = FaceLandmarkCategory.mouth;
    }

    return FaceLandmark(
      name: type.displayName,
      x: x,
      y: y,
      confidence: confidence,
      category: category,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FaceLandmark &&
          name == other.name &&
          x == other.x &&
          y == other.y &&
          confidence == other.confidence &&
          category == other.category;

  @override
  int get hashCode =>
      name.hashCode ^
      x.hashCode ^
      y.hashCode ^
      confidence.hashCode ^
      category.hashCode;

  @override
  String toString() =>
      'FaceLandmark($name: ($x, $y), conf: ${(confidence * 100).toStringAsFixed(1)}%)';
}

/// A detected face with bounding box and landmarks
@immutable
class DetectedFace {
  const DetectedFace({
    required this.boundingBox,
    required this.landmarks,
    required this.confidence,
  });

  /// Bounding box around the face
  final BoundingBox boundingBox;

  /// Face landmarks (5 for YOLO-Face, 6 for BlazeFace)
  final List<FaceLandmark> landmarks;

  /// Overall confidence score for this face detection [0, 1]
  final double confidence;

  /// Get landmarks above a confidence threshold
  List<FaceLandmark> getVisibleLandmarks({double threshold = 0.3}) {
    return landmarks.where((lm) => lm.confidence >= threshold).toList();
  }

  /// Get landmark by name
  FaceLandmark? getLandmarkByName(String name) {
    try {
      return landmarks.firstWhere((lm) => lm.name == name);
    } catch (_) {
      return null;
    }
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DetectedFace &&
          boundingBox == other.boundingBox &&
          landmarks == other.landmarks &&
          confidence == other.confidence;

  @override
  int get hashCode =>
      boundingBox.hashCode ^ landmarks.hashCode ^ confidence.hashCode;

  @override
  String toString() =>
      'DetectedFace(conf: ${(confidence * 100).toStringAsFixed(1)}%, landmarks: ${landmarks.length})';
}

/// Result of face detection
@immutable
class FaceDetectionResult {
  const FaceDetectionResult({
    required this.faces,
    this.inferenceTimeMs,
    this.preprocessingTimeMs,
    this.postprocessingTimeMs,
  });

  /// All detected faces
  final List<DetectedFace> faces;

  /// Inference time in milliseconds
  final double? inferenceTimeMs;

  /// Preprocessing time in milliseconds
  final double? preprocessingTimeMs;

  /// Postprocessing time in milliseconds
  final double? postprocessingTimeMs;

  /// Number of detected faces
  int get count => faces.length;

  /// Whether any faces were detected
  bool get hasFaces => faces.isNotEmpty;

  /// Get the face with highest confidence
  DetectedFace? get bestFace {
    if (faces.isEmpty) return null;
    return faces.reduce((a, b) => a.confidence > b.confidence ? a : b);
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FaceDetectionResult && faces == other.faces;

  @override
  int get hashCode => faces.hashCode;

  @override
  String toString() => 'FaceDetectionResult(faces: ${faces.length})';
}
