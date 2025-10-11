import 'package:flutter/material.dart';
import '../../processors/yolo_processor.dart';
import 'detection_box_painter.dart';

/// Transparent overlay that only draws bounding boxes (no background image)
/// Used for live camera detection where the camera stream is the background
class CameraBoundingBoxOverlay extends StatelessWidget {
  const CameraBoundingBoxOverlay({
    super.key,
    required this.detections,
    this.modelInputWidth = 640.0,
    this.modelInputHeight = 640.0,
    this.boxColor = Colors.green,
    this.strokeWidth = 2.5,
    this.showLabels = true,
    this.showConfidence = true,
  });

  final List<DetectedObject> detections;
  final double modelInputWidth;
  final double modelInputHeight;
  final Color boxColor;
  final double strokeWidth;
  final bool showLabels;
  final bool showConfidence;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: DetectionBoxPainter(
        detections: detections,
        modelInputWidth: modelInputWidth,
        modelInputHeight: modelInputHeight,
        boxColor: boxColor,
        strokeWidth: strokeWidth,
        showLabels: showLabels,
        showConfidence: showConfidence,
      ),
      child: Container(), // Transparent container
    );
  }
}
