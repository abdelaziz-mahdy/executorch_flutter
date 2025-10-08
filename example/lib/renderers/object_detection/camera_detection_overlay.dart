import 'package:flutter/material.dart';
import '../../processors/yolo_processor.dart';
import '../../ui/camera_view_singleton.dart';

/// Camera-specific detection overlay painter
/// Handles scaling detection boxes from normalized coordinates to camera preview space
class CameraDetectionPainter extends CustomPainter {
  final List<DetectedObject> detections;
  final Color boxColor;
  final double strokeWidth;
  final bool showLabels;
  final bool showConfidence;
  final TextStyle labelStyle;

  CameraDetectionPainter({
    required this.detections,
    this.boxColor = Colors.green,
    this.strokeWidth = 3,
    this.showLabels = true,
    this.showConfidence = true,
    this.labelStyle = const TextStyle(
      color: Colors.white,
      fontSize: 14,
      fontWeight: FontWeight.bold,
    ),
  });

  @override
  void paint(Canvas canvas, Size size) {
    final screenSize = CameraViewSingleton.actualPreviewSizeH;
    final factorX = screenSize.width;
    final factorY = screenSize.height;

    for (final detection in detections) {
      // Calculate box position and size in screen coordinates
      final left = detection.boundingBox.x * factorX;
      final top = detection.boundingBox.y * factorY;
      final width = detection.boundingBox.width * factorX;
      final height = detection.boundingBox.height * factorY;

      // Determine box color based on class
      final Color color =
          Colors.primaries[((detection.className.length +
                  detection.className.codeUnitAt(0) +
                  (detection.classIndex ?? 0)) %
              Colors.primaries.length)];

      // Draw bounding box
      final paint = Paint()
        ..color = color
        ..strokeWidth = strokeWidth
        ..style = PaintingStyle.stroke;

      final rect = Rect.fromLTWH(left, top, width, height);
      canvas.drawRect(rect, paint);

      // Draw label if enabled
      if (showLabels) {
        final labelText = showConfidence
            ? '${detection.className} ${(detection.confidence * 100).toStringAsFixed(0)}%'
            : detection.className;

        final textPainter = TextPainter(
          text: TextSpan(text: labelText, style: labelStyle),
          textDirection: TextDirection.ltr,
        );

        textPainter.layout();

        // Draw background for label
        final labelRect = Rect.fromLTWH(
          left,
          top - textPainter.height - 4,
          textPainter.width + 8,
          textPainter.height + 4,
        );

        final backgroundPaint = Paint()
          ..color = color.withOpacity(0.8)
          ..style = PaintingStyle.fill;

        canvas.drawRect(labelRect, backgroundPaint);

        // Draw text
        textPainter.paint(
          canvas,
          Offset(left + 4, top - textPainter.height - 2),
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CameraDetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.boxColor != boxColor ||
        oldDelegate.strokeWidth != strokeWidth ||
        oldDelegate.showLabels != showLabels ||
        oldDelegate.showConfidence != showConfidence;
  }
}

/// Widget for overlaying detections on camera preview
class CameraDetectionOverlay extends StatelessWidget {
  final List<DetectedObject> detections;
  final Color boxColor;
  final double strokeWidth;
  final bool showLabels;
  final bool showConfidence;

  const CameraDetectionOverlay({
    super.key,
    required this.detections,
    this.boxColor = Colors.green,
    this.strokeWidth = 3,
    this.showLabels = true,
    this.showConfidence = true,
  });

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return const SizedBox.shrink();
    }

    return CustomPaint(
      painter: CameraDetectionPainter(
        detections: detections,
        boxColor: boxColor,
        strokeWidth: strokeWidth,
        showLabels: showLabels,
        showConfidence: showConfidence,
      ),
      child: Container(),
    );
  }
}
