import 'package:flutter/material.dart';
import '../processors/yolo_processor.dart';

/// Custom painter for drawing object detection bounding boxes on a canvas.
///
/// This painter handles the scaling and rendering of detection boxes from
/// normalized coordinates [0, 1] to the actual rendered image dimensions.
/// The key insight is that we need to scale from:
/// - Image space (original model input size, e.g., 640x640)
/// - To render space (actual widget size on screen)
class DetectionBoxPainter extends CustomPainter {
  /// The list of detected objects to be painted.
  final List<DetectedObject> detections;

  /// The width of the original image, used for scaling.
  final double? imageWidth;

  /// The height of the original image, used for scaling.
  final double? imageHeight;

  /// The width of the real rendered image, used for scaling.
  final double? renderWidth;

  /// The height of the real rendered image, used for scaling.
  final double? renderHeight;

  /// The color used to paint the bounding boxes.
  final Color boxColor;

  /// The width of the strokes used to draw the boxes.
  final double strokeWidth;

  /// Whether to draw labels above the boxes.
  final bool showLabels;

  /// Whether to draw confidence scores.
  final bool showConfidence;

  /// Text style for labels.
  final TextStyle labelStyle;

  /// Constructs a DetectionBoxPainter instance.
  DetectionBoxPainter({
    required this.detections,
    this.imageWidth,
    this.imageHeight,
    this.renderWidth,
    this.renderHeight,
    this.boxColor = Colors.green,
    this.strokeWidth = 2,
    this.showLabels = true,
    this.showConfidence = true,
    this.labelStyle = const TextStyle(
      color: Colors.white,
      fontSize: 12,
      fontWeight: FontWeight.bold,
      backgroundColor: Colors.black87,
    ),
  });

  /// Scales the x-coordinate from normalized [0, 1] to render space.
  ///
  /// The detection coordinates are already normalized to [0, 1] by the processor,
  /// so we just need to scale them to the rendered widget size.
  double scaledX(double normalizedX) {
    if (renderWidth == null) return normalizedX;
    return normalizedX * renderWidth!;
  }

  /// Scales the y-coordinate from normalized [0, 1] to render space.
  double scaledY(double normalizedY) {
    if (renderHeight == null) return normalizedY;
    return normalizedY * renderHeight!;
  }

  /// Scales width from normalized [0, 1] to render space.
  double scaledWidth(double normalizedWidth) {
    if (renderWidth == null) return normalizedWidth;
    return normalizedWidth * renderWidth!;
  }

  /// Scales height from normalized [0, 1] to render space.
  double scaledHeight(double normalizedHeight) {
    if (renderHeight == null) return normalizedHeight;
    return normalizedHeight * renderHeight!;
  }

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = boxColor
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    for (final detection in detections) {
      // Scale normalized coordinates to render space
      final left = scaledX(detection.boundingBox.x);
      final top = scaledY(detection.boundingBox.y);
      final width = scaledWidth(detection.boundingBox.width);
      final height = scaledHeight(detection.boundingBox.height);

      final rect = Rect.fromLTWH(left, top, width, height);

      // Draw bounding box
      canvas.drawRect(rect, paint);

      // Draw label if enabled
      if (showLabels) {
        final labelText = showConfidence
            ? '${detection.className} ${(detection.confidence * 100).toStringAsFixed(0)}%'
            : detection.className;

        final textPainter = TextPainter(
          text: TextSpan(
            text: labelText,
            style: labelStyle,
          ),
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
          ..color = boxColor.withOpacity(0.8)
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
  bool shouldRepaint(covariant DetectionBoxPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.boxColor != boxColor ||
        oldDelegate.strokeWidth != strokeWidth ||
        oldDelegate.showLabels != showLabels ||
        oldDelegate.showConfidence != showConfidence ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight ||
        oldDelegate.renderWidth != renderWidth ||
        oldDelegate.renderHeight != renderHeight;
  }
}
