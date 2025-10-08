import 'package:flutter/material.dart';
import '../../processors/yolo_processor.dart';

/// Custom painter for drawing object detection bounding boxes on a canvas.
///
/// This painter handles the scaling and rendering of detection boxes from
/// normalized coordinates [0, 1] to the actual rendered image dimensions.
///
/// IMPORTANT: The coordinates from YOLO are normalized to the letterbox space (640x640),
/// NOT the original image space. This painter accounts for letterbox padding to correctly
/// map coordinates back to the original image space.
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

  /// Model input width (e.g., 640 for YOLO640, can be 320, 416, 512, 1280, etc.)
  final double modelInputWidth;

  /// Model input height (e.g., 640 for YOLO640, can be 320, 416, 512, 1280, etc.)
  final double modelInputHeight;

  /// Constructs a DetectionBoxPainter instance.
  DetectionBoxPainter({
    required this.detections,
    this.imageWidth,
    this.imageHeight,
    this.renderWidth,
    this.renderHeight,
    this.modelInputWidth = 640.0,
    this.modelInputHeight = 640.0,
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

  /// Transform coordinates from letterbox space to original image space
  ///
  /// Letterbox resize maintains aspect ratio by:
  /// 1. Scaling image to fit within 640x640
  /// 2. Adding gray padding to center it
  ///
  /// To reverse this:
  /// 1. Denormalize from [0, 1] to 640x640 pixel space
  /// 2. Subtract padding offsets
  /// 3. Divide by scale factor
  /// 4. Normalize to [0, 1] based on original image
  double _letterboxToOriginalX(double normalizedX) {
    if (imageWidth == null) return normalizedX;

    // Calculate letterbox parameters
    final scale =
        modelInputWidth / imageWidth! < modelInputHeight / imageHeight!
        ? modelInputWidth / imageWidth!
        : modelInputHeight / imageHeight!;

    final scaledWidth = (imageWidth! * scale);
    final offsetX = (modelInputWidth - scaledWidth) / 2;

    // Transform: letterbox [0, 1] -> letterbox pixels -> remove padding -> scale back -> original [0, 1]
    final letterboxPixel = normalizedX * modelInputWidth;
    final withoutPadding = letterboxPixel - offsetX;
    final originalPixel = withoutPadding / scale;

    return originalPixel / imageWidth!;
  }

  double _letterboxToOriginalY(double normalizedY) {
    if (imageHeight == null) return normalizedY;

    // Calculate letterbox parameters
    final scale =
        modelInputWidth / imageWidth! < modelInputHeight / imageHeight!
        ? modelInputWidth / imageWidth!
        : modelInputHeight / imageHeight!;

    final scaledHeight = (imageHeight! * scale);
    final offsetY = (modelInputHeight - scaledHeight) / 2;

    // Transform: letterbox [0, 1] -> letterbox pixels -> remove padding -> scale back -> original [0, 1]
    final letterboxPixel = normalizedY * modelInputHeight;
    final withoutPadding = letterboxPixel - offsetY;
    final originalPixel = withoutPadding / scale;

    return originalPixel / imageHeight!;
  }

  double _letterboxToOriginalWidth(double normalizedWidth) {
    if (imageWidth == null) return normalizedWidth;

    // Calculate letterbox scale
    final scale =
        modelInputWidth / imageWidth! < modelInputHeight / imageHeight!
        ? modelInputWidth / imageWidth!
        : modelInputHeight / imageHeight!;

    // Transform: letterbox [0, 1] -> letterbox pixels -> scale back -> original [0, 1]
    final letterboxPixel = normalizedWidth * modelInputWidth;
    final originalPixel = letterboxPixel / scale;

    return originalPixel / imageWidth!;
  }

  double _letterboxToOriginalHeight(double normalizedHeight) {
    if (imageHeight == null) return normalizedHeight;

    // Calculate letterbox scale
    final scale =
        modelInputWidth / imageWidth! < modelInputHeight / imageHeight!
        ? modelInputWidth / imageWidth!
        : modelInputHeight / imageHeight!;

    // Transform: letterbox [0, 1] -> letterbox pixels -> scale back -> original [0, 1]
    final letterboxPixel = normalizedHeight * modelInputHeight;
    final originalPixel = letterboxPixel / scale;

    return originalPixel / imageHeight!;
  }

  /// Scales the x-coordinate from normalized [0, 1] (original image space) to render space.
  double scaledX(double normalizedX) {
    if (renderWidth == null) return normalizedX;
    return normalizedX * renderWidth!;
  }

  /// Scales the y-coordinate from normalized [0, 1] (original image space) to render space.
  double scaledY(double normalizedY) {
    if (renderHeight == null) return normalizedY;
    return normalizedY * renderHeight!;
  }

  /// Scales width from normalized [0, 1] (original image space) to render space.
  double scaledWidth(double normalizedWidth) {
    if (renderWidth == null) return normalizedWidth;
    return normalizedWidth * renderWidth!;
  }

  /// Scales height from normalized [0, 1] (original image space) to render space.
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
      // Transform coordinates from letterbox space to original image space
      final originalX = _letterboxToOriginalX(detection.boundingBox.x);
      final originalY = _letterboxToOriginalY(detection.boundingBox.y);
      final originalWidth = _letterboxToOriginalWidth(
        detection.boundingBox.width,
      );
      final originalHeight = _letterboxToOriginalHeight(
        detection.boundingBox.height,
      );

      // Scale from original image space [0, 1] to render space
      final left = scaledX(originalX);
      final top = scaledY(originalY);
      final width = scaledWidth(originalWidth);
      final height = scaledHeight(originalHeight);

      final rect = Rect.fromLTWH(left, top, width, height);

      // Draw bounding box
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
        oldDelegate.renderHeight != renderHeight ||
        oldDelegate.modelInputWidth != modelInputWidth ||
        oldDelegate.modelInputHeight != modelInputHeight;
  }
}
