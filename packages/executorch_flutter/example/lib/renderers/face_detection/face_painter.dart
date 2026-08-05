import 'package:flutter/material.dart';
import '../../models/face_result.dart';
import '../../processors/yolo_processor.dart' show BoundingBox;

/// CustomPainter for drawing face detection results on an image
class FacePainter extends CustomPainter {
  FacePainter({
    required this.faces,
    required this.imageWidth,
    required this.imageHeight,
    required this.renderWidth,
    required this.renderHeight,
    this.showBoundingBox = true,
    this.showLandmarks = true,
    this.boxColor = Colors.green,
    this.boxStrokeWidth = 2.0,
    this.landmarkRadius = 4.0,
    this.showConfidence = true,
    this.labelStyle = const TextStyle(
      color: Colors.white,
      fontSize: 12,
      fontWeight: FontWeight.bold,
    ),
  });

  final List<DetectedFace> faces;
  final double imageWidth;
  final double imageHeight;
  final double renderWidth;
  final double renderHeight;
  final bool showBoundingBox;
  final bool showLandmarks;
  final Color boxColor;
  final double boxStrokeWidth;
  final double landmarkRadius;
  final bool showConfidence;
  final TextStyle labelStyle;

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty) return;

    // Calculate scale and offset to map from normalized coords to render space
    // The image is fit with BoxFit.contain, so we need to account for letterboxing
    final imageAspect = imageWidth / imageHeight;
    final renderAspect = renderWidth / renderHeight;

    double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (imageAspect > renderAspect) {
      // Image is wider than render area - letterbox top/bottom
      scale = renderWidth / imageWidth;
      final scaledHeight = imageHeight * scale;
      offsetY = (renderHeight - scaledHeight) / 2;
    } else {
      // Image is taller than render area - letterbox left/right
      scale = renderHeight / imageHeight;
      final scaledWidth = imageWidth * scale;
      offsetX = (renderWidth - scaledWidth) / 2;
    }

    // Draw each face
    for (final face in faces) {
      _drawFace(canvas, face, scale, offsetX, offsetY);
    }
  }

  void _drawFace(
    Canvas canvas,
    DetectedFace face,
    double scale,
    double offsetX,
    double offsetY,
  ) {
    // Draw bounding box
    if (showBoundingBox) {
      _drawBoundingBox(canvas, face.boundingBox, face.confidence, scale, offsetX, offsetY);
    }

    // Draw landmarks
    if (showLandmarks) {
      for (final landmark in face.landmarks) {
        _drawLandmark(canvas, landmark, scale, offsetX, offsetY);
      }
    }
  }

  void _drawBoundingBox(
    Canvas canvas,
    BoundingBox box,
    double confidence,
    double scale,
    double offsetX,
    double offsetY,
  ) {
    // Convert normalized coords to render space
    final left = offsetX + box.x * imageWidth * scale;
    final top = offsetY + box.y * imageHeight * scale;
    final width = box.width * imageWidth * scale;
    final height = box.height * imageHeight * scale;

    final rect = Rect.fromLTWH(left, top, width, height);

    // Draw box outline
    final boxPaint = Paint()
      ..color = boxColor
      ..strokeWidth = boxStrokeWidth
      ..style = PaintingStyle.stroke;

    canvas.drawRect(rect, boxPaint);

    // Draw label background and text
    if (showConfidence) {
      final label = '${(confidence * 100).toStringAsFixed(0)}%';
      final textSpan = TextSpan(text: label, style: labelStyle);
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      // Background for label
      final bgRect = Rect.fromLTWH(
        left,
        top - textPainter.height - 4,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final bgPaint = Paint()
        ..color = boxColor.withValues(alpha: 0.8)
        ..style = PaintingStyle.fill;

      canvas.drawRect(bgRect, bgPaint);

      // Draw text
      textPainter.paint(canvas, Offset(left + 4, top - textPainter.height - 2));
    }
  }

  void _drawLandmark(
    Canvas canvas,
    FaceLandmark landmark,
    double scale,
    double offsetX,
    double offsetY,
  ) {
    // Convert normalized coords to render space
    final x = offsetX + landmark.x * imageWidth * scale;
    final y = offsetY + landmark.y * imageHeight * scale;
    final pos = Offset(x, y);

    // Draw filled circle with landmark color
    final fillPaint = Paint()
      ..color = landmark.color
      ..style = PaintingStyle.fill;

    canvas.drawCircle(pos, landmarkRadius, fillPaint);

    // Draw outline for better visibility
    final outlinePaint = Paint()
      ..color = Colors.black
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    canvas.drawCircle(pos, landmarkRadius, outlinePaint);
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return faces != oldDelegate.faces ||
        showBoundingBox != oldDelegate.showBoundingBox ||
        showLandmarks != oldDelegate.showLandmarks;
  }
}
