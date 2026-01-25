import 'package:flutter/material.dart';
import '../../models/pose_result.dart';

/// CustomPainter for drawing pose skeletons on an image
class PosePainter extends CustomPainter {
  PosePainter({
    required this.poses,
    required this.imageWidth,
    required this.imageHeight,
    required this.renderWidth,
    required this.renderHeight,
    this.showKeypoints = true,
    this.showSkeleton = true,
    this.keypointRadius = 4.0,
    this.skeletonStrokeWidth = 2.0,
    this.confidenceThreshold = 0.3,
  });

  final List<DetectedPose> poses;
  final double imageWidth;
  final double imageHeight;
  final double renderWidth;
  final double renderHeight;
  final bool showKeypoints;
  final bool showSkeleton;
  final double keypointRadius;
  final double skeletonStrokeWidth;
  final double confidenceThreshold;

  @override
  void paint(Canvas canvas, Size size) {
    if (poses.isEmpty) return;

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

    // Draw each pose
    for (final pose in poses) {
      _drawPose(canvas, pose, scale, offsetX, offsetY);
    }
  }

  void _drawPose(
    Canvas canvas,
    DetectedPose pose,
    double scale,
    double offsetX,
    double offsetY,
  ) {
    // Build a map of keypoint positions for skeleton drawing
    final keypointPositions = <PoseKeypointType, Offset>{};

    for (final keypoint in pose.keypoints) {
      if (keypoint.confidence >= confidenceThreshold) {
        // Convert normalized coords to render space
        final x = offsetX + keypoint.x * imageWidth * scale;
        final y = offsetY + keypoint.y * imageHeight * scale;
        keypointPositions[keypoint.type] = Offset(x, y);
      }
    }

    // Draw skeleton connections first (behind keypoints)
    if (showSkeleton) {
      for (final connection in skeletonConnections) {
        final from = connection.$1;
        final to = connection.$2;

        final fromPos = keypointPositions[from];
        final toPos = keypointPositions[to];

        if (fromPos != null && toPos != null) {
          final color = getConnectionColor(from, to);
          final paint = Paint()
            ..color = color
            ..strokeWidth = skeletonStrokeWidth
            ..style = PaintingStyle.stroke
            ..strokeCap = StrokeCap.round;

          canvas.drawLine(fromPos, toPos, paint);
        }
      }
    }

    // Draw keypoints on top
    if (showKeypoints) {
      for (final keypoint in pose.keypoints) {
        if (keypoint.confidence >= confidenceThreshold) {
          final pos = keypointPositions[keypoint.type];
          if (pos != null) {
            // Draw filled circle
            final fillPaint = Paint()
              ..color = keypoint.color
              ..style = PaintingStyle.fill;

            canvas.drawCircle(pos, keypointRadius, fillPaint);

            // Draw outline for better visibility
            final outlinePaint = Paint()
              ..color = Colors.black
              ..strokeWidth = 1.0
              ..style = PaintingStyle.stroke;

            canvas.drawCircle(pos, keypointRadius, outlinePaint);
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) {
    return poses != oldDelegate.poses ||
        showKeypoints != oldDelegate.showKeypoints ||
        showSkeleton != oldDelegate.showSkeleton ||
        confidenceThreshold != oldDelegate.confidenceThreshold;
  }
}
