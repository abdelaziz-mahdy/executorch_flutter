import 'package:flutter/material.dart';
import '../../models/model_input.dart';
import '../../models/pose_result.dart';
import '../pose_detection/image_with_poses.dart';

/// Renderer for pose detection results
class PoseDetectionRenderer extends StatefulWidget {
  const PoseDetectionRenderer({
    super.key,
    required this.input,
    required this.result,
    this.showKeypoints = true,
    this.showSkeleton = true,
    this.keypointRadius = 4.0,
    this.skeletonStrokeWidth = 2.0,
    this.confidenceThreshold = 0.3,
  });

  final ModelInput input;
  final PoseDetectionResult? result;
  final bool showKeypoints;
  final bool showSkeleton;
  final double keypointRadius;
  final double skeletonStrokeWidth;
  final double confidenceThreshold;

  @override
  State<PoseDetectionRenderer> createState() => _PoseDetectionRendererState();
}

class _PoseDetectionRendererState extends State<PoseDetectionRenderer> {
  @override
  Widget build(BuildContext context) {
    // Determine which image widget to use based on input type
    final Image imageWidget;
    if (widget.input is LiveCameraInput) {
      final bytes = (widget.input as LiveCameraInput).frameBytes;
      // Use gapless playback for smooth video
      imageWidget = Image.memory(
        bytes,
        fit: BoxFit.contain,
        gaplessPlayback: true,
        excludeFromSemantics: true,
      );
    } else if (widget.input is ImageBytesInput) {
      imageWidget = Image.memory(
        (widget.input as ImageBytesInput).imageBytes,
        fit: BoxFit.contain,
      );
    } else {
      throw UnsupportedError(
        'Unsupported input type: ${widget.input.runtimeType}',
      );
    }

    if (widget.result == null || widget.result!.poses.isEmpty) {
      // No detection result, just show the image
      return imageWidget;
    }

    return ImageWithPoses(
      image: imageWidget,
      poses: widget.result!.poses,
      imageFit: BoxFit.contain,
      showKeypoints: widget.showKeypoints,
      showSkeleton: widget.showSkeleton,
      keypointRadius: widget.keypointRadius,
      skeletonStrokeWidth: widget.skeletonStrokeWidth,
      confidenceThreshold: widget.confidenceThreshold,
    );
  }
}
