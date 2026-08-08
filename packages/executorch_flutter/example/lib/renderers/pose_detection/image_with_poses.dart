import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../../models/pose_result.dart';
import 'pose_painter.dart';

/// Widget for displaying an image with pose skeleton overlays.
///
/// This widget handles:
/// - Loading and measuring the actual rendered image size
/// - Scaling pose keypoints from normalized coordinates to render space
/// - Reacting to window resize and layout changes
class ImageWithPoses extends StatefulWidget {
  /// The image to be displayed (typically Image.file or Image.memory).
  final Image image;

  /// The list of detected poses to be displayed on the image.
  final List<DetectedPose> poses;

  /// The fit for the image within its container.
  final BoxFit imageFit;

  /// Whether to show keypoint dots.
  final bool showKeypoints;

  /// Whether to show skeleton lines.
  final bool showSkeleton;

  /// Radius of keypoint dots.
  final double keypointRadius;

  /// Width of skeleton lines.
  final double skeletonStrokeWidth;

  /// Minimum confidence for a keypoint to be visible.
  final double confidenceThreshold;

  /// Constructs an ImageWithPoses widget.
  const ImageWithPoses({
    super.key,
    required this.image,
    required this.poses,
    this.imageFit = BoxFit.contain,
    this.showKeypoints = true,
    this.showSkeleton = true,
    this.keypointRadius = 4.0,
    this.skeletonStrokeWidth = 2.0,
    this.confidenceThreshold = 0.3,
  });

  @override
  State<ImageWithPoses> createState() => _ImageWithPosesState();
}

class _ImageWithPosesState extends State<ImageWithPoses>
    with WidgetsBindingObserver {
  final GlobalKey imageKey = GlobalKey();
  Completer<ui.Image> completer = Completer<ui.Image>();
  Size? imageSize;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _resolveImage();
  }

  void _resolveImage() {
    widget.image.image
        .resolve(const ImageConfiguration())
        .addListener(
          ImageStreamListener((ImageInfo info, bool _) {
            if (!completer.isCompleted) {
              completer.complete(info.image);
            }
          }),
        );
    Future.microtask(
      () => completer.future.then((_) {
        WidgetsBinding.instance.addPostFrameCallback((_) => measureSize());
      }),
    );
  }

  @override
  void didUpdateWidget(ImageWithPoses oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Reset completer when image changes (for camera frames)
    if (oldWidget.image.image != widget.image.image) {
      completer = Completer<ui.Image>();
      _resolveImage();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeMetrics() {
    super.didChangeMetrics();
    WidgetsBinding.instance.addPostFrameCallback((_) => measureSize());
  }

  void measureSize() {
    if (imageKey.currentContext != null) {
      final RenderBox renderBox =
          imageKey.currentContext!.findRenderObject() as RenderBox;
      final newSize = renderBox.size;
      if (newSize != imageSize) {
        setState(() {
          imageSize = newSize;
        });
      }
    } else {
      WidgetsBinding.instance.addPostFrameCallback((_) => measureSize());
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ui.Image>(
      future: completer.future,
      builder: (BuildContext context, AsyncSnapshot<ui.Image> snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        } else {
          return Stack(
            children: [
              Image(
                image: widget.image.image,
                fit: widget.imageFit,
                key: imageKey,
                gaplessPlayback: true,
                excludeFromSemantics: true,
              ),
              if (imageSize != null)
                Positioned.fill(
                  child: CustomPaint(
                    painter: PosePainter(
                      poses: widget.poses,
                      imageWidth: snapshot.data!.width.toDouble(),
                      imageHeight: snapshot.data!.height.toDouble(),
                      renderWidth: imageSize!.width,
                      renderHeight: imageSize!.height,
                      showKeypoints: widget.showKeypoints,
                      showSkeleton: widget.showSkeleton,
                      keypointRadius: widget.keypointRadius,
                      skeletonStrokeWidth: widget.skeletonStrokeWidth,
                      confidenceThreshold: widget.confidenceThreshold,
                    ),
                  ),
                ),
            ],
          );
        }
      },
    );
  }
}
