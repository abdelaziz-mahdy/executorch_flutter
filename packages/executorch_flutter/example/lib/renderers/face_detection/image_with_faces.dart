import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../../models/face_result.dart';
import 'face_painter.dart';

/// Widget for displaying an image with face detection overlays.
///
/// This widget handles:
/// - Loading and measuring the actual rendered image size
/// - Scaling face bounding boxes and landmarks from normalized coordinates to render space
/// - Reacting to window resize and layout changes
class ImageWithFaces extends StatefulWidget {
  /// The image to be displayed (typically Image.file or Image.memory).
  final Image image;

  /// The list of detected faces to be displayed on the image.
  final List<DetectedFace> faces;

  /// The fit for the image within its container.
  final BoxFit imageFit;

  /// Whether to show bounding boxes around faces.
  final bool showBoundingBox;

  /// Whether to show facial landmark dots.
  final bool showLandmarks;

  /// Color of bounding boxes.
  final Color boxColor;

  /// Width of bounding box strokes.
  final double boxStrokeWidth;

  /// Radius of landmark dots.
  final double landmarkRadius;

  /// Whether to show confidence scores.
  final bool showConfidence;

  /// Text style for labels.
  final TextStyle labelStyle;

  /// Constructs an ImageWithFaces widget.
  const ImageWithFaces({
    super.key,
    required this.image,
    required this.faces,
    this.imageFit = BoxFit.contain,
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

  @override
  State<ImageWithFaces> createState() => _ImageWithFacesState();
}

class _ImageWithFacesState extends State<ImageWithFaces>
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
    widget.image.image.resolve(const ImageConfiguration()).addListener(
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
  void didUpdateWidget(ImageWithFaces oldWidget) {
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
                    painter: FacePainter(
                      faces: widget.faces,
                      imageWidth: snapshot.data!.width.toDouble(),
                      imageHeight: snapshot.data!.height.toDouble(),
                      renderWidth: imageSize!.width,
                      renderHeight: imageSize!.height,
                      showBoundingBox: widget.showBoundingBox,
                      showLandmarks: widget.showLandmarks,
                      boxColor: widget.boxColor,
                      boxStrokeWidth: widget.boxStrokeWidth,
                      landmarkRadius: widget.landmarkRadius,
                      showConfidence: widget.showConfidence,
                      labelStyle: widget.labelStyle,
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
