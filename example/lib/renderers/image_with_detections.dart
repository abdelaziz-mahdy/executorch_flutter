import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../processors/yolo_processor.dart';
import 'detection_box_painter.dart';

/// Widget for displaying an image with object detection bounding boxes.
///
/// This widget handles:
/// - Loading and measuring the actual rendered image size
/// - Scaling detection boxes from normalized coordinates to render space
/// - Reacting to window resize and layout changes
class ImageWithDetections extends StatefulWidget {
  /// The image to be displayed.
  final Image image;

  /// The list of detections to be displayed on the image.
  final List<DetectedObject> detections;

  /// The fit for the image within its container.
  final BoxFit imageFit;

  /// The color of the bounding boxes.
  final Color boxColor;

  /// The width of the bounding box strokes.
  final double strokeWidth;

  /// Whether to show labels above boxes.
  final bool showLabels;

  /// Whether to show confidence scores.
  final bool showConfidence;

  /// Text style for labels.
  final TextStyle labelStyle;

  /// Constructs an ImageWithDetections widget.
  const ImageWithDetections({
    super.key,
    required this.image,
    required this.detections,
    this.imageFit = BoxFit.contain,
    this.boxColor = Colors.green,
    this.strokeWidth = 2,
    this.showLabels = true,
    this.showConfidence = true,
    this.labelStyle = const TextStyle(
      color: Colors.white,
      fontSize: 12,
      fontWeight: FontWeight.bold,
    ),
  });

  @override
  State<ImageWithDetections> createState() => _ImageWithDetectionsState();
}

class _ImageWithDetectionsState extends State<ImageWithDetections>
    with WidgetsBindingObserver {
  final GlobalKey imageKey = GlobalKey();
  Completer<ui.Image> completer = Completer<ui.Image>();
  Size? imageSize;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    widget.image.image.resolve(const ImageConfiguration()).addListener(
      ImageStreamListener((ImageInfo info, bool _) {
        if (!completer.isCompleted) {
          completer.complete(info.image);
        }
      }),
    );
    Future.microtask(() => completer.future.then((_) {
          WidgetsBinding.instance.addPostFrameCallback((_) => measureSize());
        }));
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
    }
    // run after build
    else {
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
              ),
              if (imageSize != null)
                Positioned.fill(
                  child: CustomPaint(
                    painter: DetectionBoxPainter(
                      detections: widget.detections,
                      imageWidth: snapshot.data!.width.toDouble(),
                      imageHeight: snapshot.data!.height.toDouble(),
                      renderWidth: imageSize!.width,
                      renderHeight: imageSize!.height,
                      boxColor: widget.boxColor,
                      strokeWidth: widget.strokeWidth,
                      showLabels: widget.showLabels,
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
