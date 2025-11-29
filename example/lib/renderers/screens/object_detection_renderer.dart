import 'package:flutter/material.dart';
import '../../models/model_input.dart';
import '../../processors/yolo_processor.dart';
import '../object_detection/image_with_detections.dart';

/// Renderer for object detection results
class ObjectDetectionRenderer extends StatefulWidget {
  const ObjectDetectionRenderer({
    super.key,
    required this.input,
    required this.result,
    this.modelInputWidth = 640.0,
    this.modelInputHeight = 640.0,
    this.boxColor = Colors.green,
    this.strokeWidth = 2,
    this.showLabels = true,
    this.showConfidence = true,
  });

  final ModelInput input;
  final ObjectDetectionResult? result;
  final double modelInputWidth;
  final double modelInputHeight;
  final Color boxColor;
  final double strokeWidth;
  final bool showLabels;
  final bool showConfidence;

  @override
  State<ObjectDetectionRenderer> createState() =>
      _ObjectDetectionRendererState();
}

class _ObjectDetectionRendererState extends State<ObjectDetectionRenderer> {
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
        gaplessPlayback: true, // Critical for smooth video playback
        excludeFromSemantics: true, // Performance optimization
      );
    } else if (widget.input is ImageFileInput) {
      imageWidget = Image.file(
        (widget.input as ImageFileInput).file,
        fit: BoxFit.contain,
      );
    } else {
      throw UnsupportedError(
        'Unsupported input type: ${widget.input.runtimeType}',
      );
    }

    if (widget.result == null) {
      // No detection result, just show the image
      return imageWidget;
    }

    return ImageWithDetections(
      image: imageWidget,
      detections: widget.result!.detectedObjects,
      imageFit: BoxFit.contain,
      modelInputWidth: widget.modelInputWidth,
      modelInputHeight: widget.modelInputHeight,
      boxColor: widget.boxColor,
      strokeWidth: widget.strokeWidth,
      showLabels: widget.showLabels,
      showConfidence: widget.showConfidence,
    );
  }
}
