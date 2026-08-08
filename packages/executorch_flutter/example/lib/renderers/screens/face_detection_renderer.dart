import 'package:flutter/material.dart';
import '../../models/model_input.dart';
import '../../models/face_result.dart';
import '../face_detection/image_with_faces.dart';

/// Renderer for face detection results
class FaceDetectionRenderer extends StatefulWidget {
  const FaceDetectionRenderer({
    super.key,
    required this.input,
    required this.result,
    this.showBoundingBox = true,
    this.showLandmarks = true,
    this.boxColor = Colors.green,
    this.boxStrokeWidth = 2.0,
    this.landmarkRadius = 4.0,
    this.showConfidence = true,
  });

  final ModelInput input;
  final FaceDetectionResult? result;
  final bool showBoundingBox;
  final bool showLandmarks;
  final Color boxColor;
  final double boxStrokeWidth;
  final double landmarkRadius;
  final bool showConfidence;

  @override
  State<FaceDetectionRenderer> createState() => _FaceDetectionRendererState();
}

class _FaceDetectionRendererState extends State<FaceDetectionRenderer> {
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

    if (widget.result == null || widget.result!.faces.isEmpty) {
      // No detection result, just show the image
      return imageWidget;
    }

    return ImageWithFaces(
      image: imageWidget,
      faces: widget.result!.faces,
      imageFit: BoxFit.contain,
      showBoundingBox: widget.showBoundingBox,
      showLandmarks: widget.showLandmarks,
      boxColor: widget.boxColor,
      boxStrokeWidth: widget.boxStrokeWidth,
      landmarkRadius: widget.landmarkRadius,
      showConfidence: widget.showConfidence,
    );
  }
}
