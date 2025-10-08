import 'dart:io';
import 'package:flutter/material.dart';
import '../../processors/yolo_processor.dart';
import '../image_with_detections.dart';
import 'base_result_renderer.dart';

/// Renderer for object detection results
class ObjectDetectionRenderer
    extends BaseResultRenderer<File, ObjectDetectionResult> {
  const ObjectDetectionRenderer({
    super.key,
    required super.input,
    required super.result,
    this.modelInputWidth = 640.0,
    this.modelInputHeight = 640.0,
    this.boxColor = Colors.green,
    this.strokeWidth = 2,
    this.showLabels = true,
    this.showConfidence = true,
  });

  final double modelInputWidth;
  final double modelInputHeight;
  final Color boxColor;
  final double strokeWidth;
  final bool showLabels;
  final bool showConfidence;

  @override
  Widget build(BuildContext context) {
    if (result == null) {
      // No detection result, just show the image
      return Image.file(input, fit: BoxFit.contain);
    }

    return ImageWithDetections(
      image: Image.file(input),
      detections: result!.detectedObjects,
      imageFit: BoxFit.contain,
      modelInputWidth: modelInputWidth,
      modelInputHeight: modelInputHeight,
      boxColor: boxColor,
      strokeWidth: strokeWidth,
      showLabels: showLabels,
      showConfidence: showConfidence,
    );
  }
}
