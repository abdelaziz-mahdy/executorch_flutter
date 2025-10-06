import 'dart:io';
import 'package:flutter/material.dart';
import '../../processors/image_processor.dart';
import 'base_result_renderer.dart';

/// Renderer for image classification results
class ClassificationRenderer
    extends BaseResultRenderer<File, ClassificationResult> {
  const ClassificationRenderer({
    super.key,
    required super.input,
    required super.result,
  });

  @override
  Widget build(BuildContext context) {
    return Image.file(
      input,
      fit: BoxFit.contain,
    );
    // Note: Classification results are typically shown in a separate section
    // (like a list of top predictions), not overlaid on the image
  }
}
