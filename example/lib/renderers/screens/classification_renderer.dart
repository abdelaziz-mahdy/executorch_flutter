import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../models/model_input.dart';
import '../../processors/image_processor.dart';

/// Renderer for image classification results
class ClassificationRenderer extends StatefulWidget {
  const ClassificationRenderer({
    super.key,
    required this.input,
    required this.result,
  });

  final ModelInput input;
  final ClassificationResult? result;

  @override
  State<ClassificationRenderer> createState() => _ClassificationRendererState();
}

class _ClassificationRendererState extends State<ClassificationRenderer> {
  // Cache the image bytes to enable gapless playback
  Uint8List? _cachedBytes;

  @override
  void didUpdateWidget(ClassificationRenderer oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Update cached bytes when input changes
    if (widget.input is LiveCameraInput) {
      _cachedBytes = (widget.input as LiveCameraInput).frameBytes;
    }
  }

  @override
  Widget build(BuildContext context) {
    // Build the image widget based on input type
    final Widget imageWidget;
    if (widget.input is LiveCameraInput) {
      final bytes = (widget.input as LiveCameraInput).frameBytes;
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

    // If no result, just show the image
    if (widget.result == null) {
      return imageWidget;
    }

    // Show image with classification results overlay (top prediction)
    final topPrediction = widget.result!.topK.first;
    return Stack(
      children: [
        imageWidget,
        // Top prediction overlay
        Positioned(
          top: 16,
          left: 16,
          right: 16,
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.7),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  topPrediction.className,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: LinearProgressIndicator(
                        value: topPrediction.confidence,
                        backgroundColor: Colors.white24,
                        valueColor: const AlwaysStoppedAnimation<Color>(
                          Colors.greenAccent,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      '${(topPrediction.confidence * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
