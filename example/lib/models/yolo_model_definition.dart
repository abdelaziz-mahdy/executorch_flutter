import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../processors/yolo_processor.dart';
import '../processors/opencv_processors.dart';
import '../renderers/screens/object_detection_renderer.dart';
import '../widgets/image_input_widget.dart';
import '../services/processor_preferences.dart';
import 'model_definition.dart';

/// YOLO Object Detection Model Definition
class YoloModelDefinition extends ModelDefinition<File, ObjectDetectionResult> {
  const YoloModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.assetPath,
    required super.inputSize,
    required this.labelsAssetPath,
  }) : super(icon: Icons.center_focus_strong);

  final String labelsAssetPath;

  // Cache for labels (loaded once)
  static final Map<String, List<String>> _labelsCache = {};

  Future<List<String>> _loadLabels() async {
    if (_labelsCache.containsKey(labelsAssetPath)) {
      return _labelsCache[labelsAssetPath]!;
    }

    final labelsString = await rootBundle.loadString(labelsAssetPath);
    final labels = labelsString
        .split('\n')
        .where((line) => line.isNotEmpty)
        .toList();

    _labelsCache[labelsAssetPath] = labels;
    return labels;
  }

  @override
  Widget buildInputWidget({
    required BuildContext context,
    required Function(File) onInputSelected,
  }) {
    return ImageInputWidget(onImageSelected: onInputSelected);
  }

  @override
  Future<List<TensorData>> prepareInput(File input) async {
    final bytes = await input.readAsBytes();
    final useOpenCV = await ProcessorPreferences.getUseOpenCV();

    // Dynamically select preprocessor based on user preference
    if (useOpenCV) {
      final preprocessor = OpenCVYoloPreprocessor(
        config: YoloPreprocessConfig(
          targetWidth: inputSize,
          targetHeight: inputSize,
        ),
      );
      return await preprocessor.preprocess(bytes);
    } else {
      final preprocessor = YoloPreprocessor(
        config: YoloPreprocessConfig(
          targetWidth: inputSize,
          targetHeight: inputSize,
        ),
      );
      return await preprocessor.preprocess(bytes);
    }
  }

  @override
  Future<ObjectDetectionResult> processResult({
    required File input,
    required InferenceResult inferenceResult,
  }) async {
    final labels = await _loadLabels();
    final postprocessor = YoloPostprocessor(
      classLabels: labels,
      inputWidth: inputSize,
      inputHeight: inputSize,
    );

    return await postprocessor.postprocess(
      (inferenceResult.outputs ?? []).whereType<TensorData>().toList(),
    );
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required File input,
    required ObjectDetectionResult? result,
  }) {
    return ObjectDetectionRenderer(
      input: input,
      result: result,
    );
  }

  @override
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required ObjectDetectionResult result,
    required double? processingTime,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Detected ${result.detectedObjects.length} objects',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 16),
        ...result.detectedObjects.map((obj) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      obj.className,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.primaryContainer,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      '${(obj.confidence * 100).toStringAsFixed(0)}%',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: Theme.of(context)
                                .colorScheme
                                .onPrimaryContainer,
                          ),
                    ),
                  ),
                ],
              ),
            )),
      ],
    );
  }
}
