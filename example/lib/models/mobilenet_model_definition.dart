import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../processors/image_processor.dart';
import '../processors/opencv_processors.dart';
import '../renderers/screens/classification_renderer.dart';
import '../widgets/image_input_widget.dart';
import '../services/processor_preferences.dart';
import 'model_definition.dart';

/// MobileNet Image Classification Model Definition
class MobileNetModelDefinition
    extends ModelDefinition<File, ClassificationResult> {
  const MobileNetModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.assetPath,
    required super.inputSize,
    required this.labelsAssetPath,
  }) : super(icon: Icons.image);

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
      final preprocessor = OpenCVImageNetPreprocessor(
        config: ImagePreprocessConfig(
          targetWidth: inputSize,
          targetHeight: inputSize,
          normalizeToFloat: true,
        ),
      );
      return await preprocessor.preprocess(bytes);
    } else {
      final preprocessor = ImageNetPreprocessor(
        config: ImagePreprocessConfig(
          targetWidth: inputSize,
          targetHeight: inputSize,
          normalizeToFloat: true,
        ),
      );
      return await preprocessor.preprocess(bytes);
    }
  }

  @override
  Future<ClassificationResult> processResult({
    required File input,
    required InferenceResult inferenceResult,
  }) async {
    final labels = await _loadLabels();
    final postprocessor = ImageNetPostprocessor(
      classLabels: labels,
    );

    return await postprocessor.postprocess(
      (inferenceResult.outputs ?? []).whereType<TensorData>().toList(),
    );
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required File input,
    required ClassificationResult? result,
  }) {
    return ClassificationRenderer(
      input: input,
      result: result,
    );
  }

  @override
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required ClassificationResult result,
    required double? processingTime,
  }) {
    // Get top 5 predictions
    final topPredictions = result.topK;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Top Predictions',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 16),
        ...topPredictions.map((prediction) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          prediction.className,
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
                          '${(prediction.confidence * 100).toStringAsFixed(1)}%',
                          style:
                              Theme.of(context).textTheme.bodySmall?.copyWith(
                                    fontWeight: FontWeight.bold,
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onPrimaryContainer,
                                  ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  LinearProgressIndicator(
                    value: prediction.confidence,
                    backgroundColor: Theme.of(context)
                        .colorScheme
                        .surfaceContainerHighest,
                  ),
                ],
              ),
            )),
      ],
    );
  }
}
