import 'dart:io';
import 'package:executorch_flutter_example/processors/opencv/opencv_imagenet_preprocessor.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../processors/image_processor.dart';
import '../renderers/screens/classification_renderer.dart';
import '../widgets/image_input_widget.dart';
import '../services/service_locator.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'classification_model_settings.dart';

/// MobileNet Image Classification Model Definition
class MobileNetModelDefinition
    extends ModelDefinition<ModelInput, ClassificationResult> {
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
    required Function(ModelInput) onInputSelected,
    VoidCallback? onCameraModeToggle,
    bool isCameraMode = false,
  }) {
    return ImageInputWidget(
      onImageSelected: (File file) => onInputSelected(ImageFileInput(file)),
      onCameraModeToggle: onCameraModeToggle,
      isCameraMode: isCameraMode,
    );
  }

  @override
  Future<List<TensorData>> prepareInput(ModelInput input) async {
    // Extract bytes from input (works for both ImageFileInput and LiveCameraInput)
    final Uint8List bytes;
    if (input is ImageFileInput) {
      bytes = await input.file.readAsBytes();
    } else if (input is LiveCameraInput) {
      bytes = input.frameBytes;
    } else {
      throw UnsupportedError('Unsupported input type: ${input.runtimeType}');
    }

    // Get preprocessing preference from GetIt settings
    // Default to OpenCV if no settings are registered yet
    final useOpenCV = getIt.isRegistered<ModelSettings>()
        ? getIt<ModelSettings>().preprocessingProvider == PreprocessingProvider.opencv
        : true; // Default to OpenCV

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
    required ModelInput input,
    required List<TensorData> outputs,
  }) async {
    final labels = await _loadLabels();
    final postprocessor = ImageNetPostprocessor(classLabels: labels);

    return await postprocessor.postprocess(outputs);
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required ModelInput input,
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
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        ...topPredictions.map(
          (prediction) => Padding(
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
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Theme.of(
                            context,
                          ).colorScheme.onPrimaryContainer,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                LinearProgressIndicator(
                  value: prediction.confidence,
                  backgroundColor: Theme.of(
                    context,
                  ).colorScheme.surfaceContainerHighest,
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget? buildSettingsWidget({
    required BuildContext context,
    required ModelSettings? settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    // Create default settings if none provided
    final classificationSettings = settings is ClassificationModelSettings
        ? settings
        : ClassificationModelSettings();

    // Check if we're on a platform that supports multiple camera providers
    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Performance Overlay Section
        _buildSettingsSection(
          context: context,
          title: 'Display',
          children: [
            SwitchListTile(
              title: const Text('Show Performance Overlay'),
              subtitle: const Text('Display FPS and timing metrics'),
              value: classificationSettings.showPerformanceOverlay,
              onChanged: (value) {
                classificationSettings.showPerformanceOverlay = value;
                onSettingsChanged(classificationSettings);
              },
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Camera Provider Section (only on mobile platforms)
        if (isMobile) ...[
          _buildSettingsSection(
            context: context,
            title: 'Camera Provider',
            children: [
              RadioListTile<CameraProvider>(
                title: Text(CameraProvider.platform.displayName),
                subtitle: Text(CameraProvider.platform.description),
                value: CameraProvider.platform,
                groupValue: classificationSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    classificationSettings.cameraProvider = value;
                    onSettingsChanged(classificationSettings);
                  }
                },
              ),
              RadioListTile<CameraProvider>(
                title: Text(CameraProvider.opencv.displayName),
                subtitle: Text(CameraProvider.opencv.description),
                value: CameraProvider.opencv,
                groupValue: classificationSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    classificationSettings.cameraProvider = value;
                    onSettingsChanged(classificationSettings);
                  }
                },
              ),
            ],
          ),
          const SizedBox(height: 16),
        ],

        // Preprocessing Provider Section (all platforms)
        _buildSettingsSection(
          context: context,
          title: 'Preprocessing',
          children: [
            RadioListTile<PreprocessingProvider>(
              title: Text(PreprocessingProvider.imageLib.displayName),
              subtitle: Text(PreprocessingProvider.imageLib.description),
              value: PreprocessingProvider.imageLib,
              groupValue: classificationSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  classificationSettings.preprocessingProvider = value;
                  onSettingsChanged(classificationSettings);
                }
              },
            ),
            RadioListTile<PreprocessingProvider>(
              title: Text(PreprocessingProvider.opencv.displayName),
              subtitle: Text(PreprocessingProvider.opencv.description),
              value: PreprocessingProvider.opencv,
              groupValue: classificationSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  classificationSettings.preprocessingProvider = value;
                  onSettingsChanged(classificationSettings);
                }
              },
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Classification Settings Section
        _buildSettingsSection(
          context: context,
          title: 'Classification',
          children: [
            ListTile(
              title: const Text('Top K Predictions'),
              subtitle: Text('${classificationSettings.topK} predictions'),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: classificationSettings.topK.toDouble(),
                min: 1,
                max: 10,
                divisions: 9,
                label: '${classificationSettings.topK}',
                onChanged: (value) {
                  classificationSettings.topK = value.toInt();
                  onSettingsChanged(classificationSettings);
                },
              ),
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Reset button
        Center(
          child: OutlinedButton.icon(
            onPressed: () {
              classificationSettings.reset();
              onSettingsChanged(classificationSettings);
            },
            icon: const Icon(Icons.refresh),
            label: const Text('Reset to Defaults'),
          ),
        ),
      ],
    );
  }

  Widget _buildSettingsSection({
    required BuildContext context,
    required String title,
    required List<Widget> children,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Text(
            title,
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(children: children),
        ),
      ],
    );
  }
}
