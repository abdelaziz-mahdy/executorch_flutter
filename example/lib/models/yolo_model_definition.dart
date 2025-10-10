import 'dart:io';
import 'package:executorch_flutter_example/processors/opencv/opencv_yolo_preprocessor.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../processors/yolo_processor.dart';
import '../renderers/screens/object_detection_renderer.dart';
import '../widgets/image_input_widget.dart';
import '../services/service_locator.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'yolo_model_settings.dart';

/// YOLO Object Detection Model Definition
class YoloModelDefinition extends ModelDefinition<ModelInput, ObjectDetectionResult> {
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
    required ModelInput input,
    required List<TensorData> outputs,
  }) async {
    final labels = await _loadLabels();
    final postprocessor = YoloPostprocessor(
      classLabels: labels,
      inputWidth: inputSize,
      inputHeight: inputSize,
    );

    return await postprocessor.postprocess(outputs);
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required ModelInput input,
    required ObjectDetectionResult? result,
  }) {
    return ObjectDetectionRenderer(
      input: input,
      result: result,
      modelInputWidth: inputSize.toDouble(),
      modelInputHeight: inputSize.toDouble(),
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
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        ...result.detectedObjects.map(
          (obj) => Padding(
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
                      color: Theme.of(context).colorScheme.onPrimaryContainer,
                    ),
                  ),
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
    final yoloSettings = settings is YoloModelSettings
        ? settings
        : YoloModelSettings();

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
              value: yoloSettings.showPerformanceOverlay,
              onChanged: (value) {
                yoloSettings.showPerformanceOverlay = value;
                onSettingsChanged(yoloSettings);
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
                groupValue: yoloSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    yoloSettings.cameraProvider = value;
                    onSettingsChanged(yoloSettings);
                  }
                },
              ),
              RadioListTile<CameraProvider>(
                title: Text(CameraProvider.opencv.displayName),
                subtitle: Text(CameraProvider.opencv.description),
                value: CameraProvider.opencv,
                groupValue: yoloSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    yoloSettings.cameraProvider = value;
                    onSettingsChanged(yoloSettings);
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
              groupValue: yoloSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  yoloSettings.preprocessingProvider = value;
                  onSettingsChanged(yoloSettings);
                }
              },
            ),
            RadioListTile<PreprocessingProvider>(
              title: Text(PreprocessingProvider.opencv.displayName),
              subtitle: Text(PreprocessingProvider.opencv.description),
              value: PreprocessingProvider.opencv,
              groupValue: yoloSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  yoloSettings.preprocessingProvider = value;
                  onSettingsChanged(yoloSettings);
                }
              },
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Detection Settings Section
        _buildSettingsSection(
          context: context,
          title: 'Detection',
          children: [
            ListTile(
              title: const Text('Confidence Threshold'),
              subtitle: Text('${(yoloSettings.confidenceThreshold * 100).toStringAsFixed(0)}%'),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: yoloSettings.confidenceThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label: '${(yoloSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  yoloSettings.confidenceThreshold = value;
                  onSettingsChanged(yoloSettings);
                },
              ),
            ),
            const SizedBox(height: 8),
            ListTile(
              title: const Text('NMS Threshold'),
              subtitle: Text('${(yoloSettings.nmsThreshold * 100).toStringAsFixed(0)}%'),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: yoloSettings.nmsThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label: '${(yoloSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  yoloSettings.nmsThreshold = value;
                  onSettingsChanged(yoloSettings);
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
              yoloSettings.reset();
              onSettingsChanged(yoloSettings);
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
