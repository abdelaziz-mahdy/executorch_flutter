import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:http/http.dart' as http;
import 'package:universal_platform/universal_platform.dart';

import '../processors/base_processor.dart';
import '../processors/yolo_processor.dart';
import '../processors/yolo_input_processor.dart';
import '../processors/yolo_output_processor.dart';
import '../renderers/screens/object_detection_renderer.dart';
import '../services/model_index_service.dart';
import '../widgets/image_input_widget.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'yolo_model_settings.dart';

/// YOLO Object Detection Model Definition
class YoloModelDefinition
    extends ModelDefinition<ModelInput, ObjectDetectionResult> {
  const YoloModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.remoteUrl,
    required super.inputSize,
    super.fileSizeMB,
    required this.labelsRemoteUrl,
  }) : super(icon: Icons.center_focus_strong);

  /// Remote URL to download labels file
  final String labelsRemoteUrl;

  // Cache for labels (loaded once)
  static final Map<String, List<String>> _labelsCache = {};

  Future<List<String>> _loadLabels() async {
    if (_labelsCache.containsKey(labelsRemoteUrl)) {
      return _labelsCache[labelsRemoteUrl]!;
    }

    // Download labels from remote URL (with cache buster)
    final urlWithCacheBuster = ModelIndexService.addCacheBuster(labelsRemoteUrl);
    final response = await http.get(Uri.parse(urlWithCacheBuster));
    if (response.statusCode != 200) {
      throw Exception('Failed to download labels from $labelsRemoteUrl');
    }

    final labelsString = response.body;
    final labels = labelsString
        .split('\n')
        .where((line) => line.isNotEmpty)
        .toList();

    _labelsCache[labelsRemoteUrl] = labels;
    return labels;
  }

  // Helper to load labels synchronously from cache
  List<String> _loadLabelsSync() {
    if (_labelsCache.containsKey(labelsRemoteUrl)) {
      return _labelsCache[labelsRemoteUrl]!;
    }
    // Labels should be preloaded by controller before creating processor
    throw StateError('Labels not loaded. Call loadLabels() first.');
  }

  // Make _loadLabels public so controller can preload
  Future<List<String>> loadLabels() => _loadLabels();

  @override
  Widget buildInputWidget({
    required BuildContext context,
    required Function(ModelInput) onInputSelected,
    VoidCallback? onCameraModeToggle,
    bool isCameraMode = false,
  }) {
    return ImageInputWidget(
      onImageSelected: (Uint8List bytes) =>
          onInputSelected(ImageBytesInput(bytes)),
      onCameraModeToggle: onCameraModeToggle,
      isCameraMode: isCameraMode,
    );
  }

  @override
  InputProcessor<ModelInput> createInputProcessor(ModelSettings settings) {
    final yoloSettings = settings as YoloModelSettings;
    return YoloInputProcessor(
      config: YoloPreprocessConfig(
        targetWidth: inputSize,
        targetHeight: inputSize,
      ),
      preprocessingProvider: yoloSettings.preprocessingProvider,
    );
  }

  @override
  OutputProcessor<ObjectDetectionResult> createOutputProcessor(
    ModelSettings settings,
  ) {
    final yoloSettings = settings as YoloModelSettings;

    return YoloOutputProcessor(
      classLabels: _loadLabelsSync(),
      inputWidth: inputSize,
      inputHeight: inputSize,
      confidenceThreshold: yoloSettings.confidenceThreshold,
      iouThreshold: yoloSettings.nmsThreshold,
    );
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
  ModelSettings createDefaultSettings() {
    return YoloModelSettings();
  }

  @override
  Widget buildSettingsWidget({
    required BuildContext context,
    required ModelSettings settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    // Use provided settings or create default if wrong type
    final yoloSettings = settings is YoloModelSettings
        ? settings
        : YoloModelSettings();

    // Check if we're on a platform that supports multiple camera providers
    final isMobile = UniversalPlatform.isAndroid || UniversalPlatform.isIOS;

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
              RadioGroup<CameraProvider>(
                groupValue: yoloSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    yoloSettings.cameraProvider = value;
                    onSettingsChanged(yoloSettings);
                  }
                },
                child: Column(
                  children: [
                    RadioListTile<CameraProvider>(
                      title: Text(CameraProvider.platform.displayName),
                      subtitle: Text(CameraProvider.platform.description),
                      value: CameraProvider.platform,
                    ),
                    RadioListTile<CameraProvider>(
                      title: Text(CameraProvider.opencv.displayName),
                      subtitle: Text(CameraProvider.opencv.description),
                      value: CameraProvider.opencv,
                    ),
                  ],
                ),
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
            RadioGroup<PreprocessingProvider>(
              groupValue: yoloSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  yoloSettings.preprocessingProvider = value;
                  onSettingsChanged(yoloSettings);
                }
              },
              child: Column(
                children: PreprocessingProvider.availableProviders
                    .map(
                      (provider) => RadioListTile<PreprocessingProvider>(
                        title: Text(provider.displayName),
                        subtitle: Text(provider.description),
                        value: provider,
                      ),
                    )
                    .toList(),
              ),
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
              subtitle: Text(
                '${(yoloSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: yoloSettings.confidenceThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label:
                    '${(yoloSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  yoloSettings.confidenceThreshold = value;
                  onSettingsChanged(yoloSettings);
                },
              ),
            ),
            const SizedBox(height: 8),
            ListTile(
              title: const Text('NMS Threshold'),
              subtitle: Text(
                '${(yoloSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: yoloSettings.nmsThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label:
                    '${(yoloSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
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

  @override
  String getExportCommand() {
    return 'python3 main.py export --yolo $name';
  }
}
