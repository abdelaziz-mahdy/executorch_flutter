import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:universal_platform/universal_platform.dart';

import '../processors/base_processor.dart';
import '../processors/yolo_input_processor.dart';
import '../processors/yolo_processor.dart';
import '../processors/yolo_pose_output_processor.dart';
import '../renderers/screens/pose_detection_renderer.dart';
import '../widgets/image_input_widget.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'pose_model_settings.dart';
import 'pose_result.dart';
import 'yolo_pose_model_settings.dart';

/// YOLO-Pose Model Definition
class YoloPoseModelDefinition
    extends ModelDefinition<ModelInput, PoseDetectionResult> {
  const YoloPoseModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.remoteUrl,
    required super.inputSize,
    super.fileSizeMB,
  }) : super(icon: Icons.directions_run);

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
    final poseSettings = settings as PoseModelSettings;
    return YoloInputProcessor(
      config: YoloPreprocessConfig(
        targetWidth: inputSize,
        targetHeight: inputSize,
      ),
      preprocessingProvider: poseSettings.preprocessingProvider,
    );
  }

  @override
  OutputProcessor<PoseDetectionResult> createOutputProcessor(
    ModelSettings settings,
  ) {
    final poseSettings = settings as YoloPoseModelSettings;

    return YoloPoseOutputProcessor(
      confidenceThreshold: poseSettings.confidenceThreshold,
      iouThreshold: poseSettings.nmsThreshold,
      multiPersonMode: poseSettings.multiPersonMode,
      inputWidth: inputSize,
      inputHeight: inputSize,
    );
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required ModelInput input,
    required PoseDetectionResult? result,
  }) {
    final settings = createDefaultSettings() as PoseModelSettings;
    return PoseDetectionRenderer(
      input: input,
      result: result,
      showKeypoints: settings.showKeypoints,
      showSkeleton: settings.showSkeleton,
      confidenceThreshold: settings.confidenceThreshold,
    );
  }

  @override
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required PoseDetectionResult result,
    required double? processingTime,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Detected ${result.poses.length} ${result.poses.length == 1 ? 'person' : 'people'}',
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        ...result.poses.asMap().entries.map((entry) {
          final index = entry.key;
          final pose = entry.value;
          final visibleKeypoints =
              pose.keypoints.where((k) => k.confidence >= 0.3).length;

          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    'Person ${index + 1}',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
                Text(
                  '$visibleKeypoints/17 keypoints',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(width: 8),
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
                    '${(pose.confidence * 100).toStringAsFixed(0)}%',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Theme.of(context).colorScheme.onPrimaryContainer,
                    ),
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

  @override
  ModelSettings createDefaultSettings() {
    return YoloPoseModelSettings();
  }

  @override
  Widget buildSettingsWidget({
    required BuildContext context,
    required ModelSettings settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    final poseSettings =
        settings is YoloPoseModelSettings ? settings : YoloPoseModelSettings();

    final isMobile = UniversalPlatform.isAndroid || UniversalPlatform.isIOS;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Display Settings
        _buildSettingsSection(
          context: context,
          title: 'Display',
          children: [
            SwitchListTile(
              title: const Text('Show Performance Overlay'),
              subtitle: const Text('Display FPS and timing metrics'),
              value: poseSettings.showPerformanceOverlay,
              onChanged: (value) {
                poseSettings.showPerformanceOverlay = value;
                onSettingsChanged(poseSettings);
              },
            ),
            SwitchListTile(
              title: const Text('Show Skeleton'),
              subtitle: const Text('Draw lines connecting keypoints'),
              value: poseSettings.showSkeleton,
              onChanged: (value) {
                poseSettings.showSkeleton = value;
                onSettingsChanged(poseSettings);
              },
            ),
            SwitchListTile(
              title: const Text('Show Keypoints'),
              subtitle: const Text('Draw dots at keypoint locations'),
              value: poseSettings.showKeypoints,
              onChanged: (value) {
                poseSettings.showKeypoints = value;
                onSettingsChanged(poseSettings);
              },
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Camera Provider (mobile only)
        if (isMobile) ...[
          _buildSettingsSection(
            context: context,
            title: 'Camera Provider',
            children: [
              RadioGroup<CameraProvider>(
                groupValue: poseSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    poseSettings.cameraProvider = value;
                    onSettingsChanged(poseSettings);
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

        // Preprocessing Provider
        _buildSettingsSection(
          context: context,
          title: 'Preprocessing',
          children: [
            RadioGroup<PreprocessingProvider>(
              groupValue: poseSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  poseSettings.preprocessingProvider = value;
                  onSettingsChanged(poseSettings);
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

        // Detection Settings
        _buildSettingsSection(
          context: context,
          title: 'Detection',
          children: [
            SwitchListTile(
              title: const Text('Multi-Person Mode'),
              subtitle: const Text('Detect all people instead of just one'),
              value: poseSettings.multiPersonMode,
              onChanged: (value) {
                poseSettings.multiPersonMode = value;
                onSettingsChanged(poseSettings);
              },
            ),
            ListTile(
              title: const Text('Confidence Threshold'),
              subtitle: Text(
                '${(poseSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: poseSettings.confidenceThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label:
                    '${(poseSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  poseSettings.confidenceThreshold = value;
                  onSettingsChanged(poseSettings);
                },
              ),
            ),
            ListTile(
              title: const Text('NMS Threshold'),
              subtitle: Text(
                '${(poseSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: poseSettings.nmsThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label:
                    '${(poseSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  poseSettings.nmsThreshold = value;
                  onSettingsChanged(poseSettings);
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
              poseSettings.reset();
              onSettingsChanged(poseSettings);
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
    return 'python3 main.py export --yolo-pose';
  }
}
