import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:universal_platform/universal_platform.dart';

import '../processors/base_processor.dart';
import '../processors/yolo_input_processor.dart';
import '../processors/yolo_processor.dart';
import '../processors/yolo_face_output_processor.dart';
import '../renderers/screens/face_detection_renderer.dart';
import '../widgets/image_input_widget.dart';
import 'face_model_settings.dart';
import 'face_result.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'yolo_face_model_settings.dart';

/// YOLO-Face Model Definition
class YoloFaceModelDefinition
    extends ModelDefinition<ModelInput, FaceDetectionResult> {
  const YoloFaceModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.remoteUrl,
    required super.inputSize,
    super.hash,
    super.fileSizeMB,
  }) : super(icon: Icons.face_retouching_natural);

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
    final faceSettings = settings as FaceModelSettings;
    return YoloInputProcessor(
      config: YoloPreprocessConfig(
        targetWidth: inputSize,
        targetHeight: inputSize,
      ),
      preprocessingProvider: faceSettings.preprocessingProvider,
    );
  }

  @override
  OutputProcessor<FaceDetectionResult> createOutputProcessor(
    ModelSettings settings,
  ) {
    final faceSettings = settings as YoloFaceModelSettings;

    return YoloFaceOutputProcessor(
      confidenceThreshold: faceSettings.confidenceThreshold,
      iouThreshold: faceSettings.nmsThreshold,
      multiFaceMode: faceSettings.multiFaceMode,
      inputWidth: inputSize,
      inputHeight: inputSize,
    );
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required ModelInput input,
    required FaceDetectionResult? result,
  }) {
    final settings = createDefaultSettings() as FaceModelSettings;
    return FaceDetectionRenderer(
      input: input,
      result: result,
      showBoundingBox: settings.showBoundingBox,
      showLandmarks: settings.showLandmarks,
    );
  }

  @override
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required FaceDetectionResult result,
    required double? processingTime,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Detected ${result.faces.length} ${result.faces.length == 1 ? 'face' : 'faces'}',
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        ...result.faces.asMap().entries.map((entry) {
          final index = entry.key;
          final face = entry.value;

          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    'Face ${index + 1}',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
                Text(
                  '${face.landmarks.length} landmarks',
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
                    '${(face.confidence * 100).toStringAsFixed(0)}%',
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
    return YoloFaceModelSettings();
  }

  @override
  Widget buildSettingsWidget({
    required BuildContext context,
    required ModelSettings settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    final faceSettings =
        settings is YoloFaceModelSettings ? settings : YoloFaceModelSettings();

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
              value: faceSettings.showPerformanceOverlay,
              onChanged: (value) {
                faceSettings.showPerformanceOverlay = value;
                onSettingsChanged(faceSettings);
              },
            ),
            SwitchListTile(
              title: const Text('Show Bounding Boxes'),
              subtitle: const Text('Draw boxes around detected faces'),
              value: faceSettings.showBoundingBox,
              onChanged: (value) {
                faceSettings.showBoundingBox = value;
                onSettingsChanged(faceSettings);
              },
            ),
            SwitchListTile(
              title: const Text('Show Landmarks'),
              subtitle: const Text('Draw facial landmark points'),
              value: faceSettings.showLandmarks,
              onChanged: (value) {
                faceSettings.showLandmarks = value;
                onSettingsChanged(faceSettings);
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
                groupValue: faceSettings.cameraProvider,
                onChanged: (value) {
                  if (value != null) {
                    faceSettings.cameraProvider = value;
                    onSettingsChanged(faceSettings);
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
              groupValue: faceSettings.preprocessingProvider,
              onChanged: (value) {
                if (value != null) {
                  faceSettings.preprocessingProvider = value;
                  onSettingsChanged(faceSettings);
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
              title: const Text('Multi-Face Mode'),
              subtitle: const Text('Detect all faces instead of just one'),
              value: faceSettings.multiFaceMode,
              onChanged: (value) {
                faceSettings.multiFaceMode = value;
                onSettingsChanged(faceSettings);
              },
            ),
            ListTile(
              title: const Text('Confidence Threshold'),
              subtitle: Text(
                '${(faceSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: faceSettings.confidenceThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label:
                    '${(faceSettings.confidenceThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  faceSettings.confidenceThreshold = value;
                  onSettingsChanged(faceSettings);
                },
              ),
            ),
            ListTile(
              title: const Text('NMS Threshold'),
              subtitle: Text(
                '${(faceSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: faceSettings.nmsThreshold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                label:
                    '${(faceSettings.nmsThreshold * 100).toStringAsFixed(0)}%',
                onChanged: (value) {
                  faceSettings.nmsThreshold = value;
                  onSettingsChanged(faceSettings);
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
              faceSettings.reset();
              onSettingsChanged(faceSettings);
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
    return 'python3 main.py export --yolo-face';
  }
}
