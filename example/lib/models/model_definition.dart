import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'model_input.dart';
import 'model_settings.dart';
import '../ui/widgets/performance_monitor.dart';

/// Base class for model definitions
/// Each model type (YOLO, MobileNet, etc.) will extend this to define:
/// - How to get input (camera, gallery, text field, etc.)
/// - How to process the model output
/// - How to render the results
/// - Model-specific settings
abstract class ModelDefinition<TInput extends ModelInput, TResult> {
  const ModelDefinition({
    required this.name,
    required this.displayName,
    required this.description,
    required this.icon,
    required this.assetPath,
    required this.inputSize,
    this.showPerformanceOverlay = true,
  });

  final String name;
  final String displayName;
  final String description;
  final IconData icon;
  final String assetPath;
  final int inputSize;

  /// Whether to show performance overlay (can be toggled by user)
  final bool showPerformanceOverlay;

  /// Build the input selection widget (e.g., camera/gallery picker, text input, audio recorder)
  Widget buildInputWidget({
    required BuildContext context,
    required Function(TInput) onInputSelected,
    VoidCallback? onCameraModeToggle,
    bool isCameraMode = false,
  });

  /// Prepare input for inference (convert to TensorData)
  /// Each model implementation chooses which preprocessor to use
  Future<List<TensorData>> prepareInput(TInput input);

  /// Process the model inference result
  Future<TResult> processResult({
    required TInput input,
    required List<TensorData> outputs,
  });

  /// Build the result renderer widget
  /// The input type (ImageFileInput, LiveCameraInput, etc.) determines rendering strategy
  Widget buildResultRenderer({
    required BuildContext context,
    required TInput input,
    required TResult? result,
  });

  /// Build the results details section (statistics, confidence scores, etc.)
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required TResult result,
    required double? processingTime,
  });

  /// Build the performance monitor widget
  /// Override this to provide custom performance visualization for specific model types
  Widget buildPerformanceMonitor({
    required BuildContext context,
    required PerformanceMetrics metrics,
    required PerformanceDisplayMode displayMode,
  }) {
    // Default implementation uses generic PerformanceMonitor
    return PerformanceMonitor(
      metrics: metrics,
      displayMode: displayMode,
    );
  }

  /// Build the settings widget for this model
  /// Override this to provide model-specific settings (e.g., confidence threshold, NMS threshold)
  /// Return null if the model has no settings
  Widget? buildSettingsWidget({
    required BuildContext context,
    required ModelSettings? settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    // Default implementation returns null (no settings)
    // Models can override to provide their own settings UI
    return null;
  }
}
