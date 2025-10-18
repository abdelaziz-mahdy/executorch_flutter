import 'package:flutter/material.dart';

/// Camera provider options (how to capture frames)
/// Only used by models that support camera input (e.g., image models)
enum CameraProvider {
  platform('Platform Camera', 'Uses Flutter camera plugin'),
  opencv('OpenCV Camera', 'Uses opencv_dart for camera');

  const CameraProvider(this.displayName, this.description);

  final String displayName;
  final String description;
}

/// Preprocessing provider options (how to prepare tensors)
/// Only used by models that need preprocessing (e.g., image models)
enum PreprocessingProvider {
  imageLib('Image Library', 'Uses Dart image library for preprocessing'),
  opencv('OpenCV', 'Uses opencv_dart for preprocessing'),
  gpu('GPU Shader', 'Uses GPU Fragment Shader for preprocessing');

  const PreprocessingProvider(this.displayName, this.description);

  final String displayName;
  final String description;
}

/// Base class for model-specific settings
/// Each model type can extend this to add their own configuration options
/// Only includes truly universal settings (performance overlay)
abstract class ModelSettings extends ChangeNotifier {
  ModelSettings({
    bool showPerformanceOverlay = true,
  }) : _showPerformanceOverlay = showPerformanceOverlay;

  /// Whether to show performance overlay
  bool _showPerformanceOverlay;
  bool get showPerformanceOverlay => _showPerformanceOverlay;
  set showPerformanceOverlay(bool value) {
    if (_showPerformanceOverlay != value) {
      _showPerformanceOverlay = value;
      notifyListeners();
    }
  }

  /// Copy settings (for resetting or cloning)
  ModelSettings copyWith();

  /// Reset settings to defaults
  void reset();
}

/// Common settings widget builder
/// Models can use this as a base and add their own sections
class SettingsSection {
  final String title;
  final String? subtitle;
  final Widget child;

  const SettingsSection({
    required this.title,
    this.subtitle,
    required this.child,
  });
}
