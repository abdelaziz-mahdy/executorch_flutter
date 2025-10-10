import 'package:flutter/material.dart';

/// Camera provider options (how to capture frames)
enum CameraProvider {
  platform('Platform Camera', 'Uses Flutter camera plugin'),
  opencv('OpenCV Camera', 'Uses opencv_dart for camera');

  const CameraProvider(this.displayName, this.description);

  final String displayName;
  final String description;
}

/// Preprocessing provider options (how to prepare tensors)
enum PreprocessingProvider {
  imageLib('Image Library', 'Uses Dart image library for preprocessing'),
  opencv('OpenCV', 'Uses opencv_dart for preprocessing');

  const PreprocessingProvider(this.displayName, this.description);

  final String displayName;
  final String description;
}

/// Base class for model-specific settings
/// Each model type can extend this to add their own configuration options
abstract class ModelSettings extends ChangeNotifier {
  ModelSettings({
    bool showPerformanceOverlay = true,
    CameraProvider cameraProvider = CameraProvider.opencv,
    PreprocessingProvider preprocessingProvider = PreprocessingProvider.opencv,
  }) : _showPerformanceOverlay = showPerformanceOverlay,
       _cameraProvider = cameraProvider,
       _preprocessingProvider = preprocessingProvider;

  /// Whether to show performance overlay
  bool _showPerformanceOverlay;
  bool get showPerformanceOverlay => _showPerformanceOverlay;
  set showPerformanceOverlay(bool value) {
    if (_showPerformanceOverlay != value) {
      _showPerformanceOverlay = value;
      notifyListeners();
    }
  }

  /// Camera provider selection (for models that support camera input)
  CameraProvider _cameraProvider;
  CameraProvider get cameraProvider => _cameraProvider;
  set cameraProvider(CameraProvider value) {
    if (_cameraProvider != value) {
      _cameraProvider = value;
      notifyListeners();
    }
  }

  /// Preprocessing provider selection
  PreprocessingProvider _preprocessingProvider;
  PreprocessingProvider get preprocessingProvider => _preprocessingProvider;
  set preprocessingProvider(PreprocessingProvider value) {
    if (_preprocessingProvider != value) {
      _preprocessingProvider = value;
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
