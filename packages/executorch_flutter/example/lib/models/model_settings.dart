import 'package:flutter/material.dart';
import 'package:universal_platform/universal_platform.dart';

/// Camera provider options (how to capture frames)
/// Only used by models that support camera input (e.g., image models)
enum CameraProvider {
  platform('Platform Camera', 'Uses Flutter camera plugin (iOS/Android only)'),
  opencv('OpenCV Camera', 'Uses opencv_dart for camera (recommended)');

  const CameraProvider(this.displayName, this.description);

  final String displayName;
  final String description;

  /// Returns available camera providers for the current platform
  /// - Platform camera: Only available on iOS and Android
  /// - OpenCV camera: Available on iOS, Android, macOS, Linux (not web)
  static List<CameraProvider> get availableProviders {
    if (UniversalPlatform.isWeb) {
      // Web: No camera support currently
      return [];
    }
    if (UniversalPlatform.isMacOS ||
        UniversalPlatform.isLinux ||
        UniversalPlatform.isWindows) {
      // Desktop: Only OpenCV camera works
      return [CameraProvider.opencv];
    }
    // Mobile (iOS/Android): Both available, but platform camera recommended
    // OpenCV VideoCapture has format issues on some Android devices
    return values.toList();
  }

  /// Returns the default camera provider for the current platform
  static CameraProvider get defaultProvider {
    if (UniversalPlatform.isWeb) {
      // Web has no camera support, but return platform as placeholder
      return CameraProvider.platform;
    }
    if (UniversalPlatform.isAndroid || UniversalPlatform.isIOS) {
      // Mobile: Use platform camera (Flutter camera plugin)
      // OpenCV VideoCapture doesn't support Android Camera2 API properly
      // See: https://github.com/rainyl/opencv_dart/discussions/33
      return CameraProvider.platform;
    }
    // Desktop (macOS, Linux, Windows): OpenCV VideoCapture works
    return CameraProvider.opencv;
  }

  /// Whether this provider is available on the current platform
  bool get isAvailable {
    if (UniversalPlatform.isWeb) {
      return false; // No camera on web
    }
    if (this == CameraProvider.platform) {
      // Platform camera only works on iOS and Android
      return UniversalPlatform.isIOS || UniversalPlatform.isAndroid;
    }
    // OpenCV works on all native platforms
    return true;
  }
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

  /// Returns available preprocessing providers for the current platform
  /// OpenCV is not available on web
  static List<PreprocessingProvider> get availableProviders {
    if (UniversalPlatform.isWeb) {
      return values.where((p) => p != PreprocessingProvider.opencv).toList();
    }
    return values.toList();
  }

  /// Whether this provider is available on the current platform
  bool get isAvailable {
    if (UniversalPlatform.isWeb && this == PreprocessingProvider.opencv) {
      return false;
    }
    return true;
  }
}

/// Base class for model-specific settings
/// Each model type can extend this to add their own configuration options
/// Only includes truly universal settings (performance overlay)
abstract class ModelSettings extends ChangeNotifier {
  ModelSettings({bool showPerformanceOverlay = true})
    : _showPerformanceOverlay = showPerformanceOverlay;

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
