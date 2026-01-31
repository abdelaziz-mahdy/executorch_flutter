import 'model_settings.dart';

/// Base settings for pose detection models (MoveNet, YOLO-Pose)
class PoseModelSettings extends ModelSettings {
  PoseModelSettings({
    super.showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double confidenceThreshold = 0.3,
    bool multiPersonMode = false,
    bool showSkeleton = true,
    bool showKeypoints = true,
  }) : _cameraProvider = cameraProvider ?? _defaultCameraProvider,
       _preprocessingProvider =
           preprocessingProvider ?? _defaultPreprocessingProvider,
       _confidenceThreshold = confidenceThreshold,
       _multiPersonMode = multiPersonMode,
       _showSkeleton = showSkeleton,
       _showKeypoints = showKeypoints;

  /// Default camera provider based on platform
  static CameraProvider get _defaultCameraProvider {
    return CameraProvider.defaultProvider;
  }

  /// Default preprocessing provider
  static PreprocessingProvider get _defaultPreprocessingProvider {
    return PreprocessingProvider.gpu;
  }

  /// Camera provider selection (for live camera input)
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

  /// Minimum confidence threshold for keypoints (0.0 - 1.0)
  double _confidenceThreshold;
  double get confidenceThreshold => _confidenceThreshold;
  set confidenceThreshold(double value) {
    if (_confidenceThreshold != value) {
      _confidenceThreshold = value.clamp(0.0, 1.0);
      notifyListeners();
    }
  }

  /// Whether to detect multiple people or just the best one
  bool _multiPersonMode;
  bool get multiPersonMode => _multiPersonMode;
  set multiPersonMode(bool value) {
    if (_multiPersonMode != value) {
      _multiPersonMode = value;
      notifyListeners();
    }
  }

  /// Whether to draw skeleton lines between keypoints
  bool _showSkeleton;
  bool get showSkeleton => _showSkeleton;
  set showSkeleton(bool value) {
    if (_showSkeleton != value) {
      _showSkeleton = value;
      notifyListeners();
    }
  }

  /// Whether to draw keypoint dots
  bool _showKeypoints;
  bool get showKeypoints => _showKeypoints;
  set showKeypoints(bool value) {
    if (_showKeypoints != value) {
      _showKeypoints = value;
      notifyListeners();
    }
  }

  @override
  PoseModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    bool? multiPersonMode,
    bool? showSkeleton,
    bool? showKeypoints,
  }) {
    return PoseModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? _cameraProvider,
      preprocessingProvider: preprocessingProvider ?? _preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? _confidenceThreshold,
      multiPersonMode: multiPersonMode ?? _multiPersonMode,
      showSkeleton: showSkeleton ?? _showSkeleton,
      showKeypoints: showKeypoints ?? _showKeypoints,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    cameraProvider = _defaultCameraProvider;
    preprocessingProvider = _defaultPreprocessingProvider;
    confidenceThreshold = 0.3;
    multiPersonMode = false;
    showSkeleton = true;
    showKeypoints = true;
  }
}
