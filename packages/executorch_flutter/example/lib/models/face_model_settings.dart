import 'model_settings.dart';

/// Base settings for face detection models (BlazeFace, YOLO-Face)
class FaceModelSettings extends ModelSettings {
  FaceModelSettings({
    super.showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double confidenceThreshold = 0.5,
    bool multiFaceMode = true,
    bool showBoundingBox = true,
    bool showLandmarks = true,
  }) : _cameraProvider = cameraProvider ?? _defaultCameraProvider,
       _preprocessingProvider =
           preprocessingProvider ?? _defaultPreprocessingProvider,
       _confidenceThreshold = confidenceThreshold,
       _multiFaceMode = multiFaceMode,
       _showBoundingBox = showBoundingBox,
       _showLandmarks = showLandmarks;

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

  /// Minimum confidence threshold for face detections (0.0 - 1.0)
  double _confidenceThreshold;
  double get confidenceThreshold => _confidenceThreshold;
  set confidenceThreshold(double value) {
    if (_confidenceThreshold != value) {
      _confidenceThreshold = value.clamp(0.0, 1.0);
      notifyListeners();
    }
  }

  /// Whether to detect multiple faces or just the best one
  bool _multiFaceMode;
  bool get multiFaceMode => _multiFaceMode;
  set multiFaceMode(bool value) {
    if (_multiFaceMode != value) {
      _multiFaceMode = value;
      notifyListeners();
    }
  }

  /// Whether to draw bounding boxes around faces
  bool _showBoundingBox;
  bool get showBoundingBox => _showBoundingBox;
  set showBoundingBox(bool value) {
    if (_showBoundingBox != value) {
      _showBoundingBox = value;
      notifyListeners();
    }
  }

  /// Whether to draw facial landmark dots
  bool _showLandmarks;
  bool get showLandmarks => _showLandmarks;
  set showLandmarks(bool value) {
    if (_showLandmarks != value) {
      _showLandmarks = value;
      notifyListeners();
    }
  }

  @override
  FaceModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    bool? multiFaceMode,
    bool? showBoundingBox,
    bool? showLandmarks,
  }) {
    return FaceModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? _cameraProvider,
      preprocessingProvider: preprocessingProvider ?? _preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? _confidenceThreshold,
      multiFaceMode: multiFaceMode ?? _multiFaceMode,
      showBoundingBox: showBoundingBox ?? _showBoundingBox,
      showLandmarks: showLandmarks ?? _showLandmarks,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    cameraProvider = _defaultCameraProvider;
    preprocessingProvider = _defaultPreprocessingProvider;
    confidenceThreshold = 0.5;
    multiFaceMode = true;
    showBoundingBox = true;
    showLandmarks = true;
  }
}
