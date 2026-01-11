import 'model_settings.dart';

/// Settings specific to YOLO models
class YoloModelSettings extends ModelSettings {
  YoloModelSettings({
    super.showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double confidenceThreshold = 0.5,
    double nmsThreshold = 0.45,
  }) : _cameraProvider = cameraProvider ?? _defaultCameraProvider,
       _preprocessingProvider =
           preprocessingProvider ?? _defaultPreprocessingProvider,
       _confidenceThreshold = confidenceThreshold,
       _nmsThreshold = nmsThreshold;

  /// Default camera provider based on platform
  /// Uses the centralized platform-aware logic from CameraProvider
  static CameraProvider get _defaultCameraProvider {
    return CameraProvider.defaultProvider;
  }

  /// Default preprocessing provider based on platform
  /// GPU shader is fastest and works on all platforms
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

  /// Minimum confidence threshold for detections (0.0 - 1.0)
  double _confidenceThreshold;
  double get confidenceThreshold => _confidenceThreshold;
  set confidenceThreshold(double value) {
    if (_confidenceThreshold != value) {
      _confidenceThreshold = value.clamp(0.0, 1.0);
      notifyListeners();
    }
  }

  /// NMS (Non-Maximum Suppression) threshold (0.0 - 1.0)
  double _nmsThreshold;
  double get nmsThreshold => _nmsThreshold;
  set nmsThreshold(double value) {
    if (_nmsThreshold != value) {
      _nmsThreshold = value.clamp(0.0, 1.0);
      notifyListeners();
    }
  }

  @override
  YoloModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    double? nmsThreshold,
  }) {
    return YoloModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? _cameraProvider,
      preprocessingProvider: preprocessingProvider ?? _preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? _confidenceThreshold,
      nmsThreshold: nmsThreshold ?? _nmsThreshold,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    cameraProvider = _defaultCameraProvider;
    preprocessingProvider = _defaultPreprocessingProvider;
    confidenceThreshold = 0.5;
    nmsThreshold = 0.45;
  }
}
