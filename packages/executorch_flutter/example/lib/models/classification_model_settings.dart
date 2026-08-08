import 'model_settings.dart';

/// Settings specific to Image Classification models (MobileNet, ResNet, etc.)
class ClassificationModelSettings extends ModelSettings {
  ClassificationModelSettings({
    super.showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    int topK = 5,
  }) : _cameraProvider = cameraProvider ?? _defaultCameraProvider,
       _preprocessingProvider =
           preprocessingProvider ?? _defaultPreprocessingProvider,
       _topK = topK;

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

  /// Number of top predictions to show (1-10)
  int _topK;
  int get topK => _topK;
  set topK(int value) {
    if (_topK != value) {
      _topK = value.clamp(1, 10);
      notifyListeners();
    }
  }

  @override
  ClassificationModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    int? topK,
  }) {
    return ClassificationModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? _cameraProvider,
      preprocessingProvider: preprocessingProvider ?? _preprocessingProvider,
      topK: topK ?? _topK,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    cameraProvider = _defaultCameraProvider;
    preprocessingProvider = _defaultPreprocessingProvider;
    topK = 5;
  }
}
