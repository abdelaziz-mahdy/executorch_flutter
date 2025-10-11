import 'model_settings.dart';

/// Settings specific to Image Classification models (MobileNet, ResNet, etc.)
class ClassificationModelSettings extends ModelSettings {
  ClassificationModelSettings({
    super.showPerformanceOverlay,
    super.cameraProvider,
    super.preprocessingProvider,
    int topK = 5,
  }) : _topK = topK;

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
      cameraProvider: cameraProvider ?? this.cameraProvider,
      preprocessingProvider:
          preprocessingProvider ?? this.preprocessingProvider,
      topK: topK ?? _topK,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    cameraProvider = CameraProvider.opencv;
    preprocessingProvider = PreprocessingProvider.opencv;
    topK = 5;
  }
}
