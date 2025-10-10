import 'model_settings.dart';

/// Settings specific to YOLO models
class YoloModelSettings extends ModelSettings {
  YoloModelSettings({
    super.showPerformanceOverlay,
    super.cameraProvider,
    super.preprocessingProvider,
    double confidenceThreshold = 0.5,
    double nmsThreshold = 0.45,
  }) : _confidenceThreshold = confidenceThreshold,
       _nmsThreshold = nmsThreshold;

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
      showPerformanceOverlay: showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? this.cameraProvider,
      preprocessingProvider: preprocessingProvider ?? this.preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? _confidenceThreshold,
      nmsThreshold: nmsThreshold ?? _nmsThreshold,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    cameraProvider = CameraProvider.opencv;
    preprocessingProvider = PreprocessingProvider.opencv;
    confidenceThreshold = 0.5;
    nmsThreshold = 0.45;
  }
}
