import 'model_settings.dart';
import 'face_model_settings.dart';

/// Settings specific to YOLO-Face model
class YoloFaceModelSettings extends FaceModelSettings {
  YoloFaceModelSettings({
    super.showPerformanceOverlay,
    super.cameraProvider,
    super.preprocessingProvider,
    super.confidenceThreshold = 0.5,
    super.multiFaceMode = true,
    super.showBoundingBox = true,
    super.showLandmarks = true,
    double nmsThreshold = 0.45,
  }) : _nmsThreshold = nmsThreshold;

  /// NMS (Non-Maximum Suppression) threshold for overlapping detections
  double _nmsThreshold;
  double get nmsThreshold => _nmsThreshold;
  set nmsThreshold(double value) {
    if (_nmsThreshold != value) {
      _nmsThreshold = value.clamp(0.0, 1.0);
      notifyListeners();
    }
  }

  @override
  YoloFaceModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    bool? multiFaceMode,
    bool? showBoundingBox,
    bool? showLandmarks,
    double? nmsThreshold,
  }) {
    return YoloFaceModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? this.cameraProvider,
      preprocessingProvider: preprocessingProvider ?? this.preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      multiFaceMode: multiFaceMode ?? this.multiFaceMode,
      showBoundingBox: showBoundingBox ?? this.showBoundingBox,
      showLandmarks: showLandmarks ?? this.showLandmarks,
      nmsThreshold: nmsThreshold ?? _nmsThreshold,
    );
  }

  @override
  void reset() {
    super.reset();
    nmsThreshold = 0.45;
  }
}
