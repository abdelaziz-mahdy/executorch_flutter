import 'model_settings.dart';
import 'pose_model_settings.dart';

/// Settings specific to YOLO-Pose model
class YoloPoseModelSettings extends PoseModelSettings {
  YoloPoseModelSettings({
    super.showPerformanceOverlay,
    super.cameraProvider,
    super.preprocessingProvider,
    super.confidenceThreshold = 0.3,
    super.multiPersonMode = true,
    super.showSkeleton = true,
    super.showKeypoints = true,
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
  YoloPoseModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    bool? multiPersonMode,
    bool? showSkeleton,
    bool? showKeypoints,
    double? nmsThreshold,
  }) {
    return YoloPoseModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? this.cameraProvider,
      preprocessingProvider:
          preprocessingProvider ?? this.preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      multiPersonMode: multiPersonMode ?? this.multiPersonMode,
      showSkeleton: showSkeleton ?? this.showSkeleton,
      showKeypoints: showKeypoints ?? this.showKeypoints,
      nmsThreshold: nmsThreshold ?? _nmsThreshold,
    );
  }

  @override
  void reset() {
    super.reset();
    nmsThreshold = 0.45;
  }
}
