import 'model_settings.dart';
import 'face_model_settings.dart';

/// Settings specific to BlazeFace face detection model
class BlazeFaceModelSettings extends FaceModelSettings {
  BlazeFaceModelSettings({
    super.showPerformanceOverlay,
    super.cameraProvider,
    super.preprocessingProvider,
    super.confidenceThreshold = 0.5,
    super.multiFaceMode = true,
    super.showBoundingBox = true,
    super.showLandmarks = true,
  });

  @override
  BlazeFaceModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    bool? multiFaceMode,
    bool? showBoundingBox,
    bool? showLandmarks,
  }) {
    return BlazeFaceModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? this.cameraProvider,
      preprocessingProvider:
          preprocessingProvider ?? this.preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      multiFaceMode: multiFaceMode ?? this.multiFaceMode,
      showBoundingBox: showBoundingBox ?? this.showBoundingBox,
      showLandmarks: showLandmarks ?? this.showLandmarks,
    );
  }
}
