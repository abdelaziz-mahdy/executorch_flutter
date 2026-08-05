import 'model_settings.dart';
import 'pose_model_settings.dart';

/// MoveNet model variants
enum MoveNetVariant {
  /// Lightning variant - faster, less accurate, 192x192 input
  lightning,

  /// Thunder variant - slower, more accurate, 256x256 input
  thunder,
}

/// Display name extension for MoveNet variants
extension MoveNetVariantExtension on MoveNetVariant {
  String get displayName {
    switch (this) {
      case MoveNetVariant.lightning:
        return 'Lightning (Fast)';
      case MoveNetVariant.thunder:
        return 'Thunder (Accurate)';
    }
  }

  String get description {
    switch (this) {
      case MoveNetVariant.lightning:
        return '192x192 input, optimized for speed';
      case MoveNetVariant.thunder:
        return '256x256 input, optimized for accuracy';
    }
  }

  int get inputSize {
    switch (this) {
      case MoveNetVariant.lightning:
        return 192;
      case MoveNetVariant.thunder:
        return 256;
    }
  }
}

/// Settings specific to MoveNet pose detection model
class MoveNetModelSettings extends PoseModelSettings {
  MoveNetModelSettings({
    super.showPerformanceOverlay,
    super.cameraProvider,
    super.preprocessingProvider,
    super.confidenceThreshold = 0.3,
    super.multiPersonMode = false,
    super.showSkeleton = true,
    super.showKeypoints = true,
    MoveNetVariant modelVariant = MoveNetVariant.lightning,
  }) : _modelVariant = modelVariant;

  /// MoveNet model variant (lightning or thunder)
  MoveNetVariant _modelVariant;
  MoveNetVariant get modelVariant => _modelVariant;
  set modelVariant(MoveNetVariant value) {
    if (_modelVariant != value) {
      _modelVariant = value;
      notifyListeners();
    }
  }

  @override
  MoveNetModelSettings copyWith({
    bool? showPerformanceOverlay,
    CameraProvider? cameraProvider,
    PreprocessingProvider? preprocessingProvider,
    double? confidenceThreshold,
    bool? multiPersonMode,
    bool? showSkeleton,
    bool? showKeypoints,
    MoveNetVariant? modelVariant,
  }) {
    return MoveNetModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      cameraProvider: cameraProvider ?? this.cameraProvider,
      preprocessingProvider: preprocessingProvider ?? this.preprocessingProvider,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      multiPersonMode: multiPersonMode ?? this.multiPersonMode,
      showSkeleton: showSkeleton ?? this.showSkeleton,
      showKeypoints: showKeypoints ?? this.showKeypoints,
      modelVariant: modelVariant ?? _modelVariant,
    );
  }

  @override
  void reset() {
    super.reset();
    modelVariant = MoveNetVariant.lightning;
  }
}
