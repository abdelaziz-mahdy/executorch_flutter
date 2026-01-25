# Pose & Face Detection Models Design

**Date**: 2026-01-24
**Status**: Approved
**Author**: Design session with user

## Overview

Adding four new model implementations to the executorch_flutter example app:

| Category | Models |
|----------|--------|
| **Pose Detection** | MoveNet, YOLO-Pose |
| **Face Detection** | BlazeFace, YOLO-Face |

### Key Features

- Multi-person/multi-face detection with settings toggle
- Live camera and static image input support
- Simple dots + lines visualization
- CPU + GPU preprocessing toggle via settings

---

## Result Types

### Pose Detection

```dart
// Shared by MoveNet & YOLO-Pose
class PoseKeypoint {
  final String name;        // e.g., "nose", "left_shoulder"
  final double x, y;        // Normalized 0-1 coordinates
  final double confidence;
}

class DetectedPose {
  final List<PoseKeypoint> keypoints;  // 17 keypoints
  final double confidence;              // Overall pose confidence
  final BoundingBox? boundingBox;       // Optional, YOLO-Pose provides this
}

class PoseDetectionResult {
  final List<DetectedPose> poses;       // Multiple poses supported
}
```

### Face Detection

```dart
// Shared by BlazeFace & YOLO-Face
class FaceLandmark {
  final String name;        // e.g., "left_eye", "nose_tip"
  final double x, y;
  final double confidence;
}

class DetectedFace {
  final BoundingBox boundingBox;
  final List<FaceLandmark> landmarks;   // 5-6 landmarks
  final double confidence;
}

class FaceDetectionResult {
  final List<DetectedFace> faces;
}
```

---

## Settings Classes

### Pose Detection Settings

```dart
class PoseModelSettings extends ModelSettings {
  double confidenceThreshold;      // 0.0-1.0, default 0.3
  bool multiPersonMode;            // true = detect all, false = single best
  bool showSkeleton;               // Draw connecting lines
  bool showKeypoints;              // Draw keypoint dots
  CameraProvider cameraProvider;
  PreprocessingProvider preprocessingProvider;
}

// Model-specific extensions
class MoveNetModelSettings extends PoseModelSettings {
  MoveNetVariant modelVariant;     // lightning (192) or thunder (256)
}

class YoloPoseModelSettings extends PoseModelSettings {
  double nmsThreshold;             // Non-max suppression threshold
}
```

### Face Detection Settings

```dart
class FaceModelSettings extends ModelSettings {
  double confidenceThreshold;      // 0.0-1.0, default 0.5
  bool multiFaceMode;              // true = detect all, false = single best
  bool showBoundingBox;            // Draw face box
  bool showLandmarks;              // Draw landmark dots
  CameraProvider cameraProvider;
  PreprocessingProvider preprocessingProvider;
}

// Model-specific extensions
class BlazeFaceModelSettings extends FaceModelSettings {
  // Minimal additions - uses base settings
}

class YoloFaceModelSettings extends FaceModelSettings {
  double nmsThreshold;
}
```

---

## Processors

### Pose Detection

| Model | Input Processor | Output Processor |
|-------|-----------------|------------------|
| MoveNet | Resize to 192x192 or 256x256, normalize 0-1 | Extract 17 keypoints from [1,17,3] tensor |
| YOLO-Pose | Letterbox to 640x640 (reuse YOLO preprocessing) | Extract boxes + keypoints, apply NMS |

### Face Detection

| Model | Input Processor | Output Processor |
|-------|-----------------|------------------|
| BlazeFace | Resize to 128x128, normalize -1 to 1 | Decode anchor boxes, extract 6 landmarks |
| YOLO-Face | Letterbox to 640x640 (reuse YOLO) | Extract boxes + 5 landmarks, apply NMS |

### GPU Shaders

- `movenet_preprocess.frag` - Simple resize + normalize to [0,1]
- `blazeface_preprocess.frag` - Resize + normalize to [-1,1]
- YOLO-Pose/Face reuse existing `yolo_preprocess.frag`

---

## Renderers

### Pose Detection Renderer

- Draws original image as background
- For each detected pose:
  - Keypoint dots (colored circles)
  - Skeleton lines connecting joints (if showSkeleton enabled)

**Skeleton Color Scheme:**

| Body Part | Color |
|-----------|-------|
| Face (nose, eyes, ears) | Cyan |
| Arms (shoulders, elbows, wrists) | Green |
| Torso (shoulder-hip connections) | Yellow |
| Legs (hips, knees, ankles) | Magenta |

### Face Detection Renderer

- Draws original image as background
- For each detected face:
  - Bounding box (if showBoundingBox enabled)
  - Landmark dots (if showLandmarks enabled)
  - Confidence label

**Landmark Colors:**

| Landmark | Color |
|----------|-------|
| Eyes | Blue |
| Nose | Green |
| Mouth corners | Red |
| Ears (BlazeFace only) | Yellow |

---

## Model Definitions

```dart
class MoveNetModelDefinition
    extends ModelDefinition<ModelInput, PoseDetectionResult> {
  final String name = 'movenet';
  final String displayName = 'MoveNet Pose';
  final int inputSize = 256;  // or 192 for lightning
  final IconData icon = Icons.accessibility_new;
}

class YoloPoseModelDefinition
    extends ModelDefinition<ModelInput, PoseDetectionResult> {
  final String name = 'yolo_pose';
  final String displayName = 'YOLO Pose';
  final int inputSize = 640;
  final IconData icon = Icons.directions_run;
}

class BlazeFaceModelDefinition
    extends ModelDefinition<ModelInput, FaceDetectionResult> {
  final String name = 'blazeface';
  final String displayName = 'BlazeFace';
  final int inputSize = 128;
  final IconData icon = Icons.face;
}

class YoloFaceModelDefinition
    extends ModelDefinition<ModelInput, FaceDetectionResult> {
  final String name = 'yolo_face';
  final String displayName = 'YOLO Face';
  final int inputSize = 640;
  final IconData icon = Icons.face_retouching_natural;
}
```

---

## Keypoint & Landmark Definitions

### Pose Keypoints (COCO 17-point format)

```dart
enum PoseKeypointType {
  nose,           // 0
  leftEye,        // 1
  rightEye,       // 2
  leftEar,        // 3
  rightEar,       // 4
  leftShoulder,   // 5
  rightShoulder,  // 6
  leftElbow,      // 7
  rightElbow,     // 8
  leftWrist,      // 9
  rightWrist,     // 10
  leftHip,        // 11
  rightHip,       // 12
  leftKnee,       // 13
  rightKnee,      // 14
  leftAnkle,      // 15
  rightAnkle,     // 16
}
```

### Skeleton Connections

```dart
const skeletonConnections = [
  // Face
  (leftEye, nose), (rightEye, nose),
  (leftEye, leftEar), (rightEye, rightEar),
  // Arms
  (leftShoulder, leftElbow), (leftElbow, leftWrist),
  (rightShoulder, rightElbow), (rightElbow, rightWrist),
  // Torso
  (leftShoulder, rightShoulder),
  (leftShoulder, leftHip), (rightShoulder, rightHip),
  (leftHip, rightHip),
  // Legs
  (leftHip, leftKnee), (leftKnee, leftAnkle),
  (rightHip, rightKnee), (rightKnee, rightAnkle),
];
```

### Face Landmarks

```dart
// BlazeFace (6 landmarks)
enum BlazeFaceLandmarkType {
  rightEye, leftEye, noseTip, mouthCenter, rightEar, leftEar
}

// YOLO-Face (5 landmarks)
enum YoloFaceLandmarkType {
  leftEye, rightEye, noseTip, leftMouth, rightMouth
}
```

---

## File Structure

### New Files to Create

```
example/lib/
├── models/
│   ├── pose_result.dart
│   ├── face_result.dart
│   ├── pose_model_settings.dart
│   ├── face_model_settings.dart
│   ├── movenet_model_definition.dart
│   ├── movenet_model_settings.dart
│   ├── yolo_pose_model_definition.dart
│   ├── yolo_pose_model_settings.dart
│   ├── blazeface_model_definition.dart
│   ├── blazeface_model_settings.dart
│   ├── yolo_face_model_definition.dart
│   └── yolo_face_model_settings.dart
│
├── processors/
│   ├── movenet_input_processor.dart
│   ├── movenet_output_processor.dart
│   ├── yolo_pose_input_processor.dart
│   ├── yolo_pose_output_processor.dart
│   ├── blazeface_input_processor.dart
│   ├── blazeface_output_processor.dart
│   ├── yolo_face_input_processor.dart
│   └── yolo_face_output_processor.dart
│
├── renderers/screens/
│   ├── pose_detection_renderer.dart
│   └── face_detection_renderer.dart
│
└── shaders/
    ├── movenet_preprocess.frag
    └── blazeface_preprocess.frag

models/python/
├── export_movenet.py
├── export_yolo_pose.py
├── export_blazeface.py
└── export_yolo_face.py
```

### Files to Modify

- `example/lib/models/model_registry.dart` - Add new model definitions
- `example/pubspec.yaml` - Add new shader assets
- `models/python/main.py` - Add export commands

---

## Model Export

### Model Sources

| Model | Source | Input Size | Output Format |
|-------|--------|------------|---------------|
| MoveNet Lightning | TensorFlow Hub → ONNX → ExecuTorch | 192x192 | [1,1,17,3] |
| MoveNet Thunder | TensorFlow Hub → ONNX → ExecuTorch | 256x256 | [1,1,17,3] |
| YOLO-Pose | Ultralytics (yolo11n-pose) | 640x640 | [1,56,8400] |
| BlazeFace | MediaPipe → ONNX → ExecuTorch | 128x128 | Anchors + landmarks |
| YOLO-Face | Ultralytics community | 640x640 | [1,20,8400] |

### Backend Support

All models exported with:
- XNNPACK (all platforms)
- CoreML (iOS/macOS)
- MPS (macOS)
- Vulkan (Android/Windows/Linux)

---

## Implementation Order

| Phase | Tasks |
|-------|-------|
| **Phase 1** | Shared result types (`pose_result.dart`, `face_result.dart`) |
| **Phase 2** | Shared renderers (`pose_detection_renderer.dart`, `face_detection_renderer.dart`) |
| **Phase 3** | MoveNet (settings, processors, definition) |
| **Phase 4** | BlazeFace (settings, processors, definition) |
| **Phase 5** | YOLO-Pose (settings, processors, definition) |
| **Phase 6** | YOLO-Face (settings, processors, definition) |
| **Phase 7** | GPU shaders for MoveNet/BlazeFace |
| **Phase 8** | Python export scripts |

### Testing Milestones

1. After Phase 3: MoveNet working with static images
2. After Phase 4: BlazeFace working with static images
3. After Phase 5: YOLO-Pose with multi-person detection
4. After Phase 6: YOLO-Face with multi-face detection
5. After Phase 7: Camera mode working smoothly

---

## Summary

This design adds comprehensive pose and face detection capabilities to the executorch_flutter example app, following the existing Strategy Pattern architecture. The implementation reuses existing YOLO preprocessing where possible and provides a consistent user experience across all four new models.
