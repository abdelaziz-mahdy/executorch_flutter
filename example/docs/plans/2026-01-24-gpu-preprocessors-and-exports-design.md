# GPU Preprocessors & Model Export Scripts Design

**Date**: 2026-01-24
**Status**: Approved
**Author**: Claude + User

## Overview

Add GPU and OpenCV preprocessors for MoveNet and BlazeFace models, restructure existing imageLib preprocessors for consistency, and add Python export scripts for all new model types.

## Goals

1. Complete preprocessing provider support (GPU, OpenCV, imageLib) for MoveNet and BlazeFace
2. Unify preprocessor directory structure across all models
3. Add Python export scripts for MoveNet, BlazeFace, YOLO-Pose, YOLO-Face

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Preprocessor approach | Dedicated per model | BlazeFace needs [-1,1] normalization, NHWC format differs from YOLO/MobileNet |
| OpenCV support | Full implementation | Consistency - users expect all providers for all models |
| ImageLib structure | Move to `imagelib/` dir | Match GPU and OpenCV directory patterns |
| Model sources | TF Hub (MoveNet), MediaPipe (BlazeFace), Ultralytics (YOLO variants) | Original authoritative sources |

## Model Specifications

| Model | Input Size | Format | Normalization | Notes |
|-------|-----------|--------|---------------|-------|
| MoveNet Lightning | 192x192 | NHWC | [0, 1] | Simple resize |
| MoveNet Thunder | 256x256 | NHWC | [0, 1] | Simple resize |
| BlazeFace | 128x128 | NHWC | [-1, 1] | Simple resize |
| YOLO-Pose | 640x640 | NCHW | [0, 1] | Letterbox resize |
| YOLO-Face | 640x640 | NCHW | [0, 1] | Letterbox resize |

## Architecture

### Unified Preprocessor Structure

```
processors/
├── shaders/              # All GPU preprocessors
│   ├── gpu_yolo_preprocessor.dart
│   ├── gpu_mobilenet_preprocessor.dart
│   ├── gpu_movenet_preprocessor.dart      # NEW
│   └── gpu_blazeface_preprocessor.dart    # NEW
│
├── opencv/               # All OpenCV preprocessors + stubs
│   ├── opencv_yolo_preprocessor.dart
│   ├── opencv_yolo_preprocessor_stub.dart
│   ├── opencv_mobilenet_preprocessor.dart
│   ├── opencv_mobilenet_preprocessor_stub.dart
│   ├── opencv_movenet_preprocessor.dart       # NEW
│   ├── opencv_movenet_preprocessor_stub.dart  # NEW
│   ├── opencv_blazeface_preprocessor.dart     # NEW
│   └── opencv_blazeface_preprocessor_stub.dart # NEW
│
├── imagelib/             # All imageLib preprocessors
│   ├── imagelib_yolo_preprocessor.dart      # MOVED from yolo_processor.dart
│   ├── imagelib_mobilenet_preprocessor.dart # MOVED from mobilenet_processor.dart
│   ├── imagelib_movenet_preprocessor.dart   # NEW
│   └── imagelib_blazeface_preprocessor.dart # NEW
│
├── yolo_input_processor.dart      # routing only
├── mobilenet_input_processor.dart # routing only (UPDATE)
├── movenet_input_processor.dart   # routing only (UPDATE)
└── blazeface_input_processor.dart # routing only (UPDATE)
```

### Shader Files

```
example/shaders/
├── yolo_preprocess.frag
├── mobilenet_preprocess.frag
├── movenet_preprocess.frag    # NEW
└── blazeface_preprocess.frag  # NEW
```

## Component Details

### 1. GPU Shaders

**`shaders/movenet_preprocess.frag`**
- Simple resize to target size (192x192 or 256x256)
- Normalization to [0, 1] range
- No letterbox padding needed

**`shaders/blazeface_preprocess.frag`**
- Simple resize to 128x128
- Normalization to [-1, 1] range: `(pixel / 127.5) - 1.0`
- No padding needed

### 2. GPU Preprocessors (Dart)

**`gpu_movenet_preprocessor.dart`**
```dart
class GpuMoveNetPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  // Loads movenet_preprocess.frag
  // Output: NHWC tensor [1, 192, 192, 3]
  // Normalization: [0, 1]
}
```

**`gpu_blazeface_preprocessor.dart`**
```dart
class GpuBlazeFacePreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  // Loads blazeface_preprocess.frag
  // Output: NHWC tensor [1, 128, 128, 3]
  // Normalization: [-1, 1] (done in shader)
}
```

### 3. Tensor Format Difference

```dart
// NCHW (YOLO/MobileNet): channels separated
floats[i] = r;                    // All R values first
floats[i + totalPixels] = g;      // Then all G
floats[i + totalPixels * 2] = b;  // Then all B

// NHWC (MoveNet/BlazeFace): channels interleaved per pixel
floats[i * 3] = r;
floats[i * 3 + 1] = g;
floats[i * 3 + 2] = b;
```

### 4. OpenCV Preprocessors

**`opencv_movenet_preprocessor.dart`**
- `cv.imdecodeAsync()` → `cv.cvtColorAsync()` → `cv.resizeAsync()`
- Normalization: `mat.convertTo(alpha: 1.0/255.0)`
- Output: NHWC tensor [1, 192, 192, 3]

**`opencv_blazeface_preprocessor.dart`**
- Same flow as MoveNet
- Normalization: `mat.convertTo(alpha: 1.0/127.5, beta: -1.0)`
- Output: NHWC tensor [1, 128, 128, 3]

**Web stubs**: Throw `UnsupportedError` for web platform compatibility.

### 5. ImageLib Preprocessors

Extract existing CPU preprocessing logic into dedicated files:
- Move `YoloPreprocessor` from `yolo_processor.dart` → `imagelib/imagelib_yolo_preprocessor.dart`
- Move MobileNet preprocessor → `imagelib/imagelib_mobilenet_preprocessor.dart`
- Create new `imagelib_movenet_preprocessor.dart`
- Create new `imagelib_blazeface_preprocessor.dart`

### 6. Input Processor Updates

All input processors become simple routing:

```dart
switch (preprocessingProvider) {
  case PreprocessingProvider.gpu:
    return GpuMoveNetPreprocessor(config: config).preprocess(bytes);
  case PreprocessingProvider.opencv:
    return OpenCVMoveNetPreprocessor(config: config).preprocess(bytes);
  case PreprocessingProvider.imageLib:
    return ImageLibMoveNetPreprocessor(config: config).preprocess(bytes);
}
```

### 7. Python Export Scripts

Add to `models/python/main.py`:

**`export_movenet()`**
- Source: TensorFlow Hub (`movenet/singlepose/lightning` or `thunder`)
- Flow: TF SavedModel → ONNX → PyTorch → ExecuTorch
- Output: `movenet/movenet_lightning_xnnpack.pte`

**`export_blazeface()`**
- Source: MediaPipe BlazeFace model
- Flow: TFLite → ONNX → PyTorch → ExecuTorch
- Output: `blazeface/blazeface_xnnpack.pte`

**`export_yolo_pose()`**
- Source: Ultralytics `yolo11n-pose.pt`
- Output: `yolo-pose/yolo11n-pose_xnnpack.pte`

**`export_yolo_face()`**
- Source: Ultralytics YOLO-Face community model
- Output: `yolo-face/yolo11n-face_xnnpack.pte`

**CLI additions:**
```bash
python main.py export --movenet
python main.py export --blazeface
python main.py export --yolo-pose yolo11n-pose
python main.py export --yolo-face yolo11n-face
```

### 8. Index.json Updates

Add new categories to `generate_index_json()`:

```python
model_configs = {
    "movenet": {
        "inputSize": 192,
        "labelsFile": None,
        "labelsUrl": None,
    },
    "blazeface": {
        "inputSize": 128,
        "labelsFile": None,
        "labelsUrl": None,
    },
    "yolo-pose": {
        "inputSize": 640,
        "labelsFile": None,
        "labelsUrl": None,
    },
    "yolo-face": {
        "inputSize": 640,
        "labelsFile": None,
        "labelsUrl": None,
    },
}
```

## Implementation Phases

| Phase | Tasks | Dependencies | Parallelizable |
|-------|-------|--------------|----------------|
| A | Shaders + GPU preprocessors (MoveNet, BlazeFace) | None | Yes |
| B | OpenCV preprocessors + stubs (MoveNet, BlazeFace) | None | Yes |
| C | ImageLib restructure (move existing, create new) | None | Yes |
| D | Update all input processors (routing + imports) | A, B, C | No |
| E | Python export scripts (4 functions + CLI) | None | Yes |
| F | Update pubspec.yaml + flutter analyze | A, D | No |

**Phases A, B, C, E can run in parallel using sub-agents.**

## File Counts

| Category | New Files | Moved/Updated |
|----------|-----------|---------------|
| Shaders (.frag) | 2 | 0 |
| GPU preprocessors | 2 | 0 |
| OpenCV preprocessors | 4 | 0 |
| OpenCV stubs | 2 | 0 |
| ImageLib preprocessors | 2 | 2 (move) |
| Input processors | 0 | 4 (update) |
| Python exports | 0 | 1 (add functions) |
| pubspec.yaml | 0 | 1 (register shaders) |
| **Total** | **12** | **8** |

## Testing

1. Run `flutter analyze lib/` after implementation
2. Test each preprocessing provider on each model type
3. Verify tensor shapes match expected formats
4. Test on multiple platforms (mobile for OpenCV, web for GPU/imageLib only)

## Future Considerations

- Add benchmark comparisons between providers
- Consider shared base classes if patterns stabilize
- GPU shader compilation caching for faster startup
