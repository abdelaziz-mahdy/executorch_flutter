# ExecuTorch Flutter Plugin - AI Agent Context

## Package Overview

**executorch_flutter** is a Flutter plugin package that provides on-device machine learning inference using PyTorch ExecuTorch. It enables Flutter developers to run optimized ML models on mobile and desktop platforms with a simple, type-safe Dart API.

**Package Name**: `executorch_flutter`
**Version**: 0.0.1 (pre-release)
**License**: MIT
**Platforms**: Android, iOS, macOS

## Current Development Status

- **Phase**: FFI migration complete, all platform channel code removed
- **Architecture**: Pure FFI with C/C++ wrapper, NativeFinalizer for cleanup
- **API**: Minimal surface with only `load()` and `forward()` - asset bundle loading supported
- **Code Quality**: 0 lint errors in `lib/`, all dart fixes applied
- **Build Status**: ✅ Android APK, ✅ macOS app, ✅ iOS (device only)
- **Next Step**: Test FFI on devices, then publish to pub.dev

## Core Architecture

### Technology Stack

- **Flutter Plugin**: FFI-based architecture with C/C++ wrapper
- **Platform Communication**: Dart FFI (Foreign Function Interface) - zero-overhead native interop
- **Type Generation**: Pigeon v26.0.1 for TensorData types only (no platform channels)
- **Native Wrapper**: C++ wrapper using ExecuTorch Module API
- **Android**: C++ native library (libexecutorch_flutter.so) + ExecuTorch AAR 1.0.0-rc2
- **iOS/macOS**: C++ compiled into framework + ExecuTorch XCFrameworks (SPM 1.0.0)
- **Memory Management**: NativeFinalizer for automatic cleanup + explicit dispose()

### Design Principles

1. **Minimal API Surface**: Just `load()`, `forward()`, and `dispose()` - nothing more
2. **FFI First**: Direct C interop via Dart FFI (zero-overhead, no platform channels)
3. **Type Safety**: Pigeon-generated TensorData types, ffigen-generated C bindings
4. **Async/Await**: All model operations are non-blocking via Isolate.run()
5. **Automatic Cleanup**: NativeFinalizer ensures models are freed on garbage collection
6. **User-Controlled Resources**: Developers can explicitly dispose() for immediate cleanup
7. **Structured Errors**: Exception hierarchy with clear error categories
8. **Platform Parity**: Identical behavior across Android, iOS, and macOS
9. **Asset-First**: Models loaded from `Uint8List` bytes, enabling Flutter asset bundle loading

## Platform Support

### Android
- **Minimum SDK**: API 23 (Android 6.0)
- **Architectures**: arm64-v8a (primary), x86_64 (emulator)
- **Dependencies**:
  - ExecuTorch AAR 1.0.0-rc2 (`org.pytorch:executorch-android:1.0.0-rc2`)
  - Available at: https://repo.maven.apache.org/maven2/org/pytorch/executorch-android/
- **Native Library**: libexecutorch_flutter.so (C++ wrapper compiled via CMake)
- **Implementation**:
  - C wrapper: `src/c_wrapper/executorch_flutter_wrapper.cpp`
  - Plugin stub: `android/src/main/kotlin/com/zcreations/executorch_flutter/ExecutorchFlutterPlugin.kt` (minimal, FFI only)

### iOS
- **Minimum Version**: iOS 13.0
- **Architectures**: arm64 (physical devices only, no simulator support)
- **Dependencies**: ExecuTorch XCFrameworks via Swift Package Manager (SPM 1.0.0)
  - Branch: `swiftpm-1.0.0` from https://github.com/pytorch/executorch.git
- **Native Framework**: C++ wrapper compiled into framework via CocoaPods
- **Implementation**:
  - C wrapper: `src/c_wrapper/executorch_flutter_wrapper.cpp`
  - Plugin stub: `darwin/Sources/executorch_flutter/ExecutorchFlutterPlugin.swift` (minimal, FFI only)
- **Note**: Simulator support requires x86_64 ExecuTorch builds (not currently available)

### macOS
- **Minimum Version**: macOS 11.0
- **Architectures**: arm64 (Apple Silicon only)
- **Dependencies**: ExecuTorch XCFrameworks via Swift Package Manager (SPM 1.0.0)
  - Branch: `swiftpm-1.0.0` from https://github.com/pytorch/executorch.git
- **Native Framework**: C++ wrapper compiled into framework via CocoaPods
- **Implementation**:
  - C wrapper: `src/c_wrapper/executorch_flutter_wrapper.cpp` (shared with iOS)
  - Plugin stub: `darwin/Sources/executorch_flutter/ExecutorchFlutterPlugin.swift` (minimal, FFI only)
  - Platform-specific: Conditional compilation using `#if os(iOS)` / `#if os(macOS)`

## Project Structure

```
executorch_flutter/
├── lib/
│   ├── executorch_flutter.dart              # Main library export
│   └── src/
│       ├── executorch_model.dart            # ExecuTorchModel - main FFI API (load/forward/dispose)
│       ├── executorch_errors.dart           # Exception hierarchy
│       ├── ffi/
│       │   ├── executorch_ffi_bridge.dart   # Central FFI bridge (loads native library)
│       │   ├── tensor_conversion.dart       # Dart ↔ C tensor conversion
│       │   └── error_conversion.dart        # C error codes → Dart exceptions
│       ├── processors/
│       │   ├── base_processor.dart          # BaseInputProcessor/BaseOutputProcessor
│       │   └── processors.dart              # Processor utilities
│       └── generated/
│           ├── executorch_api.dart          # Pigeon-generated types (TensorData/TensorType)
│           └── executorch_ffi_bindings.dart # ffigen-generated C bindings (793 lines)
├── src/
│   └── c_wrapper/
│       ├── executorch_flutter_wrapper.h     # C API header (exported functions)
│       └── executorch_flutter_wrapper.cpp   # C++ implementation (ExecuTorch Module wrapper)
├── android/
│   ├── src/main/
│   │   ├── kotlin/com/zcreations/executorch_flutter/
│   │   │   └── ExecutorchFlutterPlugin.kt   # Plugin stub (FFI only, no logic)
│   │   └── cpp/                             # Symlink to ../../src/c_wrapper/
│   └── CMakeLists.txt                       # NDK build config (compiles C++ wrapper)
├── darwin/
│   └── Sources/executorch_flutter/
│       ├── ExecutorchFlutterPlugin.swift    # Plugin stub (FFI only, no logic)
│       └── Generated/
│           └── ExecutorchApi.swift          # Pigeon-generated types (not used for communication)
├── macos/
│   └── Classes/                             # Symlinks to darwin/
├── pigeons/
│   └── executorch_api.dart                  # Pigeon type definitions (TensorData/TensorType only)
├── example/
│   ├── lib/
│   │   ├── main.dart                        # Example app entry
│   │   ├── screens/                         # Demo screens
│   │   ├── processors/                      # Reference processors
│   │   └── services/                        # Model management
│   └── assets/
│       ├── models/                          # .pte files (gitignored)
│       └── images/                          # Test images
├── scripts/
│   └── generate_pigeon.sh                   # Regenerate Pigeon types (not platform channels)
└── python/
    └── convert_model.py                     # PyTorch → ExecuTorch converter
```

**Key Changes from Platform Channel Architecture**:
- ❌ Removed: Kotlin/Swift platform channel implementations (ExecutorchModelManager, TensorUtils)
- ❌ Removed: Pigeon platform channel interfaces (ExecutorchHostApi)
- ✅ Added: C/C++ wrapper with ExecuTorch Module API
- ✅ Added: Dart FFI bridge with Isolate.run() for async
- ✅ Added: NativeFinalizer for automatic model cleanup
- ✅ Kept: Pigeon for TensorData type generation only

## Key APIs

### ExecuTorchModel

The primary API for loading models and running inference. Simple, minimal, and direct.

```dart
// Load a model from asset bundle
import 'package:flutter/services.dart' show rootBundle;

final modelBytes = await rootBundle.load('assets/models/model.pte');
final model = await ExecuTorchModel.load(
  modelBytes.buffer.asUint8List(),
);

// Or load from file path (if model is stored externally)
final model = await ExecuTorchModel.load(
  File('/path/to/model.pte').readAsBytesSync(),
);

// Model properties
print(model.modelId);        // Unique identifier (auto-generated)
print(model.inputShapes);    // Expected input tensor shapes
print(model.outputShapes);   // Expected output tensor shapes

// Run inference
final outputs = await model.forward([tensorData]);

// Dispose when done
await model.dispose();
```

**Key Design Points**:
- **No file paths**: Models are loaded from `Uint8List` bytes, enabling asset bundle loading
- **No options/timeouts**: Simplified API with just input tensors
- **Direct outputs**: Returns `List<TensorData>` directly (no wrapper object)
- **Asset-first**: Recommended pattern is to bundle models in `assets/` and load via `rootBundle`

### TensorData (Pigeon-generated)

Input/output tensor representation:

```dart
final tensor = TensorData(
  shape: [1, 3, 224, 224],           // [batch, channels, height, width]
  dataType: TensorType.float32,      // float32, int32, int8, uint8
  data: Uint8List(...),              // Raw bytes
  name: 'input_0',                   // Optional name
);
```

### Model Loading Pattern

**Recommended: Load from Asset Bundle**

```dart
import 'package:flutter/services.dart' show rootBundle;

// 1. Add model to pubspec.yaml assets:
//    flutter:
//      assets:
//        - assets/models/

// 2. Load model bytes from asset bundle
final modelBytes = await rootBundle.load('assets/models/model.pte');

// 3. Create model instance
final model = await ExecuTorchModel.load(
  modelBytes.buffer.asUint8List(),
);

// 4. Run inference
final outputs = await model.forward([inputTensor]);

// 5. Clean up
await model.dispose();
```

**Alternative: Load from File System**

```dart
import 'dart:io';

// Load from downloaded/cached file
final modelFile = File('/path/to/downloaded/model.pte');
final model = await ExecuTorchModel.load(
  modelFile.readAsBytesSync(),
);
```

### Exception Hierarchy

```dart
ExecuTorchException              // Base exception
├── ExecuTorchModelException     // Model loading/lifecycle errors
├── ExecuTorchInferenceException // Inference execution errors
├── ExecuTorchValidationException // Tensor validation errors
├── ExecuTorchMemoryException    // Memory/resource errors
├── ExecuTorchIOException        // File I/O errors
└── ExecuTorchPlatformException  // Platform communication errors
```

## Memory Management Philosophy

**User-Controlled Lifecycle**: This package does NOT automatically manage model memory. Developers must explicitly:

1. **Load models**: Call `ExecuTorchModel.load()` when needed
2. **Dispose models**: Call `model.dispose()` when done
3. **Handle errors**: Catch exceptions and clean up resources
4. **Monitor memory**: Use OS tools to track memory usage

**No Automatic Cleanup**: There is no singleton manager, no lifecycle manager, no memory pressure monitoring, no automatic disposal. This design gives developers full control over when models are loaded/unloaded.

**Why User-Controlled?**
- Predictable behavior (no surprise disposals mid-inference)
- Explicit resource management (developers know when models are in memory)
- Platform parity (same behavior on Android, iOS, macOS)
- Simple API (just `load()` and `dispose()`, no manager required)

## Pigeon Code Generation

### Automated Script

The package includes an automated Pigeon generation script that handles all code generation and post-processing:

```bash
./scripts/generate_pigeon.sh
```

**What it does**:
1. Runs `dart pub global run pigeon --input pigeons/executorch_api.dart`
2. Automatically makes Swift types `public` (required for SPM)
3. Makes `PigeonError` class and initializer public
4. Creates symlinks for iOS and macOS to shared darwin code

**Script Features**:
- ✅ Generates code for all platforms (Dart, Kotlin, Swift)
- ✅ Auto-fixes Swift visibility for SPM compatibility
- ✅ Keeps PigeonError for proper Swift error handling
- ✅ Creates platform symlinks automatically
- ✅ Color-coded output for easy debugging

### Manual Workflow (if needed)

1. **Edit interface**: Modify `pigeons/executorch_api.dart`
2. **Generate code**: Run `./scripts/generate_pigeon.sh` (recommended) or `dart pub global run pigeon --input pigeons/executorch_api.dart`
3. **Implement native**: Add implementations in Kotlin/Swift
4. **Test**: Run integration tests (see below)

## Integration Testing

### Automated Test Script

The example app includes an automated integration test runner that tests all platforms:

```bash
cd example
./scripts/run_integration_tests.sh           # Run tests on all platforms (default)
./scripts/run_integration_tests.sh macos     # Run tests only on macOS
./scripts/run_integration_tests.sh ios       # Run tests only on iOS
./scripts/run_integration_tests.sh android   # Run tests only on Android
```

**What it does**:
1. Checks for required model files (MobileNet, YOLO variants)
2. Runs integration tests on available platforms:
   - **macOS**: Tests on macOS device
   - **iOS**: Tests on physical device (arm64 only, no simulator)
   - **Android**: Tests on emulator or physical device (auto-launches emulator if needed)
3. Falls back to building if no device/simulator is available
4. Provides detailed summary of test results

**Script Features**:
- ✅ Multi-platform support (macOS, iOS, Android)
- ✅ Auto-detects and launches Android emulator
- ✅ Validates model files before testing
- ✅ Fallback to build if no device available
- ✅ Color-coded output with test summary
- ✅ Exit codes for CI/CD integration

**Prerequisites**:
- Models must be in `example/assets/models/`:
  - `mobilenet_v3_small_xnnpack.pte`
  - `yolo11n_xnnpack.pte`
  - `yolov5n_xnnpack.pte`
  - `yolov8n_xnnpack.pte`
- Run model setup: `cd python && python3 setup_models.py`

### Generated Files

- `lib/src/generated/executorch_api.dart` (Dart)
- `android/src/main/kotlin/com/zcreations/executorch_flutter/generated/ExecutorchApi.kt` (Kotlin)
- `darwin/Sources/executorch_flutter/Generated/ExecutorchApi.swift` (Shared Darwin)
- `ios/Classes/Generated/ExecutorchApi.swift` → symlink to darwin
- `macos/Classes/Generated/ExecutorchApi.swift` → symlink to darwin

**Important**: Generated files ARE committed to version control.

### Current Pigeon Interfaces

```dart
// Host API: Dart → Native
abstract class ExecutorchHostApi {
  ModelLoadResult load(Uint8List modelData, String modelId);
  List<TensorData> forward(String modelId, List<TensorData> inputs);
  void dispose(String modelId);
}
```

**Key Changes from Earlier Iterations**:
- `loadModel(String filePath)` → `load(Uint8List modelData)` - Models loaded from bytes, not file paths
- `runInference()` removed → Use `forward()` directly (returns `List<TensorData>`)
- No `InferenceResult` wrapper - `forward()` returns tensors directly
- No `options`, `timeoutMs`, `requestId` parameters - Simplified to just inputs
- Removed `getLoadedModels()` and `setDebugLogging()` - Minimal API surface

## Platform Implementation Details

### Android (Kotlin)

**File**: `android/src/main/kotlin/com/zcreations/executorch_flutter/ExecutorchModelManager.kt`

**Key Points**:
- Uses `Module.load()` from ExecuTorch AAR
- Coroutines with `Dispatchers.Default` for background execution
- Tensor conversion via `ExecutorchTensorUtils.kt`
- Error mapping via exception types

**Threading Model**:
```kotlin
suspend fun load(modelData: ByteArray, modelId: String): ModelLoadResult =
  withContext(Dispatchers.Default) {
    // ExecuTorch Module.load() on background thread
    // Writes bytes to temp file, loads with MMAP, deletes temp file
  }
```

**Memory Loading**: Writes model bytes to temporary file, loads with `Module.load(path, LoadMode.MMAP)`, then deletes temp file. This enables asset bundle loading while still using ExecuTorch's memory mapping.

### iOS/macOS (Swift)

**Files**:
- `darwin/ExecutorchModelManager.swift` (shared source)
- Symlinked to `ios/executorch_flutter/Sources/executorch_flutter/`
- Symlinked to `macos/executorch_flutter/Sources/executorch_flutter/`

**Key Points**:
- Uses `ExecuTorchModule` from XCFrameworks
- Swift async/await with `Task.detached` for background execution
- Tensor conversion via `ExecutorchTensorUtils.swift`
- Platform-specific APIs using `#if os(iOS)` / `#if os(macOS)`

**Threading Model**:
```swift
func load(modelData: FlutterStandardTypedData, modelId: String) async throws -> ModelLoadResult {
  return try await Task.detached {
    // ExecuTorchModule initialization on background task
    // Writes bytes to temp file, loads module, deletes temp file
  }.value
}
```

**Memory Loading**: Writes model bytes to temporary file, initializes `ExecuTorchModule`, then deletes temp file. This enables asset bundle loading while working with ExecuTorch's file-based API.

**Shared Sources**: iOS and macOS share the same Swift implementation via symlinks to avoid code duplication.

## Pre/Post Processors

The package includes reference processors for common model types:

### BaseProcessor
```dart
abstract class BaseInputProcessor<T> {
  Future<List<TensorData>> preprocess(T input);
}

abstract class BaseOutputProcessor<T, R> {
  Future<R> postprocess(List<TensorData> outputs);
}
```

### YOLOProcessor
- **Input**: Image (NCHW format, RGB, normalized)
- **Output**: Object detections with bounding boxes, class IDs, confidence scores
- **Use Case**: Object detection (YOLOv8, YOLOv11)

### ImageClassificationProcessor
- **Input**: Image (224x224, RGB, normalized)
- **Output**: Top-K class predictions with confidence scores
- **Use Case**: Image classification (MobileNetV3, ResNet)

**Location**: `lib/src/processors/` (package) and `example/lib/processors/` (reference implementations)

## GPU-Accelerated Preprocessing

The example app includes GPU-accelerated preprocessing using **Flutter Fragment Shaders** for high-performance image preprocessing on mobile and desktop platforms.

### Why GPU Preprocessing?

Traditional CPU-based preprocessing (using libraries like `image` or `opencv_dart`) can be slow for real-time applications:

- **CPU Preprocessing**: 15-25ms per frame (typical)
- **GPU Preprocessing**: 6-9ms per frame (2-3x faster)

This enables higher frame rates for camera-based inference:
- **CPU**: ~40-60 FPS
- **GPU**: ~110-160 FPS

### Architecture

GPU preprocessing uses **Flutter Fragment Shaders** (GLSL) to perform image transformations on the GPU:

1. **Native Image Decoder**: Hardware-accelerated image decoding via `ui.decodeImageFromList()`
2. **GPU Shader**: GLSL fragment shader for resize, crop, padding, and normalization
3. **Optimized Tensor Conversion**: Single-loop RGBA → NCHW conversion for cache locality

### Example: YOLO GPU Preprocessor

**Shader** (`example/shaders/yolo_preprocess.frag`):

```glsl
#version 460 core
#include <flutter/runtime_effect.glsl>

uniform vec2 uInputSize;    // Original image dimensions
uniform vec2 uOutputSize;   // Target size (640x640)
uniform sampler2D uTexture; // Input image

out vec4 fragColor;

void main() {
  vec2 fragCoord = FlutterFragCoord().xy;

  // Letterbox resize calculation (maintains aspect ratio)
  float scale = min(uOutputSize.x / uInputSize.x, uOutputSize.y / uInputSize.y);
  vec2 scaledSize = uInputSize * scale;
  vec2 offset = (uOutputSize - scaledSize) * 0.5;

  vec2 imageCoord = (fragCoord - offset) / scale;
  vec2 uv = imageCoord / uInputSize;

  // Gray padding (114, 114, 114) for letterbox borders
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    fragColor = vec4(114.0/255.0, 114.0/255.0, 114.0/255.0, 1.0);
  } else {
    fragColor = texture(uTexture, uv);
  }
}
```

**Dart Preprocessor** (`example/lib/processors/gpu_yolo_preprocessor.dart`):

```dart
import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

class GpuYoloPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  GpuYoloPreprocessor({required this.config});

  final YoloPreprocessConfig config;
  ui.FragmentProgram? _program;
  bool _isInitialized = false;

  @override
  String get inputTypeName => 'Image (Uint8List) [GPU]';

  /// Initialize the fragment shader
  Future<void> _initializeShader() async {
    if (_isInitialized) return;

    _program = await ui.FragmentProgram.fromAsset('shaders/yolo_preprocess.frag');
    _isInitialized = true;
  }

  @override
  Future<List<TensorData>> preprocess(Uint8List input) async {
    await _initializeShader();

    // 1. Hardware-accelerated image decode
    final ui.Image image = await _decodeImageNative(input);

    // 2. GPU processing (letterbox resize)
    final processedImage = await _processOnGpu(image);

    // 3. Convert to tensor (optimized single-loop)
    final tensorData = await _imageToTensor(processedImage);

    // Cleanup
    image.dispose();
    processedImage.dispose();

    return [tensorData];
  }

  /// Decode using Flutter's native decoder (hardware accelerated)
  Future<ui.Image> _decodeImageNative(Uint8List bytes) async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

  /// Process image on GPU using Fragment Shader
  Future<ui.Image> _processOnGpu(ui.Image inputImage) async {
    final shader = _program!.fragmentShader();

    // Set shader uniforms
    shader.setFloat(0, inputImage.width.toDouble());  // uInputSize.x
    shader.setFloat(1, inputImage.height.toDouble()); // uInputSize.y
    shader.setFloat(2, config.targetWidth.toDouble());  // uOutputSize.x
    shader.setFloat(3, config.targetHeight.toDouble()); // uOutputSize.y
    shader.setImageSampler(0, inputImage);             // uTexture

    // Render shader output
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final paint = Paint()..shader = shader;

    canvas.drawRect(
      Rect.fromLTWH(0, 0, config.targetWidth.toDouble(), config.targetHeight.toDouble()),
      paint,
    );

    final picture = recorder.endRecording();
    final outputImage = await picture.toImage(config.targetWidth, config.targetHeight);

    shader.dispose();
    picture.dispose();

    return outputImage;
  }

  /// Convert ui.Image to TensorData with optimized single-loop conversion
  Future<TensorData> _imageToTensor(ui.Image image) async {
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    final pixels = byteData!.buffer.asUint8List();

    final totalPixels = config.targetWidth * config.targetHeight;
    final floats = Float32List(3 * totalPixels);

    // Optimized single-loop conversion for better cache locality
    const scale = 1.0 / 255.0;
    for (int i = 0; i < totalPixels; i++) {
      final pixelIndex = i * 4;
      floats[i] = pixels[pixelIndex] * scale;                     // R channel
      floats[i + totalPixels] = pixels[pixelIndex + 1] * scale;   // G channel
      floats[i + totalPixels * 2] = pixels[pixelIndex + 2] * scale; // B channel
    }

    return TensorData(
      shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(),
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'images',
    );
  }

  void dispose() {
    _program = null;
    _isInitialized = false;
  }
}
```

### Example: MobileNet GPU Preprocessor

**Shader** (`example/shaders/mobilenet_preprocess.frag`):

```glsl
#version 460 core
#include <flutter/runtime_effect.glsl>

uniform vec2 uInputSize;
uniform vec2 uOutputSize;   // 224x224 for MobileNet
uniform sampler2D uTexture;

// ImageNet normalization constants
const vec3 mean = vec3(0.485, 0.456, 0.406);
const vec3 std = vec3(0.229, 0.224, 0.225);

out vec4 fragColor;

void main() {
  vec2 fragCoord = FlutterFragCoord().xy;

  // Center crop: Resize shortest side to 256, then crop 224x224 from center
  float scale = max(256.0 / uInputSize.x, 256.0 / uInputSize.y);
  vec2 scaledSize = uInputSize * scale;
  vec2 cropOffset = (scaledSize - uOutputSize) * 0.5;

  vec2 scaledCoord = fragCoord + cropOffset;
  vec2 inputCoord = scaledCoord / scale;
  vec2 uv = inputCoord / uInputSize;

  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    vec4 color = texture(uTexture, uv);
    vec3 normalized = (color.rgb - mean) / std;  // ImageNet normalization
    fragColor = vec4(normalized, 1.0);
  }
}
```

**Dart implementation** follows the same pattern as YOLO, just with different shader and config.

### Registering Shaders

Add shaders to `pubspec.yaml`:

```yaml
flutter:
  shaders:
    - shaders/yolo_preprocess.frag
    - shaders/mobilenet_preprocess.frag
```

### Using GPU Preprocessors

**In your model definition**:

```dart
@override
InputProcessor<ModelInput> createInputProcessor(ModelSettings settings) {
  final yoloSettings = settings as YoloModelSettings;

  switch (yoloSettings.preprocessingProvider) {
    case PreprocessingProvider.gpu:
      return GpuYoloPreprocessor(config: YoloPreprocessConfig(...));
    case PreprocessingProvider.opencv:
      return OpenCVYoloPreprocessor(config: YoloPreprocessConfig(...));
    case PreprocessingProvider.imageLib:
      return YoloPreprocessor(config: YoloPreprocessConfig(...));
  }
}
```

### Performance Characteristics

**Breakdown** (measured on mid-range mobile device):

| Stage | CPU (image lib) | GPU (Fragment Shader) |
|-------|-----------------|----------------------|
| Decode | 10-15ms | 2-3ms (native decoder) |
| Resize/Transform | 5-8ms | 1-2ms (GPU shader) |
| Tensor Conversion | 2-3ms | 2-3ms (optimized loop) |
| **Total** | **17-26ms** | **5-8ms** |

**Real-world impact**:
- **Static images**: 2-3x faster preprocessing
- **Camera streams**: Enables 110-160 FPS vs 40-60 FPS with CPU
- **Battery**: Lower CPU usage, but GPU uses more power during active inference

### When to Use GPU Preprocessing

**Use GPU preprocessing when**:
- Real-time camera inference (need high frame rates)
- Latency-critical applications
- Batch processing many images

**Use CPU preprocessing when**:
- Low power consumption is critical
- Simple preprocessing (no complex transforms)
- Debugging (easier to inspect intermediate steps)

### Implementation Notes

1. **Shader Initialization**: Load shaders once on first use (cached)
2. **Memory Management**: Always dispose `ui.Image` objects after use
3. **Tensor Conversion**: Use single-loop conversion for better cache locality
4. **Error Handling**: Wrap shader operations in try-catch for graceful degradation

### Optimization Tips

1. **Single-loop tensor conversion**: Process all channels in one loop
```dart
// Fast: Single loop (better cache locality)
for (int i = 0; i < totalPixels; i++) {
  floats[i] = pixels[i * 4] * scale;              // R
  floats[i + totalPixels] = pixels[i * 4 + 1] * scale;  // G
  floats[i + totalPixels * 2] = pixels[i * 4 + 2] * scale; // B
}

// Slow: Three separate loops (poor cache locality)
for (int c = 0; c < 3; c++) {
  for (int i = 0; i < totalPixels; i++) {
    floats[c * totalPixels + i] = pixels[i * 4 + c] * scale;
  }
}
```

2. **Native image decoder**: Always use `ui.decodeImageFromList()` instead of `image` library
3. **Shader reuse**: Initialize shader once, reuse for all frames
4. **Dispose properly**: Clean up all `ui.Image` and shader resources

**Location**: `example/lib/processors/gpu_*.dart` (GPU preprocessors) and `example/shaders/*.frag` (GLSL shaders)

## Development Workflows

### Adding New Features

1. **Update Pigeon**: Edit `pigeons/executorch_api.dart`
2. **Generate Code**: Run `./scripts/generate_pigeon.sh`
3. **Implement Android**: Add Kotlin code in `android/src/main/kotlin/`
4. **Implement iOS/macOS**: Add Swift code in `darwin/` (shared sources)
5. **Add Dart Wrapper**: Update `lib/src/executorch_inference.dart` or `lib/src/executorch_model.dart`
6. **Test**: Run example app on all platforms
7. **Document**: Update README and dartdoc comments

### Testing Strategy

- **Unit Tests**: Dart-only logic (currently minimal, needs expansion)
- **Integration Tests**: Full platform stack with real models (via example app)
- **Manual Testing**: Example app with various models (YOLO, MobileNet, etc.)

**Current Test Status**: Package has production code but minimal automated tests. Example app serves as integration test.

### Code Style

- **Dart**: Follow `dart format` and `dart analyze` recommendations
- **Kotlin**: Android Studio default formatting
- **Swift**: Xcode default formatting
- **Lint**: All lint rules enabled, `dart fix --apply` used for auto-fixes

### Publishing Workflow

1. **Analyze**: `flutter analyze` (0 errors in `lib/`)
2. **Fix Lints**: `dart fix --apply lib/`
3. **Dry Run**: `dart pub publish --dry-run`
4. **Review**: Check warnings, package size, dependencies
5. **Publish**: `dart pub publish` (when ready)

**Files Excluded from Publishing** (`.pubignore`):
- `specs/` (internal development docs)
- `CLAUDE.md` (AI agent context)
- `PIGEON_MACOS_NOTES.md` (development notes)
- `python/` (model conversion scripts)
- `tmp/` (temporary files)
- Large example assets (users generate their own models)

## Troubleshooting

### Common Issues

**1. iOS Simulator Not Supported**
- **Issue**: ExecuTorch XCFrameworks only built for arm64 (device)
- **Solution**: Test on physical iOS devices, or rebuild XCFrameworks with x86_64 support

**2. macOS Intel Not Supported**
- **Issue**: ExecuTorch XCFrameworks only built for Apple Silicon
- **Solution**: Use Apple Silicon Mac, or rebuild XCFrameworks with x86_64 support

**3. Model Loading Fails**
- **Issue**: Invalid .pte format, corrupted model, or asset not found
- **Solution**:
  - Verify asset is listed in `pubspec.yaml` under `flutter.assets`
  - Check model bytes are loaded correctly: `modelBytes.lengthInBytes > 0`
  - Verify .pte format (should be valid ExecuTorch binary)
  - Re-export model from PyTorch with correct ExecuTorch version

**4. Inference Returns Error**
- **Issue**: Wrong tensor shapes, data types, or model compatibility
- **Solution**:
  - Check `model.inputShapes` and `model.outputShapes` to verify expected formats
  - Verify tensor data types match model expectations (Float32, Int32, Int8, UInt8)
  - Ensure tensor shapes match exactly (including batch dimension)
  - Check ExecuTorch version compatibility (Android: 1.0.0-rc2, iOS/macOS: SPM 1.0.0)

**5. Memory Issues**
- **Issue**: Models not disposed, accumulating in memory
- **Solution**: Always call `dispose()` when model no longer needed

**6. Platform Channel Errors**
- **Issue**: Pigeon-generated code out of sync
- **Solution**: Regenerate with `./scripts/generate_pigeon.sh`

### Debugging Tools

- **Flutter DevTools**: Memory profiler, performance view
- **Android Studio**: Logcat with "ExecuTorch" filter
- **Xcode**: Console with "ExecuTorch" filter
- **Platform Logs**: Check native logs for detailed error messages

### Getting Help

- **Package Issues**: File issues at package repository
- **ExecuTorch Issues**: Check https://pytorch.org/executorch/
- **Flutter Issues**: Check https://flutter.dev/docs

## Performance Characteristics

### Benchmarks (Approximate)

- **Model Loading**: 50-200ms for 10-100MB models
- **Inference**: 10-50ms for MobileNetV3, 20-100ms for YOLOv8 (varies by device)
- **Memory Overhead**: ~50-100MB per loaded model (depends on model size)
- **Concurrent Models**: 2-3 models simultaneously (device-dependent)

### Optimization Tips

1. **Reuse Models**: Load once, run inference many times
2. **Use Memory Mapping**: For large models (>500MB)
3. **Quantize Models**: INT8 quantization for faster inference
4. **Optimize Input**: Resize images to exact model input size
5. **Batch Processing**: Use batch size > 1 if model supports it

## Version History

### 0.0.1 (Pre-release)
- Initial implementation with Android, iOS, macOS support
- Pigeon-based platform communication
- User-controlled memory management
- Example app with YOLO and MobileNet demos
- Reference processors for common model types
- Asset bundle loading support

**API Design**:
- Minimal API surface: Just `load()` and `forward()`
- No singleton manager, no lifecycle manager
- Models loaded from `Uint8List` bytes (enables asset bundle loading)
- Direct tensor return (no wrapper objects)
- User explicitly manages model lifecycle with `dispose()`

## Known Limitations

1. **iOS Simulator**: Not supported (arm64 device only)
2. **macOS Intel**: Not supported (Apple Silicon only)
3. **Android x86**: Not thoroughly tested (arm64-v8a primary)
4. **Model Format**: Only `.pte` files (no PyTorch `.pt` support)
5. **Desktop Platforms**: Windows/Linux not yet implemented
6. **Automated Tests**: Minimal unit tests (relies on example app for integration testing)

## Future Considerations

- Windows/Linux platform support
- iOS simulator support (x86_64 build)
- macOS Intel support (x86_64 build)
- Streaming inference for large outputs
- Model quantization utilities
- Comprehensive unit/integration test suite
- Model caching and version management
- Optional debugging/profiling APIs

## Example App Architecture

The example app (`example/`) demonstrates a complete implementation with multiple model types (YOLO, MobileNet) in a unified playground.

**For detailed example app architecture and adding new models, see: `example/CLAUDE.md`**

Key features:
- Strategy pattern for model definitions
- Unified playground supporting all model types
- Camera integration (platform and OpenCV)
- Model-specific settings and processors
- Python export scripts for PyTorch → ExecuTorch conversion

**Important**: When making changes to the example app, always refer to `example/CLAUDE.md` for architecture guidelines and step-by-step instructions for adding new model support.

## Contact and Support

- **Package Maintainer**: Check `pubspec.yaml` for author information
- **License**: MIT (see LICENSE file)
- **Repository**: Check `pubspec.yaml` for repository URL
- **Issues**: File issues at package repository
- **ExecuTorch**: https://pytorch.org/executorch/

---

**Last Updated**: 2025-10-10
**Package Version**: 0.0.1 (pre-release)
**ExecuTorch Version**: 1.0.0-rc2 (Android AAR) / 1.0.0 (iOS/macOS SPM)
**API**: Simplified to `load()` + `forward()` + `dispose()` only
**Architecture**: Generic Settings Provider with atomic utility methods
