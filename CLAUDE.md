# ExecuTorch Flutter Plugin - AI Agent Context

## Package Overview

**executorch_flutter** is a Flutter plugin package that provides on-device machine learning inference using PyTorch ExecuTorch. It enables Flutter developers to run optimized ML models on mobile and desktop platforms with a simple, type-safe Dart API.

**Package Name**: `executorch_flutter`
**Version**: 0.0.3
**License**: MIT
**Platforms**: Android, iOS, macOS, Windows, Linux, Web
**Flutter Version**: 3.38+ (requires native assets hooks)

## Current Development Status

- **Phase**: Package implementation complete, API simplified and finalized
- **API**: Minimal surface with only `load()` and `forward()` - asset bundle loading supported
- **Code Quality**: 0 lint errors in `lib/`, all dart fixes applied
- **Build Status**: ✅ Android, ✅ iOS, ✅ macOS, ✅ Windows, ✅ Linux, ✅ Web
- **Next Step**: Publish to pub.dev

## Core Architecture

### Technology Stack

- **Flutter Plugin**: dart:ffi with native assets hooks (Flutter 3.38+)
- **All Platforms**: Pre-built ExecuTorch native libraries via native assets
- **Build System**: CMake-based native asset compilation
- **Memory Management**: User-controlled lifecycle (explicit load/dispose)

### Design Principles

1. **Minimal API Surface**: Just `load()`, `forward()`, and `dispose()` - nothing more
2. **Type Safety**: dart:ffi bindings with type-safe Dart wrapper classes
3. **Async/Await**: All model operations are non-blocking
4. **User-Controlled Resources**: Developers explicitly manage model lifecycle (no automatic cleanup, no singleton)
5. **Structured Errors**: Exception hierarchy with clear error categories
6. **Platform Parity**: Identical behavior across all supported platforms
7. **Asset-First**: Models loaded from `Uint8List` bytes, enabling Flutter asset bundle loading

## Platform Support

### Android
- **Minimum SDK**: API 23 (Android 6.0)
- **Architectures**: arm64-v8a, armeabi-v7a, x86_64, x86 (all supported)
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, Vulkan

### iOS
- **Minimum Version**: iOS 13.0
- **Architectures**: arm64 (device), arm64-simulator, x86_64-simulator (all supported)
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, CoreML, Vulkan (via MoltenVK)

### macOS
- **Minimum Version**: macOS 11.0
- **Architectures**: arm64 (Apple Silicon), x86_64 (Intel) - both supported
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, CoreML, MPS (Metal Performance Shaders), Vulkan (via MoltenVK)

### Windows
- **Minimum Version**: Windows 10
- **Architectures**: x64
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, Vulkan

### Linux
- **Minimum Version**: Ubuntu 20.04+ or equivalent
- **Architectures**: x64
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, Vulkan

### Web
- **Supported Browsers**: Chrome, Firefox, Safari, Edge (modern versions)
- **Technology**: WebAssembly (WASM) build of ExecuTorch
- **Backend**: XNNPACK (WASM-optimized)

## Project Structure

```
executorch_flutter/
├── lib/
│   ├── executorch_flutter.dart              # Main library export
│   └── src/
│       ├── executorch_model.dart            # ExecuTorchModel - main API (load/forward/dispose)
│       ├── executorch_inference.dart        # ExecutorchManager facade
│       ├── executorch_errors.dart           # Exception hierarchy
│       ├── types.dart                       # TensorData, TensorType definitions
│       ├── ffi/                             # FFI layer
│       │   ├── native_tensor.dart           # NativeTensor wrapper
│       │   ├── backend.dart                 # Backend query functions
│       │   ├── version.dart                 # Version query functions
│       │   └── tensor_type_extensions.dart  # Extended tensor types
│       ├── build/
│       │   └── run_build.dart               # Native assets build hook
│       ├── generated/
│       │   └── executorch_ffi.g.dart        # ffigen-generated FFI bindings
│       └── processors/
│           ├── base_processor.dart          # BaseInputProcessor/BaseOutputProcessor
│           ├── yolo_processor.dart          # YOLOv8 pre/post processing
│           └── image_classification_processor.dart  # MobileNet processors
├── hook/
│   └── build.dart                           # Native assets build entry point
├── example/
│   ├── lib/
│   │   ├── main.dart                        # Example app entry
│   │   ├── screens/                         # Demo screens
│   │   ├── processors/                      # Reference processors
│   │   └── services/                        # Model management
│   └── assets/
│       ├── models/                          # .pte files (gitignored)
│       └── images/                          # Test images
├── native/                                   # Git submodule: executorch_native (C/C++ FFI library)
│   ├── src/
│   │   ├── executorch_ffi.cpp               # FFI implementation
│   │   └── executorch_ffi.h                 # FFI header
│   ├── cmake/
│   │   ├── download_prebuilt.cmake          # Pre-built binary download logic
│   │   └── build_from_source.cmake          # Source build logic
│   ├── scripts/
│   │   ├── build-android.sh                 # Android build script (all ABIs)
│   │   ├── build-apple.sh                   # iOS/macOS build script
│   │   ├── build-linux.sh                   # Linux build script
│   │   └── build-windows.sh                 # Windows build script
│   └── CMakeLists.txt                       # Main CMake configuration
└── models/                                   # Git submodule: executorch_flutter_models
    ├── python/                               # Model export scripts
    │   ├── main.py                           # Unified CLI for export
    │   ├── executorch_exporter.py            # Core exporter framework
    │   └── BACKENDS.md                       # Backend selection guide
    ├── mobilenet/                            # MobileNet model files
    ├── yolo/                                 # YOLO model files
    └── index.json                            # Model metadata index
```

## Git Submodules

This repository uses git submodules for native code and model assets. **Always be aware of submodule boundaries when making changes.**

### Submodules Overview

| Directory | Repository | Purpose |
|-----------|------------|---------|
| `native/` | `abdelaziz-mahdy/executorch_native` | C/C++ FFI library, CMake build system, platform build scripts |
| `models/` | `abdelaziz-mahdy/executorch_flutter_models` | Model export scripts, pre-exported .pte files, labels |

### Working with Submodules

**Initial clone with submodules:**
```bash
git clone --recursive https://github.com/user/executorch_flutter.git
# Or if already cloned:
git submodule update --init --recursive
```

**Making changes to a submodule:**
```bash
# 1. Navigate to submodule directory
cd native/  # or models/

# 2. Make your changes
# 3. Commit and push within the submodule
git add .
git commit -m "Your commit message"
git push

# 4. Go back to parent repo and update submodule reference
cd ..
git add native/  # or models/
git commit -m "Update native submodule"
git push
```

**Updating submodules to latest:**
```bash
git submodule update --remote --merge
```

### Native Submodule (`native/`)

The `native/` directory contains the C/C++ FFI library that bridges Dart to ExecuTorch. This is a separate repository because:
- It has its own CI/CD for building pre-built binaries
- Pre-built binaries are published as GitHub Releases
- Changes here require a new release to update prebuilts

**Key files:**
- `scripts/build-android.sh` - Builds all Android ABIs (arm64-v8a, armeabi-v7a, x86_64, x86)
- `scripts/build-apple.sh` - Builds iOS/macOS variants
- `cmake/download_prebuilt.cmake` - Downloads pre-built binaries from GitHub Releases
- `CMakeLists.txt` - Main build configuration

**Release workflow:**
1. Make changes in `native/`
2. Commit and push to `executorch_native` repository:
   ```bash
   cd native
   git add .
   git commit -m "feat: Your change description"
   git push
   ```
3. Create a new tag to trigger CI builds:
   ```bash
   git tag v1.0.1.7
   git push origin v1.0.1.7
   ```
4. Wait for GitHub Actions to build all platform variants
5. **Update versions in both repos:**
   - Update `_defaultPrebuiltVersion` in `lib/src/build/run_build.dart` (line ~59)
   - Update `EXECUTORCH_PREBUILT_VERSION` in `native/CMakeLists.txt` if needed
6. **Commit the submodule reference in the parent repo:**
   ```bash
   cd ..  # back to executorch_flutter root
   git add native
   git commit -m "chore: Update native submodule to v1.0.1.7"
   git push
   ```

**Important:** Always commit the updated submodule reference in the parent repo after pushing changes to the submodule. Otherwise, other developers cloning the repo will get an older version of the native code.

### CI/CD Workflow Dependencies

**CRITICAL: The `build.yml` workflow depends on pre-built native binaries being available on GitHub Releases.**

When updating the native submodule version, the following order MUST be followed:

1. **Wait for native binaries to finish building** - After tagging a new version in `executorch_native`, GitHub Actions builds binaries for all platforms (Linux, Windows, macOS, Android, iOS). This can take 30-60 minutes.

2. **Verify binaries are published** - Check that all platform variants are available at:
   `https://github.com/abdelaziz-mahdy/executorch_native/releases/tag/vX.X.X.X`

3. **Only then push changes to `executorch_flutter`** - If you push before binaries are ready, the build workflow will fail with 404 errors like:
   ```
   ERROR: downloading '.../libexecutorch_ffi-linux-x64-xnnpack-release.tar.gz' failed
   The requested URL returned error: 404
   ```

**If builds fail due to missing binaries:**
- Check the `executorch_native` GitHub Actions to see if builds are still in progress
- Wait for all platform builds to complete before re-running the workflow
- Do NOT merge PRs or push to main until binaries are available

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

### TensorData

Input/output tensor representation (defined in `lib/src/types.dart`):

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
- Platform parity (same behavior on all platforms)
- Simple API (just `load()` and `dispose()`, no manager required)

## Integration Testing

The example app includes integration tests for native platforms:

```bash
cd example
flutter test integration_test/models_integration_test.dart -d macos   # macOS
flutter test integration_test/models_integration_test.dart -d ios     # iOS
flutter test integration_test/models_integration_test.dart -d android # Android
flutter test integration_test/models_integration_test.dart -d windows # Windows
flutter test integration_test/models_integration_test.dart -d linux   # Linux
```

**Note**: Web integration tests require special handling (no `dart:io` support). Web functionality can be tested manually by running the example app in Chrome.

**Prerequisites**:
- Models are automatically downloaded from GitHub on first use
- To export models manually: `cd models/python && python3 main.py`

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

1. **Update C API**: Edit `native/src/executorch_ffi.h` and `native/src/executorch_ffi.cpp`
2. **Regenerate FFI Bindings**: Run `dart run ffigen` to regenerate `lib/src/generated/executorch_ffi.g.dart`
3. **Add Dart Wrapper**: Update `lib/src/ffi/` layer or higher-level APIs
4. **Test**: Run example app on all platforms
5. **Document**: Update README and dartdoc comments
6. **Release Native**: Tag new version in `native/` submodule if C API changed

### Testing Strategy

- **Unit Tests**: Dart-only logic (currently minimal, needs expansion)
- **Integration Tests**: Full platform stack with real models (via example app)
- **Manual Testing**: Example app with various models (YOLO, MobileNet, etc.)

**Current Test Status**: Package has production code but minimal automated tests. Example app serves as integration test.

### Code Style

- **Dart**: Follow `dart format` and `dart analyze` recommendations
- **C/C++**: Follow `clang-format` conventions (used in native/ submodule)
- **Lint**: All lint rules enabled, `dart fix --apply` used for auto-fixes

### Commit Guidelines

- **ALWAYS ask before committing** unless the user explicitly says to commit. Present the changes and proposed commit message for approval first.
- **DO NOT include AI co-author tags** in commit messages. Never use:
  - `Co-Authored-By: Claude ...`
  - `Co-Authored-By: ChatGPT ...`
  - Any other AI attribution
- **DO NOT mention AI tools** (Claude, Claude Code, ChatGPT, Copilot, etc.) in commit messages
- Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`, `ci:`
- Keep commit messages concise and focused on the "why" not the "what"

### Pre-Push Checklist

**ALWAYS run these checks before pushing:**

```bash
# 1. Run analyzer (must pass with 0 issues)
flutter analyze lib

# 2. Run formatter
dart format lib

# 3. Run tests if available
flutter test
```

If analyzer reports issues, fix them before pushing. Do NOT push code with lint errors.

### Publishing Workflow

1. **Analyze**: `flutter analyze` (0 errors in `lib/`)
2. **Fix Lints**: `dart fix --apply lib/`
3. **Dry Run**: `dart pub publish --dry-run`
4. **Review**: Check warnings, package size, dependencies
5. **Publish**: `dart pub publish` (when ready)

**Files Excluded from Publishing** (`.pubignore`):
- `specs/` (internal development docs)
- `CLAUDE.md` (AI agent context)
- `native/` (submodule, built via native assets)
- `models/` (submodule, large model files)
- `tmp/` (temporary files)
- Large example assets (users generate their own models)

## Troubleshooting

### Common Issues

**1. Model Loading Fails**
- **Issue**: Invalid .pte format, corrupted model, or asset not found
- **Solution**:
  - Verify asset is listed in `pubspec.yaml` under `flutter.assets`
  - Check model bytes are loaded correctly: `modelBytes.lengthInBytes > 0`
  - Verify .pte format (should be valid ExecuTorch binary)
  - Re-export model from PyTorch with correct ExecuTorch version

**2. Inference Returns Error**
- **Issue**: Wrong tensor shapes, data types, or model compatibility
- **Solution**:
  - Check `model.inputShapes` and `model.outputShapes` to verify expected formats
  - Verify tensor data types match model expectations (Float32, Int32, Int8, UInt8)
  - Ensure tensor shapes match exactly (including batch dimension)
  - Check ExecuTorch version compatibility (Android: 1.0.1, iOS/macOS: SPM 1.0.1)

**3. Memory Issues**
- **Issue**: Models not disposed, accumulating in memory
- **Solution**: Always call `dispose()` when model no longer needed

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

### 0.0.3 (Current)
- Full cross-platform support: Android, iOS, macOS, Windows, Linux, Web
- dart:ffi with native assets for native platforms
- WebAssembly for web platform
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

1. **Model Format**: Only `.pte` files (no PyTorch `.pt` support)
2. **Automated Tests**: Minimal unit tests (relies on example app for integration testing)

## Future Considerations

- Streaming inference for large outputs
- Model quantization utilities
- Comprehensive unit/integration test suite
- Model caching and version management
- Optional debugging/profiling APIs

## Native Assets Architecture

The package uses dart:ffi with Flutter's native assets system for cross-platform support:

- **All platforms**: Unified C/C++ FFI library for ExecuTorch integration
- **Build System**: CMake-based compilation via native assets hooks (Flutter 3.38+)
- **Pre-built binaries**: Available from GitHub Releases for faster builds
- **Source builds**: Optional for custom ExecuTorch configurations

**Key Components:**

- `native/src/executorch_ffi.cpp` - C FFI implementation wrapping ExecuTorch
- `native/CMakeLists.txt` - CMake build configuration
- `lib/src/build/run_build.dart` - Native assets build hook
- `lib/src/ffi/` - Dart FFI bindings

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

**Last Updated**: 2025-01-17
**Package Version**: 0.0.3
**Flutter Version**: 3.38+
**API**: Simplified to `load()` + `forward()` + `dispose()` only
**Architecture**: dart:ffi with native assets hooks
