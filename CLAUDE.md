# ExecuTorch Flutter Plugin - AI Agent Context

## Quick Start

```bash
# Analyze code (must pass before pushing)
flutter analyze lib

# Format code
dart format lib

# Run example app
cd example && flutter run -d macos

# Run integration tests
cd example && flutter test integration_test/models_integration_test.dart -d macos

# Regenerate FFI bindings (after native API changes)
dart run ffigen

# Publish dry run
dart pub publish --dry-run
```

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
- **Backend**: XNNPACK, Vulkan (opt-in)

### iOS
- **Minimum Version**: iOS 13.0
- **Architectures**: arm64 (device), arm64-simulator, x86_64-simulator (all supported)
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, CoreML, MPS (Metal Performance Shaders), Vulkan (opt-in, via MoltenVK)

### macOS
- **Minimum Version**: macOS 11.0
- **Architectures**: arm64 (Apple Silicon), x86_64 (Intel) - both supported
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, CoreML, MPS (Metal Performance Shaders), Vulkan (opt-in, via MoltenVK)

### Windows
- **Minimum Version**: Windows 10
- **Architectures**: x64
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, Vulkan (opt-in)

### Linux
- **Minimum Version**: Ubuntu 20.04+ or equivalent
- **Architectures**: x64
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, Vulkan (opt-in)

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

### CI/CD and Releases

**CRITICAL: All three repositories have CI/CD pipelines. NEVER create GitHub releases or tags manually - CI/CD handles all releases automatically.**

| Repository | CI/CD Trigger | What It Does |
|------------|---------------|--------------|
| `executorch_flutter` | Push to main / tags | `build.yml` builds all platforms; `release.yml` creates GitHub Release + publishes to pub.dev; `update-readme.yml` auto-updates README version links; `deploy-web.yml` deploys example to GitHub Pages |
| `executorch_native` | Tags (e.g., `v1.2.0.1`) | `release.yaml` orchestrates 4 platform build workflows → unified GitHub Release with all binary artifacts + size reports |
| `executorch_flutter_models` | Push to main (python/) or manual dispatch | `export-models.yml` exports models → generates index.json, labels, versions.json → commits to main |

**Rules:**
- **DO NOT** use `gh release create` or manually create tags/releases
- **DO NOT** use `gh release delete` to "fix" releases
- Just push code to main (or tag for native) and let CI/CD handle the rest
- If a release needs to be re-done, push a fix commit and let CI/CD create a new one

### CI/CD Workflow Dependencies

**The `build.yml` workflow depends on pre-built native binaries being available on GitHub Releases.**

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

### ExecuTorch Version Upgrade Procedure (Cross-Repo)

When a new upstream ExecuTorch version is released (e.g., 1.1.0 → 1.2.0), all three repos must be updated **in strict order**. Each step must complete before the next begins.

```
Step 1: executorch_native
  ├── Update EXECUTORCH_VERSION in CMakeLists.txt
  ├── Update default VERSION in all scripts/build-*.sh
  ├── Reset EXECUTORCH_PREBUILT_VERSION to X.Y.Z.1
  ├── Merge PR to main
  ├── Tag: git tag vX.Y.Z.1 && git push origin vX.Y.Z.1
  └── WAIT 30-60 min for all platform binaries to build
      Verify at: github.com/abdelaziz-mahdy/executorch_native/releases/tag/vX.Y.Z.1

Step 2: executorch_flutter_models
  ├── Update HARDCODED version list in .github/workflows/export-models.yml (line ~59)
  │   e.g., echo 'versions=["1.0.1", "1.1.0", "1.2.0"]' >> $GITHUB_OUTPUT
  ├── Also update workflow_dispatch input options dropdown
  ├── Merge PR to main (triggers export automatically since python/ changed)
  │   OR trigger workflow manually via GitHub Actions UI
  └── WAIT for CI to export models, generate index.json, update versions.json, and commit
      NOTE: versions.json is auto-generated OUTPUT, not input — don't edit it manually

Step 3: executorch_flutter
  ├── Update lib/src/version.dart: executorchVersion = 'X.Y.Z'
  ├── Update lib/src/build/run_build.dart: prebuilt suffix (e.g., .1)
  ├── Bump version in pubspec.yaml
  ├── Add CHANGELOG.md entry
  ├── Update native submodule ref: cd native && git pull origin main && cd ..
  ├── Update models submodule ref: cd models && git pull origin main && cd ..
  ├── Merge PR to main (update-readme.yml auto-updates README links)
  └── Tag for pub.dev release: git tag v0.X.0 && git push origin v0.X.0
      (release.yml creates GitHub Release + publishes to pub.dev)
```

**Version Sources of Truth (this repo):**
- `lib/src/version.dart` → `executorchVersion` (e.g., `'1.2.0'`)
- `lib/src/build/run_build.dart` → `_defaultPrebuiltVersion` = `'$executorchVersion.1'` (e.g., `1.2.0.1`)
- `pubspec.yaml` → `version:` (package version, e.g., `0.4.0`)

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

The example app includes GPU-accelerated preprocessing using **Flutter Fragment Shaders** (2-3x faster than CPU).

| Method | Speed | Use Case |
|--------|-------|----------|
| GPU (Fragment Shaders) | 5-8ms | Real-time camera, high FPS needed |
| CPU (image lib) | 17-26ms | Debugging, low power |
| OpenCV | 10-15ms | Desktop, complex transforms |

**Key files:**
- `example/shaders/*.frag` - GLSL shaders for letterbox/crop/normalize
- `example/lib/processors/gpu_*.dart` - GPU preprocessor implementations

**Key patterns:**
- Use `ui.decodeImageFromList()` for hardware-accelerated decode
- Single-loop tensor conversion (RGBA → NCHW) for cache locality
- Always dispose `ui.Image` objects after use

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

**Package Version**: 0.0.3
**Flutter Version**: 3.38+
**API**: Simplified to `load()` + `forward()` + `dispose()` only
**Architecture**: dart:ffi with native assets hooks
