# ExecuTorch Flutter Plugin - AI Agent Context

## Quick Start

```bash
# Analyze code (must pass before pushing) — workspace-wide, run from repo root
flutter analyze

# Format code
dart format packages/executorch_dart/lib packages/executorch_flutter/lib

# Run example app
cd packages/executorch_flutter/example && flutter run -d macos

# Run integration tests
cd packages/executorch_flutter/example && flutter test integration_test/models_integration_test.dart -d macos

# Regenerate FFI bindings (after native API changes)
cd packages/executorch_dart && dart run ffigen

# Publish dry run (each package publishes separately; core must go first)
(cd packages/executorch_dart && dart pub publish --dry-run)
(cd packages/executorch_flutter && dart pub publish --dry-run)
```

## Package Overview

**executorch_flutter** is a Flutter plugin package that provides on-device machine learning inference using PyTorch ExecuTorch. It enables Flutter developers to run optimized ML models on mobile and desktop platforms with a simple, type-safe Dart API. It's a thin wrapper over the pure-Dart core, **executorch_dart** (`packages/executorch_dart/`), which has the same `load`/`forward`/`dispose` API with no Flutter SDK required — see `packages/executorch_dart/README.md` for Dart servers and CLI tools.

**Package Names**: `executorch_flutter` (wrapper), `executorch_dart` (core)
**Version**: each package's own `pubspec.yaml` (`version:`) is the source of truth; both currently release together
**License**: MIT
**Platforms**: Android, iOS, macOS, Windows, Linux, Web (`executorch_dart` alone: everything except Web)
**Flutter Version**: 3.38+ (requires native assets hooks) — not needed at all for `executorch_dart`

## Current Development Status

- **Status**: `executorch_flutter` is published on pub.dev; `executorch_dart` is new and has not been published yet (0.6.0 is its first release). Once tagged, both release together — CI publishes `executorch_dart` before `executorch_flutter`; see "Version Sources of Truth" below
- **API**: vision inference via `ExecuTorchModel` (`load`/`forward`/`dispose`) plus experimental
  streaming LLM via `ExecuTorchLLM` (see `packages/executorch_flutter/docs/LLM.md`)
- **Code Quality**: `flutter analyze` and `dart format --set-exit-if-changed` (both packages' `lib/`) must be clean
- **Build Status**: ✅ Android, ✅ iOS, ✅ macOS, ✅ Windows, ✅ Linux, ✅ Web

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
- **Backend**: XNNPACK, CoreML, Vulkan (opt-in, via MoltenVK)

### macOS
- **Minimum Version**: macOS 11.0
- **Architectures**: arm64 (Apple Silicon), x86_64 (Intel) - both supported
- **Dependencies**: ExecuTorch pre-built native libraries via native assets
- **Backend**: XNNPACK, CoreML, Metal, MLX (opt-in, LLM GPU path), Vulkan (opt-in, via MoltenVK)

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

This is a pub workspace with two published packages under `packages/`. The
root `pubspec.yaml` lists them under `workspace:` and is also the only place
the native build's `hooks: user_defines:` block is read from in this repo —
see "Native Local Development" below.

```
executorch_flutter/                            # Repo root — workspace, not published
├── pubspec.yaml                                # Workspace members + native build user_defines
├── packages/
│   ├── executorch_dart/                        # Pure-Dart core (dart:ffi, no Flutter)
│   │   ├── lib/
│   │   │   ├── executorch_dart.dart            # Full public API (native + LLM)
│   │   │   ├── executorch_dart_shared.dart     # ffi-free subset (only consumer: web branch of executorch_flutter)
│   │   │   └── src/
│   │   │       ├── executorch_model.dart       # ExecuTorchModel - main API (load/forward/dispose)
│   │   │       ├── executorch_inference.dart   # ExecutorchManager facade
│   │   │       ├── executorch_errors.dart      # Exception hierarchy
│   │   │       ├── executorch_llm.dart         # ExecuTorchLLM (experimental streaming LLM)
│   │   │       ├── types.dart                  # TensorData, TensorType definitions
│   │   │       ├── ffi/                        # FFI layer
│   │   │       │   ├── native_module.dart      # NativeModule wrapper (load/forward/dispose)
│   │   │       │   ├── backend_query.dart      # Backend query functions
│   │   │       │   └── version.dart            # Version query functions
│   │   │       ├── build/
│   │   │       │   └── run_build.dart          # Native assets build hook (CMake orchestration)
│   │   │       ├── generated/
│   │   │       │   └── executorch_ffi.g.dart   # ffigen-generated FFI bindings
│   │   │       └── processors/
│   │   │           └── base_processor.dart     # ExecuTorchPreprocessor/ExecuTorchPostprocessor
│   │   ├── hook/build.dart                     # Native assets build entry point
│   │   ├── native/                             # Git submodule: executorch_native (C/C++ FFI library)
│   │   │   ├── src/
│   │   │   │   ├── executorch_ffi.cpp          # FFI implementation
│   │   │   │   └── executorch_ffi.h            # FFI header
│   │   │   ├── cmake/
│   │   │   │   ├── download_prebuilt.cmake     # Pre-built binary download logic
│   │   │   │   └── build_from_source.cmake     # Source build logic
│   │   │   ├── scripts/                        # Platform build scripts + compile-local.sh
│   │   │   └── CMakeLists.txt                  # Main CMake configuration
│   │   └── example/                            # Pure-Dart CLI example — dart:io only, no Flutter
│   │       └── bin/infer.dart
│   └── executorch_flutter/                     # Flutter plugin — thin wrapper over executorch_dart
│       ├── lib/
│       │   ├── executorch_flutter.dart         # Re-exports executorch_dart; routes native vs. web
│       │   └── src/
│       │       ├── assets.dart                 # loadModelFromAsset / loadModelFromAssets
│       │       └── web/                        # WebAssembly-backed implementations
│       │           ├── executorch_model_web.dart
│       │           └── wasm_module_loader.dart
│       ├── hook/build.dart                     # No native build here — just the legacy-key tripwire
│       ├── android/ ios/ macos/ linux/ windows/ web/  # Platform plugin registration + Wasm assets
│       ├── docs/LLM.md                         # On-device LLM guide
│       └── example/                            # Full Flutter demo app (YOLO, MobileNet, camera, LLM chat)
│           ├── lib/
│           │   ├── main.dart                   # Example app entry
│           │   ├── screens/                    # Demo screens
│           │   ├── processors/                 # Reference processors
│           │   └── services/                   # Model management
│           └── assets/
│               └── images/                     # Test images (models are downloaded at runtime)
└── models/                                     # Git submodule: executorch_flutter_models (still at repo root)
    ├── python/                                 # Model export scripts
    │   ├── main.py                             # Unified CLI for export
    │   ├── executorch_exporter.py              # Core exporter framework
    │   └── BACKENDS.md                         # Backend selection guide
    ├── mobilenet/                               # MobileNet model files
    ├── yolo/                                    # YOLO model files
    └── index.json                               # Model metadata index
```

## Git Submodules

This repository uses git submodules for native code and model assets. **Always be aware of submodule boundaries when making changes.**

### Submodules Overview

| Directory | Repository | Purpose |
|-----------|------------|---------|
| `packages/executorch_dart/native/` | `abdelaziz-mahdy/executorch_native` | C/C++ FFI library, CMake build system, platform build scripts. Lives under the core package since it owns the native build |
| `models/` | `abdelaziz-mahdy/executorch_flutter_models` | Model export scripts, pre-exported .pte files, labels. Stays at the repo root — shared by both packages |

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
cd packages/executorch_dart/native/  # or models/ (repo root)

# 2. Make your changes
# 3. Commit and push within the submodule
git add .
git commit -m "Your commit message"
git push

# 4. Go back to parent repo and update submodule reference
cd -
git add packages/executorch_dart/native/  # or models/
git commit -m "Update native submodule"
git push
```

**Updating submodules to latest:**
```bash
git submodule update --remote --merge
```

### Native Submodule (`packages/executorch_dart/native/`)

The `packages/executorch_dart/native/` directory contains the C/C++ FFI library that bridges Dart to ExecuTorch. This is a separate repository because:
- It has its own CI/CD for building pre-built binaries
- Pre-built binaries are published as GitHub Releases
- Changes here require a new release to update prebuilts

**Key files:**
- `scripts/build-android.sh` - Builds all Android ABIs (arm64-v8a, armeabi-v7a, x86_64, x86)
- `scripts/build-ios.sh` - Builds iOS variants
- `scripts/build-macos.sh` - Builds macOS variants
- `scripts/build-windows.ps1` - Builds Windows (PowerShell, not a shell script)
- `scripts/build-linux.sh` - Builds Linux variants
- `cmake/download_prebuilt.cmake` - Downloads pre-built binaries from GitHub Releases
- `CMakeLists.txt` - Main build configuration

**Release workflow:**
1. Make changes in `packages/executorch_dart/native/`
2. Commit and push to `executorch_native` repository:
   ```bash
   cd packages/executorch_dart/native
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
   - Update `_defaultPrebuiltVersion` in `packages/executorch_dart/lib/src/build/run_build.dart` (line ~76)
   - Update `EXECUTORCH_PREBUILT_VERSION` in `packages/executorch_dart/native/CMakeLists.txt` if needed
6. **Commit the submodule reference in the parent repo:**
   ```bash
   cd -  # back to executorch_flutter root
   git add packages/executorch_dart/native
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

Step 3: executorch_flutter repo (two packages — update both)
  ├── packages/executorch_dart (core: owns the ExecuTorch version + native build)
  │     ├── Update packages/executorch_dart/lib/src/version.dart: executorchVersion = 'X.Y.Z'
  │     ├── Update packages/executorch_dart/lib/src/build/run_build.dart: prebuilt suffix,
  │     │     reset to .1 for the new version (e.g. 'X.Y.Z.1')
  │     ├── Update native submodule ref:
  │     │     cd packages/executorch_dart/native && git pull origin main && cd -
  │     ├── Bump version in packages/executorch_dart/pubspec.yaml
  │     ├── Update the hand-maintained versions in packages/executorch_dart/README.md:
  │     │     the `executorch_dart: ^X.Y.Z` install snippet and the
  │     │     `prebuilt_version: "X.Y.Z.N"` example (both marked "bump this by hand" —
  │     │     not auto-updated by CI)
  │     └── Add a packages/executorch_dart/CHANGELOG.md entry
  ├── packages/executorch_flutter (wrapper: Web/Wasm build + asset loading)
  │     ├── Update Dockerfile.wasm: EXECUTORCH_VERSION=vX.Y.Z (stays at repo root)
  │     ├── Rebuild WebAssembly binaries: ./scripts/build_wasm.sh (stays at repo root)
  │     │     (builds Docker image, runs container, copies executorch.js + executorch.wasm
  │     │      to packages/executorch_flutter/web/wasm/ and
  │     │      packages/executorch_flutter/example/web/wasm/)
  │     ├── Bump the `executorch_dart: ^X.Y.Z` constraint in
  │     │     packages/executorch_flutter/pubspec.yaml if it needs the new core version
  │     ├── Bump version in packages/executorch_flutter/pubspec.yaml
  │     └── Add a packages/executorch_flutter/CHANGELOG.md entry
  ├── Update models submodule ref: cd models && git pull origin main && cd -
  ├── Merge PR to main (update-readme.yml auto-updates README links)
  └── Tag for pub.dev release: git tag v0.X.0 && git push origin v0.X.0
      (release.yml creates the GitHub Release, then publishes executorch_dart
      to pub.dev BEFORE executorch_flutter — see release.yml's `publish` job)
```

**Version Sources of Truth (this repo):**
- `packages/executorch_dart/lib/src/version.dart` → `executorchVersion` (currently `'1.3.1'`)
- `packages/executorch_dart/lib/src/build/run_build.dart` → `_defaultPrebuiltVersion` =
  `'$executorchVersion.<n>'`, a manually incremented prebuilt-build counter that resets to
  `.1` when `executorchVersion` bumps to a new upstream release (currently `1.3.1.9` — do
  **not** assume the suffix is always `.1`; it climbs with every prebuilt re-release of the
  same ExecuTorch version)
- `packages/executorch_dart/pubspec.yaml` and `packages/executorch_flutter/pubspec.yaml` →
  `version:` (both packages currently release together at `0.6.0`)

## Key APIs

### ExecuTorchModel

The primary API for loading models and running inference. Simple, minimal, and
direct — defined in `packages/executorch_dart` (`lib/src/executorch_model.dart`),
re-exported by `packages/executorch_flutter`.

```dart
// Flutter: load from the asset bundle
import 'package:executorch_flutter/executorch_flutter.dart';

final model = await loadModelFromAsset('assets/models/model.pte');

// Pure Dart (no Flutter): load from a file path or from bytes
import 'package:executorch_dart/executorch_dart.dart';

final model = await ExecuTorchModel.load('/path/to/model.pte');
// or:
final bytes = await File('/path/to/model.pte').readAsBytes();
final model = await ExecuTorchModel.loadFromBytes(bytes);

// Model properties
print(model.modelId);      // Unique identifier (auto-generated)
print(model.isDisposed);   // Whether dispose() has been called

// Run inference
final outputs = await model.forward([tensorData]);

// Dispose when done
await model.dispose();
```

`ExecuTorchModel` has exactly four instance members — `modelId`,
`isDisposed`, `forward`, `dispose` — plus the two static loaders above.
There's no `inputShapes`/`outputShapes` introspection; check tensor shapes
against the model's export script instead.

**Key Design Points**:
- **No options/timeouts**: Simplified API with just input tensors
- **Direct outputs**: Returns `List<TensorData>` directly (no wrapper object)
- **Asset-first for Flutter**: `loadModelFromAsset` folds the `rootBundle` read and `loadFromBytes` call into one

### TensorData

Input/output tensor representation (defined in
`packages/executorch_dart/lib/src/types.dart`):

```dart
final tensor = TensorData(
  shape: [1, 3, 224, 224],           // [batch, channels, height, width]
  dataType: TensorType.float32,      // float32, int32, int8, uint8
  data: Uint8List(...),              // Raw bytes
  name: 'input_0',                   // Optional name
);
```

### Model Loading Pattern

**Recommended (Flutter): Load from Asset Bundle**

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// 1. Add model to pubspec.yaml assets:
//    flutter:
//      assets:
//        - assets/models/

// 2. Load and create the model instance in one call
final model = await loadModelFromAsset('assets/models/model.pte');

// 3. Run inference
final outputs = await model.forward([inputTensor]);

// 4. Clean up
await model.dispose();
```

**Alternative: Load from File System (native platforms only)**

```dart
import 'dart:io';
import 'package:executorch_dart/executorch_dart.dart';

// Load from a downloaded/cached file's path
final modelFile = File('/path/to/downloaded/model.pte');
final model = await ExecuTorchModel.load(modelFile.path);

// Or from its bytes
final model = await ExecuTorchModel.loadFromBytes(
  await modelFile.readAsBytes(),
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
cd packages/executorch_flutter/example
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
abstract class ExecuTorchPreprocessor<T> {
  Future<List<TensorData>> preprocess(T input);
}

abstract class ExecuTorchPostprocessor<R> {
  Future<R> postprocess(List<TensorData> outputs);
}

// Combines both into one pipeline:
abstract class ExecuTorchProcessor<T, R> {
  ExecuTorchPreprocessor<T> get preprocessor;
  ExecuTorchPostprocessor<R> get postprocessor;
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

**Location**: `packages/executorch_dart/lib/src/processors/` (base classes
only — `ExecuTorchPreprocessor`/`ExecuTorchPostprocessor`/`ExecuTorchProcessor`)
and `packages/executorch_flutter/example/lib/processors/` (concrete
YOLO/MobileNet reference implementations)

## GPU-Accelerated Preprocessing

The example app includes GPU-accelerated preprocessing using **Flutter Fragment Shaders** (2-3x faster than CPU).

| Method | Speed | Use Case |
|--------|-------|----------|
| GPU (Fragment Shaders) | 5-8ms | Real-time camera, high FPS needed |
| CPU (image lib) | 17-26ms | Debugging, low power |
| OpenCV | 10-15ms | Desktop, complex transforms |

**Key files:**
- `packages/executorch_flutter/example/shaders/*.frag` - GLSL shaders for letterbox/crop/normalize
- `packages/executorch_flutter/example/lib/processors/shaders/gpu_*.dart` - GPU preprocessor implementations

**Key patterns:**
- Use `ui.decodeImageFromList()` for hardware-accelerated decode
- Single-loop tensor conversion (RGBA → NCHW) for cache locality
- Always dispose `ui.Image` objects after use

## Native Local Development (build modes)

**Full guide: `CONTRIBUTING.md` (Build Modes / Local Build & Testing / Source Build sections). Read it before touching native code.** Key facts:

- The plugin has three build modes, set in the **consuming app's** `pubspec.yaml`
  (`hooks: user_defines: executorch_dart: build_mode:`). The key is
  `executorch_dart` — `executorch_dart` owns the native build even when the
  app only depends on `executorch_flutter`. The old `executorch_flutter:` key
  fails the build with a message telling you to rename it (see the 0.6.0
  breaking-change entry in `packages/executorch_flutter/CHANGELOG.md`).
  - **prebuilt** (default): downloads an already-compiled `libexecutorch_ffi` from
    GitHub Releases. **Local edits to `packages/executorch_dart/native/src/*.cpp`
    have NO effect in this mode.**
  - **source**: the app's build phase compiles ExecuTorch + FFI from a local checkout —
    set `executorch_source: "/path/to/executorch"`. This is the RECOMMENDED way to test
    native changes: no manual cmake, `flutter run`/`flutter test` does everything.
    First build 15-30+ min, incremental after (ccache helps).
  - **local**: consumes `packages/executorch_dart/native/local-builds/<variant>/`
    produced by `packages/executorch_dart/native/scripts/compile-local.sh`
    (or `local_lib_dir:`).
- **When editing this repo's own example apps**, put the `user_defines` block
  in the **workspace root** `pubspec.yaml`, not
  `packages/executorch_flutter/example/pubspec.yaml`. Pub workspaces only read
  `hooks: user_defines:` from the root pubspec — the same block in a member
  package is silently ignored and the build falls back to defaults with no
  error. This doesn't affect a normal consumer app, which is its own root
  package. See `CONTRIBUTING.md` for details.
- **Do NOT hand-drive cmake/ninja in stale `packages/executorch_dart/native/build-local-*`
  dirs** — that path
  caused a cascade of traps (python wrapper self-exec on reconfigure, dead-venv torch
  paths, libomp install-name/sandbox failures, stale hooks cache needing
  `flutter clean`). All documented in `packages/executorch_dart/native/CLAUDE.md` →
  "Local Compilation".
- Backend set must match your test models: xnnpack-delegated `.pte` files fail to
  load on a build without XNNPACK.
- New native dtypes / API changes require a native release (tag `vX.Y.Z.W` in the
  `native` repo) before prebuilt mode picks them up; test via source mode meanwhile.

## Development Workflows

### Adding New Features

1. **Update C API**: Edit `packages/executorch_dart/native/src/executorch_ffi.h` and
   `packages/executorch_dart/native/src/executorch_ffi.cpp`
2. **Regenerate FFI Bindings**: From `packages/executorch_dart`, run `dart run ffigen`
   to regenerate `lib/src/generated/executorch_ffi.g.dart`
3. **Add Dart Wrapper**: Update `packages/executorch_dart/lib/src/ffi/` layer or
   higher-level APIs
4. **Test**: Run example app on all platforms
5. **Document**: Update README and dartdoc comments
6. **Release Native**: Tag new version in the `packages/executorch_dart/native/`
   submodule if C API changed

### Testing Strategy

- **Unit Tests**: Dart-only logic (currently minimal, needs expansion)
- **Integration Tests**: Full platform stack with real models (via example app)
- **Manual Testing**: Example app with various models (YOLO, MobileNet, etc.)

**Current Test Status**: Package has production code but minimal automated tests. Example app serves as integration test.

### Code Style

- **Dart**: Follow `dart format` and `dart analyze` recommendations
- **C/C++**: Follow `clang-format` conventions (used in the `packages/executorch_dart/native/` submodule)
- **Lint**: All lint rules enabled, `dart fix --apply` used for auto-fixes

### Changelog Guidelines

- **Summarized, not exhaustive**: one bullet per user-visible theme; fold
  related fixes into a single bullet. No file paths or internal function names —
  describe effects, not diffs.
- **Credit external contributors** in the entry itself: "thanks @username
  ([#NN](link-to-pr))" on the bullet their work enabled. Always link the PR.
- Use `### Added` / `### Fixed` / `### Breaking` sections (only the ones that
  apply). Breaking entries say what breaks AND what to do about it.
- **`### Fixed` is only for bugs that existed in a released version.** Bugs
  introduced and fixed within the same unreleased change are not "fixes" to the
  user — fold any relevant behavior notes into the feature's `### Added` bullet.
- Version entries match `pubspec.yaml` `version:`; check the version wasn't
  already released before bumping.

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

**ALWAYS run these checks before EVERY push.** CI (`build.yml`) checks both
packages in two different jobs: a Flutter-based job that analyzes only
`packages/executorch_flutter/lib`, and a separate no-Flutter job that copies
`packages/executorch_dart` out of the workspace and analyzes/formats/tests it
in isolation — a regression guard proving the core has zero Flutter in its
dependency graph. Locally, this simpler sequence from the workspace root
covers both packages with equivalent effect:

```bash
# 1. Analyzer — must report 0 issues, for both packages
flutter analyze

# 2. Formatter — verify there is nothing left to format
dart format --set-exit-if-changed packages/executorch_dart/lib \
  packages/executorch_flutter/lib

# 3. Tests — each package has its own test runner
(cd packages/executorch_flutter && flutter test)
(cd packages/executorch_dart && dart test)
```

Then check `git status` — **if the formatter touched ANY file, stage and commit
it in the same push.** (A formatted-but-uncommitted file is the classic way the
CI format check fails while local looks clean.)

Fork PRs: CI runs need maintainer approval — after pushing to a contributor's
branch, check `gh run list` for `action_required` and approve via
`gh api -X POST repos/<owner>/<repo>/actions/runs/<id>/approve`.

### Publishing Workflow

Each package publishes separately, core first — `packages/executorch_dart`
before `packages/executorch_flutter` (see `release.yml`'s `publish` job, or
the "ExecuTorch Version Upgrade Procedure" above). For each package:

1. **Analyze**: `flutter analyze` (0 errors)
2. **Fix Lints**: `dart fix --apply`
3. **Dry Run**: `dart pub publish --dry-run --directory=packages/<name>`
4. **Review**: Check warnings, package size, dependencies
5. **Publish**: CI does this automatically (`release.yml`) when a `vX.Y.Z` tag
   is pushed — per the CI/CD rules above, don't run `dart pub publish` by hand

**Files Excluded from Publishing** — each package has its own `.pubignore`
(`packages/executorch_dart/.pubignore`, `packages/executorch_flutter/.pubignore`):
- `CLAUDE.md` (AI agent context) — both packages; the wrapper also excludes
  `example/CLAUDE.md`, the core also excludes `native/CLAUDE.md`
- Build artifacts (`/build/`, `.dart_tool/`) — both packages; the wrapper
  additionally excludes Flutter-specific artifacts (`.flutter-plugins*`,
  `.packages`, `example/build/`, `example/.dart_tool/`,
  `example/.flutter-plugins*`) that don't exist for the pure-Dart core
- `tmp/` (temporary files) — both packages
- Large example assets the wrapper's example downloads instead of bundling
  (`example/assets/models/*.pte`, `example/assets/images/*.jpg`)

`specs/`, `.superpowers/`, and `models/` (the models submodule) live at the
**repo root**, outside both package directories, so `pub publish` never sees
them regardless of `.pubignore`. `packages/executorch_dart/native/` (the
native submodule) *is* inside a published package — only its own `CLAUDE.md`
is excluded; the C/C++ sources publish as part of `executorch_dart` (needed
for `build_mode: "source"`/`"local"`).

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
  - `ExecuTorchModel` doesn't expose shape introspection — verify the
    `shape:`/`dataType:` you pass against the model's export script instead
  - Verify tensor data types match model expectations (Float32, Int32, Int8, UInt8)
  - Ensure tensor shapes match exactly (including batch dimension)
  - Check ExecuTorch version compatibility — all platforms build against the
    same upstream version now (`executorchVersion` in
    `packages/executorch_dart/lib/src/version.dart`, currently `1.3.1`)

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

See each package's own `CHANGELOG.md` (`packages/executorch_dart/CHANGELOG.md`,
`packages/executorch_flutter/CHANGELOG.md`) — each is the source of truth for
its package and is updated every release.

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

`packages/executorch_dart` uses `dart:ffi` with Dart's native assets system
for cross-platform support — no Flutter SDK required (`dart run`, `dart test`,
and `dart build cli` all invoke the build hook directly, same as `flutter
run`). `packages/executorch_flutter` inherits this by depending on the core;
it registers no native build of its own.

- **All platforms**: Unified C/C++ FFI library for ExecuTorch integration
- **Build System**: CMake-based compilation via native assets hooks (Dart 3.10+ / Flutter 3.38+)
- **Pre-built binaries**: Available from GitHub Releases for faster builds
- **Source builds**: Optional for custom ExecuTorch configurations

**Key Components:**

- `packages/executorch_dart/native/src/executorch_ffi.cpp` - C FFI implementation wrapping ExecuTorch
- `packages/executorch_dart/native/CMakeLists.txt` - CMake build configuration
- `packages/executorch_dart/lib/src/build/run_build.dart` - Native assets build hook
- `packages/executorch_dart/lib/src/ffi/` - Dart FFI bindings

## Example App Architecture

The Flutter demo app (`packages/executorch_flutter/example/`) demonstrates a
complete implementation with multiple model types (YOLO, MobileNet) in a
unified playground. (The pure-Dart CLI example at
`packages/executorch_dart/example/` is intentionally minimal — see its own
README, not this section.)

**For detailed example app architecture and adding new models, see:
`packages/executorch_flutter/example/CLAUDE.md`**

Key features:
- Strategy pattern for model definitions
- Unified playground supporting all model types
- Camera integration (platform and OpenCV)
- Model-specific settings and processors
- Python export scripts for PyTorch → ExecuTorch conversion

**Important**: When making changes to the example app, always refer to
`packages/executorch_flutter/example/CLAUDE.md` for architecture guidelines
and step-by-step instructions for adding new model support.

## Contact and Support

- **Package Maintainer**: Check `pubspec.yaml` for author information
- **License**: MIT (see LICENSE file)
- **Repository**: Check `pubspec.yaml` for repository URL
- **Issues**: File issues at package repository
- **ExecuTorch**: https://pytorch.org/executorch/

---

**Flutter Version**: 3.38+ (`executorch_flutter` only — `executorch_dart` needs no Flutter)
**API**: Simplified to `load()` + `forward()` + `dispose()` only
**Architecture**: dart:ffi with native assets hooks
