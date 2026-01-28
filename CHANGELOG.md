# Changelog

## 0.3.0

### Changed
- **ExecuTorch 1.1.0**: Upgraded from ExecuTorch 1.0.1 to 1.1.0
  - Updated native prebuilt binaries version to 1.1.0.1
  - Rebuilt WebAssembly binaries with ExecuTorch 1.1.0
  - Updated Dockerfile.wasm to use v1.1.0 tag

### Added
- **Version-aware model loading**: Models are now loaded from version-specific directories
  - Model index URLs now include version path segment (e.g., `/1.1.0/index.json`)
  - Supports multiple ExecuTorch versions in the models repository
- **Single source of truth for version**: `executorchVersion` constant in `lib/src/version.dart`
  - Used by native build system, web platform, and model loading
  - Eliminates duplicate version constants across the codebase
- **New models support**: YOLO-Pose and YOLO-Face models available in 1.1.0
- **Example app version selector**: UI to load models from different ExecuTorch versions
  - Test compatibility of old/new models with the current runtime
  - Decorator pattern architecture for ModelIndexService (clean separation of HTTP and caching)
- **Version-aware model caching**: Models cached by ExecuTorch version
  - Cache structure: `models/{version}/{modelName}.pte`
  - SHA-256 hash verification ensures cached models match expected content
  - Automatic re-download on hash mismatch (stale cache detection)
  - Decorator pattern architecture for ModelDownloadService

### Removed
- **User-overridable ExecuTorch version**: The `executorch_version` option in
  `pubspec.yaml` user_defines has been removed
  - Package version is now tied to ExecuTorch version (1:1 mapping)
  - Users who need older ExecuTorch versions should use older package versions
  - Simplifies build system and ensures runtime/model version consistency

### Fixed
- Fixed WASM build script triggering unnecessary native builds
  - `build_wasm.sh` now copies files directly instead of using `dart run`
- Fixed model dropdown being empty when switching to versions without `platforms`
  field in index.json (backward compatibility with 1.0.1 models)

## 0.2.1

### Fixed
- Updated README with correct version references and prebuilt version (1.0.1.21)
- Fixed library size report links (now clearly labeled as downloads)

## 0.2.0

### Added
- Native Assets build system with prebuilt binary support
- Build configuration via `pubspec.yaml` (debug mode, backends, versions)
- `BackendQuery` API for runtime backend availability checks
  - `BackendQuery.isAvailable(Backend)` - Check if a specific backend is compiled in
  - `BackendQuery.available` - Get list of all available backends
- Backend filtering in example app - only shows models for available backends
- Vulkan backend available as opt-in experimental feature on all native platforms
  - macOS: MoltenVK bundled with prebuilt binaries (Vulkan-to-Metal translation)
  - Known issues documented in README (Android UBO limits, experimental status)
- Integration tests for backend query functionality
- Windows x64 platform support
- Linux x64 and arm64 platform support
- iOS Simulator support (arm64 and x86_64)
- macOS x86_64 (Intel Mac) support
- Android x86_64 architecture for emulator testing
- Windows and Linux CI workflows

### Changed
- Migrate from Pigeon to FFI for native communication
- Reduce memory copying during tensor operations with direct FFI access
- Prebuilt binaries now download automatically (faster builds)
- Default build mode is now "prebuilt" for faster development

### Removed
- Swift Package Manager dependency (replaced by Native Assets)
- Pigeon-based method channels

## 0.1.1

### Example App Improvements
- Models are now downloaded from remote GitHub repository instead of bundled in assets
- Significantly reduces app size by loading models on-demand
- Added cache-busting for GitHub model requests to ensure fresh downloads
- Updated integration tests to download models from remote URL

### Bug Fixes
- Fixed camera provider defaulting to OpenCV on all platforms for better compatibility
- Fixed broken pipe errors in integration test script

### Dependencies
- Fixed `plugin_platform_interface` version constraint

## 0.1.0

- **Web Platform Support**: WebAssembly with XNNPACK backend
- Added Docker-based Wasm build system
- Added `setup_web` CLI tool for web project setup

## 0.0.6
- **Android** fix Crash on model reuse

## 0.0.5

### Bug Fixes
- **Android**: Added ProGuard rules to prevent crashes in release builds when loading models
- **iOS**: Fixed camera initialization by using `bgra8888` image format instead of `yuv420`

## 0.0.4

### Improvements
- Converted internal Pigeon API to async for better thread safety on iOS/macOS
- Fixed race conditions in example app when disposing models during camera mode
- Fixed UI getting stuck in camera mode when model loading fails

### Code Quality
- Fixed 100+ static analysis issues
- Removed 9 deprecated lint rules (Dart 3.0-3.7)
- Migrated to Flutter 3.32+ `RadioGroup` API
- Added documentation for `ProcessorException` classes
- Removed dead code in example app renderers and controllers

## 0.0.3

- Upgraded ExecuTorch to 1.0.1 on all platforms
- No API changes from 0.0.2

## 0.0.2

- Fixed Swift 6 compilation errors for Xcode 16+

## 0.0.1

Initial release with Android, iOS, and macOS support.

- Type-safe Pigeon API
- Async model loading and inference
- XNNPACK, CoreML, MPS backends
- Example app with classification and detection demos
