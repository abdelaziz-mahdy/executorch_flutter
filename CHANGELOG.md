# Changelog

## 0.2.2

### Fixed
- Fixed `.pubignore` pattern `build/` excluding `lib/src/build/run_build.dart` from published package
  - This caused "No such file or directory" errors when using the package from pub.dev
  - Changed to `/build/` to only match root-level build directory

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
