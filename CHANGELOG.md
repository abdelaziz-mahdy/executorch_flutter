# Changelog

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
