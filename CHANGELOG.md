# Changelog

All notable changes to this project will be documented in this file.

## 0.0.3 - ExecuTorch 1.0.1 Upgrade

### Dependencies

- **Android**: Upgraded ExecuTorch from `0.7.0` to `1.0.1` stable release
- **iOS**: Upgraded Swift Package Manager dependency from `swiftpm-0.7.0` to `swiftpm-1.0.1`
- **macOS**: Upgraded Swift Package Manager dependency from `swiftpm-0.7.0` to `swiftpm-1.0.1`

### Notes

- This release updates all platforms to use the latest stable ExecuTorch 1.0.1 release
- No API changes - drop-in upgrade from 0.0.2

## 0.0.2 - Swift 6 Compatibility Fix

### Bug Fixes

- **iOS/macOS**: Fixed Swift 6 compilation errors in `ExecutorchModelManager.swift`
  - Added `try` keyword to `withUnsafeBytes` calls for Swift 6 compatibility
  - Resolves build failures on Xcode 16+ with Swift 6 language mode
  - Affects tensor conversion for Float32, Int32, and UInt8 data types

### Dependencies

- **Android**: ExecuTorch `0.7.0`
- **iOS**: Swift Package Manager `swiftpm-0.7.0`
- **macOS**: Swift Package Manager `swiftpm-0.7.0`


## 0.0.1 - Initial Release

Initial release of ExecuTorch Flutter plugin.

### Features

- ✅ Cross-platform support for Android, iOS, and macOS
- ✅ Type-safe Pigeon-generated API for platform communication
- ✅ Async model loading and inference execution
- ✅ Multiple concurrent model instances support
- ✅ Memory-efficient tensor operations
- ✅ Structured error handling with clear exceptions
- ✅ Backend support: XNNPACK, CoreML, MPS
- ✅ Actor-based concurrency (iOS/macOS) and coroutines (Android)
- ✅ Example app with image classification and object detection demos
- ✅ High-level processor interfaces for common ML tasks

### Platform Support

- **Android**: API 23+ (Android 6.0+), arm64-v8a
- **iOS**: iOS 13.0+, arm64 (device only)
- **macOS**: macOS 12.0+ (Monterey), arm64 only (Apple Silicon)

### Known Limitations

- **iOS Simulator (x86_64)**: Not supported
- **macOS Intel (x86_64)**: Not supported
- **macOS Release Builds**: Not working (tracking: [Flutter Issue #176605](https://github.com/flutter/flutter/issues/176605))

### Documentation

- Comprehensive README with usage examples
- Model export guide for converting PyTorch models
- Contributing guide for contributors
- Roadmap for future features
