# Research: macOS Platform Support

**Feature**: Add macOS support to ExecuTorch Flutter Plugin
**Date**: 2025-10-02
**Status**: Complete

## Executive Summary

ExecuTorch officially supports macOS with the same API surface and frameworks as iOS. The implementation can reuse 100% of the iOS Swift code with minimal platform-specific adjustments (UIKit → AppKit). The primary work involves configuration updates to support macOS as a target platform.

## Research Questions & Answers

### 1. ExecuTorch macOS Compatibility

**Question**: Does ExecuTorch officially support macOS? What are the requirements?

**Decision**: ExecuTorch fully supports macOS 12+ with identical APIs to iOS

**Rationale**:
- Official documentation explicitly lists macOS support alongside iOS
- Uses same .xcframework binaries for both platforms
- Swift Package Manager supports both platforms in single package definition
- Backends (XNNPACK, Core ML, MPS) available on both platforms

**Technical Evidence**:
```swift
// From ExecuTorch SPM package
platforms: [
  .iOS(.v17),
  .macOS(.v12),
]
```

### 2. Platform-Specific Code Sharing

**Question**: Can iOS Swift implementation be reused for macOS?

**Decision**: Reuse 100% of implementation with conditional compilation for UI frameworks

**Rationale**:
- Core ExecuTorch APIs are identical across platforms
- Model loading, inference, and lifecycle management are platform-agnostic
- Only UI framework references (UIKit vs AppKit) need platform checks
- Pigeon platform channels work identically on both platforms

**Implementation Pattern**:
```swift
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif
```

### 3. Flutter Plugin Structure

**Question**: Should we create separate `macos/` directory or share code?

**Decision**: Create `macos/` directory that references shared iOS implementation

**Rationale**:
- Flutter convention requires separate platform directories
- Allows platform-specific configuration (podspec, Package.swift)
- Can use symlinks or file references to share implementation
- Standard Flutter plugin structure expected by build tools

**Structure**:
```
ios/Classes/            # Shared Swift implementation
macos/Classes/          # Symlink → ios/Classes
```

### 4. Minimum macOS Version

**Decision**: macOS 12 (Monterey) minimum

**Rationale**: Aligned with ExecuTorch requirement (.macOS(.v12))

## Technology Stack

- **Language**: Swift 5.9+
- **Package Manager**: Swift Package Manager + CocoaPods
- **ExecuTorch Version**: 0.7.0 (swiftpm-0.7.0 branch)
- **Deployment Target**: macOS 12.0
- **Architectures**: arm64, x86_64 (Universal Binary)

## System Frameworks (macOS)
- Metal: GPU acceleration (MPS backend)
- Accelerate: BLAS/LAPACK operations
- Core ML: Core ML backend support
- Foundation: Base framework
- AppKit: macOS UI framework

## Risk Assessment

✅ **Low Risk**: Code reuse, official ExecuTorch support, mature Flutter macOS support
⚠️ **Medium Risk**: Platform differences (UIKit vs AppKit), Intel vs Apple Silicon performance

## References
- [ExecuTorch iOS/macOS Documentation](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)
- [Flutter macOS Plugin Development](https://docs.flutter.dev/platform-integration/macos/c-interop)
