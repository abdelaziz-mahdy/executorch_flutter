# Pigeon Multi-Platform Generation Notes

## Issue Reference
Flutter Issue #164297: [pigeon] Allow multiple output locations for generated Swift
- **Status**: Open (P2 priority)
- **URL**: https://github.com/flutter/flutter/issues/164297

## Problem
Pigeon currently doesn't support multiple output locations for Swift code in a single `@ConfigurePigeon` annotation. This is needed when supporting both iOS and macOS platforms with different implementations.

## Current Limitation
The current Pigeon configuration only supports one Swift output location:

```dart
@ConfigurePigeon(PigeonOptions(
  swiftOut: 'ios/Classes/Generated/ExecutorchApi.swift',  // Only one location
))
```

## Workaround Solution
Since Pigeon doesn't support multiple Swift outputs natively, we must:

1. **Generate once for iOS** (primary location)
2. **Copy generated file to macOS** location via script

### Implementation Steps:

1. Configure Pigeon to generate for iOS:
```dart
@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  kotlinOut: 'android/src/main/kotlin/.../ExecutorchApi.kt',
  swiftOut: 'ios/Classes/Generated/ExecutorchApi.swift',
  swiftOptions: SwiftOptions(),
))
```

2. Create a generation script that:
   - Runs `flutter pub run pigeon --input pigeons/executorch_api.dart`
   - Copies `ios/Classes/Generated/ExecutorchApi.swift` to `macos/executorch_flutter/Sources/executorch_flutter/Generated/ExecutorchApi.swift`

## Critical: PigeonError vs FlutterError

### From Pigeon Documentation
**Source**: https://github.com/flutter/packages/blob/main/packages/pigeon/example/README.md

**Important Swift Change**:
> For Swift, use `PigeonError` instead of `FlutterError` when throwing an error.

### Why This Matters
- `FlutterError` does **NOT** conform to Swift's `Error` protocol on macOS
- Pigeon generates code using `Result<Void, FlutterError>` which causes compilation errors
- The correct approach is to use `PigeonError` which properly conforms to `Error`

### Error Manifestation
When using `FlutterError` in generated Pigeon code on macOS:
```swift
// THIS FAILS ON MACOS:
func onModelLoadProgress(..., completion: @escaping (Result<Void, FlutterError>) -> Void)
// Error: type 'FlutterError' does not conform to protocol 'Error'
```

### Solution
Use `PigeonError` in the Pigeon interface definition for proper error handling across platforms.

## Directory Structure for macOS Support

### iOS Structure (CocoaPods):
```
ios/
  Classes/
    ExecutorchFlutterPlugin.swift
    ExecutorchModelManager.swift
    ExecutorchLifecycleManager.swift
    ExecutorchTensorUtils.swift
    Generated/
      ExecutorchApi.swift  (Pigeon-generated)
  executorch_flutter.podspec
```

### macOS Structure (Swift Package Manager):
```
macos/
  executorch_flutter/
    Package.swift
    Sources/
      executorch_flutter/
        ExecutorchFlutterPlugin.swift
        ExecutorchModelManager.swift
        ExecutorchLifecycleManager.swift
        ExecutorchTensorUtils.swift
        Generated/
          ExecutorchApi.swift  (Copied from iOS)
```

## Platform-Specific Considerations

### Conditional Compilation
Both iOS and macOS implementations share the same Swift source files but use conditional compilation:

```swift
#if os(iOS)
import Flutter
import UIKit
#elseif os(macOS)
import FlutterMacOS
import AppKit
#endif
```

### Pigeon Generated Code
The Pigeon-generated code already includes platform-specific imports:

```swift
#if os(iOS)
  import Flutter
#elseif os(macOS)
  import FlutterMacOS
#else
  #error("Unsupported platform.")
#endif
```

## Recommended Pigeon Generation Script

Create `scripts/generate_pigeon.sh`:

```bash
#!/bin/bash
set -e

echo "Generating Pigeon code..."
flutter pub run pigeon --input pigeons/executorch_api.dart

echo "Copying Swift code to macOS..."
mkdir -p macos/executorch_flutter/Sources/executorch_flutter/Generated
cp ios/Classes/Generated/ExecutorchApi.swift \
   macos/executorch_flutter/Sources/executorch_flutter/Generated/ExecutorchApi.swift

echo "Pigeon generation complete!"
```

## References

1. **Flutter Issue**: https://github.com/flutter/flutter/issues/164297
2. **Pigeon Examples**: https://github.com/flutter/packages/blob/main/packages/pigeon/example/README.md
3. **Pigeon Documentation**: https://pub.dev/packages/pigeon
4. **Swift Package Manager for Flutter**: https://docs.flutter.dev/packages-and-plugins/swift-package-manager/for-plugin-authors

## Debug vs Release Builds

### ExecuTorch Logging Framework
ExecuTorch provides debug logging through the `Log.shared` singleton with custom `LogSink` implementations. However:

- **Debug builds**: Link against `executorch_debug` framework (logs available)
- **Release builds**: Link against `executorch` framework (logs stripped for performance)

### Automatic Build Configuration
The Package.swift files automatically switch between debug and release frameworks:

```swift
dependencies: [
    // Debug builds use executorch_debug for detailed logging
    .product(name: "executorch_debug", package: "executorch",
             condition: .when(configuration: .debug)),
    // Release builds use executorch for optimal performance
    .product(name: "executorch", package: "executorch",
             condition: .when(configuration: .release)),
]
```

### Flutter Build Modes
- `flutter run` → Debug configuration → `executorch_debug` linked
- `flutter build ios/macos` → Release configuration → `executorch` linked
- `flutter run --release` → Release configuration → `executorch` linked

### Logging API
The plugin provides `setDebugLogging(bool enabled)` which:
- **Debug builds**: Enables/disables ExecuTorch internal logging
- **Release builds**: Prints warning (logs not available)

```swift
#if DEBUG
Log.shared.add(sink: customLogSink)  // Works
#else
print("⚠️ Debug logging only available in DEBUG builds")  // No-op
#endif
```

## Notes for Future Development

- Monitor Flutter Issue #164297 for native multi-output support
- When Pigeon adds multi-output support, update configuration to:
  ```dart
  swiftOut: ['ios/Classes/Generated/ExecutorchApi.swift',
             'macos/executorch_flutter/Sources/executorch_flutter/Generated/ExecutorchApi.swift']
  ```
- Consider using `PigeonError` instead of `FlutterError` if error conformance issues arise
- Keep iOS and macOS implementations in sync manually until better tooling exists

## Implementation Status

- [x] Pigeon configured for iOS generation
- [x] Manual copy process documented
- [ ] Automated generation script created
- [ ] PigeonError migration (if needed)
- [ ] macOS platform fully building
- [ ] iOS platform verified working
