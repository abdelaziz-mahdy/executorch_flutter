# ExecuTorch Flutter Plugin - AI Agent Context

## Package Overview

**executorch_flutter** is a Flutter plugin package that provides on-device machine learning inference using PyTorch ExecuTorch. It enables Flutter developers to run optimized ML models on mobile and desktop platforms with a simple, type-safe Dart API.

**Package Name**: `executorch_flutter`
**Version**: 0.0.1 (pre-release)
**License**: MIT
**Platforms**: Android, iOS, macOS

## Current Development Status

- **Phase**: Package implementation complete, ready for publishing
- **Code Quality**: 0 lint errors in `lib/`, all dart fixes applied
- **Build Status**: ✅ Android APK, ✅ macOS app, ✅ iOS (device only)
- **Package Size**: ~13 MB compressed for pub.dev
- **Next Step**: Commit changes and publish to pub.dev

## Core Architecture

### Technology Stack

- **Flutter Plugin**: Federated plugin architecture with platform-specific implementations
- **Platform Communication**: Pigeon v22.7.0 for type-safe method channel code generation
- **Android**: Kotlin + ExecuTorch AAR 1.0.0-rc2 + Coroutines
- **iOS/macOS**: Swift + ExecuTorch XCFrameworks (SPM 1.0.0) + async/await
- **Memory Management**: User-controlled lifecycle (explicit load/dispose)

### Design Principles

1. **Type Safety**: All platform communication via Pigeon-generated code (no manual method channels)
2. **Async/Await**: All model operations are non-blocking
3. **User-Controlled Resources**: Developers explicitly manage model lifecycle (no automatic cleanup)
4. **Structured Errors**: Exception hierarchy with clear error categories
5. **Platform Parity**: Identical behavior across Android, iOS, and macOS

## Platform Support

### Android
- **Minimum SDK**: API 23 (Android 6.0)
- **Architectures**: arm64-v8a (primary), x86_64 (emulator)
- **Dependencies**:
  - ExecuTorch AAR 1.0.0-rc2 (`org.pytorch:executorch-android:1.0.0-rc2`)
  - Available at: https://repo.maven.apache.org/maven2/org/pytorch/executorch-android/
  - FBJNI (JNI bridge)
  - SoLoader (native library loading)
- **Threading**: Kotlin coroutines on background dispatcher
- **Implementation**: `android/src/main/kotlin/com/zcreations/executorch_flutter/`

### iOS
- **Minimum Version**: iOS 13.0
- **Architectures**: arm64 (physical devices only, no simulator support)
- **Dependencies**: ExecuTorch XCFrameworks via Swift Package Manager (SPM 1.0.0)
  - Branch: `swiftpm-1.0.0` from https://github.com/pytorch/executorch.git
- **Threading**: Swift async/await with Task detachment
- **Implementation**: Shared sources via symlinks from `darwin/`
- **Note**: Simulator support requires x86_64 ExecuTorch builds (not currently available)

### macOS
- **Minimum Version**: macOS 11.0
- **Architectures**: arm64 (Apple Silicon only)
- **Dependencies**: ExecuTorch XCFrameworks via Swift Package Manager (SPM 1.0.0)
  - Branch: `swiftpm-1.0.0` from https://github.com/pytorch/executorch.git
- **Threading**: Swift async/await with Task detachment
- **Implementation**: Shared sources via symlinks from `darwin/`
- **Platform-Specific APIs**: Conditional compilation using `#if os(iOS)` / `#if os(macOS)`

## Project Structure

```
executorch_flutter/
├── lib/
│   ├── executorch_flutter.dart              # Main library export
│   └── src/
│       ├── executorch_model.dart            # ExecuTorchModel wrapper
│       ├── executorch_inference.dart        # ExecutorchManager API
│       ├── executorch_errors.dart           # Exception hierarchy
│       ├── processors/
│       │   ├── base_processor.dart          # BaseInputProcessor/BaseOutputProcessor
│       │   ├── yolo_processor.dart          # YOLOv8 pre/post processing
│       │   └── image_classification_processor.dart  # MobileNet processors
│       └── generated/
│           └── executorch_api.dart          # Pigeon-generated code
├── android/
│   └── src/main/kotlin/com/zcreations/executorch_flutter/
│       ├── ExecutorchFlutterPlugin.kt       # Plugin registration
│       ├── ExecutorchModelManager.kt        # Model lifecycle
│       ├── ExecutorchTensorUtils.kt         # Tensor conversion
│       └── Generated/ExecutorchApi.kt       # Pigeon-generated Kotlin
├── ios/
│   └── Classes/
│       ├── ExecutorchFlutterPlugin.swift    # Plugin registration
│       ├── ExecutorchModelManager.swift     # Model lifecycle
│       ├── ExecutorchTensorUtils.swift      # Tensor conversion
│       └── Generated/ExecutorchApi.swift    # Pigeon-generated Swift
├── macos/
│   └── Classes/
│       ├── ExecutorchFlutterPlugin.swift    # macOS plugin registration
│       ├── ExecutorchModelManager.swift     # macOS model lifecycle
│       ├── ExecutorchTensorUtils.swift      # macOS tensor conversion
│       └── Generated/ExecutorchApi.swift    # Pigeon-generated Swift
├── darwin/                                   # Shared iOS/macOS sources (symlinked)
│   ├── ExecutorchFlutterPlugin.swift
│   ├── ExecutorchModelManager.swift
│   └── ExecutorchTensorUtils.swift
├── pigeons/
│   └── executorch_api.dart                  # Pigeon interface definitions
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
│   ├── generate_pigeon.sh                   # Regenerate Pigeon code
│   └── setup_models.py                      # Download/convert models
└── python/
    └── convert_model.py                     # PyTorch → ExecuTorch converter
```

## Key APIs

### ExecutorchManager (Singleton)

Primary interface for model management:

```dart
// Initialize (call once at app startup)
await ExecutorchManager.instance.initialize();

// Enable debug logging
await ExecutorchManager.instance.setDebugLogging(true);

// Load a model
final model = await ExecutorchManager.instance.loadModel('/path/to/model.pte');

// Get loaded model by ID
final model = ExecutorchManager.instance.getLoadedModel('model_id');

// Run inference (convenience method)
final result = await ExecutorchManager.instance.runInference(
  modelId: 'model_id',
  inputs: [tensorData],
);

// Dispose model when done
await ExecutorchManager.instance.disposeModel('model_id');

// Cleanup all models
await ExecutorchManager.instance.disposeAllModels();

// Shutdown on app exit
await ExecutorchManager.instance.shutdown();
```

### ExecuTorchModel

Represents a loaded model instance:

```dart
// Load from file
final model = await ExecuTorchModel.loadFromFile('/path/to/model.pte');

// Model properties
print(model.modelId);        // Unique identifier
print(model.filePath);       // Original file path
print(model.inputShapes);    // Expected input tensor shapes
print(model.outputShapes);   // Expected output tensor shapes

// Run inference
final result = await model.runInference(
  inputs: [tensorData],
  options: {'key': 'value'},
  timeoutMs: 5000,
  requestId: 'unique_request_id',
);

// Dispose when done
await model.dispose();
```

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

### InferenceResult (Pigeon-generated)

Result of inference execution:

```dart
result.status;          // InferenceStatus.success / error / timeout
result.outputs;         // List<TensorData>
result.errorMessage;    // String? (if error)
result.durationMs;      // int? (execution time)
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

1. **Load models**: Call `loadModel()` when needed
2. **Dispose models**: Call `dispose()` when done
3. **Handle errors**: Catch exceptions and clean up resources
4. **Monitor memory**: Use OS tools to track memory usage

**No Automatic Cleanup**: There is no lifecycle manager, no memory pressure monitoring, no automatic disposal. This design gives developers full control over when models are loaded/unloaded.

**Why User-Controlled?**
- Predictable behavior (no surprise disposals mid-inference)
- Explicit resource management (developers know when models are in memory)
- Platform parity (same behavior on Android, iOS, macOS)

## Pigeon Code Generation

### Workflow

1. **Edit interface**: Modify `pigeons/executorch_api.dart`
2. **Generate code**: Run `./scripts/generate_pigeon.sh` or `flutter pub run pigeon --input pigeons/executorch_api.dart`
3. **Implement native**: Add implementations in Kotlin/Swift
4. **Test**: Run example app to verify

### Generated Files

- `lib/src/generated/executorch_api.dart` (Dart)
- `android/src/main/kotlin/com/zcreations/executorch_flutter/Generated/ExecutorchApi.kt` (Kotlin)
- `ios/Classes/Generated/ExecutorchApi.swift` (iOS Swift)
- `macos/Classes/Generated/ExecutorchApi.swift` (macOS Swift)

**Important**: Generated files ARE committed to version control.

### Current Pigeon Interfaces

```dart
// Host API: Dart → Native
abstract class ExecutorchHostApi {
  ModelLoadResult loadModel(String filePath, String modelId);
  InferenceResult forward(String modelId, List<TensorData> inputs, String requestId);
  void disposeModel(String modelId);
  List<String> getLoadedModels();
  void setDebugLogging(bool enabled);
}

// Flutter API: Native → Dart (unused, reserved for callbacks)
abstract class ExecutorchFlutterApi {
  void onModelLoadProgress(String modelId, double progress);
  void onInferenceProgress(String requestId, String status);
}
```

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
suspend fun loadModel(filePath: String, modelId: String): ModelLoadResult =
  withContext(Dispatchers.Default) {
    // ExecuTorch Module.load() on background thread
  }
```

**Memory Mapping**: Uses `Module.load(path, LoadMode.MMAP)` for large models

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
func loadModel(filePath: String, modelId: String) async throws -> ModelLoadResult {
  return try await Task.detached {
    // ExecuTorchModule initialization on background task
  }.value
}
```

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
- **Issue**: File not found, invalid .pte format, or corrupted model
- **Solution**:
  - Verify file path with `File(path).existsSync()`
  - Check .pte format with `file` command (should be "data")
  - Re-export model from PyTorch with correct ExecuTorch version

**4. Inference Returns Error**
- **Issue**: Wrong tensor shapes, data types, or model compatibility
- **Solution**:
  - Enable debug logging: `setDebugLogging(true)`
  - Check `model.inputShapes` and `model.outputShapes`
  - Verify tensor data types match model expectations
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
- **ExecuTorch Logging**: `setDebugLogging(true)` for detailed logs

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

**Breaking Changes from Earlier Iterations**:
- Removed automatic lifecycle management
- Removed `ExecutorchLifecycleManager` (Android/iOS/macOS)
- Users must explicitly call `dispose()` on models

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
- Asset bundle loading (currently file path only)
- Model caching and version management

## Contact and Support

- **Package Maintainer**: Check `pubspec.yaml` for author information
- **License**: MIT (see LICENSE file)
- **Repository**: Check `pubspec.yaml` for repository URL
- **Issues**: File issues at package repository
- **ExecuTorch**: https://pytorch.org/executorch/

---

**Last Updated**: 2025-10-07
**Package Version**: 0.0.1 (pre-release)
**ExecuTorch Version**: 1.0.0-rc2 (Android AAR) / 1.0.0 (iOS/macOS SPM)
