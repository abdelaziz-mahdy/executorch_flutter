# Platform API Contract: macOS Support

**Feature**: macOS Platform Support
**API Version**: 1.0.0 (matches iOS)
**Protocol**: Pigeon-generated Platform Channels

## Contract Overview

This contract defines the platform channel API between Dart (Flutter) and native macOS code. **The API is identical to iOS** - no new methods or modifications required.

## Pigeon Configuration

### Current Configuration (iOS-only)
```dart
@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  swiftOut: 'ios/Classes/Generated/ExecutorchApi.swift',
  kotlinOut: 'android/src/main/kotlin/.../ExecutorchApi.kt',
))
```

### Updated Configuration (iOS + macOS)
```dart
@ConfigurePigeon(PigeonOptions(
  dartOut: 'lib/src/generated/executorch_api.dart',
  swiftOut: 'ios/Classes/Generated/ExecutorchApi.swift',
  // macOS uses same Swift code - no separate output needed
  kotlinOut: 'android/src/main/kotlin/.../ExecutorchApi.kt',
))
```

**Note**: macOS shares the iOS Swift implementation. The generated Pigeon code is platform-agnostic and works on both iOS and macOS without modification.

## Host API (Dart → Platform)

### ExecutorchHostApi

**Purpose**: Methods called from Dart to native platform

#### loadModel
```dart
Future<String> loadModel(String modelPath)
```
**Input**: File path to .pte model
**Output**: Unique model ID string
**Errors**:
- `PlatformException("FILE_NOT_FOUND")` - Model file doesn't exist
- `PlatformException("LOAD_FAILED")` - ExecuTorch failed to load model
- `PlatformException("INVALID_MODEL")` - File is not a valid .pte model

**Platform Implementation**:
```swift
func loadModel(modelPath: String) throws -> String {
    // 1. Verify file exists
    // 2. Load with ExecuTorch Module.load()
    // 3. Store in model registry
    // 4. Return generated model ID
}
```

**macOS-Specific Notes**: File path validation uses macOS file APIs (same as iOS)

#### runInference
```dart
Future<InferenceResult> runInference(
  String modelId,
  List<TensorData> inputs
)
```
**Input**:
- `modelId`: Previously loaded model identifier
- `inputs`: List of input tensors

**Output**: InferenceResult with output tensors and metadata

**Errors**:
- `PlatformException("MODEL_NOT_FOUND")` - Invalid model ID
- `PlatformException("INVALID_INPUT")` - Tensor shape/type mismatch
- `PlatformException("INFERENCE_FAILED")` - Execution error

**Platform Implementation**:
```swift
func runInference(
    modelId: String,
    inputs: [TensorData]
) throws -> InferenceResult {
    // 1. Retrieve model from registry
    // 2. Validate input tensors
    // 3. Execute model.forward(inputs)
    // 4. Package outputs into InferenceResult
}
```

**macOS-Specific Notes**: Uses macOS threading APIs (DispatchQueue)

#### disposeModel
```dart
Future<void> disposeModel(String modelId)
```
**Input**: Model ID to dispose
**Output**: void (success)
**Errors**:
- `PlatformException("MODEL_NOT_FOUND")` - Model already disposed or invalid ID

**Platform Implementation**:
```swift
func disposeModel(modelId: String) throws {
    // 1. Remove from model registry
    // 2. Release ExecuTorch model resources
    // 3. Clean up any associated state
}
```

**macOS-Specific Notes**: ARC handles memory cleanup automatically

#### getModelMetadata
```dart
Future<ModelMetadata> getModelMetadata(String modelId)
```
**Input**: Model ID
**Output**: ModelMetadata with input/output specs

**Errors**:
- `PlatformException("MODEL_NOT_FOUND")` - Invalid model ID

**Platform Implementation**:
```swift
func getModelMetadata(modelId: String) throws -> ModelMetadata {
    // 1. Retrieve model
    // 2. Extract input/output tensor specs
    // 3. Package into ModelMetadata
}
```

**macOS-Specific Notes**: None - metadata extraction is platform-agnostic

## Flutter API (Platform → Dart)

### ExecutorchFlutterApi

**Purpose**: Callbacks from platform to Dart (currently unused, reserved for future)

*No methods currently defined*

**Future Use Cases**:
- Model download progress
- Background inference completion
- Memory pressure warnings

## Data Transfer Objects

### TensorData
```dart
class TensorData {
  final Uint8List data;
  final List<int?> shape;
  final TensorType dataType;
  final String? name;
}
```

**Serialization**: Pigeon handles automatic serialization to/from native types

### InferenceResult
```dart
class InferenceResult {
  final List<TensorData> outputs;
  final InferenceStatus status;
  final double executionTimeMs;
  final String? error;
}
```

### ModelMetadata
```dart
class ModelMetadata {
  final List<TensorSpec> inputSpecs;
  final List<TensorSpec> outputSpecs;
  final String? backend;
  final String? version;
}
```

## Error Handling Contract

### Standard Error Codes
```
FILE_NOT_FOUND - Model file doesn't exist at path
INVALID_MODEL - File is not valid .pte format
LOAD_FAILED - ExecuTorch failed to initialize model
MODEL_NOT_FOUND - Model ID not in registry
INVALID_INPUT - Tensor validation failed
INFERENCE_FAILED - Model execution error
DISPOSED - Operation on disposed model
```

### Error Format
```dart
PlatformException(
  code: "ERROR_CODE",
  message: "Human-readable description",
  details: Map<String, dynamic>? // Optional details
)
```

## Platform Detection

The API itself is platform-agnostic. Platform detection happens at build time:

```yaml
# pubspec.yaml
flutter:
  plugin:
    platforms:
      ios:
        pluginClass: ExecutorchFlutterPlugin
      macos:
        pluginClass: ExecutorchFlutterPlugin  # Same class
```

## Thread Safety Contract

1. **Model Loading**: Thread-safe, can be called concurrently
2. **Inference**: Thread-safe per model instance
3. **Disposal**: Thread-safe, idempotent
4. **Callbacks**: Always delivered on main thread/isolate

## Performance Contract

1. **Model Loading**: <200ms for models <100MB (best effort)
2. **Inference**: Depends on model size, no fixed guarantee
3. **Disposal**: <10ms (synchronous operation)
4. **Metadata Retrieval**: <1ms (cached data)

## Platform Parity Guarantee

**Contract Promise**: All API methods behave identically on iOS and macOS

**Verification**:
- Same test suite runs on both platforms
- Same error codes and messages
- Same threading behavior
- Same performance characteristics (hardware-normalized)

## Compatibility

- **Dart**: Flutter 3.0+
- **iOS**: 13.0+
- **macOS**: 12.0+
- **ExecuTorch**: 0.7.0+

## Changes Required for macOS Support

**Answer**: **ZERO API changes required**

The existing Pigeon contract works on macOS without modification. Only platform-specific files need updating:
1. Package.swift - add macOS platform
2. .podspec - add macOS configuration
3. pubspec.yaml - register macOS plugin

The API contract itself is platform-agnostic by design.
