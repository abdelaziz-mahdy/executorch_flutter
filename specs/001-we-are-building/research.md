# Research Findings: Flutter ExecuTorch Package

## Technology Decisions

### Flutter Plugin Development Approach
**Decision**: Create a federated Flutter plugin using Pigeon for type-safe method channel communication
**Rationale**:
- Pigeon generates boilerplate code for both Dart and native platforms, reducing errors
- Type-safe APIs prevent runtime serialization issues
- Federated plugins follow current Flutter best practices for platform-specific code
- Supports both Android and iOS with shared Dart interface

**Alternatives considered**:
- Manual method channels: More error-prone, requires manual serialization
- FFI (Foreign Function Interface): Complex C++ binding, harder to maintain
- Basic plugin template: No type safety, manual platform registration

### ExecuTorch Integration Strategy
**Decision**: Use official ExecuTorch libraries (Android AAR + iOS frameworks) with native wrapper classes
**Rationale**:
- Official libraries ensure compatibility and performance optimization
- Pre-built AAR/frameworks reduce build complexity
- Native wrappers provide clean abstraction for Flutter integration
- Supports latest ExecuTorch features and model formats

**Alternatives considered**:
- Custom ExecuTorch compilation: Complex build setup, maintenance overhead
- Third-party bindings: Potential compatibility issues, limited support
- Direct C++ integration: Requires extensive platform-specific code

### Platform Integration Approach
**Decision**: Kotlin for Android, Swift for iOS with unified Pigeon interface
**Rationale**:
- Modern language features for better memory management and error handling
- Kotlin coroutines and Swift async/await for non-blocking operations
- Better ExecuTorch library compatibility than Java/Objective-C
- Simplified null safety and resource management

**Alternatives considered**:
- Java + Objective-C: Older patterns, more verbose error handling
- Pure C++: Cross-platform but complex Flutter integration
- React Native approach: Not applicable to Flutter ecosystem

### Model File Handling Strategy
**Decision**: Asynchronous file loading with memory-mapped model access
**Rationale**:
- Prevents UI blocking during large model loading
- Memory mapping reduces RAM usage for large models
- Flutter's isolate-friendly for background processing
- Supports both asset bundling and external file access

**Alternatives considered**:
- Synchronous loading: Would block UI thread
- Full memory loading: Excessive RAM usage for large models
- Streaming loading: Complex implementation, limited benefit

### API Design Pattern
**Decision**: Future-based async API with structured error handling
**Rationale**:
- Fits Flutter's async/await patterns naturally
- Non-blocking inference operations maintain UI responsiveness
- Structured exceptions provide clear error information
- Supports cancellation and timeout handling

**Alternatives considered**:
- Stream-based API: Overkill for single inference requests
- Callback-based API: Less idiomatic in modern Dart
- Synchronous API: Would block UI thread

## Technical Implementation Details

### Android Integration (Updated 2025-09-20)
- **ExecuTorch Version**: 0.7.0 (latest stable AAR from Maven Central)
- **Required Dependencies**:
  - `org.pytorch:executorch-android:0.7.0`
  - `com.facebook.soloader:soloader:0.10.5`
  - `com.facebook.fbjni:fbjni:0.5.1`
- **Architecture Support**: arm64-v8a (primary), x86_64 (for emulators)
- **Build Tools**: Android SDK + NDK r27b
- **Available Backends**: XNNPACK (CPU), MediaTek NeuroPilot (NPU), Qualcomm AI Engine (NPU), Vulkan (GPU)
- **Device Compatibility**: Phones, tablets, TV boxes across Android ecosystem

### iOS Integration (Updated 2025-09-20)
- **Platform Requirements**: iOS/macOS with ARM64 architecture only
- **Development Tools**: Xcode 15+, Python 3.10+, Swift 5.9+
- **Framework Components**:
  - `executorch`: Core runtime
  - `backend_coreml`: Core ML backend
  - `backend_mps`: Metal Performance Shaders backend
  - `backend_xnnpack`: XNNPACK backend
  - `kernels_custom`: Custom kernels for LLMs
  - `kernels_optimized`: Accelerated CPU kernels
  - `kernels_quantized`: Quantized kernels
- **Integration Method**: Swift Package Manager (recommended) or building from source
- **Performance**: Use Release builds for optimal performance, Debug builds for development logging

### Verified API Integration Patterns (2025-09-20)

**Android ExecuTorch Runtime Pattern**:
```java
Module module = Module.load("/path/to/model.pte");
Tensor input = Tensor.fromBlob(inputData, inputShape);
EValue inputEValue = EValue.from(input);
float[] result = module.forward(inputEValue).toTensor().getDataAsFloatArray();
```

**iOS ExecuTorch Runtime Pattern**:
```swift
let module = Module(filePath: modelPath)
try module.load("forward")
let inputTensor = Tensor<Float>(&imageBuffer, shape: [1, 3, 224, 224])
let outputTensor = try Tensor<Float>(module.forward(inputTensor))
let logits = outputTensor.scalars()
```

**Flutter Pigeon Interface Specification**:
```dart
// Core API methods matching verified ExecuTorch patterns
@HostApi()
abstract class ExecutorchHostApi {
  @async
  ModelLoadResult loadModel(String filePath);
  @async
  InferenceResult runInference(InferenceRequest request);
  ModelMetadata? getModelMetadata(String modelId);
  void disposeModel(String modelId);
}

// Data structures aligned with ExecuTorch tensor operations
class InferenceRequest {
  String modelId;
  List<TensorData> inputs;
  Map<String, Object>? options;
  int? timeoutMs;
}
```

### Performance Considerations
- **Model Loading**: Target <200ms for models up to 100MB
- **Inference Speed**: <50ms for typical mobile models
- **Memory Usage**: <100MB additional RAM during inference
- **Concurrent Models**: Support 2-3 models simultaneously
- **Error Recovery**: Graceful handling of OOM and invalid model states

### Testing Strategy
- **Unit Tests**: Dart-only logic, mock platform channels
- **Integration Tests**: Full native stack with test models
- **Platform Tests**: Native Android/iOS unit tests for ExecuTorch integration
- **Performance Tests**: Memory and latency benchmarks
- **Example App**: Real-world usage demonstration

## Security and Deployment
- **Model Validation**: File format and size validation before loading
- **Sandboxing**: Respect platform security models (iOS app sandbox, Android permissions)
- **Asset Protection**: Support for bundled models in app assets
- **Crash Protection**: Isolate native crashes from Flutter app
- **Pub.dev Publishing**: Standard package publication with platform documentation

## Known Limitations and Mitigations
- **Model Size**: Large models (>500MB) may require streaming/chunked loading
- **Platform Support**: Initially Android/iOS only, desktop platforms possible future addition
- **ExecuTorch Updates**: Plugin versioning tied to ExecuTorch release cycle
- **Debugging**: Native debugging requires platform-specific tools (Android Studio, Xcode)
- **Hot Reload**: Model instances may need reloading during development