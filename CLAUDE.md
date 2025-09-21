# Flutter ExecuTorch Package - Development Context

## Project Overview
This is a Flutter plugin package that provides on-device machine learning inference capabilities using ExecuTorch. The package enables Flutter developers to load and run ExecuTorch models on Android and iOS platforms with a simple, type-safe Dart API.

## Current Development Status
- **Phase**: Implementation planning completed
- **Branch**: `001-we-are-building`
- **Next Step**: Task generation via `/tasks` command

## Architecture Decisions

### Core Technology Stack
- **Flutter Plugin**: Federated plugin architecture for cross-platform support
- **Native Communication**: Pigeon for type-safe method channel code generation
- **Android Integration**: Kotlin + ExecuTorch Android AAR (0.6.0+)
- **iOS Integration**: Swift + ExecuTorch iOS frameworks
- **API Pattern**: Async/await with structured error handling

### Key Design Principles
1. **Type Safety**: Pigeon generates type-safe interfaces across Dart/native boundaries
2. **Non-blocking**: All model operations are asynchronous to prevent UI blocking
3. **Resource Management**: Explicit model disposal and memory management
4. **Error Handling**: Structured exceptions with clear error categories

## Project Structure
```
├── lib/
│   ├── executorch_flutter.dart         # Main library export
│   └── src/
│       ├── executorch_model.dart       # Model wrapper class
│       ├── executorch_inference.dart   # Inference handling
│       ├── executorch_types.dart       # Data types and enums
│       └── generated/                  # Pigeon generated code
├── android/                            # Android platform implementation
├── ios/                               # iOS platform implementation
├── pigeons/                           # Pigeon interface definitions
├── example/                           # Example Flutter app
└── specs/001-we-are-building/         # Current feature specifications
```

## Key APIs and Contracts

### Main Classes
- `ExecutorchManager`: Primary interface for model management
- `ExecuTorchModel`: Represents a loaded model instance
- `InferenceRequest`: Input data and parameters for inference
- `InferenceResult`: Output data and execution metadata
- `TensorData`: Input/output tensor representation

### Pigeon Interface
Located in `specs/001-we-are-building/contracts/executorch_api.dart`:
- `ExecutorchHostApi`: Dart → Native platform calls
- `ExecutorchFlutterApi`: Native → Dart callbacks (optional)
- Type-safe data classes for all model and inference operations

## Platform Integration Details

### Android
- **Minimum SDK**: API 23 (Android 6.0)
- **Architecture**: arm64-v8a primary target
- **Dependencies**: ExecuTorch AAR, FBJNI, SoLoader
- **Implementation**: Kotlin with coroutines for async operations

### iOS
- **Minimum Version**: iOS 13.0
- **Architecture**: arm64 (device), x86_64 (simulator)
- **Dependencies**: ExecuTorch XCFrameworks
- **Implementation**: Swift with async/await

## Development Guidelines

### Code Generation
1. Pigeon interfaces defined in `pigeons/executorch_api.dart`
2. Run `flutter packages pub run pigeon` to generate platform code
3. Generated files are committed to version control

### Testing Strategy
- **Unit Tests**: Dart-only logic with mocked platform channels
- **Integration Tests**: Full native stack with test models
- **Platform Tests**: Native Android/iOS unit tests
- **Example App**: Real-world usage demonstration

### Performance Targets
- **Model Loading**: <200ms for models up to 100MB
- **Inference**: <50ms for typical mobile models
- **Memory**: <100MB additional RAM during inference
- **Concurrent Models**: Support 2-3 models simultaneously

## Common Development Tasks

### Adding New API Methods
1. Update Pigeon interface in `pigeons/executorch_api.dart`
2. Regenerate platform code with Pigeon
3. Implement native platform methods (Android Kotlin, iOS Swift)
4. Add Dart wrapper methods in `lib/src/`
5. Update tests and documentation

### Platform-Specific Features
- Android: Use Kotlin coroutines for background operations
- iOS: Use Swift async/await and proper ARC memory management
- Both: Handle ExecuTorch lifecycle and error propagation

### Debugging Tips
- Use `flutter logs` for cross-platform debugging
- Android: Check logcat for native ExecuTorch errors
- iOS: Use Xcode debugger for native Swift code
- Enable ExecuTorch logging for detailed execution info

## Known Constraints
- Model format: Only `.pte` (ExecuTorch) files supported
- Platform support: Android/iOS only (desktop platforms future consideration)
- Model size: Large models (>500MB) may require streaming approaches
- Threading: Native inference runs on background threads, callbacks to main thread

## Constitutional Requirements
**IMPORTANT**: All development must comply with `.specify/memory/constitution.md` v1.0.0

### Key Compliance Points
- **Test-First**: All features require failing tests before implementation
- **Platform Parity**: Identical behavior across Android/iOS required
- **Type-Safe APIs**: Pigeon-only communication, no manual method channels
- **Performance Targets**: <200ms loading, <50ms inference, <100MB memory
- **Official Libraries**: ExecuTorch AAR 0.6.0+ (Android), frameworks (iOS)

### Latest ExecuTorch Integration (verified 2025-09-20)
- **Android AAR**: Use `org.pytorch:executorch-android:0.6.0+` dependency
- **iOS Frameworks**: Build with `./scripts/build_apple_frameworks.sh`
- **API Pattern**: `Module.load()` → `module.forward()` → tensor operations
- **Memory**: Use memory mapping for large models (>500MB)
- **Validation**: Check .pte format, tensor shapes, and resource constraints

## Recent Specification Work
- **Feature Spec**: Flutter ExecuTorch package requirements and user stories
- **Research**: Technology decisions and integration patterns (Context7 verified)
- **Data Model**: Core entity definitions and relationships
- **API Contracts**: Pigeon interface specifications with type safety
- **Quickstart Guide**: Basic usage examples and setup instructions
- **Constitution**: Development principles and quality gates established
- **README**: Package overview with features and usage examples

## Next Development Phase
Ready for task generation and implementation. Use `/tasks` command to generate detailed implementation tasks based on current specifications and constitutional compliance requirements.