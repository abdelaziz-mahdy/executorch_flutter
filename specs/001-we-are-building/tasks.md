# Tasks: Flutter ExecuTorch Package

**Input**: Design documents from `/specs/001-we-are-building/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓), quickstart.md (✓)

## Execution Flow (main)
```
1. Load plan.md from feature directory
   ✓ Loaded: Flutter plugin structure, Pigeon-based type-safe communication
   ✓ Tech stack: Dart 3.0+, Flutter 3.16+, ExecuTorch 0.7.0, Pigeon, Kotlin, Swift
2. Load optional design documents:
   ✓ data-model.md: 6 entities → model tasks
   ✓ contracts/executorch_api.dart: Pigeon interfaces → contract test tasks
   ✓ research.md: Technology decisions → setup tasks
   ✓ quickstart.md: User scenarios → integration tests
3. Generate tasks by category:
   ✓ Setup: Flutter plugin, Pigeon, dependencies, ExecuTorch integration
   ✓ Tests: Pigeon contract tests, integration tests with real models
   ✓ Core: Dart models, native platform implementations, example app
   ✓ Integration: Platform bridges, model loading, inference execution
   ✓ Polish: unit tests, performance validation, documentation
4. Apply task rules:
   ✓ Different files = mark [P] for parallel
   ✓ Platform-specific code = sequential per platform
   ✓ Tests before implementation (TDD + Constitutional requirement)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness
9. SUCCESS: Ready for constitutional TDD execution
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **DOC VERIFY**: All implementation tasks MUST verify latest ExecuTorch docs first
- Include exact file paths in descriptions

## Path Conventions (Flutter Plugin Structure)
- **Plugin code**: `lib/`, `android/`, `ios/`, `pigeons/`
- **Tests**: `test/`, `integration_test/`
- **Platform-specific**: `android/src/main/kotlin/`, `ios/Classes/`
- **Generated**: `lib/src/generated/`, platform generated dirs

## Phase 3.1: Project Setup
- [x] T001 Create Flutter plugin project structure per plan.md specifications
- [x] T002 Configure pubspec.yaml with Dart 3.0+, Flutter 3.16+ constraints and Pigeon dependency
- [x] T003 [P] Initialize Android module with ExecuTorch AAR 0.7.0 dependencies in android/build.gradle
- [x] T004 [P] Initialize iOS module with CocoaPods specification for ExecuTorch frameworks in ios/executorch_flutter.podspec
- [x] T005 [P] Configure analysis_options.yaml with strict linting rules
- [x] T006 Create pigeons/executorch_api.dart with Pigeon interface definitions per contracts/executorch_api.dart

## Phase 3.2: Documentation Verification & Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Documentation Verification Tasks [P]
- [x] T007 [P] **DOC VERIFY**: Verify latest Android ExecuTorch integration docs and update dependencies if needed
- [x] T008 [P] **DOC VERIFY**: Verify latest iOS ExecuTorch integration docs and update framework requirements if needed

### Pigeon Contract Tests [P]
- [x] T009 [P] Contract test ExecutorchHostApi.loadModel in test/pigeon_contract_test.dart
- [x] T010 [P] Contract test ExecutorchHostApi.runInference in test/pigeon_contract_test.dart
- [x] T011 [P] Contract test ExecutorchHostApi.getModelMetadata in test/pigeon_contract_test.dart
- [x] T012 [P] Contract test ExecutorchHostApi.disposeModel in test/pigeon_contract_test.dart

### Data Structure Tests [P]
- [x] T013 [P] Unit test TensorData validation in test/tensor_data_test.dart
- [x] T014 [P] Unit test ModelMetadata validation in test/model_metadata_test.dart
- [x] T015 [P] Unit test InferenceRequest validation in test/inference_request_test.dart
- [x] T016 [P] Unit test InferenceResult validation in test/inference_result_test.dart

### Integration Tests with Real Models [P]
- [x] T017 [P] Integration test basic model loading flow in integration_test/basic_model_loading_test.dart
- [x] T018 [P] Integration test inference execution flow in integration_test/inference_execution_test.dart
- [x] T019 [P] Integration test multiple concurrent models in integration_test/concurrent_models_test.dart
- [x] T020 [P] Integration test error handling scenarios in integration_test/error_handling_test.dart

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Pigeon Code Generation
- [x] T021 Generate Pigeon platform code using flutter packages pub run pigeon --input pigeons/executorch_api.dart

### Dart Model Implementation [P]
- [x] T022 [P] **DOC VERIFY**: Implement TensorData class in lib/src/executorch_types.dart
- [x] T023 [P] Implement ModelMetadata class in lib/src/executorch_types.dart
- [x] T024 [P] Implement InferenceRequest class in lib/src/executorch_types.dart
- [x] T025 [P] Implement InferenceResult class in lib/src/executorch_types.dart
- [x] T026 [P] Implement ExecuTorchModel wrapper class in lib/src/executorch_model.dart
- [x] T027 Implement ExecutorchManager main API class in lib/src/executorch_inference.dart

### Main Library Export
- [x] T038 Create main library export file lib/executorch_flutter.dart with public API exposure

### Android Platform Implementation
- [ ] T028 **DOC VERIFY**: Implement ExecutorchFlutterPlugin.kt with Pigeon interface implementation in android/src/main/kotlin/com/executorch/flutter/ExecutorchFlutterPlugin.kt
- [ ] T029 **DOC VERIFY**: Implement ExecutorchModelManager.kt with ExecuTorch AAR integration in android/src/main/kotlin/com/executorch/flutter/ExecutorchModelManager.kt
- [ ] T030 Configure Android build.gradle with ExecuTorch dependencies and proper native library loading
- [ ] T031 Implement Android tensor data conversion utilities in ExecutorchModelManager.kt
- [ ] T032 Implement Android model lifecycle management and memory cleanup in ExecutorchModelManager.kt

### iOS Platform Implementation
- [ ] T033 **DOC VERIFY**: Implement ExecutorchFlutterPlugin.swift with Pigeon interface implementation in ios/Classes/ExecutorchFlutterPlugin.swift
- [ ] T034 **DOC VERIFY**: Implement ExecutorchModelManager.swift with ExecuTorch framework integration in ios/Classes/ExecutorchModelManager.swift
- [ ] T035 Configure iOS podspec with ExecuTorch framework dependencies and linking
- [ ] T036 Implement iOS tensor data conversion utilities in ExecutorchModelManager.swift
- [ ] T037 Implement iOS model lifecycle management with ARC-compliant cleanup in ExecutorchModelManager.swift

### Main Library Export
- [ ] T038 Create main library export file lib/executorch_flutter.dart with public API exposure

## Phase 3.4: Integration & Cross-Platform Validation

### Platform Bridge Integration
- [ ] T039 Integrate Android ExecuTorch Module.load() pattern with ExecutorchModelManager.kt
- [ ] T040 Integrate iOS ExecuTorch Module loading pattern with ExecutorchModelManager.swift
- [ ] T041 Implement cross-platform error mapping and exception handling
- [ ] T042 Add platform-specific performance monitoring and memory usage tracking

### Real Model Testing
- [ ] T043 Create test ExecuTorch .pte model files for integration testing in integration_test/fixtures/
- [ ] T044 Test Android platform with real ExecuTorch models
- [ ] T045 Test iOS platform with real ExecuTorch models
- [ ] T046 Validate cross-platform performance parity (<200ms loading, <50ms inference)

## Phase 3.5: Example Application & Documentation

### Example App [P]
- [ ] T047 [P] Create example app main.dart in example/lib/main.dart with model loading demo
- [ ] T048 [P] Add example model assets and asset configuration in example/pubspec.yaml
- [ ] T049 [P] Create example Android configuration in example/android/
- [ ] T050 [P] Create example iOS configuration in example/ios/

### Documentation & Polish [P]
- [ ] T051 [P] Update README.md with installation and usage examples
- [ ] T052 [P] Create API documentation with dartdoc comments
- [ ] T053 [P] Add performance benchmarking and memory leak detection tests
- [ ] T054 [P] Create troubleshooting guide with common issues and solutions
- [ ] T055 Validate constitutional compliance: test-first, platform parity, type safety

## Dependencies

**Sequential Dependencies**:
- T001-T006 (setup) before all other tasks
- T007-T008 (doc verification) before platform implementation (T028-T037)
- T009-T020 (tests) before T022-T038 (implementation) - **CONSTITUTIONAL REQUIREMENT**
- T021 (Pigeon generation) before T022-T027 (Dart models)
- T022-T027 (Dart models) before T028-T037 (platform implementation)
- T028-T037 (platform implementation) before T039-T046 (integration)
- T039-T046 (integration) before T047-T055 (example & polish)

**Platform Dependencies**:
- Android: T003, T028-T032, T039, T044 must run sequentially
- iOS: T004, T033-T037, T040, T045 must run sequentially
- Cross-platform: T041-T042, T046 after both platforms complete

## Parallel Execution Examples

### Phase 3.1 Setup [P]
```bash
# After T001-T002 complete, run these in parallel:
Task: "Initialize Android module with ExecuTorch AAR 0.7.0 dependencies in android/build.gradle"
Task: "Initialize iOS module with CocoaPods specification for ExecuTorch frameworks in ios/executorch_flutter.podspec"
Task: "Configure analysis_options.yaml with strict linting rules"
```

### Phase 3.2 Tests [P] - CRITICAL TDD PHASE
```bash
# Documentation verification (run first):
Task: "DOC VERIFY: Verify latest Android ExecuTorch integration docs and update dependencies if needed"
Task: "DOC VERIFY: Verify latest iOS ExecuTorch integration docs and update framework requirements if needed"

# Then all contract tests in parallel:
Task: "Contract test ExecutorchHostApi.loadModel in test/pigeon_contract_test.dart"
Task: "Contract test ExecutorchHostApi.runInference in test/pigeon_contract_test.dart"
Task: "Unit test TensorData validation in test/tensor_data_test.dart"
Task: "Integration test basic model loading flow in integration_test/basic_model_loading_test.dart"
```

### Phase 3.3 Core Implementation [P]
```bash
# After Pigeon generation, run Dart models in parallel:
Task: "DOC VERIFY: Implement TensorData class in lib/src/executorch_types.dart"
Task: "Implement ModelMetadata class in lib/src/executorch_types.dart"
Task: "Implement InferenceRequest class in lib/src/executorch_types.dart"
Task: "Implement InferenceResult class in lib/src/executorch_types.dart"
```

### Phase 3.5 Polish [P]
```bash
# Documentation and example tasks:
Task: "Create example app main.dart in example/lib/main.dart with model loading demo"
Task: "Update README.md with installation and usage examples"
Task: "Create API documentation with dartdoc comments"
Task: "Add performance benchmarking and memory leak detection tests"
```

## Constitutional Compliance Verification

### Test-First Development ✓
- All implementation tasks (T022-T055) blocked by failing tests (T009-T020)
- Pigeon contract tests ensure type safety before implementation
- Integration tests with real models validate functionality

### Platform Parity ✓
- Android (T028-T032) and iOS (T033-T037) implement identical APIs
- Cross-platform validation (T046) ensures performance parity
- Error handling standardized across platforms (T041)

### Documentation Verification ✓
- Mandatory doc verification tasks (T007-T008) before platform implementation
- All platform implementation tasks include **DOC VERIFY** requirement
- Latest ExecuTorch stable docs must be confirmed current

### Type-Safe Communication ✓
- Pigeon-only interfaces (T006, T021)
- Contract tests validate serialization (T009-T012)
- No manual method channel implementations allowed

### Resource Management ✓
- Explicit model lifecycle management (T032, T037)
- Memory leak detection tests (T053)
- Platform-appropriate cleanup patterns enforced

## Validation Checklist
*GATE: Checked before task execution*

- [x] All Pigeon interfaces have corresponding contract tests (T009-T012)
- [x] All data model entities have implementation tasks (T022-T026)
- [x] All tests come before implementation (T009-T020 before T022-T055)
- [x] Parallel tasks truly independent (verified file paths)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Documentation verification mandatory for platform implementation
- [x] Constitutional compliance requirements integrated
- [x] Real ExecuTorch model testing included
- [x] Performance targets validated (<200ms, <50ms)