# Tasks: Camera-to-Model Processor

**Input**: Design documents from `/specs/003-and-add-another/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: Camera integration, real-time processing, permission handling
2. Load design documents:
   → data-model.md: CameraProcessor, CameraConfiguration, PermissionManager
   → contracts/: camera_processor.dart → contract tests
   → research.md: Camera plugin integration, performance optimization
3. Generate tasks by category:
   → Setup: camera dependencies, permissions
   → Tests: contract tests, integration tests
   → Core: camera processor, permission handling, frame conversion
   → Integration: example app, real-time demo
   → Polish: performance tests, error handling
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Flutter package**: `lib/src/`, `test/`, `example/` at repository root
- Camera processor in `lib/src/processors/`
- Platform permissions in Android/iOS manifests

## Phase 3.1: Setup
- [ ] T001 Add camera dependencies to pubspec.yaml (camera: ^0.10.5, permission_handler: ^11.0.0)
- [ ] T002 [P] Add camera permissions to android/app/src/main/AndroidManifest.xml
- [ ] T003 [P] Add camera permissions to ios/Runner/Info.plist
- [ ] T004 [P] Update lib/executorch_flutter.dart to export camera processor
- [ ] T005 [P] Create lib/src/processors/camera/ directory structure

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T006 [P] Contract test for CameraProcessor interface in test/camera_processor_test.dart
- [ ] T007 [P] Contract test for CameraConfiguration in test/camera_configuration_test.dart
- [ ] T008 [P] Contract test for CameraPermissionManager in test/camera_permission_test.dart
- [ ] T009 [P] Contract test for ProcessingController in test/processing_controller_test.dart
- [ ] T010 [P] Contract test for FrameConverter in test/frame_converter_test.dart
- [ ] T011 [P] Integration test for camera lifecycle in integration_test/camera_lifecycle_test.dart
- [ ] T012 [P] Integration test for real-time processing in integration_test/realtime_processing_test.dart
- [ ] T013 [P] Integration test for permission handling in integration_test/permission_flow_test.dart

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T014 [P] Implement CameraConfiguration class in lib/src/processors/camera/camera_configuration.dart
- [ ] T015 [P] Implement CameraPermissionManager in lib/src/processors/camera/camera_permission_manager.dart
- [ ] T016 [P] Implement FrameConverter utility in lib/src/processors/camera/frame_converter.dart
- [ ] T017 [P] Implement ProcessingController in lib/src/processors/camera/processing_controller.dart
- [ ] T018 [P] Implement CameraStreamHandler in lib/src/processors/camera/camera_stream_handler.dart
- [ ] T019 [P] Implement CameraPreviewController in lib/src/processors/camera/camera_preview_controller.dart
- [ ] T020 Implement CameraProcessor main class in lib/src/processors/camera/camera_processor.dart
- [ ] T021 [P] Implement camera exception classes in lib/src/processors/camera/camera_exceptions.dart
- [ ] T022 Create camera processor exports in lib/src/processors/camera/camera.dart

## Phase 3.4: Integration
- [ ] T023 Create camera demo screen in example/lib/camera_demo.dart
- [ ] T024 [P] Add camera processing to example app navigation in example/lib/main.dart
- [ ] T025 [P] Create real-time inference demo in example/lib/realtime_inference_demo.dart
- [ ] T026 [P] Add camera permission request flow in example/lib/permission_helper.dart
- [ ] T027 Update example app with camera processor integration
- [ ] T028 [P] Add camera processor documentation to README.md
- [ ] T029 Create camera usage guide in example/docs/camera_guide.md

## Phase 3.5: Polish
- [ ] T030 [P] Performance test for frame processing speed in test/performance/camera_performance_test.dart
- [ ] T031 [P] Performance test for memory usage during streaming in test/performance/camera_memory_test.dart
- [ ] T032 [P] Test camera switching functionality in test/camera_switching_test.dart
- [ ] T033 [P] Test permission edge cases in test/permission_edge_cases_test.dart
- [ ] T034 [P] Test error recovery scenarios in test/camera_error_recovery_test.dart
- [ ] T035 [P] Test thermal throttling behavior in test/thermal_throttling_test.dart
- [ ] T036 Run flutter analyze and fix camera-related linting issues
- [ ] T037 Validate real-time performance targets (30fps, <100ms conversion)
- [ ] T038 Test camera functionality on various devices and orientations

## Dependencies
- Setup (T001-T005) before tests (T006-T013)
- Tests (T006-T013) before implementation (T014-T022)
- T014-T019 (components) before T020 (main processor)
- T020-T022 (core) before T023-T029 (integration)
- Implementation before polish (T030-T038)

## Parallel Example
```
# Launch T006-T013 together (contract and integration tests):
Task: "Contract test for CameraProcessor interface in test/camera_processor_test.dart"
Task: "Contract test for CameraConfiguration in test/camera_configuration_test.dart"
Task: "Contract test for CameraPermissionManager in test/camera_permission_test.dart"
Task: "Contract test for ProcessingController in test/processing_controller_test.dart"
Task: "Contract test for FrameConverter in test/frame_converter_test.dart"
Task: "Integration test for camera lifecycle in integration_test/camera_lifecycle_test.dart"
Task: "Integration test for real-time processing in integration_test/realtime_processing_test.dart"
Task: "Integration test for permission handling in integration_test/permission_flow_test.dart"

# Launch T014-T019 together (component implementation):
Task: "Implement CameraConfiguration class in lib/src/processors/camera/camera_configuration.dart"
Task: "Implement CameraPermissionManager in lib/src/processors/camera/camera_permission_manager.dart"
Task: "Implement FrameConverter utility in lib/src/processors/camera/frame_converter.dart"
Task: "Implement ProcessingController in lib/src/processors/camera/processing_controller.dart"
Task: "Implement CameraStreamHandler in lib/src/processors/camera/camera_stream_handler.dart"
Task: "Implement CameraPreviewController in lib/src/processors/camera/camera_preview_controller.dart"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Camera permissions must be configured before testing
- Focus on real-time performance and battery optimization
- Test on actual devices, not just simulators

## Task Generation Rules Applied
1. **From Contracts**:
   - camera_processor.dart → T006-T010 contract tests [P]

2. **From Data Model**:
   - CameraProcessor → T020 main implementation
   - CameraConfiguration → T014 config class [P]
   - CameraPermissionManager → T015 permission handling [P]
   - ProcessingController → T017 performance control [P]
   - FrameConverter → T016 frame processing [P]

3. **From User Stories**:
   - Real-time camera processing → T012 integration test [P]
   - Permission handling → T013 integration test [P]
   - Camera lifecycle → T011 integration test [P]

## Validation Checklist
- [x] All contracts have corresponding tests (T006-T013)
- [x] All entities have implementation tasks (T014-T022)
- [x] All tests come before implementation (T006-T013 before T014+)
- [x] Parallel tasks truly independent ([P] tasks use different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Camera permissions configured in platform manifests
- [x] Real-time performance targets specified (<100ms, 30fps)