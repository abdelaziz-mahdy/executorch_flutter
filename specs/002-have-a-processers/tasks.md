# Tasks: Model-Specific Processor Classes

**Input**: Design documents from `/specs/002-have-a-processers/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: Dart/Flutter, processor interfaces, example integration
2. Load design documents:
   → data-model.md: ExecuTorchPreprocessor, ExecuTorchPostprocessor, ProcessorTensorUtils
   → contracts/: processor_interfaces.dart, imagenet_processor.dart → contract tests
   → research.md: Generic interfaces, ImageNet example, validation patterns
3. Generate tasks by category:
   → Setup: processor architecture, dependencies
   → Tests: contract tests, integration tests
   → Core: base processors, ImageNet implementation, utilities
   → Integration: example app, documentation
   → Polish: unit tests, performance validation
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
- Processor classes in `lib/src/processors/`
- Tests in `test/` and `integration_test/`

## Phase 3.1: Setup
- [x] T001 Add image processing dependency to pubspec.yaml (image: ^4.0.17)
- [x] T002 [P] Update lib/executorch_flutter.dart to export processor classes
- [x] T003 [P] Create lib/src/processors/ directory structure

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Contract test for ExecuTorchPreprocessor interface in test/processor_interfaces_test.dart
- [x] T005 [P] Contract test for ExecuTorchPostprocessor interface in test/processor_interfaces_test.dart
- [x] T006 [P] Contract test for ExecuTorchProcessor interface in test/processor_interfaces_test.dart
- [x] T007 [P] Contract test for ProcessorTensorUtils in test/tensor_utils_test.dart
- [x] T008 [P] Contract test for ImageNet processor in test/imagenet_processor_test.dart
- [x] T009 [P] Integration test for complete processor pipeline in integration_test/processor_pipeline_test.dart
- [x] T010 [P] Integration test for example app processor usage in integration_test/example_integration_test.dart

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [x] T011 [P] Implement ExecuTorchPreprocessor<T> abstract class in lib/src/processors/base_processor.dart
- [x] T012 [P] Implement ExecuTorchPostprocessor<T> abstract class in lib/src/processors/base_processor.dart
- [x] T013 [P] Implement ExecuTorchProcessor<TInput, TOutput> abstract class in lib/src/processors/base_processor.dart
- [x] T014 [P] Implement ProcessorTensorUtils utility class in lib/src/processors/base_processor.dart
- [x] T015 [P] Implement PreprocessingException and PostprocessingException in lib/src/processors/base_processor.dart
- [x] T016 [P] Implement ImagePreprocessConfig configuration class in lib/src/processors/image_processor.dart
- [x] T017 [P] Implement ImageNetPreprocessor class in lib/src/processors/image_processor.dart
- [x] T018 [P] Implement ClassificationResult class in lib/src/processors/image_processor.dart
- [x] T019 [P] Implement ImageNetPostprocessor class in lib/src/processors/image_processor.dart
- [x] T020 Implement ImageNetProcessor complete pipeline in lib/src/processors/image_processor.dart
- [x] T021 [P] Implement TextPreprocessConfig configuration class in lib/src/processors/text_processor.dart
- [x] T022 [P] Implement AudioPreprocessConfig configuration class in lib/src/processors/audio_processor.dart
- [x] T023 Create main processors export file at lib/src/processors/processors.dart

## Phase 3.4: Integration
- [x] T024 Update example app to demonstrate processor usage in example/lib/main.dart
- [x] T025 [P] Create processor demonstration screens in example/lib/screens/
- [x] T026 Add ImageNet class labels data file in example/assets/imagenet_labels.dart
- [x] T027 Update example app pubspec.yaml with processor dependencies
- [ ] T028 [P] Add processor examples to README.md
- [ ] T029 Update package documentation with processor API docs

## Phase 3.5: Polish
- [ ] T030 [P] Performance test for preprocessing speed in test/performance/preprocessing_performance_test.dart
- [ ] T031 [P] Performance test for postprocessing speed in test/performance/postprocessing_performance_test.dart
- [ ] T032 [P] Memory usage test for processor operations in test/performance/memory_usage_test.dart
- [ ] T033 [P] Unit test for tensor utilities edge cases in test/tensor_utils_edge_cases_test.dart
- [ ] T034 [P] Unit test for image processor validation in test/image_processor_validation_test.dart
- [ ] T035 [P] Unit test for error handling scenarios in test/processor_error_handling_test.dart
- [ ] T036 Run flutter analyze and fix any linting issues
- [ ] T037 Run performance validation and ensure <50ms processing times
- [ ] T038 Create processor usage documentation in example/docs/processor_guide.md

## Dependencies
- Tests (T004-T010) before implementation (T011-T023)
- T011-T015 (base classes) before T016-T020 (image processor)
- T011-T015 (base classes) before T021-T022 (other processors)
- T020 blocks T024-T025 (example integration)
- Implementation before polish (T030-T038)

## Parallel Example
```
# Launch T004-T010 together (contract and integration tests):
Task: "Contract test for ExecuTorchPreprocessor interface in test/processor_interfaces_test.dart"
Task: "Contract test for ExecuTorchPostprocessor interface in test/processor_interfaces_test.dart"
Task: "Contract test for ExecuTorchProcessor interface in test/processor_interfaces_test.dart"
Task: "Contract test for ProcessorTensorUtils in test/tensor_utils_test.dart"
Task: "Contract test for ImageNet processor in test/imagenet_processor_test.dart"
Task: "Integration test for complete processor pipeline in integration_test/processor_pipeline_test.dart"
Task: "Integration test for example app processor usage in integration_test/example_integration_test.dart"

# Launch T011-T015 together (base implementation):
Task: "Implement ExecuTorchPreprocessor<T> abstract class in lib/src/processors/base_processor.dart"
Task: "Implement ExecuTorchPostprocessor<T> abstract class in lib/src/processors/base_processor.dart"
Task: "Implement ExecuTorchProcessor<TInput, TOutput> abstract class in lib/src/processors/base_processor.dart"
Task: "Implement ProcessorTensorUtils utility class in lib/src/processors/base_processor.dart"
Task: "Implement PreprocessingException and PostprocessingException in lib/src/processors/base_processor.dart"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts
- Focus on type safety and performance targets (<50ms processing)

## Task Generation Rules Applied
1. **From Contracts**:
   - processor_interfaces.dart → T004-T006 contract tests [P]
   - imagenet_processor.dart → T008 contract test [P]

2. **From Data Model**:
   - ExecuTorchPreprocessor → T011 model creation [P]
   - ExecuTorchPostprocessor → T012 model creation [P]
   - ExecuTorchProcessor → T013 model creation [P]
   - ProcessorTensorUtils → T014 utility creation [P]

3. **From User Stories**:
   - Basic processor usage → T009 integration test [P]
   - Example app integration → T010 integration test [P]

## Validation Checklist
- [x] All contracts have corresponding tests (T004-T008)
- [x] All entities have model tasks (T011-T015)
- [x] All tests come before implementation (T004-T010 before T011+)
- [x] Parallel tasks truly independent ([P] tasks use different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task