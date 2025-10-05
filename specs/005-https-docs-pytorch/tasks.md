# Tasks: macOS Platform Support

**Input**: Design documents from `/specs/005-https-docs-pytorch/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Summary

This task list implements macOS platform support for the ExecuTorch Flutter plugin by:
1. Creating macOS platform configuration files
2. Setting up directory structure with symlinked implementation
3. Building and validating compilation for macOS
4. Writing integration tests for platform parity
5. Updating documentation for macOS support

**Total Tasks**: 24 tasks across 5 phases
**Estimated Time**: 4-6 hours
**Parallel Opportunities**: 12 tasks can run in parallel (marked [P])

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All file paths are absolute from repository root

## Phase 3.1: Setup & Configuration

### Plugin Configuration
- [x] **T001** Update `pubspec.yaml` to register macOS platform
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/pubspec.yaml`
  - Add `macos:` section under `flutter.plugin.platforms`
  - Set `pluginClass: ExecutorchFlutterPlugin`
  - Dependencies: None

- [x] **T002** [P] Create macOS podspec at `macos/executorch_flutter.podspec`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/macos/executorch_flutter.podspec`
  - Set platform to macOS 12.0
  - Configure frameworks: Accelerate, CoreML, MetalPerformanceShaders, Foundation, AppKit
  - Set valid architectures: arm64, x86_64
  - Dependencies: None (parallel with T003)

- [x] **T003** [P] Create macOS Package.swift at `macos/executorch_flutter/Package.swift`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/macos/executorch_flutter/Package.swift`
  - Add `.macOS("12.0")` to platforms
  - Link ExecuTorch dependencies (executorch, backend_xnnpack, backend_coreml, backend_mps)
  - Dependencies: None (parallel with T002)

### Directory Structure
- [x] **T004** Create macOS directory structure
  - Create: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/macos/`
  - Create: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/macos/executorch_flutter/Sources/executorch_flutter/`
  - Dependencies: T001

- [x] **T005** Create symlink from `macos/Classes` to `ios/Classes`
  - Command: `ln -s ../ios/Classes macos/Classes`
  - Verify: Symlink points to existing iOS implementation
  - Dependencies: T004

### Example App Setup
- [x] **T006** Enable macOS support in example app
  - Directory: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/`
  - Commands:
    ```bash
    cd example
    flutter config --enable-macos-desktop
    flutter create --platforms=macos .
    ```
  - Dependencies: T001, T002, T003

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY build/implementation**

### Integration Tests
- [ ] **T007** [P] Platform parity test in `test/integration_test/platform_parity_test.dart`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/test/integration_test/platform_parity_test.dart`
  - Test: Load same model on iOS and macOS, verify identical results
  - Assert: Inference outputs match within 0.001 tolerance
  - Dependencies: T006

- [ ] **T008** [P] macOS model loading test in `test/integration_test/macos_model_loading_test.dart`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/test/integration_test/macos_model_loading_test.dart`
  - Test: Model file accessible, ExecuTorch loads successfully
  - Assert: Model ID returned, state transitions loading → ready
  - Dependencies: T006

- [ ] **T009** [P] macOS inference execution test in `test/integration_test/macos_inference_test.dart`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/test/integration_test/macos_inference_test.dart`
  - Test: Input tensors accepted, inference executes, outputs returned
  - Assert: Results match expected format, performance within range
  - Dependencies: T006

- [ ] **T010** [P] Architecture-specific test in `test/integration_test/architecture_test.dart`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/test/integration_test/architecture_test.dart`
  - Test: Verify plugin works on both ARM64 and x86_64
  - Assert: Model loads and runs on detected architecture
  - Dependencies: T006

### Validation Tests
- [ ] **T011** [P] Performance benchmark test in `test/integration_test/performance_test.dart`
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/test/integration_test/performance_test.dart`
  - Test: Measure model load time (<200ms) and inference time (<50ms)
  - Assert: Performance meets or exceeds iOS on equivalent hardware
  - Dependencies: T006

## Phase 3.3: Build & Validation (ONLY after tests are failing)

### Pigeon Code Generation
- [x] **T012** Verify Pigeon generates compatible macOS code
  - Command: `flutter pub run pigeon --input pigeons/executorch_api.dart`
  - Verify: Generated Swift code at `ios/Classes/Generated/ExecutorchApi.swift` works for both platforms
  - Assert: No macOS-specific generation needed (code is platform-agnostic)
  - Dependencies: T007-T011 (tests must fail first)

### Build Validation
- [ ] **T013** Build plugin for macOS (Debug)
  - Directory: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/`
  - Command: `flutter build macos --debug`
  - Assert: No compilation errors, frameworks linked correctly
  - Dependencies: T012

- [ ] **T014** Build plugin for macOS (Release)
  - Directory: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/`
  - Command: `flutter build macos --release`
  - Assert: Optimized build succeeds, bundle created
  - Dependencies: T013

### Runtime Validation
- [ ] **T015** Run example app on macOS and verify plugin registration
  - Command: `flutter run -d macos`
  - Verify: Plugin loads, no registration errors
  - Check: `macos/Flutter/GeneratedPluginRegistrant.swift` imports executorch_flutter
  - Dependencies: T014

- [ ] **T016** Execute quickstart validation workflow
  - Follow: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/specs/005-https-docs-pytorch/quickstart.md`
  - Steps: Load model, run inference, verify results
  - Assert: All quickstart steps pass without errors
  - Dependencies: T015

### Test Execution
- [ ] **T017** Run integration tests on macOS and verify they pass
  - Command: `flutter test integration_test/platform_parity_test.dart -d macos`
  - Command: `flutter test integration_test/macos_model_loading_test.dart -d macos`
  - Command: `flutter test integration_test/macos_inference_test.dart -d macos`
  - Command: `flutter test integration_test/architecture_test.dart -d macos`
  - Command: `flutter test integration_test/performance_test.dart -d macos`
  - Assert: All tests pass (previously failing tests now pass after implementation)
  - Dependencies: T016

## Phase 3.4: Integration & Polish

### Code Quality
- [ ] **T018** [P] Add platform detection utility if needed
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/ios/Classes/ExecutorchPlatformUtils.swift` (shared with macOS)
  - Add: Conditional compilation for iOS vs macOS specific code (if any UIKit/AppKit differences emerge)
  - Only create if platform-specific code is needed
  - Dependencies: T017

- [ ] **T019** Verify memory management on macOS
  - Test: Load multiple models, dispose, check for leaks
  - Use: Xcode Instruments to profile memory usage
  - Assert: No memory leaks, proper ARC cleanup
  - Dependencies: T017

### Documentation
- [ ] **T020** [P] Update main README with macOS support
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/README.md`
  - Add: macOS platform to supported platforms list
  - Add: macOS-specific installation/setup instructions
  - Add: Minimum version requirement (macOS 12+)
  - Dependencies: T017

- [ ] **T021** [P] Update example README with macOS instructions
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/README.md`
  - Add: How to run example app on macOS
  - Add: macOS-specific testing instructions
  - Dependencies: T017

- [ ] **T022** [P] Update MODEL_EXPORT_GUIDE if needed
  - File: `/Users/AbdelazizMahdy/flutter_projects/executorch/executorch_flutter/example/MODEL_EXPORT_GUIDE.md`
  - Add: Note that exported models work on macOS without changes
  - Add: macOS-specific file path considerations (if any)
  - Dependencies: T017

## Phase 3.5: Final Validation

### Comprehensive Testing
- [ ] **T023** Run full test suite on macOS (unit + integration)
  - Command: `flutter test`
  - Command: `flutter test integration_test/ -d macos`
  - Assert: All existing tests pass on macOS
  - Assert: No platform-specific regressions
  - Dependencies: T020-T022

### Release Readiness
- [ ] **T024** Create macOS support checklist validation
  - Verify: Plugin builds on macOS 12+
  - Verify: Plugin builds on Apple Silicon (ARM64)
  - Verify: Plugin builds on Intel Mac (x86_64) if available
  - Verify: Model loading works on macOS
  - Verify: Inference produces correct results
  - Verify: API matches iOS behavior exactly
  - Verify: Performance meets targets
  - Verify: No crashes or memory leaks
  - Verify: All tests pass
  - Verify: Documentation updated
  - Dependencies: T023

## Dependencies Graph

```
Setup & Configuration:
  T001 → T004, T006
  T002, T003 (parallel) → T006
  T004 → T005
  T006 → T007-T011

Tests First (must fail):
  T007, T008, T009, T010, T011 (all parallel) → T012

Build & Validation:
  T012 → T013 → T014 → T015 → T016 → T017

Integration & Polish:
  T017 → T018, T019, T020, T021, T022 (T020-T022 parallel)
  T020, T021, T022 → T023 → T024
```

## Parallel Execution Examples

### Batch 1: Configuration Files (after T001)
```bash
# Run T002 and T003 in parallel
Task: "Create macOS podspec at macos/executorch_flutter.podspec"
Task: "Create macOS Package.swift at macos/executorch_flutter/Package.swift"
```

### Batch 2: Integration Tests (after T006)
```bash
# Run T007-T011 in parallel
Task: "Platform parity test in test/integration_test/platform_parity_test.dart"
Task: "macOS model loading test in test/integration_test/macos_model_loading_test.dart"
Task: "macOS inference test in test/integration_test/macos_inference_test.dart"
Task: "Architecture test in test/integration_test/architecture_test.dart"
Task: "Performance benchmark test in test/integration_test/performance_test.dart"
```

### Batch 3: Documentation (after T017)
```bash
# Run T020-T022 in parallel
Task: "Update main README with macOS support"
Task: "Update example README with macOS instructions"
Task: "Update MODEL_EXPORT_GUIDE if needed"
```

## Task Execution Notes

### TDD Workflow
1. ✅ Write tests (T007-T011) - they MUST fail
2. ❌ Run tests - confirm they fail
3. ✅ Generate/build code (T012-T016)
4. ✅ Run tests again - they should now pass (T017)

### Platform-Specific Considerations
- **Symlink**: macOS uses symlink to share iOS implementation (T005)
- **Framework Names**: AppKit (macOS) vs UIKit (iOS) - handled via conditional compilation if needed (T018)
- **Architectures**: Universal binary supports both ARM64 and x86_64 (T010, T014)

### Build Validation
- Debug build (T013): Fast compilation, includes debug symbols
- Release build (T014): Optimized, smaller binary, production-ready

### Performance Targets
- Model load: <200ms for models <100MB (T011)
- Inference: <50ms for typical models (T011)
- Memory: <100MB overhead per model (T019)

## Validation Checklist
*GATE: All items must be checked before feature is complete*

Platform Support:
- [x] All contracts have corresponding tests (N/A - no new contracts)
- [x] All entities have model tasks (N/A - reusing iOS data model)
- [x] All tests come before implementation (T007-T011 before T012-T016)
- [x] Parallel tasks truly independent (verified in dependency graph)
- [x] Each task specifies exact file path (✓ absolute paths used)
- [x] No task modifies same file as another [P] task (verified)

macOS-Specific:
- [ ] macOS configuration files created (T002, T003)
- [ ] Directory structure setup (T004, T005)
- [ ] Example app runs on macOS (T006, T015, T016)
- [ ] Integration tests pass on macOS (T017)
- [ ] Documentation updated (T020-T022)
- [ ] Full test suite passes (T023)
- [ ] Release checklist complete (T024)

## Success Criteria

After completing all 24 tasks:
1. ✅ Flutter plugin builds successfully for macOS
2. ✅ Example app runs on macOS without errors
3. ✅ Model loading works identically to iOS
4. ✅ Inference executes and returns correct results
5. ✅ API behavior matches iOS exactly (platform parity)
6. ✅ Performance meets or exceeds targets
7. ✅ All tests pass on macOS
8. ✅ Documentation reflects macOS support
9. ✅ No breaking changes to existing iOS/Android support

## Estimated Timeline

| Phase | Tasks | Time | Can Parallelize |
|-------|-------|------|-----------------|
| 3.1 Setup | T001-T006 | 1h | T002, T003 |
| 3.2 Tests | T007-T011 | 1.5h | All 5 tests |
| 3.3 Build | T012-T017 | 1.5h | Sequential |
| 3.4 Polish | T018-T022 | 1h | T020-T022 |
| 3.5 Validation | T023-T024 | 0.5h | Sequential |
| **Total** | **24 tasks** | **5.5h** | **12 parallel** |

With parallel execution: **~4 hours actual time**

---

*Tasks generated from implementation plan and design documents*
*Ready for execution - each task is specific and independently completable*
