# Implementation Plan: macOS Platform Support

**Branch**: `005-https-docs-pytorch` | **Date**: 2025-10-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-https-docs-pytorch/spec.md`

## Summary

Add macOS desktop platform support to the ExecuTorch Flutter plugin by reusing the existing iOS Swift implementation. This enables Flutter developers to use the same ML inference API on macOS with zero code changes to their Flutter apps.

**Technical Approach**: Leverage ExecuTorch's official macOS support (same frameworks as iOS) and share 100% of the Swift implementation code between platforms using Flutter's standard plugin structure with separate platform directories.

## Technical Context

**Language/Version**: Swift 5.9+, Dart 3.0+
**Primary Dependencies**: ExecuTorch 0.7.0 (swiftpm-0.7.0 branch), Flutter 3.0+
**Storage**: File system (.pte model files), in-memory model registry
**Testing**: XCTest (Swift), Dart test framework, integration tests
**Target Platform**: macOS 12+ (Monterey and later), both ARM64 and x86_64
**Project Type**: Mobile (Flutter plugin with iOS + macOS platforms)
**Performance Goals**: Model load <200ms, inference <50ms, identical to iOS
**Constraints**: Must reuse iOS code, API parity required, no breaking changes
**Scale/Scope**: Single plugin supporting 2 additional platforms (macOS added to iOS/Android)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Since the constitution template is not populated, applying general software development principles:

✅ **Code Reuse**: Maximizing reuse of iOS implementation (100% Swift code shared)
✅ **API Consistency**: Same Pigeon contracts work across all platforms
✅ **Testing**: Integration tests will validate platform parity
✅ **Documentation**: Research, contracts, and quickstart guide created
✅ **Simplicity**: Minimal configuration changes, no architectural changes

**Gate Status**: PASS - No constitutional violations, follows best practices

## Project Structure

### Documentation (this feature)
```
specs/005-https-docs-pytorch/
├── plan.md              # This file
├── research.md          # ✅ Phase 0 complete
├── data-model.md        # ✅ Phase 1 complete
├── quickstart.md        # ✅ Phase 1 complete
├── contracts/           # ✅ Phase 1 complete
│   └── platform-api.md
└── tasks.md             # Phase 2 (/tasks command)
```

### Source Code (repository root)
```
ios/
├── Classes/                    # Shared Swift implementation (unchanged)
│   ├── ExecutorchFlutterPlugin.swift
│   ├── ExecutorchModelManager.swift
│   ├── ExecutorchTensorUtils.swift
│   ├── ExecutorchLifecycleManager.swift
│   └── Generated/
│       └── ExecutorchApi.swift (Pigeon-generated)
├── executorch_flutter.podspec  # iOS CocoaPods config
└── executorch_flutter/
    └── Package.swift           # iOS SPM config

macos/                          # NEW DIRECTORY
├── Classes → ../ios/Classes    # Symlink to iOS implementation
├── executorch_flutter.podspec  # macOS CocoaPods config (NEW)
└── executorch_flutter/
    └── Package.swift           # macOS SPM config (NEW)

lib/
└── src/
    └── generated/
        └── executorch_api.dart # Dart API (unchanged, works for both)

pigeons/
└── executorch_api.dart         # Pigeon source (unchanged)

example/
├── macos/                      # NEW - Flutter macOS app
│   ├── Runner.xcworkspace
│   ├── Runner/
│   └── Flutter/
└── lib/
    └── main.dart               # Example app (unchanged, works for both)

test/
└── integration_test/
    └── platform_parity_test.dart  # NEW - validates iOS/macOS consistency
```

**Structure Decision**: Using Option 3 (Mobile + API) variant with separate `ios/` and `macos/` directories as per Flutter plugin convention, with symlinked `Classes/` to share implementation code.

## Phase 0: Outline & Research

**Status**: ✅ COMPLETE

**Output**: [research.md](./research.md)

**Key Findings**:
1. ExecuTorch officially supports macOS 12+ with same APIs as iOS
2. Swift implementation can be 100% reused with conditional compilation for UI frameworks
3. Pigeon-generated code is platform-agnostic and works on both platforms
4. Both ARM64 (Apple Silicon) and x86_64 (Intel) architectures supported
5. Same backends available: XNNPACK, Core ML, MPS

**Decisions Made**:
- Use separate `macos/` directory (Flutter convention)
- Symlink `macos/Classes` → `ios/Classes` for code sharing
- Create macOS-specific Package.swift and .podspec files
- No changes to Pigeon API specification needed
- Minimum version: macOS 12 (aligned with ExecuTorch)

## Phase 1: Design & Contracts

**Status**: ✅ COMPLETE

### 1. Data Model
**Output**: [data-model.md](./data-model.md)

**Entities**: Reusing iOS data model without modification
- ExecuTorchModel
- TensorData
- InferenceResult
- ModelMetadata

**Platform Notes**: Zero platform-specific differences in data structures

### 2. API Contracts
**Output**: [contracts/platform-api.md](./contracts/platform-api.md)

**Host API Methods** (Dart → Native):
- `loadModel(modelPath) → modelId`
- `runInference(modelId, inputs) → InferenceResult`
- `disposeModel(modelId) → void`
- `getModelMetadata(modelId) → ModelMetadata`

**Contract Promise**: 100% API parity between iOS and macOS

**Changes Required**: ZERO - existing contract works on macOS

### 3. Integration Tests
Tests to be created in Phase 2/tasks.md:
- Model loading test (macOS)
- Inference execution test (macOS)
- Platform parity test (iOS vs macOS results)
- Architecture test (ARM64 vs x86_64)
- Performance benchmark test

### 4. Quickstart Guide
**Output**: [quickstart.md](./quickstart.md)

**Validation Steps**:
1. Enable macOS support in example app
2. Build and run on macOS
3. Load test model
4. Run inference
5. Verify results match iOS

**Estimated Time**: 15 minutes for basic validation

### 5. Agent Context
**Output**: CLAUDE.md updated via `.specify/scripts/bash/update-agent-context.sh claude`

**Updates**:
- Added macOS platform to development context
- Noted code reuse approach
- Documented ExecuTorch version and dependencies

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

1. **Configuration Tasks** (modify existing files):
   - Update `pubspec.yaml` to register macOS platform
   - Create `macos/executorch_flutter.podspec`
   - Create `macos/executorch_flutter/Package.swift`
   - Update Pigeon config (if needed for macOS output path)

2. **Setup Tasks** (create new directory structure):
   - Create `macos/` directory structure
   - Create symlink `macos/Classes → ios/Classes`
   - Set up macOS example app (`flutter create --platforms=macos .`)

3. **Build Tasks** (ensure compilation):
   - Generate Pigeon code for macOS
   - Build plugin for macOS architecture
   - Link ExecuTorch frameworks via SPM

4. **Test Tasks** (validate functionality):
   - Write platform parity integration test
   - Write macOS-specific model loading test
   - Write architecture-specific tests (ARM64, x86_64)
   - Write performance benchmark test

5. **Documentation Tasks**:
   - Update main README with macOS support
   - Update example README with macOS instructions
   - Add macOS section to MODEL_EXPORT_GUIDE if needed

**Ordering Strategy**:
- Configuration first (needed for build)
- Setup second (directory structure)
- Build third (validate compilation)
- Tests fourth (validate functionality)
- Documentation last (capture final state)

**Dependencies**:
- Setup depends on configuration
- Build depends on setup
- Tests depend on build
- Documentation depends on tests passing

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**Parallel Execution Opportunities**:
- [P] Multiple test files can be written in parallel
- [P] Documentation updates can happen in parallel
- [P] CocoaPods and SPM configs can be created in parallel

**IMPORTANT**: This phase is executed by the `/tasks` command, NOT by `/plan`

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following TDD approach)
**Phase 5**: Validation (run quickstart.md, integration tests, performance tests)

### Implementation Notes for Phase 4:

1. **macOS Package.swift**:
   ```swift
   platforms: [.macOS("12.0")],  // Add macOS platform
   ```

2. **macOS Podspec**:
   ```ruby
   s.platform = :osx, '12.0'
   s.osx.frameworks = ['Accelerate', 'CoreML', 'MetalPerformanceShaders', 'Foundation', 'AppKit']
   ```

3. **Conditional Compilation** (if needed):
   ```swift
   #if os(macOS)
   import AppKit
   typealias PlatformApplication = NSApplication
   #else
   import UIKit
   typealias PlatformApplication = UIApplication
   #endif
   ```

4. **Example App**:
   ```bash
   cd example
   flutter create --platforms=macos .
   flutter run -d macos
   ```

## Complexity Tracking

*No constitutional violations - no entries needed*

This implementation adds platform support by reusing existing code and following Flutter conventions. No architectural complexity added.

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning approach complete (/plan command - described above)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (N/A - none)

**Artifacts Generated**:
- [x] research.md
- [x] data-model.md
- [x] contracts/platform-api.md
- [x] quickstart.md
- [x] CLAUDE.md updated
- [ ] tasks.md (next phase)

---

## Summary

This implementation plan enables macOS support for the ExecuTorch Flutter plugin by:

1. **Reusing 100%** of iOS Swift implementation (no code duplication)
2. **Creating** macOS-specific configuration files (Package.swift, .podspec)
3. **Following** Flutter plugin conventions (separate platform directories)
4. **Maintaining** complete API parity with iOS
5. **Supporting** both Apple Silicon and Intel architectures

**Next Command**: `/tasks` to generate detailed implementation tasks

**Estimated Implementation Time**: 4-6 hours
- Configuration: 1 hour
- Setup & Build: 1 hour
- Testing: 2 hours
- Documentation: 1 hour
- Validation: 1 hour

**Risk Level**: LOW - leveraging official support and proven code reuse patterns

---

*Based on Flutter Plugin Development Best Practices and ExecuTorch Official Documentation*
