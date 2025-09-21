# Implementation Plan: Flutter ExecuTorch Package

**Branch**: `001-we-are-building` | **Date**: 2025-09-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-we-are-building/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   ✓ Loaded: Flutter ExecuTorch package specification
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   ✓ Detected Project Type: mobile package (Flutter plugin)
   ✓ Latest ExecuTorch docs verified (Android stable + iOS stable)
3. Fill the Constitution Check section based on the constitution document.
   ✓ Constitution v1.0.0 compliance verified
4. Evaluate Constitution Check section below
   ✓ All constitutional requirements satisfied
   ✓ Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   ✓ All NEEDS CLARIFICATION resolved with latest docs
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   ✓ Design artifacts updated with verified requirements
7. Re-evaluate Constitution Check section
   ✓ No constitutional violations introduced
   ✓ Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 8. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Create a Flutter plugin package that enables Flutter developers to load and run ExecuTorch machine learning models on mobile platforms (Android and iOS). The package will use Pigeon for type-safe native communication and provide a simple Dart API for model loading, inference execution, and result handling, following the latest ExecuTorch stable documentation requirements.

## Technical Context
**Language/Version**: Dart 3.0+, Flutter 3.16+, Android SDK/NDK r27b, iOS 13.0+, Xcode 15+, Swift 5.9+
**Primary Dependencies**: ExecuTorch Android AAR (0.7.0), ExecuTorch iOS frameworks, Pigeon (code generation), SoLoader (0.10.5), FBJNI (0.5.1)
**Storage**: Local file system (model files), memory-mapped model access for large files
**Testing**: flutter_test, integration_test, native unit tests (JUnit/XCTest), real device validation
**Target Platform**: Android (arm64-v8a, x86_64), iOS (arm64), macOS (arm64)
**Project Type**: mobile (Flutter plugin package)
**Performance Goals**: <200ms model loading, <50ms inference for typical models, <100MB memory overhead
**Constraints**: On-device inference only, .pte model format, async API, official library usage only
**Scale/Scope**: Single plugin package, 3-5 main API classes, multiple concurrent models, real ExecuTorch integration
**Documentation Verification**: Latest Android/iOS integration patterns verified from docs.pytorch.org/executorch/stable/

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Test-First Development**: ✓ PASS
- Pigeon contracts will have failing tests before implementation
- Native platform tests required before shipping
- Integration tests with real models mandatory

**II. Platform Parity and Performance**: ✓ PASS
- Identical API across Android/iOS platforms
- Performance targets aligned with constitutional requirements
- Official ExecuTorch libraries ensure consistency

**III. Type-Safe Cross-Platform Communication**: ✓ PASS
- Pigeon-only interfaces, no manual method channels
- Async operations prevent UI blocking
- Structured exception handling implemented

**IV. Documentation and Example-Driven Development**: ✓ PASS
- Latest ExecuTorch docs verified via WebFetch
- README and quickstart guides maintained
- Example directory with working demonstrations

**V. Resource Management and Memory Safety**: ✓ PASS
- Explicit model lifecycle management
- Platform-appropriate cleanup patterns
- Memory mapping for large models

## Project Structure

### Documentation (this feature)
```
specs/001-we-are-building/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command) - UPDATED
├── data-model.md        # Phase 1 output (/plan command) - EXISTS
├── quickstart.md        # Phase 1 output (/plan command) - EXISTS
├── contracts/           # Phase 1 output (/plan command) - EXISTS
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Flutter Plugin Package Structure
lib/
├── executorch_flutter.dart           # Main library export
├── src/
│   ├── executorch_model.dart         # Model wrapper class
│   ├── executorch_inference.dart     # Inference handling
│   ├── executorch_types.dart         # Data types and enums
│   └── generated/                    # Pigeon generated code
│       ├── executorch_api.dart
│       └── executorch_api.g.dart

android/
├── src/main/kotlin/
│   └── com/executorch/flutter/
│       ├── ExecutorchFlutterPlugin.kt
│       ├── ExecutorchModelManager.kt
│       └── generated/               # Pigeon generated code
└── build.gradle                     # ExecuTorch AAR 0.7.0 dependency

ios/
├── Classes/
│   ├── ExecutorchFlutterPlugin.swift
│   ├── ExecutorchModelManager.swift
│   └── Generated/                   # Pigeon generated code
└── executorch_flutter.podspec       # ExecuTorch frameworks dependency

example/
├── lib/main.dart                    # Example app
├── android/                         # Android example config
└── ios/                            # iOS example config

test/
├── executorch_flutter_test.dart     # Unit tests
└── integration_test/
    └── executorch_integration_test.dart

pigeons/
└── executorch_api.dart              # Pigeon interface definitions
```

**Structure Decision**: Flutter plugin package structure (mobile pattern)

## Phase 0: Outline & Research

1. **Latest Documentation Verification** (COMPLETED):
   ✓ Android integration verified: ExecuTorch AAR 0.7.0, SoLoader 0.10.5, FBJNI 0.5.1
   ✓ iOS integration verified: Xcode 15+, Swift 5.9+, ARM64 requirement
   ✓ Performance patterns and API usage confirmed
   ✓ Backend options documented (XNNPACK, Core ML, MPS, etc.)

2. **Research Updates Required**:
   - Update Android requirements to ExecuTorch 0.7.0 (was 0.6.0+)
   - Document iOS ARM64 requirement and Xcode 15+ dependency
   - Include new backend options and performance characteristics
   - Verify build script patterns for iOS frameworks

3. **Consolidate findings** in updated `research.md`:
   - Latest dependency versions and requirements
   - Updated build configurations and minimum versions
   - Platform-specific integration patterns
   - Performance benchmarks and backend options

**Output**: Updated research.md with verified latest documentation

## Phase 1: Design & Contracts

*Prerequisites: research.md updated with latest verification*

1. **Existing artifacts verified**:
   ✓ data-model.md: Core entity definitions remain valid
   ✓ contracts/executorch_api.dart: Pigeon interfaces current
   ✓ quickstart.md: Usage examples functional

2. **Updates required for latest docs**:
   - Android build.gradle: Update to ExecuTorch 0.7.0 dependencies
   - iOS podspec: Verify framework references and minimum versions
   - API contracts: Ensure compatibility with latest ExecuTorch runtime API
   - Error handling: Map to current ExecuTorch exception patterns

3. **Validation against current APIs**:
   - Module.load() pattern verification
   - Tensor.fromBlob() usage patterns
   - Backend selection mechanisms
   - Memory management approaches

**Output**: Verified and updated design artifacts

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks with mandatory documentation verification steps
- **CRITICAL**: Each implementation task MUST include pre-implementation verification of current ExecuTorch docs
- Android tasks: Verify AAR 0.7.0 integration, SoLoader/FBJNI dependencies
- iOS tasks: Verify Xcode 15+ requirements, framework build process
- Contract testing: Real ExecuTorch model integration testing

**Documentation Verification Requirements**:
Each platform implementation task MUST include:
1. Pre-implementation verification of latest ExecuTorch documentation
2. Comparison with current implementation plan
3. Update implementation if doc changes detected
4. Test with real ExecuTorch models (.pte files)

**Ordering Strategy**:
- TDD order: Documentation verification → Tests → Implementation
- Platform verification: Android docs → iOS docs → Cross-platform contracts
- Real model testing: Before any implementation is considered complete

**Estimated Output**: 25-30 numbered tasks with mandatory doc verification steps

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution with documentation verification (/tasks command creates tasks.md)
**Phase 4**: Implementation following constitutional TDD principles with real model testing
**Phase 5**: Validation including performance benchmarking and cross-platform parity verification

## Complexity Tracking
*No constitutional violations identified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - Updated with latest docs
- [x] Phase 1: Design complete (/plan command) - Verified against current APIs
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Latest ExecuTorch documentation verified
- [x] Complexity deviations documented (none)

**Documentation Verification Status**:
- [x] Android integration docs verified (docs.pytorch.org/executorch/stable/using-executorch-android.html)
- [x] iOS integration docs verified (docs.pytorch.org/executorch/stable/using-executorch-ios.html)
- [x] Dependencies and versions updated to current stable releases
- [x] API patterns verified against latest runtime interfaces

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*