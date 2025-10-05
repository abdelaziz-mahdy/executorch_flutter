# Data Model: macOS Platform Support

**Feature**: Add macOS support to ExecuTorch Flutter Plugin
**Date**: 2025-10-02

## Overview

The data model for macOS is **identical** to the iOS implementation. All entities, relationships, and data structures are reused without modification. This document references the existing iOS data model and confirms its applicability to macOS.

## Core Entities

### 1. ExecuTorchModel
**Purpose**: Represents a loaded ExecuTorch model instance

**Attributes**:
- `modelId`: String - Unique identifier for the model
- `modelPath`: String - File system path to .pte model file
- `state`: ModelState enum - Current model state (loading, ready, error, disposed)
- `metadata`: ModelMetadata? - Optional model information

**Lifecycle States**:
```
loading → ready → disposed
   ↓
 error
```

**Platform Notes**: No macOS-specific differences

### 2. TensorData
**Purpose**: Multi-dimensional array for model input/output

**Attributes**:
- `data`: Uint8List - Raw tensor data bytes
- `shape`: List<int?> - Tensor dimensions (e.g., [1, 3, 224, 224])
- `dataType`: TensorType enum - Data type (float32, int8, int32, uint8)
- `name`: String? - Optional tensor name

**Platform Notes**: Memory representation identical on both platforms

### 3. InferenceResult
**Purpose**: Output from model inference execution

**Attributes**:
- `outputs`: List<TensorData> - Output tensors
- `status`: InferenceStatus enum - Execution status
- `executionTimeMs`: double - Inference duration
- `error`: String? - Error message if failed

**Platform Notes**: No platform-specific fields needed

### 4. ModelMetadata
**Purpose**: Model information and capabilities

**Attributes**:
- `inputSpecs`: List<TensorSpec> - Expected inputs
- `outputSpecs`: List<TensorSpec> - Expected outputs
- `backend`: String? - Backend used (xnnpack, coreml, mps)
- `version`: String? - Model version

**Platform Notes**: Available backends identical (XNNPACK, Core ML, MPS)

## Enumerations

### ModelState
```dart
enum ModelState {
  loading,
  ready,
  error,
  disposed,
}
```

### TensorType
```dart
enum TensorType {
  float32,
  int8,
  int32,
  uint8,
}
```

### InferenceStatus
```dart
enum InferenceStatus {
  success,
  error,
  timeout,
  cancelled,
}
```

## Relationships

```
ExecutorchManager
    ↓ manages
ExecuTorchModel (1..*)
    ↓ accepts
TensorData (input)
    ↓ produces
InferenceResult
    ↓ contains
TensorData (output)
```

## Validation Rules

All validation rules from iOS apply to macOS:

1. **Model Loading**:
   - File must exist and be readable
   - File must have .pte extension
   - File size must be reasonable (<2GB recommended)

2. **Tensor Input**:
   - Shape must match model requirements
   - Data type must match expected type
   - Data size must equal product(shape) * sizeof(type)

3. **Inference**:
   - Model must be in `ready` state
   - All required inputs must be provided
   - Tensor shapes must match exactly

## Platform-Specific Considerations

### Memory Management
- **iOS**: Automatic memory management via ARC
- **macOS**: Same ARC behavior, but more permissive memory limits

### File System Access
- **iOS**: Sandboxed app directories
- **macOS**: Broader file system access (if user grants permission)

### Both Platforms
- Models memory-mapped for efficient loading
- Background inference on dedicated threads
- Main thread callbacks for results

## State Transitions

### Model Lifecycle
```
[Initial] → loading → ready → [Active]
                ↓
              error → [Terminal]

[Active] → disposed → [Terminal]
```

### Inference Execution
```
[Ready Model] → execute() → InferenceResult
                    ↓
              success/error/timeout/cancelled
```

## Data Flow

```
1. User provides model path
   ↓
2. Platform loads .pte file
   ↓
3. ExecuTorch initializes model
   ↓
4. Model enters 'ready' state
   ↓
5. User provides TensorData inputs
   ↓
6. Platform executes inference
   ↓
7. InferenceResult returned with output TensorData
```

## Constraints

1. **Thread Safety**:
   - Model instances are thread-safe
   - Concurrent inference supported per model
   - Platform manages threading internally

2. **Resource Limits**:
   - No hard limit on number of loaded models
   - Memory constrained by device capabilities
   - Recommended max 3-5 large models simultaneously

3. **Performance Targets**:
   - Model load: <200ms for models <100MB
   - Inference: <50ms for typical mobile models
   - Memory overhead: <100MB per loaded model

## No macOS-Specific Extensions Required

This data model requires **zero modifications** for macOS support. All entities, relationships, validation rules, and constraints apply identically to both platforms.

The implementation reuses existing Pigeon-generated data classes which are platform-agnostic by design.
