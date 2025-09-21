# Data Model: Flutter ExecuTorch Package

## Core Entities

### ExecuTorchModel
Represents a loaded machine learning model instance.

**Fields**:
- `String id`: Unique identifier for the model instance
- `String filePath`: Original file path of the model
- `ModelState state`: Current loading/ready/error state
- `ModelMetadata metadata`: Model information and capabilities
- `DateTime loadedAt`: Timestamp when model was loaded
- `int? memoryUsage`: Estimated memory usage in bytes

**States**:
- `ModelState.loading`: Model is being loaded from file
- `ModelState.ready`: Model is loaded and ready for inference
- `ModelState.error`: Model failed to load or encountered error
- `ModelState.disposed`: Model has been disposed and is no longer usable

**Validation Rules**:
- `id` must be non-empty and unique within application
- `filePath` must be valid file system path
- `state` transitions: loading → ready/error, ready → disposed, error → disposed
- `memoryUsage` must be positive when available

### InferenceRequest
Contains input data and options for model inference.

**Fields**:
- `String modelId`: Reference to target ExecuTorchModel
- `List<TensorData> inputs`: Input tensor data for inference
- `Map<String, dynamic> options`: Optional inference parameters
- `Duration? timeout`: Maximum execution time before cancellation
- `String? requestId`: Optional identifier for tracking

**Validation Rules**:
- `modelId` must reference an existing, ready model
- `inputs` must match model's expected input specification
- `timeout` must be positive if specified
- Input tensor shapes must be compatible with model requirements

### InferenceResult
Contains output data and metadata from model inference.

**Fields**:
- `String requestId`: Matches request identifier if provided
- `List<TensorData> outputs`: Output tensor data from inference
- `Duration executionTime`: Actual inference execution time
- `InferenceStatus status`: Success/failure status
- `String? errorMessage`: Error description if inference failed
- `Map<String, dynamic> metadata`: Additional execution information

**States**:
- `InferenceStatus.success`: Inference completed successfully
- `InferenceStatus.error`: Inference failed with error
- `InferenceStatus.timeout`: Inference exceeded timeout limit
- `InferenceStatus.cancelled`: Inference was cancelled by user

**Validation Rules**:
- `outputs` must be present when status is success
- `executionTime` must be positive
- `errorMessage` must be present when status is error
- Output tensor count must match model specification

### TensorData
Represents input or output tensor data.

**Fields**:
- `List<int> shape`: Tensor dimensions [height, width, channels, etc.]
- `TensorType dataType`: Data type (float32, int8, etc.)
- `Uint8List data`: Raw tensor data in bytes
- `String? name`: Optional tensor name for identification

**Supported Types**:
- `TensorType.float32`: 32-bit floating point
- `TensorType.int8`: 8-bit signed integer
- `TensorType.int32`: 32-bit signed integer
- `TensorType.uint8`: 8-bit unsigned integer

**Validation Rules**:
- `shape` must contain positive dimensions
- `data` byte length must match shape × dataType size
- `dataType` must be supported by ExecuTorch runtime
- Tensor memory layout must be contiguous

### ModelMetadata
Provides information about model capabilities and requirements.

**Fields**:
- `String modelName`: Human-readable model name
- `String version`: Model version string
- `List<TensorSpec> inputSpecs`: Expected input tensor specifications
- `List<TensorSpec> outputSpecs`: Expected output tensor specifications
- `Map<String, dynamic> properties`: Additional model properties
- `int estimatedMemoryMB`: Estimated memory requirement

**Validation Rules**:
- `inputSpecs` and `outputSpecs` must be non-empty
- `estimatedMemoryMB` must be positive
- Tensor specifications must be consistent with actual model

### TensorSpec
Specification for input or output tensor requirements.

**Fields**:
- `String name`: Tensor name or identifier
- `List<int> shape`: Expected tensor dimensions (-1 for dynamic)
- `TensorType dataType`: Required data type
- `bool optional`: Whether tensor is optional for inference
- `List<int>? validRange`: Valid value range for tensor data

**Validation Rules**:
- `shape` dimensions must be positive or -1 for dynamic
- Dynamic dimensions (-1) allowed only for specific positions
- `validRange` must contain exactly 2 elements [min, max] if specified

## Entity Relationships

```
ExecuTorchModel (1) -----> (1) ModelMetadata
       |
       | (1)
       |
       v
InferenceRequest -----> (1) InferenceResult
       |                        |
       | (*)                    | (*)
       v                        v
   TensorData              TensorData
       |
       | (1)
       v
   TensorSpec (via validation)
```

## State Transitions

### Model Lifecycle
```
[Created] → [Loading] → [Ready] → [Disposed]
                ↓
            [Error] → [Disposed]
```

### Inference Lifecycle
```
[Requested] → [Executing] → [Success]
                   ↓
               [Timeout/Error/Cancelled]
```

## Data Serialization

### Pigeon Interface Mapping
- Dart entities map to Pigeon data classes
- Native platforms use generated equivalent classes
- Tensor data transmitted as byte arrays with metadata
- Complex nested structures flattened for cross-platform compatibility

### Memory Management
- Large tensor data shared via memory mapping when possible
- Native platforms responsible for ExecuTorch tensor lifecycle
- Dart side maintains references and metadata only
- Automatic disposal on model destruction or app termination

## Error Handling

### Exception Types
- `ModelLoadException`: Model file access or format errors
- `InferenceException`: Runtime inference errors
- `ValidationException`: Input validation failures
- `ResourceException`: Memory or resource constraints
- `PlatformException`: Native platform communication errors

### Error Context
All exceptions include:
- Error code for programmatic handling
- Human-readable error message
- Context information (model ID, file path, etc.)
- Platform-specific error details when available