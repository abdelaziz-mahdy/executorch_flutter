# Data Model: Model-Specific Processor Classes

**Feature**: Model-Specific Processor Classes
**Date**: 2025-09-22
**Status**: Complete

## Core Entities

### 1. ExecuTorchPreprocessor<T>
**Purpose**: Abstract interface for preprocessing input data before model inference
**Type**: Abstract class with generic type parameter
**Lifecycle**: Stateless, instantiated per model type

**Key Properties**:
- `inputTypeName`: String identifier for debugging/logging
- Generic type `T`: Input data type (e.g., Uint8List for images, String for text)

**Key Methods**:
- `preprocess(T input, {ModelMetadata? metadata})`: Transform input into List<TensorData>
- `validateInput(T input)`: Validate input compatibility
- `Future<List<TensorData>>`: Returns tensor data ready for inference

**Validation Rules**:
- Input must pass validateInput() check before preprocessing
- Output tensors must have valid shape and dataType
- Processing must complete within performance targets (<50ms)

### 2. ExecuTorchPostprocessor<T>
**Purpose**: Abstract interface for postprocessing model outputs into meaningful results
**Type**: Abstract class with generic type parameter
**Lifecycle**: Stateless, instantiated per model type

**Key Properties**:
- `outputTypeName`: String identifier for debugging/logging
- Generic type `T`: Output result type (e.g., ClassificationResult, Map<String, double>)

**Key Methods**:
- `postprocess(List<TensorData> outputs, {ModelMetadata? metadata})`: Transform model outputs
- `validateOutputs(List<TensorData> outputs)`: Validate output compatibility
- `Future<T>`: Returns processed results in desired format

**Validation Rules**:
- Outputs must pass validateOutputs() check before postprocessing
- Input tensors must have expected shape and dataType
- Processing must complete within performance targets (<50ms)

### 3. ExecuTorchProcessor<TInput, TOutput>
**Purpose**: Combined processor handling complete input-to-output pipeline
**Type**: Abstract class with dual generic type parameters
**Lifecycle**: Stateless, composition of preprocessor and postprocessor

**Key Properties**:
- `preprocessor`: ExecuTorchPreprocessor<TInput> instance
- `postprocessor`: ExecuTorchPostprocessor<TOutput> instance

**Key Methods**:
- `process(TInput input, dynamic model, {ModelMetadata? metadata})`: Complete pipeline
- Returns `Future<TOutput>`: Final processed output

**Relationships**:
- Composes ExecuTorchPreprocessor and ExecuTorchPostprocessor
- Orchestrates: preprocess → inference → postprocess

### 4. ProcessorTensorUtils
**Purpose**: Utility class for common tensor operations in processors
**Type**: Static utility class
**Lifecycle**: Stateless utility functions

**Key Methods**:
- `createTensor({shape, dataType, data, name})`: Create TensorData from values
- `extractFloat32Data(TensorData)`: Extract float data from tensor
- `extractInt32Data(TensorData)`: Extract integer data from tensor
- `calculateElementCount(List<int> shape)`: Calculate total elements

**Validation Rules**:
- Data length must match calculated element count
- DataType must match extraction method
- Shape must be valid (positive dimensions)

### 5. Configuration Objects

#### ImagePreprocessConfig
**Purpose**: Configuration for image preprocessing operations
**Type**: Immutable data class with const constructor
**Properties**:
- `targetWidth`, `targetHeight`: int (resize dimensions)
- `normalizeToFloat`: bool (convert to float values)
- `meanSubtraction`: List<double> (normalization means)
- `standardDeviation`: List<double> (normalization std devs)
- `channelOrder`: ChannelOrder enum (RGB/BGR)
- `dataLayout`: DataLayout enum (NCHW/NHWC)

#### TextPreprocessConfig
**Purpose**: Configuration for text preprocessing operations
**Type**: Immutable data class with const constructor
**Properties**:
- `maxLength`: int (maximum sequence length)
- `paddingToken`: int (token ID for padding)
- `truncationStrategy`: TruncationStrategy enum
- `includeCLS`, `includeSEP`: bool (special tokens)

#### AudioPreprocessConfig
**Purpose**: Configuration for audio preprocessing operations
**Type**: Immutable data class with const constructor
**Properties**:
- `sampleRate`: int (target sample rate)
- `targetLength`: int (target audio length)
- `normalizeAudio`: bool (apply normalization)
- `applyPreemphasis`: bool (apply preemphasis filter)

### 6. Exception Types

#### PreprocessingException
**Purpose**: Exception for preprocessing failures
**Properties**:
- `message`: String (error description)
- `details`: Map<String, dynamic>? (additional context)

#### PostprocessingException
**Purpose**: Exception for postprocessing failures
**Properties**:
- `message`: String (error description)
- `details`: Map<String, dynamic>? (additional context)

## Entity Relationships

```
ExecuTorchProcessor<TInput, TOutput>
├── preprocessor: ExecuTorchPreprocessor<TInput>
└── postprocessor: ExecuTorchPostprocessor<TOutput>

ExecuTorchPreprocessor<T>
├── input: T (e.g., Uint8List, String)
├── config: ProcessorConfig (e.g., ImagePreprocessConfig)
└── output: List<TensorData>

ExecuTorchPostprocessor<T>
├── input: List<TensorData>
├── config: ProcessorConfig (optional)
└── output: T (e.g., ClassificationResult)

ProcessorTensorUtils
├── createTensor() → TensorData
├── extractFloat32Data() → List<double>
└── extractInt32Data() → List<int>
```

## State Transitions

### Preprocessing Flow
1. **Input Validation**: validateInput(input) → bool
2. **Configuration**: Apply processor configuration
3. **Data Transformation**: Convert input to tensor format
4. **Tensor Creation**: Create TensorData with proper shape/type
5. **Output**: Return List<TensorData> for inference

### Postprocessing Flow
1. **Output Validation**: validateOutputs(outputs) → bool
2. **Data Extraction**: Extract numeric data from tensors
3. **Result Transformation**: Apply domain-specific logic
4. **Format Conversion**: Convert to user-friendly format
5. **Output**: Return typed result (T)

### Error States
- **Validation Failure**: Throw PreprocessingException or PostprocessingException
- **Processing Timeout**: Exceed performance targets
- **Memory Exhaustion**: Large inputs exceed available memory
- **Format Mismatch**: Incompatible tensor shapes or data types

## Implementation Constraints

### Type Safety
- All processors must use generic type parameters
- Compile-time validation of input/output types
- No dynamic typing or unsafe casts

### Performance
- Preprocessing/postprocessing must complete within 50ms
- Memory usage should be minimal and temporary
- No blocking operations on UI thread

### Compatibility
- Must work with existing TensorData and InferenceResult types
- No breaking changes to current ExecuTorch API
- Support for optional ModelMetadata parameter

### Testing
- Each entity must have comprehensive unit tests
- Integration tests for complete processor pipelines
- Performance tests for timing and memory usage