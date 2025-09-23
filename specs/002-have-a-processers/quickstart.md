# Quickstart: Model-Specific Processor Classes

**Feature**: Model-Specific Processor Classes
**Date**: 2025-09-22
**Purpose**: Demonstrate clean processor usage patterns in the example app

## Overview
This quickstart guide demonstrates how to use model-specific processor classes to cleanly separate data transformation logic from model inference in the ExecuTorch Flutter package.

## Prerequisites
- ExecuTorch Flutter package set up with processor classes
- Example ImageNet model (.pte file)
- Sample image for classification testing
- Flutter development environment

## Quick Start Scenarios

### Scenario 1: Basic ImageNet Classification
**Goal**: Classify an image using ImageNet processor with clean code separation

**Setup**:
```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// Initialize the processor with ImageNet configuration
final processor = ImageNetProcessor(
  preprocessConfig: const ImagePreprocessConfig(
    targetWidth: 224,
    targetHeight: 224,
    normalizeToFloat: true,
    meanSubtraction: [0.485, 0.456, 0.406], // ImageNet normalization
    standardDeviation: [0.229, 0.224, 0.225],
    channelOrder: ChannelOrder.rgb,
    dataLayout: DataLayout.nchw,
  ),
  classLabels: ImageNetLabels.labels, // 1000 ImageNet class labels
);
```

**Usage**:
```dart
// Load model (one-time setup)
final model = await ExecutorchManager.instance.loadModel('assets/imagenet_model.pte');

// Load image bytes (from camera, file, etc.)
final imageBytes = await loadImageBytes('path/to/image.jpg');

// Process through complete pipeline
try {
  final result = await processor.process(imageBytes, model);
  print('Classification: ${result.className}');
  print('Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%');
} catch (e) {
  print('Classification failed: $e');
}
```

**Expected Output**:
```
Classification: golden retriever
Confidence: 87.3%
```

### Scenario 2: Manual Preprocessing/Postprocessing
**Goal**: Use preprocessor and postprocessor separately for custom workflows

**Setup**:
```dart
// Create preprocessor and postprocessor separately
final preprocessor = ImageNetPreprocessor(
  config: const ImagePreprocessConfig(
    targetWidth: 224,
    targetHeight: 224,
    normalizeToFloat: true,
    meanSubtraction: [0.485, 0.456, 0.406],
    standardDeviation: [0.229, 0.224, 0.225],
  ),
);

final postprocessor = ImageNetPostprocessor(
  classLabels: ImageNetLabels.labels,
);
```

**Usage**:
```dart
// Manual preprocessing
final imageBytes = await loadImageBytes('path/to/image.jpg');
final tensors = await preprocessor.preprocess(imageBytes);

// Run inference manually
final inferenceResult = await model.runInference(inputs: tensors);

// Manual postprocessing
if (inferenceResult.status == InferenceStatus.success) {
  final result = await postprocessor.postprocess(inferenceResult.outputs!);
  print('Top prediction: ${result.className} (${result.confidence})');

  // Access all probabilities for custom analysis
  final topIndices = getTopKIndices(result.allProbabilities, k: 5);
  for (int i = 0; i < topIndices.length; i++) {
    final idx = topIndices[i];
    print('${i + 1}. ${ImageNetLabels.labels[idx]}: ${result.allProbabilities[idx].toStringAsFixed(3)}');
  }
}
```

**Expected Output**:
```
Top prediction: golden retriever (0.873)
1. golden retriever: 0.873
2. Nova Scotia duck tolling retriever: 0.089
3. Labrador retriever: 0.023
4. beagle: 0.008
5. cocker spaniel: 0.004
```

### Scenario 3: Custom Processor Implementation
**Goal**: Implement a custom processor for a different model type

**Setup**:
```dart
// Custom preprocessor for grayscale images
class GrayscaleImagePreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  @override
  String get inputTypeName => 'Grayscale Image Bytes';

  @override
  bool validateInput(Uint8List input) => input.isNotEmpty;

  @override
  Future<List<TensorData>> preprocess(Uint8List input, {ModelMetadata? metadata}) async {
    // Custom grayscale preprocessing logic
    final processedData = await convertToGrayscale(input);

    final tensor = ProcessorTensorUtils.createTensor(
      shape: [1, 1, 224, 224], // Batch, Channel, Height, Width
      dataType: TensorType.float32,
      data: processedData,
      name: 'grayscale_input',
    );

    return [tensor];
  }
}

// Custom postprocessor for binary classification
class BinaryClassificationPostprocessor extends ExecuTorchPostprocessor<Map<String, double>> {
  const BinaryClassificationPostprocessor({required this.classNames});

  final List<String> classNames;

  @override
  String get outputTypeName => 'Binary Classification Result';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    return outputs.isNotEmpty &&
           outputs.first.dataType == TensorType.float32;
  }

  @override
  Future<Map<String, double>> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
    final scores = ProcessorTensorUtils.extractFloat32Data(outputs.first);

    return {
      classNames[0]: scores[0],
      classNames[1]: scores[1],
    };
  }
}
```

**Usage**:
```dart
// Use custom processors
final customPreprocessor = GrayscaleImagePreprocessor();
final customPostprocessor = BinaryClassificationPostprocessor(
  classNames: ['cat', 'dog'],
);

final imageBytes = await loadImageBytes('path/to/pet.jpg');

// Process with custom pipeline
final tensors = await customPreprocessor.preprocess(imageBytes);
final inferenceResult = await model.runInference(inputs: tensors);
final result = await customPostprocessor.postprocess(inferenceResult.outputs!);

print('Classification scores: $result');
```

**Expected Output**:
```
Classification scores: {cat: 0.823, dog: 0.177}
```

### Scenario 4: Error Handling and Validation
**Goal**: Demonstrate proper error handling with processor validation

**Setup**:
```dart
final processor = ImageNetProcessor(/* configuration */);
```

**Usage**:
```dart
// Test input validation
final emptyInput = Uint8List(0);
if (!processor.preprocessor.validateInput(emptyInput)) {
  print('Input validation failed: empty image data');
  return;
}

// Test preprocessing error handling
try {
  final corruptedImage = Uint8List.fromList([1, 2, 3]); // Invalid image data
  final tensors = await processor.preprocessor.preprocess(corruptedImage);
} on PreprocessingException catch (e) {
  print('Preprocessing failed: ${e.message}');
  if (e.details != null) {
    print('Details: ${e.details}');
  }
}

// Test postprocessing validation
final invalidOutputs = <TensorData>[];
if (!processor.postprocessor.validateOutputs(invalidOutputs)) {
  print('Output validation failed: no output tensors');
}
```

**Expected Output**:
```
Input validation failed: empty image data
Preprocessing failed: Invalid image format
Details: {inputSize: 3, expectedMinimum: 100}
Output validation failed: no output tensors
```

## Performance Validation

### Benchmark Test
**Goal**: Verify processor performance meets constitutional requirements

```dart
// Performance test for preprocessing
final stopwatch = Stopwatch()..start();
final tensors = await processor.preprocessor.preprocess(imageBytes);
stopwatch.stop();

assert(stopwatch.elapsedMilliseconds < 50, 'Preprocessing too slow: ${stopwatch.elapsedMilliseconds}ms');
print('Preprocessing completed in ${stopwatch.elapsedMilliseconds}ms ✓');

// Performance test for postprocessing
stopwatch.reset()..start();
final result = await processor.postprocessor.postprocess(modelOutputs);
stopwatch.stop();

assert(stopwatch.elapsedMilliseconds < 50, 'Postprocessing too slow: ${stopwatch.elapsedMilliseconds}ms');
print('Postprocessing completed in ${stopwatch.elapsedMilliseconds}ms ✓');
```

### Memory Usage Test
**Goal**: Verify minimal memory overhead during processing

```dart
// Memory usage monitoring (pseudo-code for demonstration)
final initialMemory = await getMemoryUsage();

await processor.process(imageBytes, model);

final finalMemory = await getMemoryUsage();
final overhead = finalMemory - initialMemory;

assert(overhead < 100 * 1024 * 1024, 'Memory overhead too high: ${overhead ~/ (1024 * 1024)}MB');
print('Memory overhead: ${overhead ~/ (1024 * 1024)}MB ✓');
```

## Integration Test Checklist

### ✅ Basic Functionality
- [ ] ImageNet processor loads with default configuration
- [ ] Complete pipeline processes test image successfully
- [ ] Classification result contains valid class name and confidence
- [ ] Preprocessing completes within 50ms performance target
- [ ] Postprocessing completes within 50ms performance target

### ✅ Error Handling
- [ ] Invalid input properly rejected by validation
- [ ] Preprocessing errors throw PreprocessingException
- [ ] Postprocessing errors throw PostprocessingException
- [ ] Invalid model outputs handled gracefully
- [ ] Memory cleanup after processing errors

### ✅ Custom Implementation
- [ ] Custom preprocessor follows interface contract
- [ ] Custom postprocessor follows interface contract
- [ ] Custom configuration objects work correctly
- [ ] Type safety maintained with generic parameters
- [ ] Integration with existing ExecuTorch inference pipeline

### ✅ Example App Integration
- [ ] Example app demonstrates clean processor usage
- [ ] Code separation between transformation and inference
- [ ] Clear examples for different processor patterns
- [ ] Documentation updated with processor examples
- [ ] Working example with real ImageNet model

## Next Steps
1. Run all integration tests to validate processor functionality
2. Test with real ImageNet model and various image inputs
3. Benchmark performance on target devices (Android/iOS)
4. Update example app with clean processor implementation
5. Add comprehensive documentation and code examples

## Success Criteria
- ✅ All quickstart scenarios work as documented
- ✅ Performance targets met (<50ms preprocessing/postprocessing)
- ✅ Clean code separation demonstrated in example app
- ✅ Type safety maintained throughout processor pipeline
- ✅ Error handling works correctly for edge cases
- ✅ Custom processor implementation is straightforward