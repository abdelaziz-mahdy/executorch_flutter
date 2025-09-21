# Quickstart Guide: Flutter ExecuTorch Package

This guide demonstrates how to integrate and use the Flutter ExecuTorch package for on-device ML inference.

## Installation

### 1. Add Dependency
Add to your `pubspec.yaml`:
```yaml
dependencies:
  executorch_flutter: ^1.0.0
```

### 2. Platform Setup

#### Android
Ensure minimum SDK version and dependencies in `android/app/build.gradle`:
```gradle
android {
    defaultConfig {
        minSdkVersion 23  // Required for ExecuTorch
    }
}

dependencies {
    // ExecuTorch dependencies (automatically included with plugin)
    implementation 'org.pytorch:executorch-android:0.7.0'
    implementation 'com.facebook.soloader:soloader:0.10.5'
    implementation 'com.facebook.fbjni:fbjni:0.5.1'
}
```

#### iOS
Set minimum deployment target and ensure Xcode version in `ios/Podfile`:
```ruby
platform :ios, '13.0'  # Required for ExecuTorch
```
**Requirements**: Xcode 15+, Swift 5.9+, ARM64 devices only

## Basic Usage

### 1. Import the Package
```dart
import 'package:executorch_flutter/executorch_flutter.dart';
```

### 2. Initialize ExecuTorch
```dart
class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late ExecutorchManager _executorch;

  @override
  void initState() {
    super.initState();
    _executorch = ExecutorchManager();
  }

  @override
  void dispose() {
    _executorch.dispose();
    super.dispose();
  }
}
```

### 3. Load a Model
```dart
Future<void> loadModel() async {
  try {
    // Load model from assets
    final modelPath = 'assets/models/my_model.pte';
    final model = await _executorch.loadModel(modelPath);

    print('Model loaded: ${model.id}');
    print('Input specs: ${model.metadata.inputSpecs}');
    print('Output specs: ${model.metadata.outputSpecs}');
  } catch (e) {
    print('Failed to load model: $e');
  }
}
```

### 4. Prepare Input Data
```dart
TensorData createInputTensor() {
  // Example: Create a float32 tensor with shape [1, 224, 224, 3]
  final shape = [1, 224, 224, 3];
  final elementCount = shape.reduce((a, b) => a * b);

  // Create sample data (replace with actual image data)
  final float32Data = Float32List(elementCount);
  for (int i = 0; i < elementCount; i++) {
    float32Data[i] = i.toDouble() / elementCount; // Normalized sample data
  }

  // Convert to bytes
  final bytes = Uint8List.view(float32Data.buffer);

  return TensorData(
    shape: shape,
    dataType: TensorType.float32,
    data: bytes,
    name: 'input_image',
  );
}
```

### 5. Run Inference
```dart
Future<void> runInference(String modelId) async {
  try {
    // Prepare input
    final inputTensor = createInputTensor();

    // Create inference request
    final request = InferenceRequest(
      modelId: modelId,
      inputs: [inputTensor],
      timeoutMs: 5000, // 5 second timeout
    );

    // Run inference
    final result = await _executorch.runInference(request);

    if (result.status == InferenceStatus.success) {
      print('Inference completed in ${result.executionTimeMs}ms');
      print('Output tensors: ${result.outputs?.length}');

      // Process outputs
      for (final output in result.outputs ?? []) {
        print('Output shape: ${output.shape}');
        print('Output type: ${output.dataType}');
        // Process output.data as needed
      }
    } else {
      print('Inference failed: ${result.errorMessage}');
    }
  } catch (e) {
    print('Inference error: $e');
  }
}
```

## Complete Example

```dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

class MLInferenceScreen extends StatefulWidget {
  @override
  _MLInferenceScreenState createState() => _MLInferenceScreenState();
}

class _MLInferenceScreenState extends State<MLInferenceScreen> {
  final ExecutorchManager _executorch = ExecutorchManager();
  ExecuTorchModel? _model;
  String _status = 'Ready';
  String _results = '';

  @override
  void dispose() {
    _model?.dispose();
    _executorch.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    setState(() => _status = 'Loading model...');

    try {
      final model = await _executorch.loadModelFromAssets('models/my_model.pte');
      setState(() {
        _model = model;
        _status = 'Model loaded successfully';
      });
    } catch (e) {
      setState(() => _status = 'Failed to load model: $e');
    }
  }

  Future<void> _runInference() async {
    if (_model == null) {
      setState(() => _status = 'No model loaded');
      return;
    }

    setState(() => _status = 'Running inference...');

    try {
      // Create sample input (replace with actual data)
      final input = TensorData(
        shape: [1, 224, 224, 3],
        dataType: TensorType.float32,
        data: _createSampleImageData(),
      );

      final request = InferenceRequest(
        modelId: _model!.id,
        inputs: [input],
        timeoutMs: 5000,
      );

      final result = await _executorch.runInference(request);

      if (result.status == InferenceStatus.success) {
        setState(() {
          _status = 'Inference completed';
          _results = 'Execution time: ${result.executionTimeMs.toStringAsFixed(2)}ms\n'
              'Output tensors: ${result.outputs?.length ?? 0}';
        });
      } else {
        setState(() {
          _status = 'Inference failed';
          _results = result.errorMessage ?? 'Unknown error';
        });
      }
    } catch (e) {
      setState(() {
        _status = 'Error during inference';
        _results = e.toString();
      });
    }
  }

  Uint8List _createSampleImageData() {
    // Create sample RGB image data (224x224x3 float32 values)
    final elementCount = 224 * 224 * 3;
    final float32Data = Float32List(elementCount);

    for (int i = 0; i < elementCount; i++) {
      float32Data[i] = (i % 256) / 255.0; // Normalized sample data
    }

    return Uint8List.view(float32Data.buffer);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('ExecuTorch Demo')),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Status: $_status', style: Theme.of(context).textTheme.headlineSmall),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadModel,
              child: Text('Load Model'),
            ),
            SizedBox(height: 8),
            ElevatedButton(
              onPressed: _model != null ? _runInference : null,
              child: Text('Run Inference'),
            ),
            SizedBox(height: 16),
            if (_results.isNotEmpty) ...[
              Text('Results:', style: Theme.of(context).textTheme.titleMedium),
              SizedBox(height: 8),
              Container(
                padding: EdgeInsets.all(8),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(_results),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
```

## Model Preparation

### Converting Models to ExecuTorch Format
```bash
# Example: Convert PyTorch model to ExecuTorch
python -m executorch.exir.lowered_backend_modules.exir_export_linalg \
  --model my_model.py \
  --output my_model.pte
```

### Adding Models to Flutter Assets
1. Create `assets/models/` directory
2. Add model files to the directory
3. Update `pubspec.yaml`:
```yaml
flutter:
  assets:
    - assets/models/
```

## Error Handling

```dart
try {
  final result = await _executorch.runInference(request);
  // Handle result
} on ModelLoadException catch (e) {
  // Model loading errors
  print('Model load error: ${e.message}');
} on InferenceException catch (e) {
  // Inference execution errors
  print('Inference error: ${e.message}');
} on ValidationException catch (e) {
  // Input validation errors
  print('Validation error: ${e.message}');
} on ResourceException catch (e) {
  // Memory or resource errors
  print('Resource error: ${e.message}');
} catch (e) {
  // Other errors
  print('Unexpected error: $e');
}
```

## Performance Tips

1. **Model Optimization**: Use quantized models for better performance
2. **Input Preprocessing**: Optimize image/data preprocessing on the Dart side
3. **Memory Management**: Dispose models when no longer needed
4. **Batch Processing**: Process multiple inputs in a single inference call when supported
5. **Async Operations**: Always use async/await to avoid blocking the UI

## Next Steps

- Check the [API documentation](../data-model.md) for detailed class references
- See [platform integration guide](../research.md) for advanced platform-specific features
- Review [example implementations](../../example/) for more complex use cases