# ExecuTorch Model Export Guide

This guide explains how to export your PyTorch models to ExecuTorch format (.pte files) for use with the `executorch_flutter` package.

## Overview

The `executorch_flutter` package requires models in ExecuTorch's optimized `.pte` format. This guide covers:
- Model export using official PyTorch ExecuTorch documentation
- Platform-specific optimizations for Flutter apps
- Model validation and testing
- Integration with Flutter apps
- Example export scripts for reference

## Recommended Approach

**We recommend following the official PyTorch ExecuTorch export documentation**: [PyTorch ExecuTorch Export Guide](https://docs.pytorch.org/executorch/stable/using-executorch-export.html)

This ensures you have the most up-to-date export process and full control over optimizations for your specific use case.

### Why Use the Official Documentation?

- **Always up-to-date**: Latest ExecuTorch features and optimizations
- **Complete control**: Customize export for your specific model and requirements
- **Official support**: Direct access to PyTorch team documentation and examples
- **Best practices**: Learn the canonical way to work with ExecuTorch

---

## Export Process

### 1. Install ExecuTorch

```bash
# Create virtual environment (recommended)
python -m venv executorch_env
source executorch_env/bin/activate  # On Windows: executorch_env\Scripts\activate

# Install ExecuTorch
pip install executorch
```

### 2. Export Your Model

Follow the [official PyTorch ExecuTorch export guide](https://docs.pytorch.org/executorch/stable/using-executorch-export.html) for detailed instructions. Here's a basic example:

```python
import torch
from executorch.exir import to_edge

# Load your PyTorch model
model = torch.jit.load('your_model.pt')
model.eval()

# Create example input matching your model's expected input
example_input = torch.randn(1, 3, 224, 224)

# Export to ExecuTorch
exported_program = torch.export.export(model, (example_input,))
edge_program = to_edge(exported_program)
executorch_program = edge_program.to_executorch()

# Save as .pte file
with open("your_model.pte", "wb") as f:
    executorch_program.write_to_file(f)
```

### 3. Add to Flutter App

Copy the generated `.pte` files to your Flutter app's assets:

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/
```

```dart
// Load in Flutter
final model = await ExecutorchManager.instance.loadModel(
  'assets/models/my_model_ios_coreml.pte'
);
```

## Detailed Export Process

### Supported Model Formats

The exporter accepts PyTorch models in these formats:
- **`.pth`**: PyTorch state dict (recommended)
- **`.pt`**: PyTorch saved model
- **`.torchscript`**: TorchScript model

### Input Shape Requirements

You must specify the exact input shapes your model expects:

```bash
# Single input (common for image models)
--input-shapes 1,3,224,224

# Multiple inputs
--input-shapes "1,3,224,224;1,100"

# Dynamic batch size (use 1 for mobile)
--input-shapes 1,3,224,224  # Fixed batch size of 1
```

### Backend Selection

Different backends provide different optimizations:

#### ✅ Universal Backends
- **Portable**: Works everywhere, basic performance
- **XNNPACK**: CPU optimization, recommended for mobile

#### 🍎 iOS-Specific Backends
- **CoreML**: Apple Neural Engine acceleration (best for iOS)
- **MPS**: Metal Performance Shaders GPU acceleration

#### 🤖 Android-Specific Backends
- **Vulkan**: GPU acceleration (requires Vulkan support)
- **QNN**: Qualcomm Snapdragon optimization (requires SDK)

#### Platform Recommendations

**For iOS apps:**
```bash
python executorch_exporter.py model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends coreml mps xnnpack
```

**For Android apps:**
```bash
python executorch_exporter.py model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends xnnpack vulkan
```

**For cross-platform:**
```bash
python executorch_exporter.py model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends xnnpack
```

## Model Types and Examples

### Image Classification

```python
# Export a custom image classifier
python executorch_exporter.py my_classifier.pth \
  --model-name image_classifier \
  --input-shapes 1,3,224,224 \
  --backends coreml xnnpack
```

Expected input: `[batch_size, channels, height, width]` (NCHW format)

### Object Detection

```python
# Export object detection model
python executorch_exporter.py yolo_model.pth \
  --model-name object_detector \
  --input-shapes 1,3,640,640 \
  --backends xnnpack
```

### Text Processing

```python
# Export text model (e.g., sentiment analysis)
python executorch_exporter.py text_model.pth \
  --model-name text_classifier \
  --input-shapes 1,512 \
  --backends xnnpack
```

Expected input: Token IDs as integer tensor

### Custom Models

For models with non-standard architectures:

```python
# Export with verbose logging
python executorch_exporter.py custom_model.pth \
  --model-name custom \
  --input-shapes 1,256,256 \
  --backends portable \
  --verbose
```

## Integration Patterns

### Loading Models in Flutter

```dart
class ModelManager {
  static Future<ExecuTorchModel> loadOptimalModel(String baseName) async {
    // Platform-specific model selection
    String modelPath;
    if (Platform.isIOS) {
      modelPath = 'assets/models/${baseName}_ios_coreml.pte';
    } else if (Platform.isAndroid) {
      modelPath = 'assets/models/${baseName}_android_xnnpack.pte';
    } else {
      modelPath = 'assets/models/${baseName}_portable.pte';
    }

    return await ExecutorchManager.instance.loadModel(modelPath);
  }
}
```

### Preprocessing Input Data

```dart
// Image preprocessing for classification
TensorDataWrapper preprocessImage(Uint8List imageBytes) {
  final image = img.decodeImage(imageBytes)!;
  final resized = img.copyResize(image, width: 224, height: 224);

  // ImageNet normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  final floats = Float32List(1 * 3 * 224 * 224);
  int index = 0;

  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        final value = [pixel.r, pixel.g, pixel.b][c] / 255.0;
        floats[index++] = (value - mean[c]) / std[c];
      }
    }
  }

  return TensorDataWrapper(
    shape: [1, 3, 224, 224],
    dataType: TensorType.float32,
    data: floats.buffer.asUint8List(),
    name: 'input',
  );
}
```

### Using ImageNet Class Labels

For image classification models, load the ImageNet labels to get meaningful results:

```dart
class ImageClassifier {
  List<String> _imageNetLabels = [];

  Future<void> loadLabels() async {
    final labelsData = await rootBundle.loadString('assets/models/imagenet_classes.txt');
    _imageNetLabels = labelsData.trim().split('\n');
  }

  String parseClassificationResult(List<TensorDataWrapper> outputs) {
    final output = outputs.first;
    final byteData = ByteData.sublistView(output.data);
    final probabilities = <double>[];

    for (int i = 0; i < output.data.length ~/ 4; i++) {
      probabilities.add(byteData.getFloat32(i * 4, Endian.host));
    }

    // Find top-5 predictions
    final indexed = List.generate(probabilities.length, (i) => MapEntry(i, probabilities[i]));
    indexed.sort((a, b) => b.value.compareTo(a.value));
    final top5 = indexed.take(5).toList();

    final results = <String>[];
    for (int i = 0; i < top5.length; i++) {
      final index = top5[i].key;
      final prob = top5[i].value;
      final label = index < _imageNetLabels.length ? _imageNetLabels[index] : 'Unknown';
      final confidence = (prob * 100).toStringAsFixed(1);
      results.add('${i + 1}. $label ($confidence%)');
    }

    return 'Top predictions:\n${results.join('\n')}';
  }
}
```

### Handling Multiple Models

```dart
class MultiModelInference {
  final Map<String, ExecuTorchModel> _models = {};

  Future<void> loadModels() async {
    _models['classifier'] = await ModelManager.loadOptimalModel('classifier');
    _models['detector'] = await ModelManager.loadOptimalModel('detector');
  }

  Future<ClassificationResult> classify(Uint8List image) async {
    final input = preprocessImage(image);
    final result = await _models['classifier']!.runInference(inputs: [input]);
    return parseClassificationResult(result.outputs!);
  }
}
```

## Performance Optimization

### Model Size Optimization

```bash
# Use quantization for smaller models
python executorch_exporter.py model.pth \
  --model-name quantized_model \
  --input-shapes 1,3,224,224 \
  --backends xnnpack \
  --quantize
```

### Memory Management

```dart
// Proper model disposal
class ModelLifecycleManager {
  ExecuTorchModel? _model;

  Future<void> loadModel(String path) async {
    await _model?.dispose(); // Clean up previous model
    _model = await ExecutorchManager.instance.loadModel(path);
  }

  Future<void> dispose() async {
    await _model?.dispose();
    _model = null;
  }
}
```

## Troubleshooting

### Common Export Issues

**Error: "Unsupported operator"**
```bash
# Use portable backend for compatibility
python executorch_exporter.py model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends portable
```

**Error: "Input shape mismatch"**
- Verify your input shapes match the model's expected input
- Use `model.forward(torch.randn(1, 3, 224, 224))` to test locally

**Backend not available:**
- Check platform compatibility (CoreML only on macOS/iOS)
- Install required SDKs for specialized backends

### Flutter Integration Issues

**Model not loading:**
```dart
// Check if file exists in assets
try {
  await rootBundle.load('assets/models/my_model.pte');
  print('Model file found');
} catch (e) {
  print('Model file not found: $e');
}
```

**Performance issues:**
- Use platform-optimized backends (CoreML for iOS, XNNPACK for Android)
- Consider model quantization for large models
- Run inference on background isolates for heavy models

## Advanced Topics

### Custom Operator Support

If your model uses custom operators:

```python
# Register custom operators before export
import my_custom_ops  # Your custom operator implementation
python executorch_exporter.py model.pth \
  --model-name custom_ops_model \
  --input-shapes 1,256 \
  --backends portable
```

### Batch Processing

```dart
// Process multiple inputs efficiently
Future<List<InferenceResult>> processBatch(List<Uint8List> images) async {
  final results = <InferenceResult>[];

  for (final image in images) {
    final input = preprocessImage(image);
    final result = await model.runInference(inputs: [input]);
    results.add(result);
  }

  return results;
}
```

### Model Versioning

```dart
// Handle model updates gracefully
class VersionedModelManager {
  static const String modelVersion = '1.2.0';

  Future<ExecuTorchModel> loadModel() async {
    final path = 'assets/models/classifier_v${modelVersion}_ios_coreml.pte';
    return await ExecutorchManager.instance.loadModel(path);
  }
}
```

---

## Example Export Scripts

We provide example export scripts to demonstrate common patterns and platform-specific optimizations. You can use these as reference when creating your own export process.

### When to Reference Our Scripts

- **Learning patterns**: See examples of platform-specific optimizations
- **Quick prototyping**: Use our scripts as a starting point for your own export pipeline
- **Common use cases**: Our scripts cover typical mobile ML scenarios
- **Best practices**: See how to structure export code with proper error handling

### Getting the Example Scripts

```bash
# Clone repository to see example scripts
git clone https://github.com/abdelaziz-mahdy/executorch_flutter.git
cd executorch_flutter/python

# Or download individual scripts
curl -O https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter/main/python/executorch_exporter.py
curl -O https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter/main/python/generate_test_models.py
curl -O https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter/main/python/export_examples.py

# Download ImageNet class labels (for classification models)
curl -O https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

### Example Script Features

Our example scripts demonstrate:
- **Auto-backend detection**: Automatically select optimal backends for the platform
- **Platform targeting**: iOS (CoreML, MPS) vs Android (XNNPACK, Vulkan) optimizations
- **Error handling**: Robust export process with meaningful error messages
- **Metadata generation**: Export summaries and model information

### Example Usage

You can use our example scripts as a starting point:

```bash
# Example: Export a model with platform-specific optimizations
python executorch_exporter.py your_model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends coreml xnnpack  # iOS and Android optimized

# Example: Generate test models like our example app uses
python generate_test_models.py
```

### Key Export Principles for Flutter

When creating your own export process, keep these Flutter-specific considerations in mind:

1. **Platform Optimization**:
   - **iOS**: Prefer CoreML and MPS backends for best performance
   - **Android**: Use XNNPACK for CPU optimization
   - **Cross-platform**: Use portable backend as fallback

2. **Asset Management**:
   - Export multiple platform-specific versions of your model
   - Use descriptive filenames (e.g., `model_ios_coreml.pte`, `model_android_xnnpack.pte`)
   - Keep models under 100MB for optimal app store distribution

3. **Input Validation**:
   - Match exact input shapes your Flutter app will provide
   - Test with representative data before integrating

---

## References

- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)
- [PyTorch ExecuTorch Export Guide](https://docs.pytorch.org/executorch/stable/using-executorch-export.html) ⭐
- [ExecuTorch Backend Documentation](https://docs.pytorch.org/executorch/stable/backends.html)
- [Flutter Asset Management](https://docs.flutter.dev/ui/assets/assets-and-images)

## Support

For questions and issues:
- Check the [main README](README.md) for package usage
- Review [example app](example/) for implementation patterns
- Open issues on the GitHub repository
- Consult ExecuTorch documentation for model export specifics