# ExecuTorch Model Export Scripts

This directory contains Python scripts for exporting PyTorch models to ExecuTorch format (.pte files) optimized for different backends and platforms.

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Virtual Environment**: Recommended (conda or venv)
- **Platform**: macOS ARM64 (for CoreML), Linux x86_64, Windows

## Quick Start

### 1. Install Dependencies

```bash
cd python

# Recommended: Create virtual environment
python -m venv executorch_env
source executorch_env/bin/activate  # On Windows: executorch_env\Scripts\activate

# Install ExecuTorch and dependencies
pip install -r requirements.txt
```

### 2. Generate Test Models for Flutter App

```bash
# Generate all test models (simple demo + classification)
python generate_test_models.py

# OR use the generic exporter for custom models
python executorch_exporter.py your_model.pth --model-name my_model --input-shapes 1,3,224,224
```

### 3. Use in Flutter App

The exported `.pte` files are automatically saved to `../example/assets/models/`. Add them to your Flutter app's `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/
```

**Note**: The `.pte` files are not committed to git. You need to generate them locally.

## Available Scripts

### `generate_test_models.py` 🚀

**Recommended**: One-click generation of test models for Flutter plugin validation.

```bash
python generate_test_models.py
```

**Generates:**
- Simple demo model (portable backend)
- MobileNetV3 for iOS (CoreML, MPS backends)
- MobileNetV3 for Android (XNNPACK backend)

### `executorch_exporter.py` 🔧

**Generic exporter** for any PyTorch model with automatic backend detection.

```bash
# Export custom model
python executorch_exporter.py your_model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends xnnpack coreml

# Auto-select backends for platform
python executorch_exporter.py model.pth \
  --model-name mobile_model \
  --input-shapes 1,3,224,224 \
  --target-platform android
```

**Features:**
- Supports any PyTorch model (.pth, .pt, .torchscript)
- Auto-detects available backends
- Platform-specific optimizations
- JSON export summaries
- Progress tracking and error handling

### `export_examples.py` 📚

**Examples** showing how to use the generic exporter with different model types.

```bash
python export_examples.py
```

## Flutter Integration

### Loading Models in Flutter

```dart
// Load exported model
final model = await ExecutorchManager.instance.loadModel(
  'assets/models/mobilenet_v3_small_ios_coreml.pte'
);

// Check model metadata
print('Backend: ${model.metadata.properties['backend']}');
print('Input shape: ${model.metadata.inputSpecs.first.shape}');

// Run inference
final result = await model.runInference(
  inputs: [inputTensor],
  timeoutMs: 5000,
);
```

### Camera Frame Preprocessing

```dart
// Preprocess camera frame for MobileNetV3 inference
TensorDataWrapper preprocessCameraFrame(CameraImage cameraImage) {
  // Convert camera format (YUV420/NV21) to RGB
  final rgbBytes = convertYUV420ToRGB(cameraImage);

  // Resize to model input size (224x224)
  final resizedBytes = resizeImage(rgbBytes, 224, 224);

  // Apply ImageNet normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  final floats = Float32List(1 * 3 * 224 * 224);
  for (int i = 0; i < floats.length; i++) {
    final channel = i % 3;
    final pixelValue = resizedBytes[i] / 255.0;
    floats[i] = (pixelValue - mean[channel]) / std[channel];
  }

  return TensorDataWrapper(
    shape: [1, 3, 224, 224],
    dataType: TensorType.float32,
    data: floats.buffer.asUint8List(),
    name: 'input',
  );
}

// Platform-specific model selection
String getOptimalModelPath() {
  if (Platform.isIOS) {
    return 'assets/models/mobilenet_v3_small_ios_coreml.pte';
  } else {
    return 'assets/models/mobilenet_v3_small_android_cpu_xnnpack.pte';
  }
}
```

## Available Test Models

After running `python generate_test_models.py`, you'll have these models in `../example/assets/models/`:

| Model | Backend | Platform | Size | Use Case |
|-------|---------|----------|------|----------|
| `simple_demo_portable.pte` | Portable | Any | ~1.6KB | Basic testing |
| `mobilenet_v3_small_ios_coreml.pte` | CoreML | iOS | ~5.3MB | iOS Neural Engine |
| `mobilenet_v3_small_ios_mps.pte` | MPS | iOS | ~9.8MB | iOS Metal GPU |
| `mobilenet_v3_small_android_cpu_xnnpack.pte` | XNNPACK | Android | ~9.8MB | Android CPU |

## Backend Availability

The exporter automatically detects available backends:

### ✅ Always Available
- **Portable**: Universal compatibility, basic performance
- **XNNPACK**: CPU optimization for mobile devices

### 🍎 macOS/iOS Only
- **CoreML**: Apple Neural Engine optimization
- **MPS**: Metal Performance Shaders GPU acceleration

### 🤖 Platform Specific
- **Vulkan**: Linux/Android/Windows GPU acceleration
- **QNN**: Qualcomm Snapdragon optimization (requires SDK)
- **ARM**: ARM Ethos-U NPU (embedded devices)

## Troubleshooting

### Model Generation Fails
1. Ensure Python 3.10-3.12 and virtual environment
2. Check ExecuTorch installation: `pip list | grep executorch`
3. Try generating individual models with the generic exporter

### Backend Not Available
- Check platform compatibility in the table above
- Some backends require additional SDKs or hardware support
- Use `executorch_exporter.py --help` to see available backends

### Flutter Integration
1. Ensure models are in `assets/models/` directory
2. Add to `pubspec.yaml` under `flutter: assets:`
3. Use exact model filenames in Flutter code
4. Models are not committed to git - generate locally

## References

- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)
- [Official ExecuTorch Examples](https://github.com/meta-pytorch/executorch-examples)
- [Backend Optimization Guide](https://docs.pytorch.org/executorch/stable/backends.html)
- [Flutter Plugin API Documentation](../README.md)