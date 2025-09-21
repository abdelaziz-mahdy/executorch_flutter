# ExecuTorch Test Models

This directory contains ExecuTorch model files (.pte) for testing the Flutter plugin. **These models are not committed to git** and must be generated locally.

## 🚀 Quick Start

### Generate Test Models

From the project root, run:

```bash
cd python
pip install -r requirements.txt
python generate_test_models.py
```

This will create the following models in this directory:

| Model File | Backend | Platform | Use Case |
|-----------|---------|----------|----------|
| `simple_demo_portable.pte` | Portable | Any | Basic API testing |
| `mobilenet_v3_small_ios_coreml.pte` | CoreML | iOS | Neural Engine optimization |
| `mobilenet_v3_small_ios_mps.pte` | MPS | iOS | Metal GPU acceleration |
| `mobilenet_v3_small_android_cpu_xnnpack.pte` | XNNPACK | Android | CPU optimization |

Additional files automatically included:
- `imagenet_classes.txt` - ImageNet class labels for meaningful classification results

## 📱 Usage in Example App

The example app uses camera feed for real-time inference. Models are loaded from local assets:

```dart
// Load platform-specific model
String getModelPath() {
  if (Platform.isIOS) {
    return 'assets/models/mobilenet_v3_small_ios_coreml.pte';
  } else {
    return 'assets/models/mobilenet_v3_small_android_cpu_xnnpack.pte';
  }
}

// Load model in Flutter
final model = await ExecutorchManager.instance.loadModel(getModelPath());

// Process camera frame for inference
Future<void> processCameraFrame(CameraImage cameraImage) async {
  final inputTensor = preprocessCameraImage(cameraImage);

  final result = await model.runInference(
    inputs: [inputTensor],
    timeoutMs: 100, // Fast inference for camera
  );

  if (result.isSuccess) {
    // Update UI with classification results
    updateUI(result.outputs);
  }
}
```

## 🔧 Custom Models

To export your own PyTorch models:

```bash
cd python
python executorch_exporter.py your_model.pth \
  --model-name my_model \
  --input-shapes 1,3,224,224 \
  --backends xnnpack coreml
```

## ❓ Troubleshooting

### No Models Generated?
1. Check Python version (3.10-3.12 required)
2. Ensure virtual environment: `python -m venv executorch_env && source executorch_env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

### Models Not Loading in Flutter?
1. Ensure models are in this directory (`example/assets/models/`)
2. Check `pubspec.yaml` includes `assets/models/`
3. Use exact filenames in Flutter code
4. Models must be generated locally (not downloaded)

### Camera Inference Issues?
- Use appropriate timeouts (100ms for real-time)
- Preprocess camera frames to match model input (224x224 for MobileNetV3)
- Handle inference on background isolate for performance
- Consider reducing inference frequency for better UX

### Backend Issues?
- **CoreML/MPS**: Only available on iOS devices
- **XNNPACK**: Universal fallback, works on all platforms
- **Portable**: Always available, basic performance

## 🏗️ Example App Architecture

The example app demonstrates:
- **Real-time camera inference** with ExecuTorch models
- **Platform-specific model loading** (CoreML for iOS, XNNPACK for Android)
- **Background processing** to avoid blocking UI
- **Performance monitoring** with inference timing
- **Model switching** between different backends

## 📖 Documentation

- [Python Export Scripts](../../python/README.md)
- [Flutter Plugin API](../../README.md)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)

---

**Note**: Model files (.pte) are ignored by git. Each developer must generate them locally using the Python scripts.