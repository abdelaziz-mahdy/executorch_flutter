# GPU Shader Preprocessing

This directory contains GPU-accelerated image preprocessing implementations using Flutter Fragment Shaders.

## Overview

GPU shader preprocessing offers **performance comparable to OpenCV** (very close on macOS) while using native Flutter APIs with zero external dependencies. These preprocessors leverage hardware acceleration on all platforms (mobile + desktop) for high-performance real-time inference.

## Files

### Preprocessors

- **[gpu_yolo_preprocessor.dart](gpu_yolo_preprocessor.dart)** - YOLO preprocessing with letterbox resize
  - Maintains aspect ratio with gray padding (114, 114, 114)
  - Target size: 640x640
  - Normalization: [0, 1] range (divide by 255)
  - Shader: **[../../../shaders/yolo_preprocess.frag](../../../shaders/yolo_preprocess.frag)**

- **[gpu_mobilenet_preprocessor.dart](gpu_mobilenet_preprocessor.dart)** - MobileNet/ImageNet preprocessing
  - Center crop with shortest side = 256
  - Target size: 224x224
  - Normalization: ImageNet mean/std
  - Shader: **[../../../shaders/mobilenet_preprocess.frag](../../../shaders/mobilenet_preprocess.frag)**

### Shaders (GLSL)

Located in `shaders/` directory (project root):

- **[yolo_preprocess.frag](../../../shaders/yolo_preprocess.frag)** - Letterbox resize with padding on GPU
- **[mobilenet_preprocess.frag](../../../shaders/mobilenet_preprocess.frag)** - Center crop with ImageNet normalization on GPU

## Architecture

Each GPU preprocessor follows this pipeline:

```
Input Bytes (JPEG/PNG)
    ↓
Native Decoder (ui.decodeImageFromList)
    ↓ [hardware accelerated]
ui.Image
    ↓
Fragment Shader Processing
    ↓ [GPU: resize, crop, normalize]
ui.Image (processed)
    ↓
Tensor Conversion (optimized single-loop)
    ↓ [RGBA → NCHW float32]
TensorData
```

## Key Features

### 1. Native Image Decoding
Uses Flutter's hardware-accelerated `ui.decodeImageFromList()` instead of software-based image libraries.

### 2. GPU Processing
Fragment shaders perform:
- Resize operations (letterbox or center crop)
- Padding (letterbox borders)
- Normalization (per-pixel or ImageNet)

### 3. Optimized Tensor Conversion
Single-loop conversion for better cache locality:
```dart
for (int i = 0; i < totalPixels; i++) {
  final pixelIndex = i * 4;
  floats[i] = pixels[pixelIndex] * scale;                     // R
  floats[i + totalPixels] = pixels[pixelIndex + 1] * scale;   // G
  floats[i + totalPixels * 2] = pixels[pixelIndex + 2] * scale; // B
}
```

## Usage

### In Model Definition

```dart
@override
InputProcessor<ModelInput> createInputProcessor(ModelSettings settings) {
  final yoloSettings = settings as YoloModelSettings;

  switch (yoloSettings.preprocessingProvider) {
    case PreprocessingProvider.gpu:
      return YoloInputProcessor(
        config: YoloPreprocessConfig(
          targetWidth: inputSize,
          targetHeight: inputSize,
        ),
        preprocessingProvider: PreprocessingProvider.gpu,
      );
    // ... other cases
  }
}
```

The input processor will automatically route to the appropriate GPU preprocessor based on settings.

## Performance

- **Comparable to OpenCV** (very close on macOS)
- **Hardware accelerated** on all platforms
- **Great for real-time** camera inference and high frame rates
- **Zero dependencies** - no external packages required

## When to Use

✅ **Use GPU shader preprocessing when:**
- Real-time camera inference
- High frame rates needed
- Want OpenCV-like performance without external dependencies
- Targeting mobile and desktop platforms

❌ **Use CPU preprocessing when:**
- Low power consumption is critical
- Simple preprocessing needs
- Debugging (easier to inspect steps)

## Learn More

📖 **[Complete Tutorial](../../../GPU_PREPROCESSING.md)** - Step-by-step guide for implementing custom GPU preprocessors

## Implementation Details

### Shader Initialization

Shaders are loaded once on first use and cached:

```dart
Future<void> _initializeShader() async {
  if (_isInitialized) return;  // Prevent re-initialization

  _program = await ui.FragmentProgram.fromAsset('shaders/yolo_preprocess.frag');
  _isInitialized = true;
}
```

### Resource Management

Always dispose `ui.Image` objects to prevent memory leaks:

```dart
final image = await _decodeImageNative(input);
final processedImage = await _processOnGpu(image);
final tensorData = await _imageToTensor(processedImage);

// Cleanup
image.dispose();
processedImage.dispose();
```

### Error Handling

All preprocessors wrap operations in try-catch blocks and throw `PreprocessingException` on failure:

```dart
try {
  // Preprocessing steps
} catch (e) {
  if (e is ProcessorException) rethrow;
  throw PreprocessingException('GPU preprocessing failed: $e', e);
}
```

## Contributing

When adding new GPU preprocessors:

1. Create GLSL shader in `assets/shaders/`
2. Register shader in `pubspec.yaml`
3. Implement preprocessor class extending `ExecuTorchPreprocessor<Uint8List>`
4. Follow the existing architecture pattern
5. Add documentation and usage examples

---

Built with ❤️ for high-performance on-device ML inference.
