# ExecuTorch Flutter Example App

A comprehensive demonstration of the `executorch_flutter` plugin featuring:

- 🎯 **Unified Model Playground** - Single interface for multiple model types
- 📸 **Live Camera Inference** - Real-time object detection and classification
- 🖼️ **Static Image Processing** - Upload and analyze images from gallery
- ⚙️ **Configurable Settings** - Adjust thresholds, preprocessing methods, and more
- 📊 **Performance Monitoring** - Real-time FPS and inference time tracking

## Supported Models

### Image Classification
- **MobileNet V3 Small** - Efficient ImageNet classification
- Performance: ~10-15ms inference time
- 1000 ImageNet classes

### Object Detection
- **YOLO11 Nano** - Latest YOLO architecture
- **YOLOv8 Nano** - Fast and accurate detection
- **YOLOv5 Nano** - Lightweight object detection
- Performance: ~20-50ms inference time
- 80 COCO classes

## Quick Start

### 1. Install Dependencies

```bash
# Install Flutter dependencies
flutter pub get

# Setup models (downloads and converts PyTorch models to .pte format)
cd python
python3 setup_models.py
cd ..
```

### 2. Run the App

```bash
# macOS
flutter run -d macos

# iOS (requires physical device, simulator not supported)
flutter run -d <device-id>

# Android
flutter run -d <device-id>
```

### 3. Choose a Model

1. Select a model from the dropdown (e.g., "YOLO11 Nano" or "MobileNet V3")
2. Pick an image from gallery OR enable camera mode
3. View results with bounding boxes (YOLO) or class predictions (MobileNet)

## Preprocessing Options

The example app demonstrates **three preprocessing approaches**:

### 1. GPU Preprocessing (Recommended) ⭐

**Hardware-accelerated preprocessing using Flutter Fragment Shaders:**

- ⚡ **Performance comparable to OpenCV** (very close on macOS)
- 📦 **Zero dependencies** - uses native Flutter APIs
- 🌍 **All platforms** - mobile and desktop
- 🎨 **Customizable** - write your own GLSL shaders
- 🎯 **Great for real-time** - camera inference and high frame rates

**📖 [Complete GPU Preprocessing Tutorial](GPU_PREPROCESSING.md)** - Learn how to implement GPU-accelerated preprocessing with step-by-step guide and shader examples.

**Reference implementations:**
- **[lib/processors/shaders/](lib/processors/shaders/)** - GPU preprocessor implementations with README
- **[shaders/](../shaders/)** - GLSL fragment shaders (yolo_preprocess.frag, mobilenet_preprocess.frag)

### 2. OpenCV Preprocessing

**High-performance C++ library:**

- ⚡ **High performance** (very close to GPU on macOS)
- 🌍 **Cross-platform** - works on mobile and desktop
- 📦 Requires `opencv_dart` package
- 🔧 Advanced image processing and computer vision capabilities

### 3. CPU Preprocessing (image library)

**Pure Dart implementation:**

- ⏱️ **Slower than GPU/OpenCV**, suitable for non-realtime use
- 🌍 **All platforms**
- 📦 Uses `image` package
- 🐛 **Best for debugging** - easier to inspect steps

**To switch preprocessing methods:**
1. Open Settings in the app
2. Select "Preprocessing Provider"
3. Choose: GPU Shader, OpenCV, or Image Library

## Project Structure

```
example/
├── lib/
│   ├── main.dart                          # App entry point
│   ├── models/                            # Model definitions
│   │   ├── model_definition.dart          # Abstract base class
│   │   ├── model_registry.dart            # Available models list
│   │   ├── yolo_model_definition.dart     # YOLO implementation
│   │   └── mobilenet_model_definition.dart # MobileNet implementation
│   ├── processors/                        # Preprocessing strategies
│   │   ├── shaders/                       # GPU shader preprocessing
│   │   │   ├── README.md                  # Shader preprocessing guide
│   │   │   ├── gpu_yolo_preprocessor.dart # GPU YOLO preprocessing
│   │   │   └── gpu_mobilenet_preprocessor.dart # GPU MobileNet preprocessing
│   │   ├── yolo_processor.dart            # CPU YOLO preprocessing
│   │   └── opencv/                        # OpenCV implementations
│   ├── renderers/                         # Result visualization
│   │   └── screens/
│   │       ├── yolo_renderer.dart         # Bounding box overlay
│   │       └── classification_renderer.dart # Class predictions
│   ├── services/
│   │   └── model_controller.dart          # Model state management
│   └── screens/
│       └── unified_model_playground.dart  # Main playground screen
├── shaders/                               # GPU shaders (GLSL)
│   ├── yolo_preprocess.frag              # YOLO letterbox resize
│   └── mobilenet_preprocess.frag         # MobileNet center crop
├── assets/
│   ├── models/                           # .pte model files (generated)
│   ├── imagenet_classes.txt              # MobileNet labels
│   └── coco_labels.txt                   # YOLO labels
└── python/                               # Model export scripts
    ├── setup_models.py                   # One-command setup
    ├── export_mobilenet.py               # MobileNet export
    └── export_yolo.py                    # YOLO export
```

## Exporting Your Own Models

The example includes Python scripts for converting PyTorch models to ExecuTorch format:

### One-Command Setup (Recommended)

```bash
cd python
python3 setup_models.py
```

This will:
- ✅ Install all dependencies (torch, ultralytics, executorch)
- ✅ Export MobileNet V3 Small
- ✅ Export YOLO11n, YOLOv8n, YOLOv5n
- ✅ Generate label files
- ✅ Verify all models

### Manual Export

#### MobileNet
```bash
cd python
python3 export_mobilenet.py
```

#### YOLO
```bash
cd python
python3 export_yolo.py --model yolo11n  # or yolov8n, yolov5n
```

**📖 See [Python Export Scripts](python/README.md)** for detailed export instructions and custom model conversion.

## Testing

Run integration tests on all platforms:

```bash
cd example

# Test all platforms
./scripts/run_integration_tests.sh

# Test specific platform
./scripts/run_integration_tests.sh macos
./scripts/run_integration_tests.sh ios
./scripts/run_integration_tests.sh android
```

## Learn More

- **[GPU Preprocessing Tutorial](GPU_PREPROCESSING.md)** - Implement GPU-accelerated preprocessing with Fragment Shaders
- **[Main Package README](../README.md)** - Core API documentation and usage
- **[Example App Architecture](CLAUDE.md)** - Detailed architecture guide for developers

## Requirements

### Android
- Minimum SDK: API 23 (Android 6.0)
- Architecture: arm64-v8a

### iOS
- Minimum Version: iOS 17.0+
- Architecture: arm64 (physical devices only)
- ⚠️ **Simulator NOT supported**

### macOS
- Minimum Version: macOS 12.0+ (Monterey)
- Architecture: arm64 (Apple Silicon only)
- ⚠️ **Intel Macs NOT supported**
- ⚠️ **Release builds NOT working** (debug builds work fine)

## Troubleshooting

### Models Not Loading

**Issue**: "Failed to load model" error

**Solution**:
```bash
# Re-export models
cd python
python3 setup_models.py

# Verify files exist
ls -lh ../assets/models/
```

### Camera Not Working

**Issue**: Black screen or no camera feed

**Solution**: Check camera permissions in device settings

### Low Frame Rates

**Issue**: FPS below 30

**Solutions**:
1. Switch to **GPU preprocessing** in settings
2. Use smaller model (e.g., YOLO11n instead of YOLO11m)
3. Reduce camera resolution
4. Run in release mode: `flutter run --release`

### iOS Build Errors

**Issue**: "requires minimum platform version 17.0"

**Solution**: Update deployment target in Xcode (see main README)

## Contributing

Contributions are welcome! When adding new models:

1. Create model definition in `lib/models/`
2. Implement preprocessors in `lib/processors/`
3. Add result renderer in `lib/renderers/screens/`
4. Register in `lib/models/model_registry.dart`
5. Add export script in `python/`

See **[Example App Architecture Guide](CLAUDE.md)** for detailed instructions.

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

Built with ❤️ to demonstrate the power of on-device ML with ExecuTorch and Flutter.
