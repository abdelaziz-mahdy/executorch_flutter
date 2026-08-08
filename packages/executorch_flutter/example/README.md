# ExecuTorch Flutter Example App

A comprehensive demonstration of the `executorch_flutter` plugin featuring multiple model types, live camera inference, and configurable preprocessing.

---

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
- [Preprocessing Options](#preprocessing-options)
- [Project Structure](#project-structure)
- [Exporting Models](#exporting-your-own-models)
- [Testing](#testing)
- [Platform Requirements](#platform-requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Features

- **Unified Model Playground** - Single interface for multiple model types
- **Live Camera Inference** - Real-time object detection and classification
- **Static Image Processing** - Upload and analyze images from gallery
- **Configurable Settings** - Adjust thresholds, preprocessing methods, and more
- **Performance Monitoring** - Real-time FPS and inference time tracking

---

## Supported Models

### Image Classification

| Model | Performance | Classes |
|-------|-------------|---------|
| **MobileNet V3 Small** | ~10-15ms | 1000 ImageNet |

### Object Detection

| Model | Performance | Classes |
|-------|-------------|---------|
| **YOLO11 Nano** | ~20-50ms | 80 COCO |
| **YOLOv8 Nano** | ~20-50ms | 80 COCO |
| **YOLOv5 Nano** | ~20-50ms | 80 COCO |

---

## Quick Start

### 1. Install Dependencies

```bash
flutter pub get
```

**Models are automatically downloaded from GitHub** on first use.

**Copying this example out of the repo?** `pubspec.yaml` pins
`resolution: workspace`, which only resolves inside this repo's pub
workspace — delete that line first. The workspace root also supplies this
example's native build config, which a standalone copy loses silently (no
LLM runner, no MLX backend); add it back here as a
`hooks: user_defines: executorch_dart:` block with `llm: true` and
`backends: [xnnpack, mlx]`.

### 2. Run the App

```bash
flutter run -d macos      # macOS
flutter run -d <device>   # iOS/Android
flutter run -d windows    # Windows
flutter run -d linux      # Linux
flutter run -d chrome     # Web
```

### 3. Use the App

1. Select a model from the dropdown (e.g., "YOLO11 Nano")
2. Pick an image OR enable camera mode
3. View results with bounding boxes (YOLO) or class predictions (MobileNet)

---

## Preprocessing Options

The app demonstrates **three preprocessing approaches**:

| Strategy | Performance | Platforms | Dependencies | Best For |
|----------|-------------|-----------|--------------|----------|
| **GPU Shader** | Fast | All | None | Real-time, web |
| **OpenCV** | Very fast | Native only | opencv_dart | High performance |
| **CPU (image lib)** | Slower | All | image | Debugging |

**To switch:** Settings → Preprocessing Provider

### GPU Preprocessing (Recommended)

Hardware-accelerated using Flutter Fragment Shaders:
- Zero dependencies
- Works on all platforms including web
- Customizable GLSL shaders

**[Complete GPU Preprocessing Tutorial](GPU_PREPROCESSING.md)**

**Reference implementations:** [lib/processors/shaders/](lib/processors/shaders/)

---

## Project Structure

```
example/
├── lib/
│   ├── main.dart                    # App entry point
│   ├── models/                      # Model definitions
│   │   ├── model_definition.dart    # Abstract base class
│   │   ├── model_registry.dart      # Dynamic model loading
│   │   ├── yolo_model_definition.dart
│   │   └── mobilenet_model_definition.dart
│   ├── processors/                  # Preprocessing strategies
│   │   ├── shaders/                 # GPU shader preprocessing
│   │   ├── opencv/                  # OpenCV implementations
│   │   └── yolo_processor.dart      # CPU implementations
│   ├── renderers/screens/           # Result visualization
│   │   ├── yolo_renderer.dart       # Bounding box overlay
│   │   └── classification_renderer.dart
│   ├── services/
│   │   ├── model_controller.dart    # Model state management
│   │   └── model_index_service.dart # Fetches models from GitHub
│   └── screens/
│       └── unified_model_playground.dart
├── shaders/                         # GLSL fragment shaders
│   ├── yolo_preprocess.frag
│   └── mobilenet_preprocess.frag
└── assets/images/                   # Test images
```

**Models are hosted separately:** [executorch_flutter_models](https://github.com/abdelaziz-mahdy/executorch_flutter_models)

---

## Exporting Your Own Models

Models are downloaded automatically. To export manually:

### Export All Models

```bash
cd ../models/python
python3 main.py
```

### Export Specific Models

```bash
python3 main.py export --mobilenet
python3 main.py export --yolo yolo11n
python3 main.py export --all --backends xnnpack coreml
```

**[Model Export Tools Documentation](../models/python/README.md)**

---

## Testing

### Integration Tests

```bash
cd example

# All platforms
./scripts/run_integration_tests.sh

# Specific platform
./scripts/run_integration_tests.sh macos
./scripts/run_integration_tests.sh ios
./scripts/run_integration_tests.sh android
```

---

## Platform Requirements

| Platform | Min Version | Architectures |
|----------|-------------|---------------|
| **Android** | API 23 | arm64-v8a, armeabi-v7a, x86_64, x86 |
| **iOS** | 13.0+ | arm64, arm64-simulator, x86_64-simulator |
| **macOS** | 11.0+ | arm64, x86_64 |
| **Windows** | 10 | x64 |
| **Linux** | Ubuntu 20.04+ | x64 |
| **Web** | Modern browsers | WebAssembly |

---

## Troubleshooting

<details>
<summary><b>Models not loading</b></summary>

- Models are downloaded from GitHub on first use
- Check internet connection
- Clear app cache and restart

Verify manually:
```bash
curl https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/index.json
```
</details>

<details>
<summary><b>Camera not working</b></summary>

Check camera permissions in device settings.
</details>

<details>
<summary><b>Low frame rates</b></summary>

1. Switch to **GPU preprocessing** in settings
2. Use smaller model (e.g., YOLO11n)
3. Reduce camera resolution
4. Run in release mode: `flutter run --release`
</details>

<details>
<summary><b>iOS build errors</b></summary>

Ensure iOS deployment target is set to 13.0+ in Xcode.
</details>

---

## Contributing

When adding new models:

1. Create model definition in `lib/models/`
2. Implement preprocessors in `lib/processors/`
3. Add result renderer in `lib/renderers/screens/`
4. Register in `lib/models/model_registry.dart`
5. Add export function in `../models/python/main.py`

**[Detailed Architecture Guide](CLAUDE.md)**

---

## Learn More

- **[GPU Preprocessing Tutorial](GPU_PREPROCESSING.md)**
- **[Main Package README](../README.md)**
- **[Architecture Guide](CLAUDE.md)**
- **[Model Export Tools](../models/python/README.md)**
- **[Backend Selection Guide](../models/python/BACKENDS.md)**

---

## License

MIT License - see [LICENSE](../LICENSE).

---

Built with love to demonstrate on-device ML with ExecuTorch and Flutter.
