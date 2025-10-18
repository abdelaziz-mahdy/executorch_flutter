# ExecuTorch Flutter - Example App Architecture

**Version**: 1.0
**Last Updated**: 2025-10-11

This document describes the architecture of the example app and provides step-by-step guides for adding new model types.

## Architecture Overview

The example app uses a **Strategy Pattern** with **Model Definitions** to support multiple model types (YOLO, MobileNet, etc.) in a unified playground. Each model is completely self-contained and knows how to process its inputs/outputs and render its results.

### Key Design Principles

1. **Model as Strategy**: Each model type is a complete strategy that encapsulates all its behavior
2. **Unified Playground**: Single screen supports all model types through polymorphism
3. **Settings Per Model**: Each model has its own settings class with type-safe defaults
4. **Processor Pattern**: Input/output processing is separate from model definition
5. **Reactive UI**: Settings changes immediately recreate processors with new configuration

---

## Core Architecture Components

```
┌────────────────────────────────────────────────────────┐
│                  UnifiedModelPlayground                │
│            (Single screen for all models)              │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                   ModelController                      │
│         (Owns model lifecycle & state)                 │
│  - execuTorchModel (loaded model)                     │
│  - definition (which model type)                       │
│  - settings (model-specific config)                    │
│  - processors (input/output strategies)                │
│  - camera (optional live stream)                       │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                  ModelDefinition<TInput, TResult>      │
│         (Abstract base for all model types)            │
│                                                        │
│  Methods:                                              │
│  - createInputProcessor(settings)                      │
│  - createOutputProcessor(settings)                     │
│  - buildInputWidget(...)                               │
│  - buildResultRenderer(...)                            │
│  - buildSettingsWidget(...)                            │
│  - buildResultsDetailsSection(...)                     │
└────────────────────────┬───────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌────────────────────┐         ┌────────────────────┐
│ MobileNetModelDef  │         │  YoloModelDef      │
│                    │         │                    │
│ Uses:              │         │ Uses:              │
│ - ImageFileInput   │         │ - ImageFileInput   │
│ - LiveCameraInput  │         │ - LiveCameraInput  │
│                    │         │                    │
│ Returns:           │         │ Returns:           │
│ - Classification   │         │ - DetectionResult  │
│   Result           │         │                    │
└────────────────────┘         └────────────────────┘
```

---

## File Structure

```
example/
├── lib/
│   ├── main.dart                              # App entry, model registry loader
│   │
│   ├── models/                                # Model definitions
│   │   ├── model_definition.dart              # Abstract base class
│   │   ├── model_registry.dart                # List of all available models
│   │   ├── model_input.dart                   # Input types (ImageFile, LiveCamera)
│   │   ├── model_settings.dart                # Base settings class
│   │   ├── mobilenet_model_definition.dart    # MobileNet implementation
│   │   ├── yolo_model_definition.dart         # YOLO implementation
│   │   ├── classification_model_settings.dart # Settings for classification
│   │   └── yolo_model_settings.dart           # Settings for YOLO
│   │
│   ├── processors/                            # Input/Output processing strategies
│   │   ├── base_processor.dart                # Abstract InputProcessor/OutputProcessor
│   │   ├── mobilenet_input_processor.dart     # MobileNet preprocessing
│   │   ├── mobilenet_output_processor.dart    # MobileNet postprocessing
│   │   ├── yolo_input_processor.dart          # YOLO preprocessing
│   │   ├── yolo_output_processor.dart         # YOLO postprocessing
│   │   ├── image_processor.dart               # Common image processing
│   │   ├── opencv/                            # OpenCV-based preprocessors
│   │   │   ├── opencv_imagenet_preprocessor.dart
│   │   │   └── opencv_yolo_preprocessor.dart
│   │   └── camera_image_converter.dart        # Camera frame conversion
│   │
│   ├── renderers/                             # Result visualization
│   │   └── screens/
│   │       ├── classification_renderer.dart   # Classification results
│   │       └── yolo_renderer.dart             # Detection boxes overlay
│   │
│   ├── services/                              # Business logic
│   │   └── model_controller.dart              # Central model state manager
│   │
│   ├── screens/                               # UI screens
│   │   └── unified_model_playground.dart      # Main playground screen
│   │
│   ├── controllers/                           # Camera management
│   │   ├── camera_controller.dart             # Abstract camera interface
│   │   ├── platform_camera_controller.dart    # Flutter camera plugin
│   │   └── opencv_camera_controller.dart      # OpenCV camera (desktop)
│   │
│   └── widgets/                               # Reusable UI components
│       ├── image_input_widget.dart            # Image picker + camera toggle
│       └── performance_monitor.dart           # FPS/timing display
│
├── assets/
│   ├── models/                                # .pte model files
│   │   ├── mobilenet_v3_small_xnnpack.pte
│   │   ├── yolo11n_xnnpack.pte
│   │   ├── yolov8n_xnnpack.pte
│   │   └── yolov5n_xnnpack.pte
│   ├── imagenet_classes.txt                   # MobileNet labels
│   └── coco_labels.txt                        # YOLO labels
│
├── python/                                    # Model export scripts
│   ├── setup_models.py                        # One-command model setup
│   ├── export_mobilenet.py                    # Export MobileNet to .pte
│   ├── export_yolo.py                         # Export YOLO to .pte
│   └── requirements.txt                       # Python dependencies
│
└── scripts/
    └── run_integration_tests.sh               # Multi-platform testing
```

---

## How It Works: Data Flow

### 1. App Startup
```dart
main.dart
  └─> ModelRegistry.loadAll()
       └─> Returns list of ModelDefinitions
            └─> UnifiedModelPlayground displays model selector
```

### 2. Model Selection
```dart
User selects "YOLO11 Nano"
  └─> UnifiedModelPlayground._selectModel(yoloDefinition)
       └─> Load model bytes from assets
       └─> ExecuTorchModel.load(bytes)
       └─> ModelController.create(definition, model, settings)
            └─> Controller creates processors
            └─> UI renders model-specific input widget
```

### 3. Running Inference (Static Image)
```dart
User selects image from gallery
  └─> buildInputWidget() receives ImageFileInput
       └─> ModelController.processInput(input)
            └─> 1. inputProcessor.process(input) → List<TensorData>
            └─> 2. execuTorchModel.forward(tensors) → List<TensorData>
            └─> 3. outputProcessor.process(outputs) → TResult
            └─> 4. buildResultRenderer(input, result) → Widget
```

### 4. Running Inference (Live Camera)
```dart
User toggles camera mode
  └─> ModelController.enableCameraMode()
       └─> Create CameraController (Platform or OpenCV)
       └─> Start streaming frames
            └─> For each frame:
                 └─> Convert to LiveCameraInput
                 └─> Same as static: preprocess → forward → postprocess
                 └─> Update UI with result
```

### 5. Changing Settings
```dart
User adjusts confidence threshold
  └─> buildSettingsWidget() → onSettingsChanged(newSettings)
       └─> ModelController.updateSettings(newSettings)
            └─> Recreate processors with new settings
            └─> Next inference uses new configuration
```

---

## Adding a New Model Type

Follow these steps to add support for a new model type (e.g., Segmentation, Pose Estimation, Text Generation).

### Step 1: Export Your Model to .pte Format

**Create Python export script** in `example/python/`:

```python
# example/python/export_segmentation.py
import torch
from executorch.exir import to_edge
import torchvision.models.segmentation as models

def export_segmentation_model():
    # Load PyTorch model
    model = models.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.eval()

    # Example input
    example_input = (torch.randn(1, 3, 512, 512),)

    # Export to ExecuTorch
    edge_program = to_edge(torch.export.export(model, example_input))
    executorch_program = edge_program.to_executorch()

    # Save .pte file
    with open("../assets/models/deeplabv3_xnnpack.pte", "wb") as f:
        f.write(executorch_program.buffer)

    print("✅ Segmentation model exported!")

if __name__ == "__main__":
    export_segmentation_model()
```

**Add to setup script** in `example/python/setup_models.py`:

```python
# Add to setup_models.py
from export_segmentation import export_segmentation_model

def setup_all_models():
    # ... existing models ...

    print("📦 Exporting Segmentation model...")
    export_segmentation_model()
```

**Run export**:
```bash
cd example/python
python3 setup_models.py  # Exports all models including new one
```

### Step 2: Create Model Settings Class

**Create** `example/lib/models/segmentation_model_settings.dart`:

```dart
import 'model_settings.dart';

class SegmentationModelSettings extends ModelSettings {
  double maskThreshold;
  bool showOverlay;

  SegmentationModelSettings({
    this.maskThreshold = 0.5,
    this.showOverlay = true,
    super.preprocessingProvider,
    super.cameraProvider,
    super.showPerformanceOverlay,
  });

  @override
  void reset() {
    maskThreshold = 0.5;
    showOverlay = true;
    super.reset();
  }
}
```

### Step 3: Create Result Class

**Create** `example/lib/models/segmentation_result.dart`:

```dart
class SegmentationResult {
  final List<int> maskData;      // Segmentation mask (HxW pixels)
  final int height;
  final int width;
  final List<String> classLabels; // Label for each class

  const SegmentationResult({
    required this.maskData,
    required this.height,
    required this.width,
    required this.classLabels,
  });
}
```

### Step 4: Create Input/Output Processors

**Create** `example/lib/processors/segmentation_input_processor.dart`:

```dart
import 'package:executorch_flutter/executorch_flutter.dart';
import 'base_processor.dart';
import '../models/model_input.dart';

class SegmentationInputProcessor extends InputProcessor<ModelInput> {
  final int targetWidth;
  final int targetHeight;

  const SegmentationInputProcessor({
    required this.targetWidth,
    required this.targetHeight,
  });

  @override
  Future<List<TensorData>> process(ModelInput input) async {
    // Convert image to tensor (512x512, RGB, normalized)
    // ... preprocessing logic ...

    return [tensorData];
  }
}
```

**Create** `example/lib/processors/segmentation_output_processor.dart`:

```dart
import 'package:executorch_flutter/executorch_flutter.dart';
import 'base_processor.dart';
import '../models/segmentation_result.dart';

class SegmentationOutputProcessor extends OutputProcessor<SegmentationResult> {
  final List<String> classLabels;
  final double maskThreshold;

  const SegmentationOutputProcessor({
    required this.classLabels,
    required this.maskThreshold,
  });

  @override
  Future<SegmentationResult> process(List<TensorData> outputs) async {
    // Extract mask from output tensor
    // Apply threshold
    // Return SegmentationResult

    return SegmentationResult(...);
  }
}
```

### Step 5: Create Result Renderer

**Create** `example/lib/renderers/screens/segmentation_renderer.dart`:

```dart
import 'package:flutter/material.dart';
import '../../models/model_input.dart';
import '../../models/segmentation_result.dart';

class SegmentationRenderer extends StatelessWidget {
  final ModelInput input;
  final SegmentationResult? result;

  const SegmentationRenderer({
    super.key,
    required this.input,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    // Render image with segmentation mask overlay
    return Stack(
      children: [
        // Original image
        _buildInputImage(),

        // Segmentation mask overlay
        if (result != null) _buildMaskOverlay(result!),
      ],
    );
  }

  Widget _buildInputImage() { /* ... */ }
  Widget _buildMaskOverlay(SegmentationResult result) { /* ... */ }
}
```

### Step 6: Create Model Definition

**Create** `example/lib/models/segmentation_model_definition.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'segmentation_model_settings.dart';
import 'segmentation_result.dart';
import '../processors/base_processor.dart';
import '../processors/segmentation_input_processor.dart';
import '../processors/segmentation_output_processor.dart';
import '../renderers/screens/segmentation_renderer.dart';
import '../widgets/image_input_widget.dart';

class SegmentationModelDefinition
    extends ModelDefinition<ModelInput, SegmentationResult> {

  const SegmentationModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.assetPath,
    required super.inputSize,
    required this.labelsAssetPath,
  }) : super(icon: Icons.layers);

  final String labelsAssetPath;

  // Cache for labels
  static final Map<String, List<String>> _labelsCache = {};

  Future<List<String>> _loadLabels() async {
    if (_labelsCache.containsKey(labelsAssetPath)) {
      return _labelsCache[labelsAssetPath]!;
    }

    final labelsString = await rootBundle.loadString(labelsAssetPath);
    final labels = labelsString.split('\n')
        .where((line) => line.isNotEmpty)
        .toList();

    _labelsCache[labelsAssetPath] = labels;
    return labels;
  }

  List<String> _loadLabelsSync() {
    if (_labelsCache.containsKey(labelsAssetPath)) {
      return _labelsCache[labelsAssetPath]!;
    }
    throw StateError('Labels not loaded. Call loadLabels() first.');
  }

  Future<List<String>> loadLabels() => _loadLabels();

  @override
  ModelSettings createDefaultSettings() {
    return SegmentationModelSettings();
  }

  @override
  Widget buildInputWidget({
    required BuildContext context,
    required Function(ModelInput) onInputSelected,
    VoidCallback? onCameraModeToggle,
    bool isCameraMode = false,
  }) {
    return ImageInputWidget(
      onImageSelected: (file) => onInputSelected(ImageFileInput(file)),
      onCameraModeToggle: onCameraModeToggle,
      isCameraMode: isCameraMode,
    );
  }

  @override
  InputProcessor<ModelInput> createInputProcessor(ModelSettings settings) {
    return SegmentationInputProcessor(
      targetWidth: inputSize,
      targetHeight: inputSize,
    );
  }

  @override
  OutputProcessor<SegmentationResult> createOutputProcessor(
    ModelSettings settings,
  ) {
    final segSettings = settings as SegmentationModelSettings;
    return SegmentationOutputProcessor(
      classLabels: _loadLabelsSync(),
      maskThreshold: segSettings.maskThreshold,
    );
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required ModelInput input,
    required SegmentationResult? result,
  }) {
    return SegmentationRenderer(input: input, result: result);
  }

  @override
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required SegmentationResult result,
    required double? processingTime,
  }) {
    // Show segmentation statistics (class distribution, etc.)
    return Column(
      children: [
        Text('Mask Size: ${result.width}x${result.height}'),
        // ... more details ...
      ],
    );
  }

  @override
  Widget buildSettingsWidget({
    required BuildContext context,
    required ModelSettings settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    final segSettings = settings as SegmentationModelSettings;

    return Column(
      children: [
        // Mask threshold slider
        ListTile(
          title: Text('Mask Threshold'),
          subtitle: Slider(
            value: segSettings.maskThreshold,
            onChanged: (value) {
              segSettings.maskThreshold = value;
              onSettingsChanged(segSettings);
            },
          ),
        ),

        // Show overlay toggle
        SwitchListTile(
          title: Text('Show Overlay'),
          value: segSettings.showOverlay,
          onChanged: (value) {
            segSettings.showOverlay = value;
            onSettingsChanged(segSettings);
          },
        ),
      ],
    );
  }
}
```

### Step 7: Register Model in Registry

**Edit** `example/lib/models/model_registry.dart`:

```dart
import 'segmentation_model_definition.dart';

class ModelRegistry {
  static Future<List<ModelDefinition>> loadAll() async {
    return [
      // ... existing models ...

      // Segmentation Models
      const SegmentationModelDefinition(
        name: 'deeplabv3_mobilenet',
        displayName: 'DeepLabV3 MobileNet',
        description: 'Semantic segmentation model',
        assetPath: 'assets/models/deeplabv3_xnnpack.pte',
        inputSize: 512,
        labelsAssetPath: 'assets/segmentation_classes.txt',
      ),
    ];
  }
}
```

### Step 8: Add Assets

**Update** `example/pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/
    - assets/segmentation_classes.txt  # Add labels file
```

**Create** `example/assets/segmentation_classes.txt`:
```
background
person
car
...
```

### Step 9: Test

```bash
cd example
flutter run -d macos  # Or ios, android

# In the app:
# 1. Select "DeepLabV3 MobileNet" from dropdown
# 2. Pick an image
# 3. View segmentation mask overlay
# 4. Adjust settings (mask threshold, overlay visibility)
```

---

## Python Model Export Workflow

The `example/python/` directory contains scripts for exporting PyTorch models to ExecuTorch format.

### Directory Structure

```
example/python/
├── setup_models.py                 # One-command setup (all models)
├── export_mobilenet.py             # MobileNet V3 export
├── export_yolo.py                  # YOLO export (v5, v8, v11)
├── requirements.txt                # Python dependencies
└── README.md                       # Python setup instructions
```

### Quick Setup

```bash
cd example/python
python3 setup_models.py
```

This will:
1. Install Python dependencies (`torch`, `ultralytics`, `executorch`)
2. Export all models to `example/assets/models/`
3. Generate label files (`imagenet_classes.txt`, `coco_labels.txt`)
4. Verify all models are ready

### Manual Export

#### Exporting MobileNet

```bash
cd example/python
python3 export_mobilenet.py
```

**What it does**:
- Downloads MobileNet V3 Small (pretrained on ImageNet)
- Exports to ExecuTorch format with XNNPACK delegation
- Saves to `../assets/models/mobilenet_v3_small_xnnpack.pte`
- Generates `../assets/imagenet_classes.txt` (1000 classes)

#### Exporting YOLO

```bash
cd example/python
python3 export_yolo.py --model yolo11n  # or yolov8n, yolov5n
```

**What it does**:
- Downloads YOLO model from Ultralytics
- Exports to ExecuTorch format with XNNPACK delegation
- Saves to `../assets/models/{model}_xnnpack.pte`
- Generates `../assets/coco_labels.txt` (80 classes)

### Custom Model Export

**Template** for exporting custom models:

```python
# example/python/export_custom.py
import torch
from executorch.exir import to_edge
from executorch.exir.backend.canonical_partitioners import CanonicalPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

def export_custom_model():
    # 1. Load your PyTorch model
    model = YourModel()
    model.load_state_dict(torch.load('your_model.pth'))
    model.eval()

    # 2. Create example input with correct shape
    example_input = (torch.randn(1, 3, 224, 224),)

    # 3. Export to ExecuTorch
    exported_program = torch.export.export(model, example_input)
    edge_program = to_edge(exported_program)

    # 4. Apply XNNPACK backend delegation (optional but recommended)
    edge_program = edge_program.to_backend(XnnpackPartitioner())

    # 5. Generate ExecuTorch program
    executorch_program = edge_program.to_executorch()

    # 6. Save .pte file
    output_path = "../assets/models/custom_model_xnnpack.pte"
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    print(f"✅ Model exported to {output_path}")
    print(f"   Input shape: {example_input[0].shape}")
    print(f"   Model size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export_custom_model()
```

### Troubleshooting Model Export

#### Issue: Model export fails with "operator not supported"

**Solution**: Check ExecuTorch supported operators:
```python
from executorch.exir import EdgeCompileConfig

# List unsupported ops
config = EdgeCompileConfig(_check_ir_validity=True)
edge_program = to_edge(exported_program, compile_config=config)
```

#### Issue: Model is too large

**Solutions**:
1. **Quantize to INT8**:
```python
from torch.ao.quantization import quantize_dynamic

quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

2. **Use smaller backbone**:
```python
# Instead of MobileNetV3 Large, use Small
model = models.mobilenet_v3_small(pretrained=True)
```

#### Issue: Inference is slow

**Solutions**:
1. **Enable XNNPACK delegation** (already shown above)
2. **Reduce input size**:
```python
# Instead of 640x640, use 320x320 for YOLO
example_input = (torch.randn(1, 3, 320, 320),)
```

---

## Best Practices

### Model Organization

1. **One definition per file**: `{model_name}_model_definition.dart`
2. **Dedicated settings class**: `{model_name}_model_settings.dart`
3. **Separate processors**: `{model_name}_input_processor.dart` and `{model_name}_output_processor.dart`
4. **Custom renderer**: `{model_name}_renderer.dart` in `renderers/screens/`

### Settings Management

1. **Extend ModelSettings**: Always inherit from base class
2. **Provide defaults**: Constructor should have sensible defaults
3. **Implement reset()**: Reset all settings to defaults
4. **Type-safe casting**: Cast `ModelSettings` to specific type in processors

### Performance Optimization

1. **Cache labels**: Load once, store in static map
2. **Reuse processors**: Controller recreates on settings change only
3. **Preload models**: Load model bytes before creating controller
4. **Dispose properly**: Always call `model.dispose()` in controller cleanup

### Testing New Models

1. **Test with static images first**: Easier to debug
2. **Verify tensor shapes**: Check `model.inputShapes` and `model.outputShapes`
3. **Test settings changes**: Ensure processors recreate correctly
4. **Test camera mode**: Verify frame processing performance

---

## Quick Reference: File Checklist

When adding a new model, create these files:

- [ ] `python/export_{model}.py` - Export script
- [ ] `lib/models/{model}_model_definition.dart` - Main definition
- [ ] `lib/models/{model}_model_settings.dart` - Settings class
- [ ] `lib/models/{model}_result.dart` - Result type
- [ ] `lib/processors/{model}_input_processor.dart` - Preprocessing
- [ ] `lib/processors/{model}_output_processor.dart` - Postprocessing
- [ ] `lib/renderers/screens/{model}_renderer.dart` - Visualization
- [ ] `assets/models/{model}_xnnpack.pte` - Model file
- [ ] `assets/{model}_labels.txt` - Class labels (if applicable)
- [ ] Update `lib/models/model_registry.dart` - Register model
- [ ] Update `pubspec.yaml` - Add assets

---

## Contact and Support

For questions about the example app architecture:
- Check this document first
- Review existing model implementations (MobileNet, YOLO)
- File issues at the package repository

**Last Updated**: 2025-10-11
**Example App Version**: 1.0
**Package Version**: 0.0.2
