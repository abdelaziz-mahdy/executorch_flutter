# ExecuTorch Model Export Guide

This guide explains how to export and use ML models with the ExecuTorch Flutter example app.

## Supported Models

### 1. Image Classification (MobileNet V3)
- **Input Size**: 224x224
- **Format**: RGB, normalized with ImageNet stats
- **Output**: 1000 ImageNet classes

### 2. Object Detection (YOLO)
- **Supported Models**: YOLOv5, YOLOv8, YOLO11, YOLO12 (all variants: n, s, m, l, x)
- **Input Size**: 640x640
- **Format**: RGB, normalized [0, 1]
- **Output**: Bounding boxes + 80 COCO classes
- **Recommended**: YOLO11n or yolov8n for mobile devices
- **Automatic Format Detection**: Processor handles YOLOv5 (85 channels) and YOLOv8/v11 (84 channels) formats

## Quick Start

### Option 1: Use the Unified Export Script (Recommended)

Generate all models needed for the example app:

```bash
cd python
python3 export_models.py
```

This will:
- ✅ Export MobileNet V3 Small for image classification
- ✅ Create COCO class labels file (80 classes)
- ℹ️ Provide YOLO export instructions

Generated files:
- `example/assets/models/mobilenet_v3_small_xnnpack.pte` (9.8 MB)
- `example/assets/coco_labels.txt` (80 COCO classes)

### Option 2: Export Models Individually

#### Export MobileNet V3

```bash
cd python
python3 export_models.py
```

This exports MobileNet V3 Small with XNNPACK backend optimized for mobile devices.

#### Export YOLO Models (Direct Method)

**Supports: YOLOv5, YOLOv8, YOLO11**

```bash
cd python

# Export YOLO11 Nano (default, recommended)
python3 export_yolo.py

# Or specify any YOLO model:
python3 export_yolo.py yolo11n.pt
python3 export_yolo.py yolov8n.pt
python3 export_yolo.py yolov5s.pt
```

This script uses PyTorch's `torch.export` API to directly convert YOLO models to ExecuTorch format (no ONNX intermediate step needed).

## YOLO Export Details

### Supported YOLO Models

**YOLOv5** (Ultralytics):
- `yolov5n.pt` - Nano (~4MB) - Fastest
- `yolov5s.pt` - Small (~14MB) - Balanced
- `yolov5m.pt` - Medium (~40MB) - More accurate
- `yolov5l.pt` - Large (~90MB) - High accuracy
- `yolov5x.pt` - XLarge (~170MB) - Best accuracy (not recommended for mobile)

**YOLOv8** (Ultralytics):
- `yolov8n.pt` - Nano (~6MB) - Fastest, improved over v5
- `yolov8s.pt` - Small (~22MB) - Best balance for mobile
- `yolov8m.pt` - Medium (~52MB) - High accuracy
- `yolov8l.pt` - Large (~88MB) - Very high accuracy
- `yolov8x.pt` - XLarge (~138MB) - Best accuracy (not recommended for mobile)

**YOLO11** (Latest):
- `yolo11n.pt` - Nano (~6MB) - **Recommended for mobile**
- `yolo11s.pt` - Small (~22MB) - Best balance
- `yolo11m.pt` - Medium (~52MB) - High accuracy
- `yolo11l.pt` - Large (~88MB) - Very high accuracy
- `yolo11x.pt` - XLarge (~138MB) - Best accuracy (not recommended for mobile)

**Recommendation**: Use Nano (n) or Small (s) variants for mobile devices. YOLO11 offers the best accuracy/speed tradeoff.

### Direct Export Method (Recommended)

The `export_yolo.py` script uses `torch.export` to directly convert YOLO models:

```python
import torch
from ultralytics import YOLO
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Load YOLO model (downloads automatically)
yolo = YOLO("yolo11n.pt")
model = yolo.model.eval().cpu()

# Export to ExecuTorch
example_input = (torch.zeros(1, 3, 640, 640),)
exported_program = torch.export.export(model, example_input)
edge_program = to_edge(exported_program)
edge_program = edge_program.to_backend(XnnpackPartitioner())
executorch_program = edge_program.to_executorch()

# Save
with open("yolo11n_xnnpack.pte", "wb") as f:
    executorch_program.write_to_file(f)
```

### Fallback: ONNX Method

If direct export fails, the script falls back to ONNX:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
onnx_path = model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    dynamic=False  # ExecuTorch requires static shapes
)
```

Then follow the [ExecuTorch ONNX guide](https://pytorch.org/executorch/stable/tutorial-onnx-to-executorch.html) to convert ONNX to `.pte`.

### Place Model File

Move the `.pte` file to:
```
example/assets/models/yolo11n_xnnpack.pte
```

## Model Configuration in Flutter

### Add New Model to Example App

Edit `example/lib/screens/model_playground.dart`:

```dart
const List<ModelConfig> availableModels = [
  ModelConfig(
    name: 'MobileNet V3',
    type: ModelType.imageClassification,
    assetPath: 'assets/models/mobilenet_v3_small_xnnpack.pte',
    inputSize: 224,
  ),
  ModelConfig(
    name: 'YOLOv8 Nano',
    type: ModelType.objectDetection,
    assetPath: 'assets/models/yolov8n_xnnpack.pte',
    inputSize: 640,
  ),
  // Add your custom model here
  ModelConfig(
    name: 'My Custom Model',
    type: ModelType.imageClassification,
    assetPath: 'assets/models/my_model_xnnpack.pte',
    inputSize: 256,
  ),
];
```

### Create Custom Processor

Processors handle the complete ML pipeline. Create one for your model:

```dart
// 1. Create preprocessor
class MyPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  @override
  String get inputTypeName => 'Image (Uint8List)';

  @override
  bool validateInput(Uint8List input) {
    return input.isNotEmpty;
  }

  @override
  Future<List<TensorData>> preprocess(Uint8List input, {ModelMetadata? metadata}) async {
    // Your preprocessing logic: resize, normalize, convert to tensor
    final processedData = await processImage(input);

    return [TensorData(
      shape: [1, 3, 256, 256].cast<int?>(),
      dataType: TensorType.float32,
      data: processedData,
      name: 'input',
    )];
  }
}

// 2. Create postprocessor
class MyPostprocessor extends ExecuTorchPostprocessor<MyResult> {
  @override
  String get outputTypeName => 'My Result Type';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    return outputs.isNotEmpty && outputs.first.dataType == TensorType.float32;
  }

  @override
  Future<MyResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
    // Your postprocessing logic: parse tensors, apply softmax, etc.
    final result = parseOutput(outputs.first);
    return MyResult(data: result);
  }
}

// 3. Combine into processor
class MyProcessor extends ExecuTorchProcessor<Uint8List, MyResult> {
  MyProcessor() :
    _preprocessor = MyPreprocessor(),
    _postprocessor = MyPostprocessor();

  final MyPreprocessor _preprocessor;
  final MyPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<Uint8List> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<MyResult> get postprocessor => _postprocessor;
}
```

See `example/lib/processors/` for complete working examples:
- `image_processor.dart` - ImageNet classification
- `yolo_processor.dart` - YOLO object detection

## Testing Your Model

### 1. Update pubspec.yaml

Ensure your model is listed in assets:

```yaml
flutter:
  assets:
    - assets/models/
    - assets/imagenet_classes.txt
    - assets/coco_labels.txt
```

### 2. Run the App

```bash
cd example
flutter run
```

### 3. Select Your Model

1. Launch the app
2. Select your model from the list
3. Pick an image (camera or gallery)
4. View the results

## Troubleshooting

### Model Loading Fails

**Error**: `Failed to load model from ...`

**Solutions**:
- Verify the `.pte` file exists in `assets/models/`
- Check file permissions
- Ensure model was exported with correct backend (xnnpack for mobile)

### Tensor Shape Mismatch

**Error**: `Input 0 dimension X mismatch: expected Y, got Z`

**Solutions**:
- Check your preprocessor output shape matches model input
- Verify model was exported with correct input size
- Update `ModelConfig.inputSize` to match your model

### Wrong Results

**Solutions**:
- Verify preprocessing normalization matches training
- Check class labels match model output
- Ensure proper data format (RGB vs BGR, NCHW vs NHWC)

## Model Optimization

### Quantization

Reduce model size and improve performance:

```python
config = ExportConfig(
    model_name="my_model",
    backends=["xnnpack"],
    quantize=True  # Enable INT8 quantization
)
```

### Backend Selection

Choose the right backend for your platform:

- **xnnpack**: Best for mobile CPU (recommended)
- **vulkan**: GPU acceleration (if available)
- **portable**: Fallback, works everywhere
- **coreml**: Apple Neural Engine (iOS only)
- **mps**: Apple Metal Performance Shaders (iOS only)

## Performance Tips

1. **Model Size**: Start with nano/small models, upgrade if needed
2. **Input Resolution**: Lower resolution = faster inference
   - Classification: 224x224 is standard
   - Detection: 640x640 for accuracy, 416x416 for speed
3. **Quantization**: Use INT8 for 4x smaller size with minimal accuracy loss
4. **Backend**: XNNPACK provides best mobile CPU performance

## Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [ExecuTorch GitHub](https://github.com/pytorch/executorch)
- [Model Zoo](https://pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review ExecuTorch documentation
3. Open an issue on GitHub with:
   - Model type and size
   - Error messages
   - Steps to reproduce
