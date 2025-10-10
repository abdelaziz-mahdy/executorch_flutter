# ExecuTorch Model Export Guide

This guide explains how to export PyTorch models to ExecuTorch format (`.pte` files) for use with the `executorch_flutter` package.

## Overview

The `executorch_flutter` package requires models in ExecuTorch's optimized `.pte` format. This guide covers:
- Basic model export process
- Platform-specific backend optimization
- Model validation and testing
- Integration with Flutter apps

## Prerequisites

```bash
# Create virtual environment (recommended)
python -m venv executorch_env
source executorch_env/bin/activate  # On Windows: executorch_env\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install executorch
```

## Basic Export Process

### Step 1: Prepare Your PyTorch Model

Your model must be in evaluation mode and have a known input shape:

```python
import torch
import torchvision.models as models

# Load your model
model = models.mobilenet_v3_small(weights='DEFAULT')
model.eval()

# Define example input (required for export)
example_input = (torch.randn(1, 3, 224, 224),)
```

### Step 2: Export to ExecuTorch

Use `torch.export` to convert your model:

```python
from executorch.exir import to_edge

# Export to torch.export format
exported_program = torch.export.export(model, example_input)

# Convert to Edge IR (ExecuTorch intermediate representation)
edge_program = to_edge(exported_program)

# Generate ExecuTorch program
executorch_program = edge_program.to_executorch()

# Save to .pte file
with open("model.pte", "wb") as f:
    executorch_program.write_to_file(f)
```

### Step 3: Add Backend Optimization (Recommended)

For mobile deployment, use XNNPACK backend for CPU optimization:

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Export with XNNPACK optimization
exported_program = torch.export.export(model, example_input)
edge_program = to_edge(exported_program)
edge_program = edge_program.to_backend(XnnpackPartitioner())
executorch_program = edge_program.to_executorch()

# Save
with open("model_xnnpack.pte", "wb") as f:
    executorch_program.write_to_file(f)
```

## Platform-Specific Backends

Choose the right backend for your target platform:

### Android (CPU)
```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

edge_program = edge_program.to_backend(XnnpackPartitioner())
```
**Best for**: CPU inference on Android devices

### iOS (Apple Neural Engine)
```python
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner

edge_program = edge_program.to_backend(CoreMLPartitioner())
```
**Best for**: Leveraging Apple Neural Engine (A12+)

### iOS (Metal GPU)
```python
from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner

edge_program = edge_program.to_backend(MPSPartitioner())
```
**Best for**: GPU acceleration on iOS devices

### Portable (Fallback)
```python
# No backend partitioner needed
edge_program = to_edge(exported_program)
executorch_program = edge_program.to_executorch()
```
**Best for**: Testing, debugging, or unsupported platforms

## Complete Example Script

```python
#!/usr/bin/env python3
"""
Example: Export MobileNet V3 to ExecuTorch with XNNPACK
"""

import torch
import torchvision.models as models
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

def export_mobilenet():
    # Load model
    model = models.mobilenet_v3_small(weights='DEFAULT').eval()

    # Create example input
    example_input = (torch.randn(1, 3, 224, 224),)

    # Export
    print("Exporting model...")
    exported_program = torch.export.export(model, example_input)

    # Apply XNNPACK backend
    print("Applying XNNPACK optimization...")
    edge_program = to_edge(exported_program)
    edge_program = edge_program.to_backend(XnnpackPartitioner())

    # Generate ExecuTorch program
    print("Generating ExecuTorch program...")
    executorch_program = edge_program.to_executorch()

    # Save
    output_file = "mobilenet_v3_small_xnnpack.pte"
    with open(output_file, "wb") as f:
        executorch_program.write_to_file(f)

    print(f"✅ Model exported to: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    export_mobilenet()
```

## Using Exported Models in Flutter

### 1. Add Model to Assets

Place your `.pte` file in your Flutter project:

```
your_flutter_app/
  assets/
    models/
      your_model.pte
```

Update `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/
```

### 2. Load and Use Model

```dart
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

// Load model from assets (copy to temporary directory first)
final byteData = await rootBundle.load('assets/models/your_model.pte');
final tempDir = await getTemporaryDirectory();
final file = File('${tempDir.path}/your_model.pte');
await file.writeAsBytes(byteData.buffer.asUint8List());

// Load model
final model = await ExecuTorchModel.load(file.path);

// Prepare input tensor
final inputTensor = TensorData(
  shape: [1, 3, 224, 224].cast<int?>(),
  dataType: TensorType.float32,
  data: yourInputData,  // Uint8List
  name: 'input',
);

// Run inference (returns List<TensorData> directly)
final outputs = await model.forward([inputTensor]);

// Process outputs
for (var output in outputs) {
  print('Output shape: ${output.shape}');
  print('Data type: ${output.dataType}');
  // Process tensor data...
}

// Clean up
await model.dispose();
```

## Model Requirements

### Input/Output Specifications

- **Static Shapes**: ExecuTorch requires fixed tensor dimensions
- **Supported Types**: float32, int32, int8, uint8
- **Memory**: Keep models under 100MB for optimal mobile performance

### Best Practices

1. **Use Quantization**: Reduce model size with INT8 quantization
2. **Optimize for Mobile**: Use lightweight architectures (MobileNet, EfficientNet)
3. **Test on Device**: Always validate performance on actual hardware
4. **Profile Inference**: Monitor execution time and memory usage

## Example Models

For working examples and reference implementations, see:

### Image Classification
- **Location**: `python/export_models.py`
- **Model**: MobileNet V3 Small
- **Features**: ImageNet preprocessing, XNNPACK backend

### Object Detection (YOLO)
- **Location**: `python/export_yolo.py`
- **Models**: YOLOv5, YOLOv8, YOLO11 (all variants)
- **Features**: Direct torch.export, no ONNX needed

### Example App
- **Location**: `example/MODEL_EXPORT_GUIDE.md`
- **Complete integration examples** with processors and UI

## Troubleshooting

### Export Fails with "torch.export not found"
Update PyTorch to version 2.1.0+:
```bash
pip install --upgrade torch
```

### XNNPACK Backend Not Available
Install ExecuTorch with XNNPACK support:
```bash
pip install executorch[xnnpack]
```

### Model Too Large
Consider:
- Using quantization (INT8)
- Choosing a smaller architecture
- Removing unused operations

### Inference Too Slow
Try:
- Different backend (XNNPACK, CoreML, MPS)
- Lower input resolution
- Quantization
- Smaller model variant

## Additional Resources

- **Official Documentation**: [PyTorch ExecuTorch Docs](https://pytorch.org/executorch/)
- **Export Tutorial**: [ExecuTorch Export Guide](https://pytorch.org/executorch/stable/using-executorch-export.html)
- **Example Models**: `python/` directory in this repository
- **Flutter Integration**: `example/MODEL_EXPORT_GUIDE.md`

## Quick Setup for Example App

To quickly set up and generate all example models:

```bash
cd python
python3 setup_models.py
```

This script will:
- Install all required dependencies
- Export MobileNet V3 for image classification
- Export YOLO11n for object detection
- Generate COCO labels file
- Verify all models are ready for the example app

## Support

For issues or questions:
- Check the example implementations in `python/` directory
- Review the example app guide: `example/MODEL_EXPORT_GUIDE.md`
- File issues on GitHub with detailed reproduction steps

---

**Note**: This guide focuses on the export process. For complete Flutter integration examples including preprocessing and postprocessing, see the example app documentation.
