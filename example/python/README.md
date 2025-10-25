# ExecuTorch Flutter - Python Tools

Unified command-line tool for model export and validation with **multi-backend support**.

## Quick Start

```bash
# Export all models with all available backends (default mode)
python main.py

# Export specific model with all backends
python main.py export --mobilenet

# Export with specific backends only
python main.py export --mobilenet --backends xnnpack coreml

# Export all models with XNNPACK only (fast)
python main.py export --all --backends xnnpack

# Validate all models
python main.py validate
```

## Backend Support

This tool now exports models for multiple ExecuTorch backends:

- **XNNPACK**: CPU-optimized (Android, iOS, macOS, Linux) - *default*
- **CoreML**: Apple Neural Engine (iOS, macOS) - *best performance on Apple devices*
- **MPS**: Metal GPU acceleration (iOS, macOS) - *GPU-accelerated*
- **Vulkan**: Cross-platform GPU (Android, Linux) - *GPU-accelerated*

📚 **See [BACKENDS.md](BACKENDS.md) for complete backend selection guide**

## Installation

```bash
pip install torch torchvision executorch ultralytics opencv-python torchao
```

## Commands

### Export (Default)

Export models to ExecuTorch format for the Flutter app.

```bash
# Export all models (MobileNet + YOLO11n + labels)
python main.py
python main.py export --all

# Export MobileNet only
python main.py export --mobilenet

# Export YOLO only
python main.py export --yolo yolo11n
python main.py export --yolo yolo11n yolov8n  # Multiple models

# Export labels only
python main.py export --labels
```

**Supported YOLO models**: yolo11n, yolov8n, yolov5n (nano versions only)

**Output**: `../assets/models/`

### Validate

Validate exported models with test images and save results.

```bash
# Validate all models with all test images
python main.py validate

# Custom directories
python main.py validate --models-dir ../assets/models \
                        --images-dir ../assets/images \
                        --output-file ../assets/results.json
```

**Output**: `../assets/model_test_results.json`

## File Structure

```
python/
├── main.py                      # Main CLI tool (use this!)
├── executorch_exporter.py       # Generic ExecuTorch exporter framework (legacy)
├── validate_all_models.py       # Model validation framework
└── README.md                    # This file
```

**Note**: Both MobileNet and YOLO exports now use the official Ultralytics-style ExecuTorch export pattern directly in `main.py`.

## Examples

### Export Workflow

```bash
# 1. Export all models
python main.py

# 2. Verify files exist
ls -lh ../assets/models/

# 3. Run Flutter app
cd ../example
flutter run
```

### Validation Workflow

```bash
# 1. Export models first
python main.py export --all

# 2. Validate models
python main.py validate

# 3. Check results
cat ../assets/model_test_results.json
```

## Exported Models

### MobileNet V3 Small
- **Files**: `mobilenet_v3_small_{backend}.pte` (~5-6 MB per backend)
  - `mobilenet_v3_small_xnnpack.pte` (CPU)
  - `mobilenet_v3_small_coreml.pte` (Apple NPU)
  - `mobilenet_v3_small_mps.pte` (Apple GPU)
  - `mobilenet_v3_small_vulkan.pte` (GPU)
- **Input**: [1, 3, 224, 224] (RGB, ImageNet normalized)
- **Output**: [1, 1000] logits (requires softmax)
- **Use**: Image classification (1000 ImageNet classes)

### YOLO Nano Models
All YOLO models export to multiple backends:
- **Files**: `{model}_{backend}.pte` (~8-9 MB per backend)
  - `yolo11n_xnnpack.pte`, `yolo11n_coreml.pte`, `yolo11n_mps.pte`, `yolo11n_vulkan.pte`
  - `yolov8n_xnnpack.pte`, `yolov8n_coreml.pte`, `yolov8n_mps.pte`, `yolov8n_vulkan.pte`
  - `yolov5n_xnnpack.pte`, `yolov5n_coreml.pte`, `yolov5n_mps.pte`, `yolov5n_vulkan.pte`
- **Input**: [1, 3, 640, 640] (RGB, normalized to [0,1])
- **Output**: [1, 84, 8400] (4 bbox coords + 80 COCO classes, raw format)
- **Use**: Object detection (80 COCO classes)
- **Note**: Requires post-processing (DFL, sigmoid, NMS) in Flutter app

## Validation Results

The validation script tests each model with 5 test images:
- Cat
- Dog
- Car
- Person
- Street

**Results include**:
- Top-5 predictions (classification)
- All detected objects with bounding boxes (detection)
- Confidence scores
- Inference times
- Model metadata

## Troubleshooting

### Missing dependencies
```bash
pip install torch torchvision executorch ultralytics opencv-python torchao
```

### YOLO export fails
Use the official export script directly:
```bash
python export_yolo_official.py --model_name yolo11n.pt --backend xnnpack
```

### Models don't load in Flutter
1. Verify .pte files exist in `../assets/models/`
2. Check file sizes (should be ~10MB each)
3. Re-export with `python main.py export --all`

## Advanced Usage

### Custom Output Directory
```bash
python main.py export --all --output-dir /path/to/output
```

### Export Multiple YOLOs
```bash
python main.py export --yolo yolo11n yolo11s yolov8n
```

### Validation with Custom Paths
```bash
python main.py validate \
  --models-dir /custom/models \
  --images-dir /custom/images \
  --output-file /custom/results.json
```

## Model Export Details

### Export Method (Ultralytics-style)
Both MobileNet and YOLO models use the official Ultralytics ExecuTorch export pattern:

```python
et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()
```

### MobileNet V3 Small
- **Input**: [1, 3, 224, 224] (RGB, ImageNet normalized)
- **Output**: [1, 1000] logits (requires softmax)
- **Preprocessing**: Resize(256) → CenterCrop(224) → Normalize
- **File**: `mobilenet_v3_small_xnnpack.pte` (~9.8 MB)

### YOLO11 Nano
- **Input**: [1, 3, 640, 640] (RGB, normalized to [0,1])
- **Output**: [1, 84, 8400] (4 bbox + 80 classes, raw format)
- **Preprocessing**: Letterbox resize to 640x640
- **File**: `yolo11n_xnnpack.pte` (~10.2 MB)
- **Note**: Output requires post-processing (DFL, sigmoid, NMS) in Flutter app

## CI/CD Integration

```bash
# In your CI pipeline
cd python
python main.py export --all
python main.py validate

# Check exit code
if [ $? -eq 0 ]; then
  echo "✅ All models valid"
else
  echo "❌ Validation failed"
  exit 1
fi
```

## Support

For issues or questions:
- Check Flutter app logs: `flutter logs`
- Re-export models: `python main.py export --all`
- Run validation: `python main.py validate`
- Review results: `cat ../assets/model_test_results.json`
