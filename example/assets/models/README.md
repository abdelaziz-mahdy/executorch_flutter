# ExecuTorch Models Directory

This directory contains ExecuTorch model files (`.pte`) for the example app. **Model files are not committed to git** and must be generated locally.

## 🚀 Quick Start

### One-Command Setup (Recommended)

From the project root, run:

```bash
cd python
python3 setup_models.py
```

This automated script will:
- ✅ Install all required Python dependencies
- ✅ Export MobileNet V3 Small for image classification
- ✅ Export YOLO11 Nano for object detection
- ✅ Generate COCO labels file (80 classes)
- ✅ Verify all models are ready

### Manual Model Export

If you prefer to export models individually:

```bash
cd python
python3 export_models.py  # MobileNet V3 + COCO labels
python3 export_yolo.py     # YOLO models
```

## 📦 Current Models

| Model File | Type | Backend | Size | Use Case |
|-----------|------|---------|------|----------|
| `mobilenet_v3_small_xnnpack.pte` | Classification | XNNPACK | 9.8 MB | ImageNet 1000 classes |
| `yolo12n_xnnpack.pte` | Detection | XNNPACK | 10 MB | COCO 80 classes (latest) |
| `yolo11n_xnnpack.pte` | Detection | XNNPACK | 10 MB | COCO 80 classes |
| `yolov8n_xnnpack.pte` | Detection | XNNPACK | 12 MB | COCO 80 classes |
| `yolov5n_xnnpack.pte` | Detection | XNNPACK | 10 MB | COCO 80 classes |

Additional files:
- `imagenet_classes.txt` - 1000 ImageNet class labels
- `../coco_labels.txt` - 80 COCO object class labels

## 📱 Usage in Example App

The example app demonstrates a modern single-page playground where you:

1. **Select a Model** - Choose between image classification or object detection
2. **Pick an Image** - From camera or gallery
3. **View Results** - See predictions with confidence scores

```dart
// Models are automatically loaded based on user selection
final model = await ExecutorchManager.instance.loadModel(
  config.assetPath // e.g., 'assets/models/mobilenet_v3_small_xnnpack.pte'
);

// Processors handle preprocessing and postprocessing
final processor = ImageNetProcessor(
  preprocessConfig: ImagePreprocessConfig(
    targetWidth: 224,
    targetHeight: 224,
  ),
  classLabels: classLabels,
);

final result = await processor.process(imageBytes, model);
```

## 🔧 Export Your Own Models

### For Image Classification

Use the unified export script:

```bash
cd python
python3 export_models.py
```

Or customize with `executorch_exporter.py`:

```bash
python3 executorch_exporter.py
```

### For Object Detection (YOLO)

Follow the detailed guide in `python/export_yolo.py` or the complete [Model Export Guide](../MODEL_EXPORT_GUIDE.md).

Quick summary:
1. Install ultralytics: `pip install ultralytics`
2. Export to ONNX: `model.export(format='onnx')`
3. Convert ONNX to ExecuTorch using ExecuTorch tools
4. Place `.pte` file in this directory

## ❓ Troubleshooting

### Models Not Generated?
1. Check Python version: `python3 --version` (3.10+ required)
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure PyTorch and ExecuTorch are installed correctly

### Models Not Loading in App?
1. Verify models exist in this directory
2. Check `pubspec.yaml` includes `assets/models/`
3. Ensure model file names match the app configuration
4. Check Flutter console for specific error messages

### Wrong Results?
- Verify preprocessing matches model training (normalization, input size)
- Check class labels file matches model output
- Ensure tensor format is correct (NCHW vs NHWC)

## 📖 Complete Documentation

For detailed model export instructions, see:
- **[Model Export Guide](../MODEL_EXPORT_GUIDE.md)** - Complete conversion guide
- **[Python Scripts README](../../python/README.md)** - Export script documentation
- **[ExecuTorch Docs](https://pytorch.org/executorch/)** - Official PyTorch ExecuTorch documentation

## 🎯 Next Steps

1. Generate the MobileNet model: `cd python && python3 export_models.py`
2. (Optional) Export YOLO model following the guide
3. Run the example app: `cd example && flutter run`
4. Try different images and models in the playground

---

**Note**: Model files (`.pte`) are in `.gitignore`. Each developer generates them locally using the Python export scripts.