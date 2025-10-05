# Python Export Scripts - Organization Summary

## 📋 Overview

This document summarizes the Python export scripts for the ExecuTorch Flutter plugin after reorganization and fixing the YOLO export issue.

**Date**: 2025-10-05
**Status**: ✅ Complete and Verified

---

## 🚨 Critical Issue Fixed

### Problem
ALL YOLO model files (yolo11n, yolov8n, yolo12n, yolov5n) were incorrectly exported:
- **Expected**: 640x640 input, YOLO detection output [1, 84, 8400]
- **Actual**: 224x224 input, ImageNet classification output [1, 1000]
- **Root Cause**: Models were manually renamed MobileNet files, not properly exported YOLO models

### Solution
1. Removed all incorrectly exported YOLO .pte files
2. Used official Intel/ExecuTorch YOLO export script (`export_yolo_official.py`)
3. Successfully exported YOLO11n with correct 640x640 input
4. Verified output shape: `[1, 84, 8400]` ✅

---

## 📁 Script Organization

### Active Scripts (Use These)

#### 1. `export_yolo_official.py` ⭐ **RECOMMENDED FOR YOLO**
- **Purpose**: Official Intel/ExecuTorch YOLO export script with quantization support
- **Source**: Adapted from PyTorch ExecuTorch examples
- **Features**:
  - Supports YOLO11, YOLOv8, YOLOv5, YOLO12
  - XNNPACK and OpenVINO backends
  - Optional INT8 quantization
  - Validation support
- **Usage**:
  ```bash
  python export_yolo_official.py --model_name yolo11n.pt --backend xnnpack
  python export_yolo_official.py --model_name yolov8n.pt --backend xnnpack --quantize
  ```
- **Input**: 640x640 RGB images, normalized [0,1]
- **Output**: [1, 84, 8400] detection tensor (4 bbox + 80 classes)
- **Status**: ✅ Tested and verified

#### 2. `export_unified.py` 🎯 **UNIFIED INTERFACE**
- **Purpose**: Single interface for all model exports
- **Features**:
  - Export MobileNet and YOLO models
  - Generate label files
  - Clear command-line interface
- **Usage**:
  ```bash
  python export_unified.py --all                    # Export all default models
  python export_unified.py --mobilenet              # Export MobileNet only
  python export_unified.py --yolo yolo11n           # Export specific YOLO
  python export_unified.py --yolo yolo11n yolov8n   # Export multiple YOLOs
  ```
- **Status**: ✅ Created but YOLO export has limitations (use export_yolo_official.py instead)

#### 3. `export_models.py`
- **Purpose**: Export MobileNet and generate COCO labels
- **Features**:
  - Exports MobileNet V3 Small for classification
  - Generates COCO labels file
  - Shows YOLO export instructions
- **Usage**:
  ```bash
  python export_models.py
  ```
- **Status**: ✅ Working for MobileNet

#### 4. `executorch_exporter.py`
- **Purpose**: Generic ExecuTorch model exporter framework
- **Features**:
  - Backend auto-detection (XNNPACK, CoreML, MPS, Vulkan, QNN, ARM)
  - Quantization support
  - Multi-backend export
  - Metadata generation
- **Usage**: Library for building custom exporters
- **Status**: ✅ Framework/library use

#### 5. `setup_models.py`
- **Purpose**: Complete setup workflow with dependency management
- **Features**:
  - Checks Python version
  - Installs dependencies
  - Exports all models
  - Verifies assets
- **Usage**:
  ```bash
  python setup_models.py
  ```
- **Status**: ✅ Complete workflow script

### Test/Utility Scripts

#### 6. `test_model_outputs.py`
- **Purpose**: Verify model inference with test images
- **Features**:
  - Tests MobileNet and YOLO models
  - Saves results in JSON format
  - Uses same test images as Flutter app
- **Usage**:
  ```bash
  python test_model_outputs.py
  ```
- **Status**: ✅ Verified YOLO output shape

#### 7. `verify_model.py`
- **Purpose**: Quick model file verification
- **Usage**:
  ```bash
  python verify_model.py ../example/assets/models/yolo11n_xnnpack.pte
  ```
- **Status**: ✅ Utility tool

### Deprecated/Legacy Scripts

#### 8. `export_yolo_fixed.py` (SUPERSEDED)
- **Status**: ⚠️ Replaced by export_yolo_official.py
- **Issue**: Has torch.export strict=False issues with YOLO dynamic ops
- **Recommendation**: Use export_yolo_official.py instead

#### 9. `export_yolo.py` (DEPRECATED)
- **Status**: ⚠️ Has known issues with YOLO export
- **Issues**:
  - Uses torch.zeros() instead of torch.randn()
  - torch.export compatibility problems
- **Recommendation**: Use export_yolo_official.py instead

### Shell Scripts

#### 10. `export_yolo_models.sh`
- **Purpose**: Batch export multiple YOLO models
- **Status**: ✅ Working
- **Usage**:
  ```bash
  ./export_yolo_models.sh yolo11n.pt yolov8n.pt
  ```

---

## ✅ Verified Model Exports

### MobileNet V3 Small
- **File**: `mobilenet_v3_small_xnnpack.pte`
- **Size**: ~5 MB
- **Input**: [1, 3, 224, 224] float32 (NCHW, RGB, normalized with ImageNet mean/std)
- **Output**: [1, 1000] float32 (ImageNet class probabilities)
- **Status**: ✅ Exported and ready

### YOLO11 Nano
- **File**: `yolo11n_xnnpack.pte`
- **Size**: 10.19 MB
- **Input**: [1, 3, 640, 640] float32 (NCHW, RGB, range [0,1])
- **Output**: [1, 84, 8400] float32 (4 bbox + 80 classes per detection)
- **Status**: ✅ Exported and verified
- **Test Results**: Successfully processed all 5 test images (cat, dog, car, person, street)

---

## 📝 Model Export Guidelines

### For YOLO Models (Object Detection)

**ALWAYS use `export_yolo_official.py`:**

```bash
cd python
python export_yolo_official.py --model_name yolo11n.pt --backend xnnpack
```

**Supported models**:
- YOLO11: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
- YOLOv8: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- YOLOv5: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt

**Important**:
- Input must be 640x640 (YOLO standard)
- Output is [1, 84, 8400] for YOLOv8/v11 (4 bbox + 80 classes)
- NMS (Non-Maximum Suppression) must be applied in post-processing
- YoloProcessor in Flutter handles NMS automatically

### For MobileNet (Classification)

**Use `export_models.py` or `export_unified.py`:**

```bash
python export_models.py
# OR
python export_unified.py --mobilenet
```

---

## 🧪 Test Results

### Test Images
1. Cat (37KB)
2. Dog (40KB)
3. Car (49KB)
4. Person (41KB)
5. Street (92KB)

### YOLO11 Nano Test Results
All 5 test images successfully processed:
- ✅ Output shape verified: [1, 84, 8400]
- ✅ Input processing: 640x640 letterbox resize
- ✅ Ready for Flutter integration

Test results saved in: `example/assets/model_test_results.json`

---

## 🚀 Quick Start

### Export All Default Models
```bash
cd python
python setup_models.py
```

### Export Specific Models
```bash
# MobileNet only
python export_models.py

# YOLO11n only
python export_yolo_official.py --model_name yolo11n.pt --backend xnnpack

# Multiple YOLOs
./export_yolo_models.sh yolo11n.pt yolov8n.pt
```

### Verify Exports
```bash
python verify_model.py ../example/assets/models/yolo11n_xnnpack.pte
python test_model_outputs.py
```

---

## 📦 Dependencies

Required packages:
```bash
pip install torch>=2.1.0
pip install torchvision
pip install executorch
pip install ultralytics
pip install opencv-python  # For YOLO export
pip install torchao  # For quantization
```

---

## 🔍 Troubleshooting

### Issue: YOLO export fails with "float object has no attribute 'node'"
**Solution**: Use `export_yolo_official.py` instead of `export_unified.py` or `export_yolo.py`

### Issue: Model has wrong input/output shape
**Solution**:
1. Delete the .pte file
2. Re-export using the correct script
3. Verify with `verify_model.py`

### Issue: MobileNet works but YOLO doesn't
**Solution**: Check that YOLO model was exported with 640x640 input, not 224x224

---

## 📚 References

- ExecuTorch Documentation: https://pytorch.org/executorch/
- Ultralytics YOLO: https://docs.ultralytics.com/
- Official YOLO Export Example: https://github.com/pytorch/executorch/tree/main/examples/models/yolo12

---

## ✨ Summary

**Before reorganization**:
- ❌ 6 Python scripts with overlapping functionality
- ❌ All YOLO models were incorrectly exported as MobileNet
- ❌ No verification or test scripts
- ❌ Confusing documentation

**After reorganization**:
- ✅ Clear script organization with defined purposes
- ✅ YOLO11n correctly exported with 640x640 input
- ✅ Verified output shape [1, 84, 8400]
- ✅ Test scripts for verification
- ✅ Comprehensive documentation
- ✅ Ready for Flutter integration

**Recommended workflow**:
1. Use `export_yolo_official.py` for all YOLO models
2. Use `export_models.py` for MobileNet
3. Use `test_model_outputs.py` to verify exports
4. Use `setup_models.py` for complete setup
