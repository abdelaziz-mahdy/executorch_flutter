# Multi-Backend Export Summary

## Successfully Exported Models

All models have been exported with multiple backend variants for optimal performance on different platforms.

### Backend Coverage

✅ **XNNPACK** (CPU) - Works on all platforms
✅ **CoreML** (Apple NPU) - iOS/macOS only
✅ **MPS** (Metal GPU) - iOS/macOS only
⚠️ **Vulkan** (GPU) - Not available on macOS (Android/Linux only)

---

## Exported Model Files

### MobileNet V3 Small (Image Classification)

| Backend | File | Size | Platform | Performance |
|---------|------|------|----------|-------------|
| XNNPACK | `mobilenet_v3_small_xnnpack.pte` | 9.7 MB | All | Good ⭐⭐⭐ |
| CoreML | `mobilenet_v3_small_coreml.pte` | 5.2 MB | iOS/macOS | Excellent ⭐⭐⭐⭐⭐ |
| MPS | `mobilenet_v3_small_mps.pte` | 9.8 MB | iOS/macOS | Very Good ⭐⭐⭐⭐ |

**Total**: 3 variants, 24.7 MB

### YOLO11 Nano (Object Detection)

| Backend | File | Size | Platform | Performance |
|---------|------|------|----------|-------------|
| XNNPACK | `yolo11n_xnnpack.pte` | 10 MB | All | Good ⭐⭐⭐ |
| CoreML | `yolo11n_coreml.pte` | 5.7 MB | iOS/macOS | Excellent ⭐⭐⭐⭐⭐ |
| MPS | `yolo11n_mps.pte` | 10 MB | iOS/macOS | Very Good ⭐⭐⭐⭐ |

**Total**: 3 variants, 25.7 MB

### YOLOv8 Nano (Object Detection)

| Backend | File | Size | Platform | Performance |
|---------|------|------|----------|-------------|
| XNNPACK | `yolov8n_xnnpack.pte` | 12.2 MB | All | Good ⭐⭐⭐ |
| CoreML | `yolov8n_coreml.pte` | 6.5 MB | iOS/macOS | Excellent ⭐⭐⭐⭐⭐ |
| MPS | `yolov8n_mps.pte` | 12.2 MB | iOS/macOS | Very Good ⭐⭐⭐⭐ |

**Total**: 3 variants, 30.9 MB

### YOLOv5 Nano (Object Detection)

| Backend | File | Size | Platform | Performance |
|---------|------|------|----------|-------------|
| XNNPACK | `yolov5n_xnnpack.pte` | 10.3 MB | All | Good ⭐⭐⭐ |
| CoreML | `yolov5n_coreml.pte` | 5.6 MB | iOS/macOS | Excellent ⭐⭐⭐⭐⭐ |
| MPS | `yolov5n_mps.pte` | 10.3 MB | iOS/macOS | Very Good ⭐⭐⭐⭐ |

**Total**: 3 variants, 26.2 MB

---

## Summary Statistics

- **Total Models Exported**: 4 base models
- **Total Variants**: 12 model files (3 backends × 4 models)
- **Total Size**: 107.5 MB
- **Average Size per Variant**: 9.0 MB

### Backend Distribution

| Backend | Model Count | Total Size | Avg Size |
|---------|-------------|------------|----------|
| XNNPACK | 4 | 42.2 MB | 10.6 MB |
| CoreML | 4 | 23.0 MB | 5.8 MB |
| MPS | 4 | 42.3 MB | 10.6 MB |

**Note**: CoreML models are smaller due to aggressive optimization by Apple's compiler.

---

## Flutter App Integration

All 12 model variants are now registered in the Flutter app's `model_registry.dart`:

### Available in UI

**MobileNet V3 Small**:
- MobileNet V3 Small (XNNPACK)
- MobileNet V3 Small (CoreML)
- MobileNet V3 Small (MPS)
- ~~MobileNet V3 Small (Vulkan)~~ - Not exported (macOS)

**YOLO11 Nano**:
- YOLO11 Nano (XNNPACK)
- YOLO11 Nano (CoreML)
- YOLO11 Nano (MPS)
- ~~YOLO11 Nano (Vulkan)~~ - Not exported (macOS)

**YOLOv8 Nano**:
- YOLOv8 Nano (XNNPACK)
- YOLOv8 Nano (CoreML)
- YOLOv8 Nano (MPS)
- ~~YOLOv8 Nano (Vulkan)~~ - Not exported (macOS)

**YOLOv5 Nano**:
- YOLOv5 Nano (XNNPACK)
- YOLOv5 Nano (CoreML)
- YOLOv5 Nano (MPS)
- ~~YOLOv5 Nano (Vulkan)~~ - Not exported (macOS)

Users can switch between backends in the model dropdown to compare performance.

---

## Performance Expectations

### iOS/macOS Devices

**MobileNet V3 Small** (224×224 image):
- CoreML: ~4-6ms (best)
- MPS: ~7-9ms
- XNNPACK: ~13-17ms

**YOLO11 Nano** (640×640 image):
- CoreML: ~8-12ms (best)
- MPS: ~15-20ms
- XNNPACK: ~30-40ms

### Android Devices (if Vulkan exported)

**With GPU support**:
- Vulkan: ~10-15ms
- XNNPACK: ~25-35ms

**Without GPU**:
- XNNPACK: ~25-35ms (CPU only)

---

## Next Steps

### For Development
1. Run Flutter app: `cd ../example && flutter run`
2. Select any model with backend suffix from dropdown
3. Compare inference times between backends
4. Observe performance overlay for FPS and timing

### For Production

**iOS/macOS apps**:
```yaml
# Include in pubspec.yaml
assets:
  - assets/models/mobilenet_v3_small_coreml.pte  # Best performance
  - assets/models/mobilenet_v3_small_xnnpack.pte  # Fallback
  - assets/models/yolo11n_coreml.pte
  - assets/models/yolo11n_xnnpack.pte
```

**Android apps** (when Vulkan exported):
```yaml
assets:
  - assets/models/mobilenet_v3_small_vulkan.pte  # Best performance
  - assets/models/mobilenet_v3_small_xnnpack.pte  # Fallback
  - assets/models/yolo11n_vulkan.pte
  - assets/models/yolo11n_xnnpack.pte
```

### To Export Vulkan (on Linux/Android)

Vulkan backend requires Linux or Android development environment:

```bash
# On Linux or WSL
python main.py export --all --backends vulkan xnnpack

# Or specific model
python main.py export --mobilenet --backends vulkan xnnpack
```

This will create:
- `mobilenet_v3_small_vulkan.pte`
- `yolo11n_vulkan.pte`
- `yolov8n_vulkan.pte`
- `yolov5n_vulkan.pte`

---

## Troubleshooting

### Model doesn't load
1. Check the model file exists in `assets/models/`
2. Verify the model is listed in `pubspec.yaml` under `assets:`
3. Try XNNPACK variant (works everywhere)

### Poor performance on iOS/macOS
1. Use CoreML backend for best performance
2. Ensure device has Apple Silicon or A-series chip
3. Check iOS version (iOS 13+ required)

### Missing Vulkan models
Vulkan backend is only available on Linux/Android. Export on:
- Linux development machine
- WSL (Windows Subsystem for Linux)
- Android CI/CD pipeline

---

## References

- [Backend Selection Guide](BACKENDS.md) - Complete guide to choosing backends
- [README](README.md) - Export tool documentation
- [ExecuTorch Backends](https://pytorch.org/executorch/main/backends-overview.html) - Official docs
- [Model Registry](../lib/models/model_registry.dart) - Flutter app integration
