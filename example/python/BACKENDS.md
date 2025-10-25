# ExecuTorch Backends Guide

This guide explains how to choose the right backend for your ExecuTorch Flutter models.

## Quick Start

Export models with multiple backends:

```bash
# Export all models with all available backends (default)
python main.py

# Export MobileNet with all backends
python main.py export --mobilenet

# Export MobileNet with specific backends only
python main.py export --mobilenet --backends xnnpack coreml

# Export all YOLO models with XNNPACK only (fastest export)
python main.py export --all --backends xnnpack
```

## Backend Overview

| Backend | Platform(s) | Hardware | Use Case | Performance |
|---------|------------|----------|----------|-------------|
| **XNNPACK** | All | CPU | General-purpose, fallback | Good ⭐⭐⭐ |
| **Core ML** | iOS, macOS | NPU/GPU/CPU | Apple devices, best performance | Excellent ⭐⭐⭐⭐⭐ |
| **MPS** | iOS, macOS | GPU | Apple GPU acceleration | Very Good ⭐⭐⭐⭐ |
| **Vulkan** | Android, Linux | GPU | Android GPU acceleration | Very Good ⭐⭐⭐⭐ |
| **QNN** | Android | NPU | Qualcomm SoCs (Snapdragon) | Excellent ⭐⭐⭐⭐⭐ |
| **ARM** | Embedded | NPU | ARM Ethos-U MCUs | Good ⭐⭐⭐ |

## Backend Selection Guide

### iOS/macOS Devices

**Recommended priority**: CoreML > MPS > XNNPACK

```bash
python main.py export --mobilenet --backends coreml mps xnnpack
```

**Why?**
- **CoreML**: Uses Apple Neural Engine for maximum efficiency and best battery life
- **MPS**: GPU acceleration using Metal, good for models not fully supported by CoreML
- **XNNPACK**: CPU fallback, works everywhere but slower

**Performance example (iPhone 14 Pro)**:
- CoreML: ~5ms inference (MobileNet)
- MPS: ~8ms inference
- XNNPACK: ~15ms inference

### Android Devices

**Recommended priority**: QNN (Qualcomm) > Vulkan > XNNPACK

```bash
# For Qualcomm devices (Snapdragon)
python main.py export --yolo yolo11n --backends qnn vulkan xnnpack

# For other Android devices
python main.py export --yolo yolo11n --backends vulkan xnnpack
```

**Why?**
- **QNN**: Best performance on Qualcomm Snapdragon devices (uses Hexagon DSP/NPU)
- **Vulkan**: Cross-vendor GPU acceleration, works on most modern Android devices
- **XNNPACK**: CPU fallback

**Performance example (Pixel 7)**:
- QNN: ~7ms inference (YOLO11n) - Snapdragon devices only
- Vulkan: ~12ms inference
- XNNPACK: ~25ms inference

### Desktop/Server (Linux/Windows)

**Recommended priority**: Vulkan > XNNPACK

```bash
python main.py export --mobilenet --backends vulkan xnnpack
```

**Why?**
- **Vulkan**: GPU acceleration on most modern GPUs (NVIDIA, AMD, Intel)
- **XNNPACK**: CPU execution, good for servers without GPU

### Embedded Devices (ARM MCUs)

**Recommended priority**: ARM > XNNPACK

```bash
python main.py export --mobilenet --backends arm xnnpack
```

**Why?**
- **ARM**: Optimized for ARM Ethos-U NPU in microcontrollers
- **XNNPACK**: CPU fallback for non-NPU ARM devices

## Backend Capabilities

### XNNPACK (CPU)
✅ Works on: Android, iOS, macOS, Linux, Windows
✅ Supported models: All
✅ Setup: No additional setup required
⚠️ Performance: Good, but not optimal

**When to use**:
- Development/testing
- Devices without GPU/NPU support
- Battery-critical applications (CPU may be more efficient than GPU for small models)

### Core ML (Apple NPU)
✅ Works on: iOS 13+, macOS 11+
✅ Supported models: Most vision models (classification, detection, segmentation)
⚠️ Limitations: Some ops may fall back to CPU
⚠️ Setup: Requires `coremltools>=9.0` for export

**When to use**:
- iOS/macOS production apps
- Battery-critical applications
- Best possible performance on Apple devices

**Export requirements**:
```bash
pip install 'coremltools>=9.0'
```

### MPS (Metal Performance Shaders)
✅ Works on: iOS 13+, macOS 11+
✅ Supported models: Most models
✅ Setup: Built into ExecuTorch
⚠️ Power: Higher power consumption than CoreML

**When to use**:
- iOS/macOS apps when CoreML doesn't support specific ops
- GPU-intensive workloads
- Real-time video processing

### Vulkan (GPU)
✅ Works on: Android 7+, Linux, Windows
✅ Supported models: Most models
⚠️ Setup: Vulkan drivers required
⚠️ Power: Higher power consumption than NPU

**When to use**:
- Android apps with GPU support
- Desktop apps with GPU
- Real-time inference

**Setup on device**:
- Android: Vulkan drivers included in Android 7+
- Linux/Windows: Install GPU drivers with Vulkan support

### QNN (Qualcomm)
✅ Works on: Qualcomm Snapdragon devices (Android)
✅ Supported models: Most vision models
⚠️ Setup: Requires Qualcomm QNN SDK
⚠️ Platform: Snapdragon devices only

**When to use**:
- Production Android apps targeting Qualcomm devices
- Maximum performance on Snapdragon devices
- Battery-efficient inference

**Supported devices**:
- Snapdragon 8 Gen 1/2/3
- Snapdragon 888/888+
- Snapdragon 778G/780G/870/865

**Export requirements**:
```bash
# Requires Qualcomm QNN SDK (advanced setup)
# See: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html
```

### ARM (Ethos-U NPU)
✅ Works on: ARM Cortex-M MCUs with Ethos-U NPU
✅ Supported models: Quantized models (INT8)
⚠️ Setup: Requires ARM Ethos-U driver
⚠️ Platform: Embedded devices only

**When to use**:
- Embedded ML on ARM MCUs
- Ultra-low power inference
- Edge devices

## Export Time Comparison

Export time varies by backend complexity:

| Backend | Export Time (MobileNet) | Export Time (YOLO11n) |
|---------|------------------------|---------------------|
| XNNPACK | ~30 seconds | ~45 seconds |
| CoreML | ~2 minutes | ~4 minutes |
| MPS | ~30 seconds | ~45 seconds |
| Vulkan | ~1 minute | ~2 minutes |
| QNN | ~3 minutes | ~5 minutes |
| ARM | ~2 minutes | ~3 minutes |

**Tip**: Export XNNPACK first for quick testing, then export optimized backends for production.

## File Size Comparison

Backend models have similar file sizes (±10%):

| Model | XNNPACK | CoreML | MPS | Vulkan |
|-------|---------|--------|-----|--------|
| MobileNet V3 Small | 5.5 MB | 5.8 MB | 5.5 MB | 5.6 MB |
| YOLO11 Nano | 8.2 MB | 8.5 MB | 8.2 MB | 8.3 MB |
| YOLOv8 Nano | 8.0 MB | 8.3 MB | 8.0 MB | 8.1 MB |

## Troubleshooting

### Backend Not Available

If you get "Backend not available" error:

```bash
⚠️  No available backends from requested: ['coreml']
   Available backends: ['xnnpack', 'mps', 'vulkan']
```

**Solution**: Install missing backend dependencies

For CoreML:
```bash
pip install 'coremltools>=9.0'
```

For QNN:
```bash
# Requires Qualcomm QNN SDK
# Download from: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
```

### Export Fails for Specific Backend

If export succeeds for some backends but fails for others:

```
✅ Successfully exported 2/3 backends
   • xnnpack: mobilenet_v3_small_xnnpack.pte (5.5 MB)
   • mps: mobilenet_v3_small_mps.pte (5.5 MB)

⚠️  Failed 1 backend(s):
   • coreml: Operator not supported by CoreML
```

**Solution**: Use the backends that worked. Some models have ops not supported by all backends.

### Model Doesn't Load on Device

If model exports successfully but fails to load on device:

**Check**:
1. Backend support on target platform (e.g., CoreML only works on iOS/macOS)
2. Device OS version (e.g., CoreML requires iOS 13+, Vulkan requires Android 7+)
3. Model file is included in app assets

**Solution**: Use XNNPACK as fallback (works everywhere)

## Advanced: Multi-Backend Deployment

For production apps, export multiple backends and select at runtime:

```dart
// Dart example
Future<ExecuTorchModel> loadBestModel() async {
  // Try CoreML first (best performance on iOS)
  if (Platform.isIOS || Platform.isMacOS) {
    try {
      return await ExecuTorchModel.load(
        await rootBundle.load('assets/models/mobilenet_v3_small_coreml.pte'),
      );
    } catch (e) {
      print('CoreML failed, trying MPS...');
    }

    // Try MPS fallback
    try {
      return await ExecuTorchModel.load(
        await rootBundle.load('assets/models/mobilenet_v3_small_mps.pte'),
      );
    } catch (e) {
      print('MPS failed, falling back to XNNPACK...');
    }
  }

  // Android: Try Vulkan
  if (Platform.isAndroid) {
    try {
      return await ExecuTorchModel.load(
        await rootBundle.load('assets/models/mobilenet_v3_small_vulkan.pte'),
      );
    } catch (e) {
      print('Vulkan failed, falling back to XNNPACK...');
    }
  }

  // CPU fallback (always works)
  return await ExecuTorchModel.load(
    await rootBundle.load('assets/models/mobilenet_v3_small_xnnpack.pte'),
  );
}
```

**Benefits**:
- Best performance on each platform
- Automatic fallback to CPU if GPU/NPU fails
- Single codebase supports all devices

**Trade-off**:
- Larger app size (multiple model files)
- More complex deployment logic

**Recommendation**:
- Development: Use XNNPACK only (fastest iteration)
- Production iOS: Include CoreML + XNNPACK
- Production Android: Include Vulkan + XNNPACK

## Performance Testing

To measure backend performance on your device:

1. Export model with multiple backends:
```bash
python main.py export --mobilenet --backends xnnpack coreml mps vulkan
```

2. Run example app and switch between backends in UI

3. Compare inference times in performance overlay

**Expected results** (iPhone 14 Pro, MobileNet V3):
- CoreML: 4-6ms
- MPS: 7-9ms
- XNNPACK: 13-17ms

**Expected results** (Pixel 7, YOLO11n):
- Vulkan: 10-15ms
- XNNPACK: 23-28ms

## Summary

**Quick recommendations**:

- **iOS production**: `--backends coreml xnnpack`
- **Android production**: `--backends vulkan xnnpack`
- **Development/testing**: `--backends xnnpack` (fast export)
- **Maximum compatibility**: `--backends xnnpack coreml mps vulkan`

**Rule of thumb**:
- Always include XNNPACK (works everywhere)
- Add platform-specific backends for better performance
- Test on real devices before deployment

## References

- ExecuTorch Backends: https://pytorch.org/executorch/main/backends-overview.html
- CoreML: https://developer.apple.com/documentation/coreml
- Vulkan: https://www.vulkan.org/
- Qualcomm QNN: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html
