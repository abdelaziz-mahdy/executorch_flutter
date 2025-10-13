# GPU-Accelerated Preprocessing with Flutter Shaders

A step-by-step tutorial for implementing GPU-accelerated image preprocessing for ExecuTorch models using Flutter Fragment Shaders.

> 📁 **Reference Implementations**: All working code is in [lib/processors/shaders/](lib/processors/shaders/) - this tutorial explains how to build your own based on these examples.

## Why GPU Preprocessing?

GPU preprocessing offers **performance comparable to OpenCV** (very close on macOS) while using native Flutter APIs:

- ✅ **High performance** - comparable to OpenCV preprocessing
- ✅ **Hardware acceleration** - on all platforms (mobile & desktop)
- ✅ **No external dependencies** - no opencv_dart required
- ✅ **Easy to customize** - write custom GLSL shaders
- ✅ **Great for real-time** - camera inference and high frame rates

## Architecture Overview

GPU preprocessing uses three components working together:

1. **Native Image Decoder**: `ui.decodeImageFromList()` - Hardware-accelerated
2. **Fragment Shader (GLSL)**: GPU-based image transformations
3. **Optimized Tensor Conversion**: Single-loop RGBA → NCHW format

```
Input Bytes → Native Decode → GPU Shader → Tensor Convert → TensorData
  (JPEG/PNG)   [2-3ms]         [1-2ms]        [2-3ms]      (Float32)
```

---

## Step-by-Step Tutorial

### Step 1: Create a GLSL Fragment Shader

Fragment shaders run on the GPU and perform image transformations. You'll create a `.frag` file in your `shaders/` directory.

**What the shader needs to do:**
- Accept input image dimensions and target dimensions as uniforms
- Sample pixels from the input texture
- Apply transformations (resize, crop, padding)
- Optionally apply normalization
- Output transformed pixels

**Example YOLO shader** (letterbox resize with gray padding):
- 📄 **See**: [shaders/yolo_preprocess.frag](../shaders/yolo_preprocess.frag)
- **Key operations**:
  - Calculates scale to fit image while maintaining aspect ratio
  - Applies letterbox padding with gray (114, 114, 114)
  - Samples from input texture with correct UV coordinates

**Example MobileNet shader** (center crop with ImageNet normalization):
- 📄 **See**: [shaders/mobilenet_preprocess.frag](../shaders/mobilenet_preprocess.frag)
- **Key operations**:
  - Resizes shortest side to 256
  - Center crops to 224x224
  - Applies ImageNet normalization (mean/std subtraction)

**Shader basics you need to know:**
- Uniforms: `uniform vec2 uInputSize` - values passed from Dart
- Samplers: `uniform sampler2D uTexture` - input image
- Output: `out vec4 fragColor` - resulting pixel color
- Coordinates: `FlutterFragCoord().xy` - current pixel position

### Step 2: Register Shaders in pubspec.yaml

Tell Flutter about your shaders by adding them to `pubspec.yaml`:

```yaml
flutter:
  shaders:
    - shaders/yolo_preprocess.frag
    - shaders/mobilenet_preprocess.frag
```

Flutter will compile these shaders when you build your app.

### Step 3: Implement the Preprocessor Class

Create a Dart class that loads and uses your shader.

> 📄 **Full Implementation Examples**:
> - [lib/processors/shaders/gpu_yolo_preprocessor.dart](lib/processors/shaders/gpu_yolo_preprocessor.dart)
> - [lib/processors/shaders/gpu_mobilenet_preprocessor.dart](lib/processors/shaders/gpu_mobilenet_preprocessor.dart)

**Your preprocessor class needs to:**

#### 3.1: Load the Shader (Once)

Shaders should be loaded once and cached. Flutter's `FragmentProgram.fromAsset()` loads the compiled shader:

```dart
Future<void> _initializeShader() async {
  if (_isInitialized) return;  // Only load once
  _program = await ui.FragmentProgram.fromAsset('shaders/your_shader.frag');
  _isInitialized = true;
}
```

> 📍 **See**: [gpu_yolo_preprocessor.dart:29-34](lib/processors/shaders/gpu_yolo_preprocessor.dart#L29-L34)

#### 3.2: Decode Images with Native Decoder

Use Flutter's hardware-accelerated decoder instead of software libraries:

```dart
Future<ui.Image> _decodeImageNative(Uint8List bytes) async {
  final completer = Completer<ui.Image>();
  ui.decodeImageFromList(bytes, completer.complete);
  return completer.future;
}
```

**Why this matters**: Native decoding is 3-5x faster than software decoding with the `image` package.

> 📍 **See**: [gpu_yolo_preprocessor.dart:55-59](lib/processors/shaders/gpu_yolo_preprocessor.dart#L55-L59)

#### 3.3: Execute Shader on GPU

Pass your image and parameters to the shader, then render the result:

**Key steps:**
1. Get shader instance from program
2. Set shader uniforms (image dimensions, target size)
3. Set image sampler
4. Draw to canvas with shader
5. Convert canvas to image

> 📍 **See full implementation**: [gpu_yolo_preprocessor.dart:62-87](lib/processors/shaders/gpu_yolo_preprocessor.dart#L62-L87)

**Setting uniforms example:**
```dart
shader.setFloat(0, inputImage.width.toDouble());   // uInputSize.x
shader.setFloat(1, inputImage.height.toDouble());  // uInputSize.y
shader.setFloat(2, targetWidth.toDouble());        // uOutputSize.x
shader.setFloat(3, targetHeight.toDouble());       // uOutputSize.y
shader.setImageSampler(0, inputImage);             // uTexture
```

**Important**: Set floats before setting the image sampler!

#### 3.4: Convert to Tensor (Optimized)

Convert the processed `ui.Image` to NCHW tensor format using a single-loop approach:

**Why single-loop is faster:**
- Better CPU cache locality
- Processes all channels in one pass
- 2-3x faster than separate channel loops

> 📍 **See full implementation**: [gpu_yolo_preprocessor.dart:90-116](lib/processors/shaders/gpu_yolo_preprocessor.dart#L90-L116)

**Single-loop pattern:**
```dart
for (int i = 0; i < totalPixels; i++) {
  final pixelIndex = i * 4;  // RGBA, so 4 bytes per pixel
  floats[i] = pixels[pixelIndex] * scale;                     // R
  floats[i + totalPixels] = pixels[pixelIndex + 1] * scale;   // G
  floats[i + totalPixels * 2] = pixels[pixelIndex + 2] * scale; // B
}
```

#### 3.5: Clean Up Resources

**Always dispose images** to prevent memory leaks:

```dart
image.dispose();
processedImage.dispose();
```

> 📍 **See error handling and cleanup**: [gpu_yolo_preprocessor.dart:37-53](lib/processors/shaders/gpu_yolo_preprocessor.dart#L37-L53)

### Step 4: Use in Your Model Definition

Integrate your GPU preprocessor into your model's input processing:

**Pattern:**
1. Add GPU option to your `PreprocessingProvider` enum
2. Create a switch case for GPU preprocessing
3. Instantiate your GPU preprocessor with configuration

> 📍 **See example**: Model definitions use input processors that route to GPU preprocessors based on settings

```dart
switch (settings.preprocessingProvider) {
  case PreprocessingProvider.gpu:
    return YoloInputProcessor(
      config: YoloPreprocessConfig(targetWidth: 640, targetHeight: 640),
      preprocessingProvider: PreprocessingProvider.gpu,
    );
  // ... other cases
}
```

---

## Key Concepts Explained

### Shader Uniforms

Uniforms are values you pass from Dart to your shader:
- `vec2` = 2 floats (e.g., width and height)
- `sampler2D` = image texture
- Set with `shader.setFloat()` and `shader.setImageSampler()`

### Texture Coordinates (UV)

Shaders use normalized coordinates (0.0 to 1.0):
- (0.0, 0.0) = top-left corner
- (1.0, 1.0) = bottom-right corner
- Calculate UV by dividing pixel position by image dimensions

### Letterbox vs Center Crop

**Letterbox** (YOLO):
- Scale image to fit target size
- Maintain aspect ratio
- Add padding on sides
- Good for: Object detection models

**Center Crop** (MobileNet):
- Scale shortest side to intermediate size
- Crop center square
- Good for: Classification models

> 📍 **See implementations**:
> - Letterbox: [shaders/yolo_preprocess.frag](../shaders/yolo_preprocess.frag)
> - Center crop: [shaders/mobilenet_preprocess.frag](../shaders/mobilenet_preprocess.frag)

### NCHW Tensor Format

Neural networks expect tensors in **NCHW** format:
- **N**: Batch size (usually 1)
- **C**: Channels (3 for RGB)
- **H**: Height
- **W**: Width

Memory layout: `[R-plane][G-plane][B-plane]` not `RGBRGBRGB...`

---

## Common Shader Patterns

### Pattern 1: Letterbox Resize

Maintains aspect ratio with padding:

```glsl
float scale = min(targetWidth / inputWidth, targetHeight / inputHeight);
vec2 scaledSize = inputSize * scale;
vec2 offset = (targetSize - scaledSize) * 0.5;
```

> 📍 **Full example**: [shaders/yolo_preprocess.frag](shaders/yolo_preprocess.frag)

### Pattern 2: Center Crop

Resize then crop center:

```glsl
float scale = max(256.0 / inputWidth, 256.0 / inputHeight);
vec2 cropOffset = (scaledSize - targetSize) * 0.5;
```

> 📍 **Full example**: [shaders/mobilenet_preprocess.frag](shaders/mobilenet_preprocess.frag)

### Pattern 3: Normalization

Apply mean/std normalization in shader:

```glsl
const vec3 mean = vec3(0.485, 0.456, 0.406);
const vec3 std = vec3(0.229, 0.224, 0.225);
vec3 normalized = (color.rgb - mean) / std;
```

---

## Performance Optimization Tips

### 1. Shader Initialization

**Do**: Load shader once, cache it
```dart
if (_isInitialized) return;
```

**Don't**: Load shader every frame

### 2. Image Decoding

**Do**: Use native decoder
```dart
ui.decodeImageFromList(bytes, completer.complete);
```

**Don't**: Use `image` package for decoding

### 3. Tensor Conversion

**Do**: Single-loop conversion
```dart
for (int i = 0; i < totalPixels; i++) {
  floats[i] = pixels[i * 4] * scale;  // All channels in one pass
}
```

**Don't**: Separate loops per channel

### 4. Resource Management

**Do**: Always dispose images
```dart
image.dispose();
processedImage.dispose();
```

**Don't**: Let images accumulate in memory

---

## When to Use GPU Preprocessing

✅ **Use GPU preprocessing when:**
- Real-time camera inference
- High frame rates needed (>60 FPS)
- Want OpenCV-like performance without dependencies
- Targeting mobile and desktop platforms

❌ **Use CPU preprocessing when:**
- Low power consumption is critical
- Simple preprocessing needs
- Debugging (easier to inspect intermediate steps)

---

## Comparison with OpenCV

| Feature | GPU Preprocessing | OpenCV |
|---------|------------------|---------|
| Performance | Comparable (very close on macOS) | High-performance |
| Dependencies | None (native Flutter) | opencv_dart package |
| Platform Support | All (mobile + desktop) | All (cross-platform) |
| Customization | GLSL shaders | C++ OpenCV API |
| Memory Usage | Moderate | Low |
| GPU Utilization | High | Low (CPU-based) |

---

## Complete Example Implementations

### Working Code

All GPU preprocessing code is organized in the shaders directory:

📁 **[lib/processors/shaders/](lib/processors/shaders/)** - GPU preprocessors directory
  - 📄 **[README.md](lib/processors/shaders/README.md)** - Architecture and usage guide
  - 📄 **[gpu_yolo_preprocessor.dart](lib/processors/shaders/gpu_yolo_preprocessor.dart)** - YOLO preprocessing
  - 📄 **[gpu_mobilenet_preprocessor.dart](lib/processors/shaders/gpu_mobilenet_preprocessor.dart)** - MobileNet preprocessing

### GLSL Shaders

📁 **[shaders/](../shaders/)** - Fragment shader implementations
  - 📄 **[yolo_preprocess.frag](shaders/yolo_preprocess.frag)** - Letterbox resize with gray padding
  - 📄 **[mobilenet_preprocess.frag](shaders/mobilenet_preprocess.frag)** - Center crop with ImageNet normalization

---

## Troubleshooting

### Issue: Shader fails to load

**Error**: `Failed to load shader` or `Shader not found`

**Solution**:
1. Verify shader is registered in `pubspec.yaml`:
   ```yaml
   flutter:
     shaders:
       - shaders/your_shader.frag
   ```
2. Run `flutter clean && flutter pub get`
3. Rebuild your app

### Issue: Black or corrupted output

**Possible causes**:
- Uniforms set in wrong order
- Image sampler set before floats
- Incorrect UV coordinate calculation

**Solution**:
1. Always set floats before image sampler
2. Check uniform indices match shader definition
3. Verify UV coordinates are in 0.0-1.0 range

> 📍 **See correct order**: [gpu_yolo_preprocessor.dart:66-71](lib/processors/shaders/gpu_yolo_preprocessor.dart#L66-L71)

### Issue: Memory leaks

**Symptoms**: App memory grows over time, eventual crashes

**Solution**: Always dispose `ui.Image` objects:
```dart
image.dispose();
processedImage.dispose();
```

### Issue: Slow performance

**Check**:
1. Are you using native decoder? (`ui.decodeImageFromList`)
2. Is shader initialization cached? (don't reload every frame)
3. Is tensor conversion using single-loop pattern?

---

## Next Steps

1. **Study the examples**: Start with [lib/processors/shaders/gpu_yolo_preprocessor.dart](lib/processors/shaders/gpu_yolo_preprocessor.dart)
2. **Read the shaders**: Understand GLSL code in [shaders/yolo_preprocess.frag](../shaders/yolo_preprocess.frag)
3. **Customize for your model**: Create your own shader and preprocessor
4. **Test performance**: Compare with OpenCV and CPU preprocessing

---

## References

- **[Flutter Fragment Shaders](https://docs.flutter.dev/development/ui/advanced/shaders)** - Official Flutter documentation
- **[GLSL Documentation](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)** - OpenGL Shading Language reference
- **[ExecuTorch Documentation](https://pytorch.org/executorch/)** - PyTorch ExecuTorch docs
- **[Processors Directory](lib/processors/shaders/README.md)** - Implementation guide

---

**Built with ❤️ for high-performance on-device ML inference**
