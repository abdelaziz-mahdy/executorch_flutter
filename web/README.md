##  ExecuTorch Flutter Web Platform

This directory contains the WebAssembly implementation of ExecuTorch for Flutter web applications.

## Architecture

```
Flutter Dart Code
    ↓
lib/src/web/executorch_model_web.dart (Dart web plugin)
    ↓ (dart:js_interop)
web/js/executorch_wrapper.js (JavaScript wrapper)
    ↓
web/wasm/executor_runner.js (Emscripten glue code)
    ↓
web/wasm/executor_runner.wasm (ExecuTorch WebAssembly binary)
```

## Directory Structure

```
web/
├── js/
│   ├── executorch_wrapper.js     # JavaScript API wrapper
│   └── README.md                  # JavaScript wrapper documentation
├── wasm/
│   ├── executor_runner.js         # Emscripten-generated JavaScript (73 KB)
│   ├── executor_runner.wasm       # WebAssembly binary (2.4 MB)
│   └── tmp/                       # Gitignored - contains HTML example
│       └── executor_runner.html   # Reference implementation
└── README.md                      # This file
```

## Usage

### Loading Models

Unlike mobile platforms that load models from file paths, the web platform loads models from bytes:

```dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:executorch_flutter/executorch_flutter.dart';

// Load model from asset bundle
final modelBytes = await rootBundle.load('assets/models/model.pte');
final model = await ExecuTorchModel.load(
  modelBytes.buffer.asUint8List(),
);

// Run inference
final outputs = await model.forward([inputTensor]);

// Clean up
await model.dispose();
```

### Platform Detection

The package automatically uses the web implementation when compiled for web:

```dart
import 'package:executorch_flutter/executorch_flutter.dart';

// Works on both mobile and web - platform detection is automatic
final model = await ExecuTorchModel.load(...);
```

- **Mobile (Android, iOS, macOS)**: Expects `String` file path
- **Web**: Expects `Uint8List` model bytes

## Building

The WebAssembly binaries are pre-built and included in this directory. To rebuild them:

```bash
# From the package root
./scripts/build_wasm.sh

# This will:
# 1. Build ExecuTorch C++ → WebAssembly using Docker or native Emscripten
# 2. Output executor_runner.js and executor_runner.wasm to web/wasm/
# 3. Include reference HTML example in web/wasm/tmp/
```

## Implementation Status

### ✅ Completed
- JavaScript wrapper API (`web/js/executorch_wrapper.js`)
- Dart web plugin with dart:js_interop bindings
- WebAssembly module initialization
- Virtual filesystem management (Emscripten FS)
- Model loading from bytes
- Platform detection with conditional imports
- Web plugin registration

### ⏳ C++ Bindings Needed

The current implementation **returns mock tensor data** from `forward()` because the C++ bindings are not yet complete. To enable real inference:

1. Add Emscripten bindings to ExecuTorch C++ code:
   ```cpp
   // In ExecuTorch source (not this package)
   EMSCRIPTEN_BINDINGS(executorch) {
     emscripten::function("loadModel", &loadModel);
     emscripten::function("forward", &forward);
     emscripten::function("dispose", &dispose);
     emscripten::function("getInputShapes", &getInputShapes);
     emscripten::function("getOutputShapes", &getOutputShapes);
   }
   ```

2. Rebuild WebAssembly with new bindings:
   ```bash
   ./scripts/build_wasm.sh
   ```

3. Update JavaScript wrapper to call C++ functions instead of returning mocks

## Current Behavior

**Without C++ bindings**, the web implementation:

- ✅ Initializes WebAssembly module correctly
- ✅ Writes model bytes to virtual filesystem
- ✅ Manages model lifecycle (load/dispose)
- ⚠️ Returns mock zero-filled tensors from `forward()`
- ⚠️ Cannot extract real input/output shapes

**This is sufficient for**:
- Testing Dart ↔ JavaScript ↔ Wasm communication
- Developing UI and preprocessing logic
- Verifying web deployment works

**This is NOT sufficient for**:
- Real inference
- Production use

## Performance Characteristics

### File Sizes

- **executor_runner.wasm**: 2.4 MB uncompressed
- **executor_runner.js**: 73 KB
- **executorch_wrapper.js**: ~8 KB
- **Total download**: ~2.5 MB (compresses to ~1 MB with gzip/brotli)

### Inference Performance

Expected performance (once C++ bindings are complete):

- **Model loading**: 100-300ms (depends on model size and network)
- **Inference**: 20-100ms (depends on model complexity)
- **Memory**: ~50-100MB per loaded model

## Browser Compatibility

| Browser | Version | Status | Notes |
|---------|---------|--------|-------|
| **Chrome** | 91+ | ✅ Full support | Best performance |
| **Firefox** | 89+ | ✅ Full support | Good performance |
| **Safari** | 15+ | ✅ Full support | Requires CORS headers |
| **Edge** | 91+ | ✅ Full support | Chromium-based |

### Requirements

- WebAssembly support (all modern browsers)
- JavaScript enabled
- Sufficient memory (depends on model size)
- CORS headers for model files (if loaded from CDN)

## Debugging

### Browser Console

Check browser console for ExecuTorch logs:

```javascript
// In browser console
window.ExecuTorchRunner.isInitialized  // Should be true after initialization
window.ExecuTorchRunner.getLoadedModelIds()  // List of loaded models
```

### Network Tab

Verify WebAssembly files are loaded:

- `executor_runner.js` should load with 200 status
- `executor_runner.wasm` should load with 200 status
- Check for CORS errors if loading fails

### Memory Usage

Monitor memory usage in browser DevTools:

- Model loading creates large typed arrays
- Each model requires ~50-100MB of heap memory
- Call `dispose()` to free memory when done

## Deployment

### Flutter Web Build

```bash
flutter build web --release
```

The Wasm files and JavaScript wrapper are automatically included in the build output.

### Web Server Configuration

Ensure your web server serves Wasm files with correct MIME type:

```nginx
# Nginx example
location ~* \.wasm$ {
    types {
        application/wasm wasm;
    }
    add_header 'Access-Control-Allow-Origin' '*';
}
```

```apache
# Apache example
AddType application/wasm .wasm
Header set Access-Control-Allow-Origin "*"
```

## Known Limitations

1. **No Real Inference**: Currently returns mock data (C++ bindings needed)
2. **Large Bundle Size**: 2.4 MB Wasm file (compresses to ~1 MB)
3. **Browser Only**: Does not work in Flutter desktop web builds
4. **No Streaming**: Models must be fully loaded before inference
5. **Memory**: No automatic memory management (user must call `dispose()`)

## Future Enhancements

- C++ bindings for real inference
- Streaming model loading for large models
- Web Worker support for background inference
- Shared model caching across tabs
- Progressive loading for better UX

## Resources

- **ExecuTorch Wasm Example**: https://github.com/pytorch/executorch/tree/main/examples/wasm
- **Emscripten Bindings**: https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html
- **Dart JS Interop**: https://dart.dev/web/js-interop
- **Build System**: `scripts/WASM_BUILD_README.md`
- **Design Document**: `WEB_PLATFORM_DESIGN.md`

---

**Status**: Web plugin implementation complete, C++ bindings needed for inference
**Last Updated**: 2026-01-04
