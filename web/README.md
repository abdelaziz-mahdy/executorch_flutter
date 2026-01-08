## ExecuTorch Flutter Web Platform

This directory contains the WebAssembly implementation of ExecuTorch for Flutter web applications.

## Architecture

```
Flutter Dart Code
    ↓
lib/src/web/executorch_model_web.dart (Dart web plugin)
    ↓ (dart:js_interop)
web/js/executorch_wrapper.js (JavaScript wrapper)
    ↓
web/wasm/executorch.js (Emscripten glue code)
    ↓
web/wasm/executorch.wasm (ExecuTorch WebAssembly binary with XNNPACK)
```

## Directory Structure

```
web/
├── js/
│   ├── executorch_wrapper.js     # JavaScript API wrapper
│   └── README.md                  # JavaScript wrapper documentation
├── wasm/
│   ├── executorch.js              # Emscripten-generated JavaScript (~140 KB)
│   └── executorch.wasm            # WebAssembly binary with XNNPACK (~3.3 MB)
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
# 1. Build ExecuTorch C++ → WebAssembly using Docker (with XNNPACK)
# 2. Output executorch.js and executorch.wasm to web/wasm/
# 3. Run setup_web.dart to copy files to example project
```

## Implementation Status

### ✅ Fully Working
- JavaScript wrapper API (`web/js/executorch_wrapper.js`)
- Dart web plugin with dart:js_interop bindings
- WebAssembly module with XNNPACK backend (Wasm SIMD)
- Virtual filesystem management (Emscripten FS)
- Model loading from bytes or remote URLs
- Real inference with actual tensor outputs
- Platform detection with conditional imports
- Web plugin registration

## Performance Characteristics

### File Sizes

- **executorch.wasm**: ~3.3 MB uncompressed (includes XNNPACK)
- **executorch.js**: ~140 KB
- **executorch_wrapper.js**: ~8 KB
- **Total download**: ~3.5 MB (compresses to ~1.2 MB with gzip/brotli)

### Inference Performance (YOLO11n)

| Stage | Time | % of Total |
|-------|------|------------|
| Preprocessing | ~154ms | 18% |
| Inference | ~622ms | 73% |
| Postprocessing | ~79ms | 9% |
| **Total** | **~855ms** | 100% |

- **Model loading**: 100-500ms (depends on model size and network)
- **Memory**: ~50-100MB per loaded model

**Comparison to Native**:
| Platform | Inference Time |
|----------|----------------|
| Native (XNNPACK) | ~50-100ms |
| Web (XNNPACK Wasm SIMD) | ~622ms |

Web is ~6-10x slower than native but fully functional for interactive use.

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

1. **Bundle Size**: ~3.3 MB Wasm file (compresses to ~1.2 MB with gzip)
2. **No Streaming**: Models must be fully loaded before inference
3. **Memory**: No automatic memory management (user must call `dispose()`)
4. **No Camera**: Live camera inference not supported on web
5. **Performance**: ~6-10x slower than native XNNPACK

## Future Enhancements

- Streaming model loading for large models
- Web Worker support for background inference
- Shared model caching across tabs
- Progressive loading for better UX

## Resources

- **ExecuTorch Wasm Example**: https://github.com/pytorch/executorch/tree/main/examples/wasm
- **Emscripten Bindings**: https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html
- **Dart JS Interop**: https://dart.dev/web/js-interop
- **Build System**: `scripts/WASM_BUILD_README.md`

---

**Status**: Fully working with XNNPACK backend
**Last Updated**: 2026-01-08
