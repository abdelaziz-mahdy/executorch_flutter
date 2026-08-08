# ExecuTorch JavaScript Wrapper

This directory contains the JavaScript wrapper that provides a clean API around the Emscripten-generated ExecuTorch WebAssembly module.

## Architecture

```
Flutter Dart
    ↓ (dart:js_interop)
JavaScript Wrapper (executorch_wrapper.js)
    ↓
Emscripten Module (executor_runner.js)
    ↓
WebAssembly Binary (executor_runner.wasm)
    ↓
ExecuTorch C++ Runtime
```

## Files

- **`executorch_wrapper.js`** - Main wrapper class exposing clean API
- **`../wasm/executor_runner.js`** - Emscripten-generated JavaScript glue code
- **`../wasm/executor_runner.wasm`** - Compiled ExecuTorch runtime (2.4 MB)

## ExecuTorchRunner API

### Initialization

```javascript
// Create global instance (done automatically when script loads)
window.ExecuTorchRunner = new ExecuTorchRunner();

// Initialize Wasm module (call once before using)
await window.ExecuTorchRunner.initialize();
```

### Model Loading

```javascript
// Load model from bytes
const modelBytes = new Uint8Array([...]);  // .pte file bytes
const result = await window.ExecuTorchRunner.loadModel(modelBytes);

console.log(result.modelId);        // Unique model ID
console.log(result.inputShapes);    // Expected input shapes
console.log(result.outputShapes);   // Expected output shapes
```

### Running Inference

```javascript
// Prepare input tensors
const inputs = [{
  shape: [1, 3, 224, 224],           // [batch, channels, height, width]
  dataType: 'float32',               // 'float32', 'int32', 'int8', 'uint8'
  data: new Uint8Array([...]),       // Raw tensor bytes
  name: 'input_0',                   // Optional name
}];

// Run inference
const outputs = await window.ExecuTorchRunner.forward(modelId, inputs);

// Process outputs (same format as inputs)
outputs.forEach(output => {
  console.log(output.shape);
  console.log(output.dataType);
  console.log(output.data);         // Uint8Array of results
});
```

### Cleanup

```javascript
// Dispose model when done
await window.ExecuTorchRunner.dispose(modelId);
```

### Utility Methods

```javascript
// Check if model is loaded
const isLoaded = window.ExecuTorchRunner.isModelLoaded(modelId);

// Get model metadata
const metadata = window.ExecuTorchRunner.getModelMetadata(modelId);

// Get all loaded model IDs
const modelIds = window.ExecuTorchRunner.getLoadedModelIds();
```

## Implementation Status

### ✅ Completed
- JavaScript class structure
- Wasm module initialization
- Virtual filesystem management (Emscripten FS)
- Model loading (write bytes to virtual FS)
- Basic error handling
- Logging and debugging

### ⏳ TODO (Requires C++ Bindings)
- **Actual model loading** - Currently only writes to FS, needs C++ `loadModel()` call
- **Actual inference** - Currently returns mock data, needs C++ `forward()` call
- **Shape extraction** - Currently returns empty arrays, needs C++ metadata extraction
- **Proper disposal** - Currently only cleans up FS, needs C++ `dispose()` call

## C++ Bindings Needed

To complete the implementation, we need to expose C++ functions to JavaScript using Emscripten's `EMSCRIPTEN_BINDINGS`:

```cpp
// In executor_runner.cpp (or new bindings file)
EMSCRIPTEN_BINDINGS(executorch) {
  emscripten::function("loadModel", &loadModel);
  emscripten::function("forward", &forward);
  emscripten::function("dispose", &dispose);
  emscripten::function("getInputShapes", &getInputShapes);
  emscripten::function("getOutputShapes", &getOutputShapes);
}
```

These bindings would:
1. Load `.pte` model files from virtual FS
2. Execute inference with tensor inputs
3. Return tensor outputs
4. Free model resources

## Current Behavior

**Without C++ bindings**, the wrapper currently:
- ✅ Initializes Wasm module
- ✅ Writes model bytes to virtual filesystem
- ✅ Manages model lifecycle
- ⚠️ Returns mock empty tensors from `forward()`
- ⚠️ Cannot extract real input/output shapes

**This is sufficient for**:
- Testing Dart <-> JavaScript interop
- Testing model loading flow
- Verifying initialization works

**This is NOT sufficient for**:
- Real inference
- Production use

## Next Steps

### Option 1: Add C++ Bindings (Complete Solution)
1. Modify `executor_runner.cpp` to expose functions via `EMSCRIPTEN_BINDINGS`
2. Rebuild Wasm with new bindings
3. Update JavaScript wrapper to call exposed C++ functions
4. Test end-to-end inference

### Option 2: Use Existing Wasm Runner (Quick Test)
1. Keep current mock implementation
2. Focus on Dart web plugin implementation
3. Verify Flutter -> JS -> Wasm communication works
4. Replace with real bindings later

## Testing

### Load HTML directly
```bash
cd /path/to/executorch_flutter/web
python3 -m http.server 8000
# Open http://localhost:8000/wasm/tmp/executor_runner.html
```

### Test JavaScript wrapper
```javascript
// In browser console
await window.ExecuTorchRunner.initialize();
const model = await window.ExecuTorchRunner.loadModel(new Uint8Array([...]));
console.log(model);  // Should show modelId
```

### Test from Flutter
```bash
cd /path/to/executorch_flutter/example
flutter run -d chrome
```

## Resources

- **Emscripten Bindings**: https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html
- **Emscripten FS API**: https://emscripten.org/docs/api_reference/Filesystem-API.html
- **ExecuTorch Wasm Example**: https://github.com/pytorch/executorch/tree/main/examples/wasm
- **Dart JS Interop**: https://dart.dev/web/js-interop

## Notes

- Models are stored in virtual filesystem at `/models/model_<id>.pte`
- Virtual FS is in-memory only (cleared on page reload)
- Wasm binary is ~2.4 MB (will compress to ~1 MB with gzip)
- All operations are async (returns Promises)
- Logging goes to browser console with `[ExecuTorch]` prefix

---

**Status**: JavaScript wrapper complete, C++ bindings needed for real inference
**Next**: Implement Dart web plugin with `dart:js_interop`
