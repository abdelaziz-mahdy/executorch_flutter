# Web Platform Support Design Document

**Package**: executorch_flutter
**Feature**: Web platform support via ExecuTorch WebAssembly
**Status**: Design Phase
**Author**: AI Assistant
**Date**: 2026-01-04

## Executive Summary

This document outlines the design for adding web platform support to the `executorch_flutter` package using ExecuTorch's WebAssembly (Wasm) build. The implementation will maintain API parity with existing platforms (Android, iOS, macOS) while adapting to web-specific constraints.

**Key Decisions**:
- ✅ Include pre-built ExecuTorch Wasm binary in package
- ✅ Load models via HTTP requests (Flutter asset bundle)
- ✅ Use `dart:js_interop` for JavaScript communication
- ✅ Maintain identical Dart API surface (`load()`, `forward()`, `dispose()`)

## Background

### Current State

The package currently supports:
- **Android**: Kotlin + ExecuTorch AAR 1.0.1
- **iOS**: Swift + ExecuTorch XCFrameworks (SPM 1.0.1)
- **macOS**: Swift + ExecuTorch XCFrameworks (SPM 1.0.1)

All platforms use **Pigeon** for type-safe platform channel communication.

### ExecuTorch Wasm Capabilities

ExecuTorch provides Wasm support via Emscripten:
- **Binary**: `executor_runner.js` + `executor_runner.wasm`
- **Runtime**: Browser (Chrome, Firefox, Safari) and Node.js
- **Model Loading**: Virtual filesystem (embedded) or HTTP requests
- **API**: JavaScript wrapper around C++ ExecuTorch runtime

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────┐
│          Flutter Dart Layer                 │
│  ┌─────────────────────────────────────┐   │
│  │   ExecuTorchModel (Unified API)     │   │
│  │   - load(Uint8List)                 │   │
│  │   - forward(List<TensorData>)       │   │
│  │   - dispose()                        │   │
│  └─────────────────────────────────────┘   │
│                   │                          │
│         ┌─────────┴─────────┐               │
│         │                    │               │
│    Mobile/Desktop          Web               │
│    (Pigeon)           (JS Interop)          │
│         │                    │               │
└─────────┼────────────────────┼───────────────┘
          │                    │
          ▼                    ▼
   ┌─────────────┐    ┌──────────────────┐
   │   Native    │    │  JavaScript +    │
   │  Platform   │    │  Wasm Runtime    │
   │  Channels   │    │                  │
   └─────────────┘    └──────────────────┘
          │                    │
          ▼                    ▼
   ┌─────────────┐    ┌──────────────────┐
   │ ExecuTorch  │    │ ExecuTorch Wasm  │
   │ Native Libs │    │  (executor_runner│
   │             │    │   .js + .wasm)   │
   └─────────────┘    └──────────────────┘
```

### Web Plugin Structure

**New Files**:
```
executorch_flutter/
├── lib/
│   ├── src/
│   │   ├── executorch_model.dart              # Updated with conditional web import
│   │   ├── executorch_model_web.dart          # Web-specific implementation
│   │   └── web/
│   │       ├── executorch_web_plugin.dart     # Web plugin registration
│   │       ├── wasm_module_loader.dart        # Wasm binary loader
│   │       └── js_interop.dart                # JS <-> Dart bridge
│   └── executorch_flutter_web.dart            # Web platform export
├── web/
│   ├── wasm/
│   │   ├── executor_runner.js                 # Pre-built ExecuTorch JS wrapper
│   │   └── executor_runner.wasm               # Pre-built ExecuTorch Wasm binary
│   └── index.html                             # Web entry point (example)
└── WEB_PLATFORM_DESIGN.md                     # This document
```

## Implementation Strategy

### 1. Wasm Binary Integration

**Approach**: Ship pre-built ExecuTorch Wasm binaries with the package.

**Build Process** (run once by maintainers):
```bash
# In ExecuTorch repository
cd executorch
source .ci/scripts/setup-emscripten.sh
./install_executorch.sh --clean

mkdir -p cmake-out-wasm
cd cmake-out-wasm
emcmake cmake -DEXECUTORCH_PAL_DEFAULT=posix ..
cmake --build . -j32 --target executor_runner

# Copy outputs to Flutter package
cp executor_runner.js /path/to/executorch_flutter/web/wasm/
cp executor_runner.wasm /path/to/executorch_flutter/web/wasm/
```

**Packaging**:
- Include `executor_runner.js` and `executor_runner.wasm` in `web/wasm/`
- Total size: ~2-5 MB (acceptable for web package)
- Users get working Wasm binary out-of-the-box

**Versioning**:
- Track ExecuTorch version in `WEB_PLATFORM_DESIGN.md`
- Rebuild when upgrading ExecuTorch dependency
- Document build date and commit hash

### 2. JavaScript Interop Layer

**Technology**: Use `dart:js_interop` (Flutter 3.16+) for modern, type-safe JS communication.

**Key Components**:

#### a. `js_interop.dart` - Low-level JS bindings

```dart
import 'dart:js_interop';
import 'dart:typed_data';

/// JavaScript representation of ExecuTorch Module
@JS('Module')
extension type ExecuTorchModule(JSObject _) implements JSObject {
  /// Load Wasm module
  external static JSPromise<ExecuTorchModule> loadWasm(JSString wasmPath);

  /// Load model from bytes
  external JSPromise<JSNumber> loadModel(JSUint8Array modelData);

  /// Run inference
  external JSPromise<JSArray<JSAny>> forward(JSNumber modelId, JSArray<JSObject> inputs);

  /// Dispose model
  external void disposeModel(JSNumber modelId);
}

/// JavaScript representation of Tensor
@JS()
extension type JSTensor(JSObject _) implements JSObject {
  external factory JSTensor({
    JSArray<JSNumber> shape,
    JSString dataType,
    JSUint8Array data,
    JSString? name,
  });

  external JSArray<JSNumber> get shape;
  external JSString get dataType;
  external JSUint8Array get data;
  external JSString? get name;
}
```

#### b. `wasm_module_loader.dart` - Wasm binary loader

```dart
import 'dart:async';
import 'dart:js_interop';
import 'package:flutter/services.dart' show rootBundle;
import 'js_interop.dart';

class WasmModuleLoader {
  static ExecuTorchModule? _module;
  static bool _isInitialized = false;

  /// Initialize Wasm module (call once on first model load)
  static Future<void> initialize() async {
    if (_isInitialized) return;

    // Load Wasm binary from package assets
    final wasmBytes = await rootBundle.load('packages/executorch_flutter/web/wasm/executor_runner.wasm');

    // Initialize Emscripten runtime
    await _initializeEmscripten();

    // Load Wasm module
    _module = await ExecuTorchModule.loadWasm('executor_runner.wasm'.toJS).toDart;
    _isInitialized = true;
  }

  static Future<void> _initializeEmscripten() async {
    // Load executor_runner.js script
    final scriptUrl = 'packages/executorch_flutter/web/wasm/executor_runner.js';
    await _loadScript(scriptUrl);

    // Wait for Emscripten runtime to initialize
    await _waitForEmscriptenReady();
  }

  static Future<void> _loadScript(String url) async {
    final completer = Completer<void>();

    final script = document.createElement('script'.toJS) as HTMLScriptElement;
    script.src = url.toJS;
    script.onload = (JSAny event) {
      completer.complete();
    }.toJS;
    script.onerror = (JSAny event, JSString source, JSNumber lineno, JSNumber colno, JSAny error) {
      completer.completeError('Failed to load script: $url');
    }.toJS;

    document.head!.appendChild(script);
    return completer.future;
  }

  static Future<void> _waitForEmscriptenReady() async {
    // Poll for Module.ready or similar Emscripten initialization flag
    while (!_isEmscriptenReady()) {
      await Future.delayed(Duration(milliseconds: 100));
    }
  }

  static bool _isEmscriptenReady() {
    // Check if Module object exists and is ready
    return globalThis.has('Module') &&
           (globalThis['Module'] as JSObject).has('ready');
  }

  static ExecuTorchModule get module {
    if (!_isInitialized || _module == null) {
      throw StateError('Wasm module not initialized. Call WasmModuleLoader.initialize() first.');
    }
    return _module!;
  }
}
```

#### c. `executorch_model_web.dart` - Web implementation

```dart
import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';
import 'package:executorch_flutter/src/executorch_errors.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';
import 'wasm_module_loader.dart';
import 'js_interop.dart';

class ExecuTorchModelWeb {
  ExecuTorchModelWeb._({
    required this.modelId,
    required this.inputShapes,
    required this.outputShapes,
  });

  final String modelId;
  final List<List<int>> inputShapes;
  final List<List<int>> outputShapes;

  int? _nativeModelId;

  /// Load model from bytes
  static Future<ExecuTorchModelWeb> load(Uint8List modelData) async {
    try {
      // Initialize Wasm module if not already done
      await WasmModuleLoader.initialize();

      // Convert Uint8List to JS Uint8Array
      final jsModelData = modelData.toJS;

      // Load model in Wasm
      final nativeModelId = await WasmModuleLoader.module
          .loadModel(jsModelData)
          .toDart;

      // Generate unique model ID
      final modelId = 'web_${DateTime.now().millisecondsSinceEpoch}_${nativeModelId.toDartInt}';

      // Get model metadata (shapes)
      final metadata = await _getModelMetadata(nativeModelId.toDartInt);

      return ExecuTorchModelWeb._(
        modelId: modelId,
        inputShapes: metadata['inputShapes'] as List<List<int>>,
        outputShapes: metadata['outputShapes'] as List<List<int>>,
      ).._nativeModelId = nativeModelId.toDartInt;

    } catch (e) {
      throw ExecuTorchModelException('Failed to load model: $e');
    }
  }

  /// Run inference
  Future<List<TensorData>> forward(List<TensorData> inputs) async {
    if (_nativeModelId == null) {
      throw ExecuTorchModelException('Model has been disposed');
    }

    try {
      // Convert Dart TensorData to JS tensors
      final jsTensors = inputs.map(_tensorToJS).toList().toJS;

      // Run inference
      final jsOutputs = await WasmModuleLoader.module
          .forward(_nativeModelId!.toJS, jsTensors)
          .toDart;

      // Convert JS tensors back to Dart TensorData
      final outputs = <TensorData>[];
      for (int i = 0; i < jsOutputs.length; i++) {
        final jsTensor = jsOutputs[i] as JSTensor;
        outputs.add(_tensorFromJS(jsTensor));
      }

      return outputs;

    } catch (e) {
      throw ExecuTorchInferenceException('Inference failed: $e');
    }
  }

  /// Dispose model
  Future<void> dispose() async {
    if (_nativeModelId == null) return;

    try {
      WasmModuleLoader.module.disposeModel(_nativeModelId!.toJS);
      _nativeModelId = null;
    } catch (e) {
      throw ExecuTorchModelException('Failed to dispose model: $e');
    }
  }

  // Helper: Get model metadata from Wasm
  static Future<Map<String, dynamic>> _getModelMetadata(int nativeModelId) async {
    // Call Wasm function to get input/output shapes
    // For now, return empty shapes (TODO: expose metadata API in Wasm)
    return {
      'inputShapes': <List<int>>[],
      'outputShapes': <List<int>>[],
    };
  }

  // Helper: Convert Dart TensorData to JS tensor
  JSTensor _tensorToJS(TensorData tensor) {
    return JSTensor(
      shape: tensor.shape.map((s) => (s ?? 0).toJS).toList().toJS,
      dataType: _tensorTypeToString(tensor.dataType).toJS,
      data: tensor.data.toJS,
      name: tensor.name?.toJS,
    );
  }

  // Helper: Convert JS tensor to Dart TensorData
  TensorData _tensorFromJS(JSTensor jsTensor) {
    return TensorData(
      shape: jsTensor.shape.toDart.map((js) => (js as JSNumber).toDartInt).toList().cast<int?>(),
      dataType: _stringToTensorType(jsTensor.dataType.toDart),
      data: jsTensor.data.toDart.buffer.asUint8List(),
      name: jsTensor.name?.toDart,
    );
  }

  String _tensorTypeToString(TensorType type) {
    switch (type) {
      case TensorType.float32:
        return 'float32';
      case TensorType.int32:
        return 'int32';
      case TensorType.int8:
        return 'int8';
      case TensorType.uint8:
        return 'uint8';
      default:
        throw ExecuTorchValidationException('Unsupported tensor type: $type');
    }
  }

  TensorType _stringToTensorType(String type) {
    switch (type) {
      case 'float32':
        return TensorType.float32;
      case 'int32':
        return TensorType.int32;
      case 'int8':
        return TensorType.int8;
      case 'uint8':
        return TensorType.uint8;
      default:
        throw ExecuTorchValidationException('Unsupported tensor type: $type');
    }
  }
}
```

### 3. Model Loading Strategy

**HTTP Request Approach** (Recommended):

```dart
import 'package:flutter/services.dart' show rootBundle;

// 1. Add model to pubspec.yaml assets
//    flutter:
//      assets:
//        - assets/models/

// 2. Load model via rootBundle (works on web via HTTP)
final modelBytes = await rootBundle.load('assets/models/model.pte');

// 3. Create model instance (identical to other platforms)
final model = await ExecuTorchModel.load(
  modelBytes.buffer.asUint8List(),
);

// 4. Run inference (identical API)
final outputs = await model.forward([inputTensor]);

// 5. Dispose (identical API)
await model.dispose();
```

**Key Points**:
- ✅ No changes needed to existing Dart API
- ✅ Works with Flutter asset bundle (automatic HTTP on web)
- ✅ Same user experience across all platforms
- ✅ Models can be cached by browser

### 4. Platform Detection and Conditional Imports

**Update** `lib/src/executorch_model.dart`:

```dart
import 'executorch_model_interface.dart'
    if (dart.library.io) 'executorch_model_native.dart'
    if (dart.library.js_interop) 'executorch_model_web.dart';

/// ExecuTorch model for on-device inference
///
/// Supports Android, iOS, macOS, and Web platforms
class ExecuTorchModel {
  // Factory constructor delegates to platform-specific implementation
  static Future<ExecuTorchModel> load(Uint8List modelData) {
    return ExecuTorchModelImpl.load(modelData);
  }

  // Rest of API remains unchanged...
}
```

**Create** `lib/src/executorch_model_interface.dart`:

```dart
/// Platform-agnostic interface for ExecuTorchModel
abstract class ExecuTorchModelImpl {
  static Future<ExecuTorchModel> load(Uint8List modelData) {
    throw UnimplementedError('Platform not supported');
  }

  Future<List<TensorData>> forward(List<TensorData> inputs);
  Future<void> dispose();
}
```

**Rename existing implementation** to `lib/src/executorch_model_native.dart`:

```dart
// Current implementation for Android, iOS, macOS
class ExecuTorchModelNative extends ExecuTorchModelImpl {
  // Existing Pigeon-based implementation...
}
```

### 5. Wasm Module JavaScript API

ExecuTorch's `executor_runner.js` provides a JavaScript interface. We need to verify the exposed API and potentially wrap it.

**Expected Wasm API** (based on ExecuTorch docs):

```javascript
// Emscripten Module object
Module = {
  // Load model from virtual filesystem
  loadModel: function(modelPath) { ... },

  // Run inference
  forward: function(modelId, inputs) { ... },

  // Dispose model
  disposeModel: function(modelId) { ... },

  // Filesystem operations
  FS: {
    writeFile: function(path, data) { ... },
    readFile: function(path) { ... },
    unlink: function(path) { ... },
  }
};
```

**Our Approach**:
1. Write model bytes to Emscripten virtual filesystem
2. Call `Module.loadModel()` with virtual path
3. Run inference with `Module.forward()`
4. Clean up with `Module.disposeModel()` + `FS.unlink()`

**Wrapper Implementation** (in `executor_runner_wrapper.js`, bundled with package):

```javascript
// Wrapper around Emscripten Module for easier Dart interop
class ExecuTorchRunner {
  constructor(module) {
    this.module = module;
    this.nextModelId = 0;
  }

  async loadModel(modelBytes) {
    const modelId = this.nextModelId++;
    const modelPath = `/models/model_${modelId}.pte`;

    // Write bytes to virtual filesystem
    this.module.FS.writeFile(modelPath, new Uint8Array(modelBytes));

    // Load model
    this.module.loadModel(modelPath);

    return modelId;
  }

  async forward(modelId, inputs) {
    // Convert inputs to Wasm-compatible format
    const wasmInputs = inputs.map(tensor => ({
      shape: Array.from(tensor.shape),
      dataType: tensor.dataType,
      data: new Uint8Array(tensor.data),
    }));

    // Run inference
    const outputs = this.module.forward(modelId, wasmInputs);

    return outputs;
  }

  disposeModel(modelId) {
    const modelPath = `/models/model_${modelId}.pte`;
    this.module.disposeModel(modelId);
    this.module.FS.unlink(modelPath);
  }
}

// Export for Dart access
window.ExecuTorchRunner = ExecuTorchRunner;
```

## Memory Management

### Web Platform Constraints

1. **JavaScript Garbage Collection**: No manual memory control
2. **Wasm Linear Memory**: Fixed-size memory, can grow dynamically
3. **Browser Limits**: Typically 2-4 GB per tab

### Strategy

1. **Explicit Disposal**: User calls `dispose()` (same as other platforms)
2. **Wasm Memory Cleanup**: Free Wasm-allocated buffers on dispose
3. **Model Limit**: Recommend max 2-3 models simultaneously (same as mobile)
4. **No Automatic Cleanup**: Consistent with package philosophy

### Implementation

```dart
// In ExecuTorchModelWeb
Future<void> dispose() async {
  if (_nativeModelId == null) return;

  try {
    // 1. Dispose model in Wasm (frees C++ memory)
    WasmModuleLoader.module.disposeModel(_nativeModelId!.toJS);

    // 2. Clean up virtual filesystem (removes .pte file)
    // (handled by executor_runner_wrapper.js)

    // 3. Mark model as disposed
    _nativeModelId = null;

  } catch (e) {
    throw ExecuTorchModelException('Failed to dispose model: $e');
  }
}
```

## Browser Compatibility

### Target Browsers

| Browser | Version | Status | Notes |
|---------|---------|--------|-------|
| Chrome | 90+ | ✅ Full support | Best performance |
| Firefox | 88+ | ✅ Full support | Good performance |
| Safari | 14+ | ⚠️ Partial support | Slower Wasm execution |
| Edge | 90+ | ✅ Full support | Chromium-based |

### Required Web Features

- **WebAssembly**: All modern browsers (2017+)
- **WebAssembly SIMD**: Optional but improves performance (Chrome 91+, Firefox 89+)
- **SharedArrayBuffer**: Not required for single-threaded execution
- **Web Workers**: Optional for background inference (future enhancement)

### Performance Considerations

**Expected Performance** (vs Native):
- **Loading**: 1.5-2x slower (HTTP + Wasm compilation)
- **Inference**: 2-4x slower (interpreted Wasm vs native)
- **Memory**: 1.2-1.5x higher overhead (JS + Wasm)

**Optimizations**:
- Enable Wasm SIMD when available
- Use WebAssembly streaming compilation
- Cache compiled Wasm modules in IndexedDB (future)

## Testing Strategy

### 1. Unit Tests

**File**: `test/executorch_model_web_test.dart`

```dart
@TestOn('browser')
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

void main() {
  group('ExecuTorchModel Web', () {
    test('loads model from bytes', () async {
      final modelBytes = await loadTestModel();
      final model = await ExecuTorchModel.load(modelBytes);

      expect(model.modelId, isNotEmpty);
      await model.dispose();
    });

    test('runs inference', () async {
      final model = await loadTestModel();
      final input = createTestTensor();

      final outputs = await model.forward([input]);

      expect(outputs, isNotEmpty);
      await model.dispose();
    });

    test('disposes model', () async {
      final model = await loadTestModel();
      await model.dispose();

      expect(
        () => model.forward([createTestTensor()]),
        throwsA(isA<ExecuTorchModelException>()),
      );
    });
  });
}
```

### 2. Integration Tests

**File**: `example/integration_test/web_platform_test.dart`

```dart
@TestOn('browser')
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter_example/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Web Platform Integration', () {
    testWidgets('loads and runs MobileNet on web', (tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Select MobileNet model
      await tester.tap(find.text('MobileNetV3'));
      await tester.pumpAndSettle();

      // Load model
      await tester.tap(find.text('Load Model'));
      await tester.pumpAndSettle(Duration(seconds: 5));

      // Verify model loaded
      expect(find.text('Model loaded'), findsOneWidget);

      // Run inference
      await tester.tap(find.text('Run Inference'));
      await tester.pumpAndSettle(Duration(seconds: 2));

      // Verify results
      expect(find.textContaining('Class:'), findsWidgets);
    });
  });
}
```

### 3. Browser Testing

**Script**: `example/scripts/test_web.sh`

```bash
#!/bin/bash
set -e

echo "Testing on Chrome..."
flutter test integration_test/web_platform_test.dart \
  --platform chrome \
  --dart-define=FLUTTER_WEB_USE_SKIA=true

echo "Testing on Firefox..."
flutter test integration_test/web_platform_test.dart \
  --platform firefox

echo "Testing on Safari..."
flutter test integration_test/web_platform_test.dart \
  --platform safari
```

## Known Limitations

### 1. Performance

- **Inference Speed**: 2-4x slower than native (Wasm overhead)
- **Model Loading**: 1.5-2x slower (network + compilation)
- **Memory Usage**: 1.2-1.5x higher (JS + Wasm overhead)

### 2. Browser Support

- **Safari**: Slower Wasm execution than Chrome/Firefox
- **iOS Safari**: Limited memory (may struggle with large models)
- **Mobile Browsers**: Slower performance than native apps

### 3. Feature Parity

- **No GPU Acceleration**: Wasm runs on CPU only (no WebGL/WebGPU integration yet)
- **No SIMD Universally**: Not all browsers support Wasm SIMD
- **No Threading**: Single-threaded execution (no Web Workers support initially)

### 4. Model Size

- **Large Models**: May hit browser memory limits (2-4 GB)
- **Network Loading**: Initial load slower for large models over HTTP
- **Caching**: No automatic caching (users must implement)

## Migration Path

### For Package Users

**No API changes required**:

```dart
// Same code works on Android, iOS, macOS, and Web
final modelBytes = await rootBundle.load('assets/models/model.pte');
final model = await ExecuTorchModel.load(modelBytes);
final outputs = await model.forward([input]);
await model.dispose();
```

**Only pubspec.yaml update needed**:

```yaml
dependencies:
  executorch_flutter: ^0.1.0  # Version with web support
```

### For Package Maintainers

**Build Wasm binaries**:
```bash
./scripts/build_wasm.sh  # New script to build ExecuTorch Wasm
```

**Update documentation**:
- README: Add web platform to supported platforms
- CHANGELOG: Document web support
- Example app: Test web deployment

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Phase 1: Setup** | Build Wasm, set up project structure | 1-2 days |
| **Phase 2: Core Implementation** | JS interop, model loading, inference | 3-4 days |
| **Phase 3: Testing** | Unit tests, browser testing, debugging | 2-3 days |
| **Phase 4: Documentation** | README, example app, migration guide | 1-2 days |
| **Phase 5: Polish** | Performance optimization, bug fixes | 1-2 days |
| **Total** | | **8-13 days** |

## Success Criteria

- ✅ Same Dart API works on all platforms (Android, iOS, macOS, Web)
- ✅ Example app runs on web with MobileNet and YOLO models
- ✅ Tests pass in Chrome, Firefox, and Safari
- ✅ Documentation updated with web-specific notes
- ✅ Performance within 2-4x of native (acceptable for web)
- ✅ Package size < 10 MB (with Wasm binaries)

## Future Enhancements

1. **Web Workers**: Offload inference to background threads
2. **WebGPU**: GPU acceleration when browser support matures
3. **IndexedDB Caching**: Cache compiled Wasm modules
4. **Streaming Models**: Load models progressively (chunk by chunk)
5. **Service Worker**: Enable offline inference

## References

- [ExecuTorch Wasm Documentation](https://github.com/pytorch/executorch/tree/main/examples/wasm)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [Flutter Web Platform Plugins](https://docs.flutter.dev/platform-integration/web/building-a-plugin-for-web)
- [Dart JS Interop](https://dart.dev/web/js-interop)

---

**Next Steps**:
1. Review and approve this design
2. Build ExecuTorch Wasm binaries
3. Implement web plugin following this architecture
4. Test across browsers
5. Update documentation and publish

