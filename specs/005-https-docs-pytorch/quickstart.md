# Quickstart: macOS Platform Support

**Feature**: Add macOS support to ExecuTorch Flutter Plugin
**Target Audience**: Developers testing macOS support
**Estimated Time**: 15 minutes

## Prerequisites

- macOS 12+ (Monterey or later)
- Xcode 15+
- Flutter 3.0+ with macOS support enabled
- Command Line Tools installed (`xcode-select --install`)

## Quick Validation Test

This quickstart validates that the ExecuTorch Flutter plugin works on macOS with the same API as iOS.

### Step 1: Enable macOS Support

```bash
cd example
flutter config --enable-macos-desktop
flutter create --platforms=macos .
```

**Expected Output**:
```
Creating macOS application...
  macos/Runner.xcworkspace
  macos/Runner/Configs/AppInfo.xcconfig
  ...
✓ macOS application created
```

### Step 2: Verify Plugin Registration

Check that the plugin is registered for macOS:

```bash
cat macos/Flutter/GeneratedPluginRegistrant.swift
```

**Expected Content**:
```swift
import executorch_flutter

func RegisterGeneratedPlugins(registry: FlutterPluginRegistry) {
  ExecutorchFlutterPlugin.register(with: registry.registrar(forPlugin: "ExecutorchFlutterPlugin"))
}
```

### Step 3: Add Test Model

```bash
# Copy a test model to macOS app assets
cp assets/models/mobilenet_v3_small_xnnpack.pte \
   macos/Runner/Assets/
```

### Step 4: Run Example App on macOS

```bash
flutter run -d macos
```

**Expected Output**:
```
Launching lib/main.dart on macOS in debug mode...
Building macOS application...
✓ Built build/macos/Build/Products/Debug/Runner.app
...
Flutter run key commands.
```

### Step 5: Test Model Loading

In the running app:

1. Click "Load Model"
2. Select `mobilenet_v3_small_xnnpack.pte`
3. Verify status shows "Model loaded successfully"

**Expected Behavior**:
```
✅ Model loads without errors
✅ Model ID displayed (e.g., "executorch_model_a1b2c3d4")
✅ Status: "Ready"
```

### Step 6: Test Inference

1. Click "Pick Image"
2. Select a test image (e.g., cat, dog, car)
3. Click "Run Inference"
4. Verify results displayed

**Expected Behavior**:
```
✅ Inference completes in <100ms
✅ Top-5 predictions displayed with confidence scores
✅ No errors or crashes
```

### Step 7: Verify Platform Parity

Run the same test on iOS (if available):

```bash
flutter run -d ios
```

**Validation**:
- ✅ Same model loads successfully
- ✅ Same inference results (within floating-point tolerance)
- ✅ Similar performance characteristics

## Automated Test Script

Create `test_macos_support.sh`:

```bash
#!/bin/bash
set -e

echo "🧪 Testing macOS Support for ExecuTorch Flutter Plugin"

# Test 1: Plugin builds for macOS
echo "📦 Test 1: Building plugin for macOS..."
cd example
flutter build macos --debug
echo "✅ Build successful"

# Test 2: Model loading test
echo "🔧 Test 2: Running model loading test..."
flutter test integration_test/model_loading_test.dart -d macos
echo "✅ Model loading test passed"

# Test 3: Inference test
echo "🧠 Test 3: Running inference test..."
flutter test integration_test/inference_test.dart -d macos
echo "✅ Inference test passed"

# Test 4: Platform parity test
echo "⚖️  Test 4: Testing iOS/macOS parity..."
flutter test integration_test/platform_parity_test.dart
echo "✅ Platform parity confirmed"

echo "🎉 All tests passed! macOS support is working."
```

Run the test suite:

```bash
chmod +x test_macos_support.sh
./test_macos_support.sh
```

## Expected Test Results

### Build Test
```
✓ macOS build completed
✓ No compilation errors
✓ All frameworks linked correctly
✓ Bundle created: Runner.app
```

### Model Loading Test
```
✓ Model file accessible
✓ ExecuTorch Module.load() succeeds
✓ Model ID returned
✓ Model state transitions: loading → ready
```

### Inference Test
```
✓ Input tensors accepted
✓ Model.forward() executes
✓ Output tensors returned
✓ Results match expected format
✓ Performance within acceptable range
```

### Platform Parity Test
```
✓ Same model works on iOS and macOS
✓ Inference results identical (±0.001)
✓ API calls behave identically
✓ Error handling consistent
```

## Troubleshooting

### Issue: "Plugin not found"
```
Solution: Run `flutter pub get` and rebuild
```

### Issue: "Model file not found"
```
Solution: Verify model path and file permissions
```

### Issue: "Architecture mismatch"
```
Solution: Ensure building for correct architecture (arm64 or x86_64)
```

### Issue: "Framework not found"
```
Solution: Clean build and re-fetch dependencies:
  flutter clean
  flutter pub get
  flutter build macos
```

## Performance Benchmarks

Run performance tests on different Macs:

```bash
flutter test integration_test/performance_test.dart -d macos
```

**Expected Results**:

| Mac Model | Architecture | Model Load | Inference |
|-----------|-------------|------------|-----------|
| M1 Mac | ARM64 | ~100ms | ~30ms |
| M2 Mac | ARM64 | ~80ms | ~25ms |
| Intel Mac | x86_64 | ~150ms | ~45ms |

## Success Criteria

✅ Plugin builds for macOS without errors
✅ Model loads successfully on macOS
✅ Inference executes and returns correct results
✅ API behaves identically to iOS
✅ Performance meets acceptable ranges
✅ No platform-specific crashes or errors

## Next Steps

After quickstart validation:

1. Test with multiple model types (YOLO, MobileNet, custom models)
2. Test on both Apple Silicon and Intel Macs
3. Verify memory management under load
4. Test with macOS-specific features (file dialogs, drag-drop)
5. Run full test suite on macOS

## Resources

- [ExecuTorch macOS Documentation](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)
- [Flutter macOS Plugin Guide](https://docs.flutter.dev/platform-integration/macos/c-interop)
- Example App: `example/lib/main.dart`
- Integration Tests: `test/integration_test/`

## Validation Checklist

Before considering macOS support complete:

- [ ] Plugin builds on macOS 12+
- [ ] Plugin builds on Apple Silicon
- [ ] Plugin builds on Intel Mac
- [ ] Model loading works
- [ ] Inference produces correct results
- [ ] API matches iOS behavior
- [ ] Performance is acceptable
- [ ] Memory management is sound
- [ ] No crashes or errors
- [ ] Tests pass on macOS
- [ ] Example app runs successfully
- [ ] Documentation updated

**Estimated Completion Time**: 15-20 minutes for basic validation

---

*This quickstart provides a fast path to verify macOS support is working correctly*
