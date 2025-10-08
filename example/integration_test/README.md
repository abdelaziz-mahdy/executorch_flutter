# Integration Tests

This directory contains integration tests for the ExecuTorch Flutter example app. These tests verify that all models load and run correctly on all supported platforms.

## Test Coverage

The integration tests verify:

### Model Loading
- ✅ MobileNet V3 Small (image classification)
- ✅ YOLO11n (object detection)
- ✅ YOLOv5n (object detection)
- ✅ YOLOv8n (object detection)

### Functionality
- ✅ ExecutorchManager initialization
- ✅ Model loading from file paths
- ✅ Inference execution with dummy input tensors
- ✅ Multiple concurrent models
- ✅ Model disposal and resource cleanup
- ✅ Model reload handling

## Prerequisites

### 1. Install Models

Before running integration tests, ensure all required models are available:

```bash
cd ../python
python3 setup_models.py
```

This will download and export all required models to `assets/models/`.

### 2. Platform Requirements

**macOS** (arm64 only - Apple Silicon):
- macOS 12.0+ (Monterey)
- No additional setup required

**iOS** (arm64 only - Physical device):
- iOS 13.0+ physical device connected
- ⚠️ iOS Simulator is NOT supported

**Android**:
- Android device or emulator (API 23+, arm64-v8a)
- Device/emulator must be running and connected

## Running Tests

### Run on All Available Platforms

Use the automated test runner script:

```bash
./scripts/run_integration_tests.sh
```

This script will:
1. Check that all required models exist
2. Detect available platforms/devices
3. Run integration tests on each available platform
4. Display a summary of results

### Run on Specific Platform

**macOS:**
```bash
flutter test integration_test/models_integration_test.dart -d macos
```

**iOS (with physical device connected):**
```bash
flutter test integration_test/models_integration_test.dart -d <device-id>
```

To get your iOS device ID:
```bash
flutter devices
```

**Android:**
```bash
flutter test integration_test/models_integration_test.dart -d <device-id>
```

## Test Output

Successful test output will show:
```
✓ ExecutorchManager should initialize successfully
✓ Should load MobileNet V3 model successfully
✓ Should load YOLO11n model successfully
✓ Should load YOLOv5n model successfully
✓ Should load YOLOv8n model successfully
✓ Should run inference on MobileNet V3 model
✓ Should run inference on YOLO11n model
✓ Should run inference on YOLOv5n model
✓ Should run inference on YOLOv8n model
✓ Should handle multiple models concurrently
✓ Should properly dispose models and free resources
✓ Should handle model reload correctly
```

## Current Test Status

**✅ macOS**: All tests passing (12/12)
**⚠️ iOS**: Requires physical device - Simulator NOT supported (ExecuTorch architecture limitation)
**⚠️ Android**: Build errors in plugin code (needs fixing separately from tests)

## Troubleshooting

### "Model file not found"
- Run `python3 setup_models.py` in the `python/` directory
- Verify models exist in `assets/models/`

### "No devices available"
- **iOS**: Connect a physical iOS device (Simulator NOT supported - ExecuTorch doesn't provide simulator-compatible libraries)
- **Android**: Emulator launches automatically, or connect a physical device
- **macOS**: Ensure you're on Apple Silicon (arm64)

### iOS Simulator "Unsupported Swift architecture"
- This is expected - ExecuTorch does not provide iOS Simulator-compatible binaries
- Use a physical iOS device for testing

### Android build failures
- Check for Kotlin compilation errors in the plugin code
- Ensure Android SDK and NDK are properly configured

### Tests timing out
- Some models may take time to load on first run
- Ensure device has sufficient resources
- Check device logs for errors

## Adding New Tests

To add tests for new models:

1. Add the model file to `assets/models/`
2. Add model to `REQUIRED_MODELS` in `run_integration_tests.sh`
3. Add test cases in `models_integration_test.dart`:
   - Model loading test
   - Inference execution test
4. Run tests to verify

## CI/CD Integration

To integrate these tests in CI/CD:

```bash
# Ensure models are available
cd python && python3 setup_models.py

# Run tests
cd example
./scripts/run_integration_tests.sh
```

For platform-specific CI runners, run tests individually on each platform.
