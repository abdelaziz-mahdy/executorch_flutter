# Quickstart: Camera-to-Model Processor

**Feature**: Camera-to-Model Processor
**Date**: 2025-09-22
**Purpose**: Demonstrate real-time camera processing for ML inference

## Overview
This quickstart guide demonstrates how to use the camera-to-model processor for real-time machine learning inference on live camera feeds in the ExecuTorch Flutter package.

## Prerequisites
- ExecuTorch Flutter package with camera processor extension
- Camera-enabled device (Android/iOS)
- Example ML model compatible with image input
- Camera permissions configured in app manifest

## Quick Start Scenarios

### Scenario 1: Basic Real-Time Classification
**Goal**: Classify objects in real-time using device camera

**Setup**:
```dart
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:camera/camera.dart';

// Initialize camera processor with real-time configuration
final cameraProcessor = CameraProcessor(
  configuration: const CameraConfiguration(
    cameraDirection: CameraLensDirection.back,
    resolution: ResolutionPreset.medium,
    targetFrameRate: 30,
    processingFrequency: 5.0, // Process 5 frames per second
    enablePreview: true,
    previewSize: Size(300, 400),
  ),
);
```

**Usage**:
```dart
// Load model for inference
final model = await ExecutorchManager.instance.loadModel('assets/classification_model.pte');

// Start camera and processing
try {
  await cameraProcessor.startCamera();

  // Listen to processed frames
  cameraProcessor.frameStream.listen((tensorData) async {
    // Run inference on processed frame
    final result = await model.runInference(inputs: [tensorData]);

    if (result.status == InferenceStatus.success) {
      // Process classification results
      final classification = await postprocessor.postprocess(result.outputs!);
      print('Detected: ${classification.className} (${classification.confidence})');

      // Update UI with results
      await cameraProcessor.previewController.showProcessingOverlay({
        'class': classification.className,
        'confidence': classification.confidence,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
      });
    }
  });
} catch (e) {
  print('Camera processing failed: $e');
}
```

**Expected Output**:
```
Detected: golden retriever (0.89)
Detected: person (0.92)
Detected: car (0.87)
```

### Scenario 2: Performance-Optimized Processing
**Goal**: Optimize camera processing for battery life and thermal management

**Setup**:
```dart
// Create performance-aware camera processor
final optimizedProcessor = CameraProcessor(
  configuration: const CameraConfiguration(
    cameraDirection: CameraLensDirection.back,
    resolution: ResolutionPreset.medium,
    targetFrameRate: 30,
    processingFrequency: 2.0, // Reduced for efficiency
    enablePreview: true,
    previewSize: Size(224, 224), // Smaller preview
  ),
);

// Enable adaptive throttling
optimizedProcessor.processingController.enableThrottling(true);
```

**Usage**:
```dart
// Monitor and adjust performance automatically
optimizedProcessor.processingController.monitorPerformance();

// Get recommended processing rate for current device
final recommendedRate = await optimizedProcessor.processingController.getRecommendedRate();
await optimizedProcessor.processingController.adjustProcessingRate(recommendedRate);

print('Recommended processing rate: ${recommendedRate} fps');
print('Current throttling: ${optimizedProcessor.processingController.isThrottling}');

// Start optimized processing
await optimizedProcessor.startCamera();
```

**Expected Output**:
```
Recommended processing rate: 3.5 fps
Current throttling: false
Thermal state: normal
Battery optimization: active
```

### Scenario 3: Permission Handling and Error Recovery
**Goal**: Handle camera permissions and error states gracefully

**Setup**:
```dart
final permissionManager = CameraPermissionManager();
```

**Usage**:
```dart
// Check and request camera permissions
final permissionStatus = await permissionManager.checkPermissionStatus();

if (!permissionManager.isPermissionGranted) {
  // Show rationale if needed
  if (await permissionManager.shouldShowRationale()) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Camera Access Required'),
        content: Text('This app needs camera access for real-time ML processing.'),
        actions: [
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              final status = await permissionManager.requestCameraPermission();
              if (status == PermissionStatus.granted) {
                await startCameraProcessing();
              } else {
                handlePermissionDenied();
              }
            },
            child: Text('Grant Permission'),
          ),
        ],
      ),
    );
  } else {
    // Request permission directly
    final status = await permissionManager.requestCameraPermission();
    if (status != PermissionStatus.granted) {
      handlePermissionDenied();
    }
  }
}

void handlePermissionDenied() {
  if (permissionManager.permissionStatus == PermissionStatus.permanentlyDenied) {
    // Direct user to settings
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Permission Required'),
        content: Text('Please enable camera access in app settings.'),
        actions: [
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await permissionManager.openAppSettings();
            },
            child: Text('Open Settings'),
          ),
        ],
      ),
    );
  }
}
```

**Expected Output**:
```
Permission status: granted
Camera access: enabled
Processing started: true
```

### Scenario 4: Camera Switching and Lifecycle Management
**Goal**: Switch between cameras and manage processor lifecycle

**Setup**:
```dart
final dualCameraProcessor = CameraProcessor(
  configuration: CameraConfiguration(
    cameraDirection: CameraLensDirection.back,
    resolution: ResolutionPreset.high,
    targetFrameRate: 30,
    processingFrequency: 4.0,
    enablePreview: true,
    previewSize: Size(320, 240),
  ),
);
```

**Usage**:
```dart
// Start with rear camera
await dualCameraProcessor.startCamera();
print('Started with rear camera');

// Process for a while...
await Future.delayed(Duration(seconds: 5));

// Switch to front camera
try {
  await dualCameraProcessor.switchCamera(CameraLensDirection.front);
  print('Switched to front camera');
} catch (e) {
  print('Camera switch failed: $e');
}

// Pause processing temporarily
await dualCameraProcessor.pauseProcessing();
print('Processing paused - camera preview continues');

// Resume processing
await dualCameraProcessor.resumeProcessing();
print('Processing resumed');

// Proper cleanup when done
await dualCameraProcessor.stopCamera();
print('Camera stopped and resources cleaned up');
```

**Expected Output**:
```
Started with rear camera
Switched to front camera
Processing paused - camera preview continues
Processing resumed
Camera stopped and resources cleaned up
```

### Scenario 5: Custom Frame Processing with Overlays
**Goal**: Implement custom frame processing with visual result overlays

**Setup**:
```dart
// Custom processor with overlay support
final overlayProcessor = CameraProcessor(
  configuration: const CameraConfiguration(
    cameraDirection: CameraLensDirection.back,
    resolution: ResolutionPreset.high,
    targetFrameRate: 30,
    processingFrequency: 8.0,
    enablePreview: true,
    previewSize: Size(400, 300),
  ),
);
```

**Usage**:
```dart
// Start camera with overlay support
await overlayProcessor.startCamera();

// Process frames with custom logic
overlayProcessor.frameStream.listen((tensorData) async {
  // Run multiple models or custom processing
  final detectionResult = await detectionModel.runInference(inputs: [tensorData]);
  final classificationResult = await classificationModel.runInference(inputs: [tensorData]);

  // Combine results for rich overlay
  final overlayData = {
    'detections': extractDetections(detectionResult.outputs!),
    'classification': extractClassification(classificationResult.outputs!),
    'processing_time': DateTime.now().millisecondsSinceEpoch,
    'confidence_threshold': 0.7,
  };

  // Display rich overlay with multiple results
  await overlayProcessor.previewController.showProcessingOverlay(overlayData);
});

// Capture processed frame when needed
final capturedFrame = await overlayProcessor.previewController.capturePreviewFrame();
if (capturedFrame != null) {
  print('Captured frame with overlay: ${capturedFrame.length} bytes');
}
```

**Expected Output**:
```
Overlay updated: 3 detections, 1 classification
Processing time: 45ms
Captured frame with overlay: 245760 bytes
```

## Performance Validation

### Real-Time Processing Test
**Goal**: Verify camera processor meets real-time performance requirements

```dart
// Performance monitoring setup
final stopwatch = Stopwatch();
int frameCount = 0;
double totalProcessingTime = 0;

cameraProcessor.frameStream.listen((tensorData) async {
  stopwatch.start();

  // Process frame
  final result = await model.runInference(inputs: [tensorData]);

  stopwatch.stop();
  frameCount++;
  totalProcessingTime += stopwatch.elapsedMilliseconds;

  final avgProcessingTime = totalProcessingTime / frameCount;
  print('Frame $frameCount - Processing time: ${stopwatch.elapsedMilliseconds}ms, Average: ${avgProcessingTime.toStringAsFixed(1)}ms');

  // Validate performance targets
  assert(stopwatch.elapsedMilliseconds < 100, 'Frame processing too slow');
  assert(avgProcessingTime < 80, 'Average processing time exceeds target');

  stopwatch.reset();
});
```

### Memory Usage Test
**Goal**: Verify bounded memory usage during continuous processing

```dart
// Memory monitoring (pseudo-code for demonstration)
final initialMemory = await getMemoryUsage();
int maxMemoryUsage = initialMemory;

// Run continuous processing for extended period
Timer.periodic(Duration(seconds: 10), (timer) async {
  final currentMemory = await getMemoryUsage();
  maxMemoryUsage = math.max(maxMemoryUsage, currentMemory);

  final memoryIncrease = currentMemory - initialMemory;
  print('Memory usage: ${currentMemory ~/ (1024*1024)}MB (+${memoryIncrease ~/ (1024*1024)}MB)');

  // Validate memory bounds
  assert(memoryIncrease < 50 * 1024 * 1024, 'Memory usage too high: ${memoryIncrease ~/ (1024*1024)}MB');

  if (timer.tick >= 60) { // Stop after 10 minutes
    timer.cancel();
    print('Memory test completed. Max usage: ${maxMemoryUsage ~/ (1024*1024)}MB');
  }
});
```

## Integration Test Checklist

### ✅ Core Functionality
- [ ] Camera processor initializes with valid configuration
- [ ] Camera permissions are requested and handled correctly
- [ ] Real-time frame processing works at target frequency
- [ ] Camera preview displays correctly with processing overlays
- [ ] Camera switching between front/rear works smoothly
- [ ] Processing can be paused and resumed without issues

### ✅ Performance Requirements
- [ ] Frame-to-tensor conversion completes within 100ms
- [ ] Processing maintains target frequency without dropping frames
- [ ] Memory usage stays bounded during extended processing
- [ ] CPU usage allows UI to remain responsive
- [ ] Battery drain is reasonable for continuous processing

### ✅ Error Handling
- [ ] Permission denial handled gracefully with user guidance
- [ ] Camera hardware failures result in clear error messages
- [ ] Processing errors don't crash the application
- [ ] Resource cleanup works properly on all error conditions
- [ ] Network or model loading failures are handled correctly

### ✅ Platform Compatibility
- [ ] Identical behavior on Android and iOS devices
- [ ] Different camera hardware configurations supported
- [ ] Various screen sizes and orientations handled
- [ ] Performance optimization adapts to device capabilities
- [ ] Platform-specific permission flows work correctly

### ✅ Integration Requirements
- [ ] Camera processor extends ExecuTorchPreprocessor correctly
- [ ] Output tensors compatible with existing model inference
- [ ] Integration with existing error handling patterns
- [ ] Example app demonstrates all major camera features
- [ ] Documentation covers all camera processor capabilities

## Next Steps
1. Test camera processor with various device types and capabilities
2. Optimize frame processing performance for target devices
3. Implement comprehensive error handling and recovery
4. Add camera processor demonstrations to example app
5. Create performance benchmarks for different configurations

## Success Criteria
- ✅ Real-time camera processing at 30fps on mid-range devices
- ✅ <100ms frame-to-tensor conversion consistently achieved
- ✅ Smooth camera switching and lifecycle management
- ✅ Clear permission handling with user-friendly guidance
- ✅ Bounded memory usage during extended processing sessions
- ✅ Consistent cross-platform behavior on Android and iOS