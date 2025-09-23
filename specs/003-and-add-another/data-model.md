# Data Model: Camera-to-Model Processor

**Feature**: Camera-to-Model Processor
**Date**: 2025-09-22
**Status**: Complete

## Core Entities

### 1. CameraProcessor
**Purpose**: Main processor class that handles real-time camera feed processing for ML inference
**Type**: Concrete class extending ExecuTorchPreprocessor<CameraImage>
**Lifecycle**: Instantiated per camera session, manages camera lifecycle

**Key Properties**:
- `cameraConfiguration`: CameraConfiguration instance for camera settings
- `isProcessing`: bool indicating current processing state
- `currentCamera`: CameraController instance
- `processingFrequency`: double for frames per second processing rate

**Key Methods**:
- `startCamera()`: Initialize and start camera with configured settings
- `stopCamera()`: Stop camera and cleanup resources
- `pauseProcessing()`: Temporarily pause frame processing
- `resumeProcessing()`: Resume frame processing
- `switchCamera()`: Switch between front/rear cameras
- `preprocess(CameraImage input)`: Convert camera frame to TensorData

**Validation Rules**:
- Camera must be available and permissions granted before starting
- Processing frequency must be between 1-30 fps
- Camera configuration must be compatible with device capabilities

### 2. CameraConfiguration
**Purpose**: Configuration object for camera settings and processing parameters
**Type**: Immutable data class with const constructor
**Lifecycle**: Created once per processor instance

**Key Properties**:
- `cameraDirection`: CameraLensDirection (front/rear)
- `resolution`: ResolutionPreset for camera resolution
- `targetFrameRate`: int for desired camera frame rate
- `processingFrequency`: double for ML processing frequency
- `enablePreview`: bool for preview display
- `previewSize`: Size for preview display dimensions

**Validation Rules**:
- Processing frequency cannot exceed target frame rate
- Resolution must be supported by device camera
- Preview size must be positive if preview enabled

### 3. FrameConverter
**Purpose**: Utility class for converting camera frames to tensor format
**Type**: Static utility class
**Lifecycle**: Stateless utility functions

**Key Methods**:
- `convertToTensor(CameraImage image, {ImageFormat? format})`: Convert frame to TensorData
- `resizeFrame(CameraImage image, Size targetSize)`: Resize camera frame
- `normalizeFrame(CameraImage image, NormalizationConfig config)`: Apply normalization
- `extractBytesFromFrame(CameraImage image)`: Extract raw bytes for processing

**Validation Rules**:
- Input image must have valid format and dimensions
- Target size must be positive dimensions
- Normalization parameters must be valid ranges

### 4. CameraPermissionManager
**Purpose**: Handles camera permission requests and status management
**Type**: Singleton service class
**Lifecycle**: Application-scoped singleton

**Key Properties**:
- `permissionStatus`: PermissionStatus current status
- `isPermissionGranted`: bool convenience getter
- `hasRequestedPermission`: bool tracking request history

**Key Methods**:
- `requestCameraPermission()`: Request camera access from user
- `checkPermissionStatus()`: Check current permission state
- `openAppSettings()`: Navigate to app settings for manual permission
- `shouldShowRationale()`: Check if permission rationale should be shown

**State Transitions**:
1. **Initial**: Permission status unknown
2. **Requesting**: Permission request in progress
3. **Granted**: Permission granted, camera access allowed
4. **Denied**: Permission denied, show rationale or settings
5. **PermanentlyDenied**: User denied with "don't ask again"

### 5. ProcessingController
**Purpose**: Manages adaptive frame processing and performance optimization
**Type**: Service class with performance monitoring
**Lifecycle**: Tied to camera processor lifecycle

**Key Properties**:
- `currentProcessingRate`: double actual processing frequency
- `targetProcessingRate`: double desired processing frequency
- `deviceCapabilities`: DeviceCapabilities performance metrics
- `isThrottling`: bool indicating if throttling is active
- `thermalState`: ThermalState device thermal condition

**Key Methods**:
- `adjustProcessingRate(double targetRate)`: Dynamically adjust processing frequency
- `monitorPerformance()`: Track processing performance metrics
- `enableThrottling(bool enabled)`: Enable/disable adaptive throttling
- `getRecommendedRate()`: Get recommended rate for current device state

**Validation Rules**:
- Processing rate must be positive and <= camera frame rate
- Throttling parameters must be within acceptable ranges
- Performance monitoring must not impact processing performance

### 6. CameraPreviewController
**Purpose**: Manages camera preview display and processing result overlays
**Type**: Widget controller class
**Lifecycle**: Tied to preview widget lifecycle

**Key Properties**:
- `previewSize`: Size of preview display
- `aspectRatio`: double for preview aspect ratio
- `overlayEnabled`: bool for processing result overlays
- `overlayData`: Map<String, dynamic> current overlay information

**Key Methods**:
- `updatePreviewSize(Size newSize)`: Adjust preview dimensions
- `showProcessingOverlay(Map<String, dynamic> results)`: Display ML results
- `hideProcessingOverlay()`: Remove result overlays
- `capturePreviewFrame()`: Capture current preview frame

**Validation Rules**:
- Preview size must match camera output aspect ratio
- Overlay data must be serializable for display
- Preview updates must not block camera processing

### 7. CameraStreamHandler
**Purpose**: Handles camera image stream processing and frame distribution
**Type**: Stream controller class
**Lifecycle**: Active during camera processing session

**Key Properties**:
- `frameStream`: Stream<CameraImage> camera frame stream
- `processedFrameStream`: Stream<TensorData> processed tensor stream
- `isStreaming`: bool indicating stream activity
- `bufferSize`: int for frame buffer management

**Key Methods**:
- `startFrameStream()`: Begin camera frame streaming
- `stopFrameStream()`: End frame streaming and cleanup
- `processNextFrame()`: Process next available frame
- `skipFrames(int count)`: Skip frames for throttling

**Validation Rules**:
- Frame stream must be active before processing
- Buffer size must be reasonable for memory constraints
- Frame skipping must maintain minimum processing rate

## Entity Relationships

```
CameraProcessor
├── cameraConfiguration: CameraConfiguration
├── permissionManager: CameraPermissionManager
├── processingController: ProcessingController
├── frameConverter: FrameConverter
├── streamHandler: CameraStreamHandler
└── previewController: CameraPreviewController (optional)

CameraConfiguration
├── validation rules
└── device compatibility checks

ProcessingController
├── monitors: DeviceCapabilities
├── adjusts: processingFrequency
└── reports: PerformanceMetrics

CameraStreamHandler
├── input: Camera.imageStream
├── processing: FrameConverter
└── output: Stream<TensorData>
```

## State Transitions

### Camera Processor Flow
1. **Initialized**: Configuration set, ready to start
2. **Permission Check**: Checking/requesting camera permissions
3. **Starting**: Initializing camera with configuration
4. **Active**: Camera running, processing frames
5. **Paused**: Camera active but processing suspended
6. **Stopping**: Cleaning up camera resources
7. **Stopped**: Camera stopped, resources released

### Frame Processing Flow
1. **Frame Capture**: Camera provides new frame
2. **Frame Validation**: Check frame format and quality
3. **Frame Conversion**: Convert to tensor format
4. **Tensor Output**: Provide tensor for ML inference
5. **Result Overlay**: Optional display of ML results

### Permission Flow
1. **Unknown**: Initial permission state
2. **Requesting**: User permission dialog shown
3. **Granted**: Permission approved, proceed with camera
4. **Denied**: Permission rejected, show rationale
5. **Settings**: Direct user to app settings for manual permission

## Implementation Constraints

### Performance Requirements
- Frame processing must maintain target frame rate
- Memory usage must be bounded with frame buffering
- CPU usage must not cause UI blocking
- Battery impact must be reasonable for continuous use

### Platform Compatibility
- Must work identically on Android and iOS
- Handle device-specific camera capabilities
- Support different camera resolutions and formats
- Graceful degradation on lower-end devices

### Integration Requirements
- Must extend existing ExecuTorchPreprocessor interface
- Compatible with existing TensorData format
- Integrates with existing model inference pipeline
- Follows established error handling patterns

### Testing Requirements
- Unit tests for all core processing logic
- Integration tests with simulated camera feeds
- Permission handling tests across different states
- Performance tests on various device capabilities