/// API Contract: Camera-to-Model Processor
///
/// This file defines the contracts for camera-based processor implementation.
/// These contracts must be fulfilled by the implementation for real-time camera processing.

import 'dart:typed_data';
import 'package:camera/camera.dart';

// Note: These are contract definitions only - actual types come from dependencies
abstract class TensorData {
  List<int?> get shape;
  TensorType get dataType;
  Uint8List get data;
  String? get name;
}

abstract class ExecuTorchPreprocessor<T> {
  Future<List<TensorData>> preprocess(T input, {ModelMetadata? metadata});
  String get inputTypeName;
  bool validateInput(T input);
}

enum TensorType { float32, int32, int8, uint8 }
enum CameraLensDirection { front, back, external }
enum ResolutionPreset { low, medium, high, veryHigh, ultraHigh, max }

/// Contract: Camera configuration for processor
abstract class CameraConfiguration {
  /// Camera direction (front/rear)
  ///
  /// Contract requirements:
  /// - Must be valid CameraLensDirection
  /// - Device must support selected camera direction
  CameraLensDirection get cameraDirection;

  /// Camera resolution setting
  ///
  /// Contract requirements:
  /// - Must be supported by device camera
  /// - Higher resolutions may impact performance
  ResolutionPreset get resolution;

  /// Target camera frame rate
  ///
  /// Contract requirements:
  /// - Must be positive integer
  /// - Must be supported by camera hardware
  /// - Typical values: 15, 30, 60 fps
  int get targetFrameRate;

  /// ML processing frequency (frames per second)
  ///
  /// Contract requirements:
  /// - Must be positive number
  /// - Cannot exceed targetFrameRate
  /// - Lower values improve performance
  double get processingFrequency;

  /// Whether to enable camera preview display
  ///
  /// Contract requirements:
  /// - Must be boolean value
  /// - Preview may impact processing performance
  bool get enablePreview;

  /// Preview display size
  ///
  /// Contract requirements:
  /// - Must have positive width and height
  /// - Should maintain camera aspect ratio
  /// - Only relevant if enablePreview is true
  Size get previewSize;
}

/// Contract: Camera processor for real-time ML inference
abstract class CameraProcessor extends ExecuTorchPreprocessor<CameraImage> {
  /// Configuration for camera and processing settings
  ///
  /// Contract requirements:
  /// - Must return valid CameraConfiguration instance
  /// - Must be consistent across calls
  /// - Configuration should not change during processing
  CameraConfiguration get configuration;

  /// Current camera processing state
  ///
  /// Contract requirements:
  /// - Must reflect actual processing status
  /// - Updated immediately when state changes
  bool get isProcessing;

  /// Start camera and begin frame processing
  ///
  /// Contract requirements:
  /// - Must request camera permissions if not granted
  /// - Must initialize camera with configuration settings
  /// - Must throw CameraException if camera unavailable
  /// - Must throw PermissionException if permissions denied
  /// - Must return quickly (non-blocking operation)
  Future<void> startCamera();

  /// Stop camera and cleanup resources
  ///
  /// Contract requirements:
  /// - Must stop camera preview and frame capture
  /// - Must cleanup all resources and streams
  /// - Must be safe to call multiple times
  /// - Must complete cleanup even if camera in error state
  Future<void> stopCamera();

  /// Temporarily pause frame processing
  ///
  /// Contract requirements:
  /// - Camera remains active but processing stops
  /// - Must be resumable with resumeProcessing()
  /// - UI preview continues if enabled
  /// - Must not throw exceptions
  Future<void> pauseProcessing();

  /// Resume frame processing after pause
  ///
  /// Contract requirements:
  /// - Resumes processing at configured frequency
  /// - Must handle case where camera state changed during pause
  /// - Must restore all processing settings
  /// - Must not throw exceptions
  Future<void> resumeProcessing();

  /// Switch between front and rear cameras
  ///
  /// Contract requirements:
  /// - Must stop current camera cleanly
  /// - Must start new camera with same configuration
  /// - Must maintain processing state (paused/active)
  /// - Must throw CameraException if requested camera unavailable
  Future<void> switchCamera(CameraLensDirection direction);

  /// Process camera frame to tensor data
  ///
  /// Contract requirements:
  /// - Must validate camera frame format and quality
  /// - Must convert frame to model-compatible tensor format
  /// - Must complete within performance targets (<100ms)
  /// - Must handle different camera image formats
  /// - Must throw ProcessingException on conversion failure
  @override
  Future<List<TensorData>> preprocess(CameraImage input, {ModelMetadata? metadata});

  /// Validate camera frame input
  ///
  /// Contract requirements:
  /// - Must return false for null or invalid frames
  /// - Must check frame format compatibility
  /// - Must verify frame dimensions are reasonable
  /// - Must not throw exceptions during validation
  @override
  bool validateInput(CameraImage input);

  /// Input type identifier
  ///
  /// Contract requirements:
  /// - Must return "Camera Image Frame"
  /// - Must be consistent across instances
  @override
  String get inputTypeName;
}

/// Contract: Camera permission management
abstract class CameraPermissionManager {
  /// Current camera permission status
  ///
  /// Contract requirements:
  /// - Must reflect actual system permission state
  /// - Updated when permission status changes
  PermissionStatus get permissionStatus;

  /// Convenience getter for permission granted state
  ///
  /// Contract requirements:
  /// - Must return true only if permission fully granted
  /// - Must be consistent with permissionStatus
  bool get isPermissionGranted;

  /// Request camera permission from user
  ///
  /// Contract requirements:
  /// - Must show system permission dialog
  /// - Must handle user grant/deny/dismiss actions
  /// - Must update permissionStatus after user action
  /// - Must not show dialog if permission already granted
  Future<PermissionStatus> requestCameraPermission();

  /// Check current permission status without requesting
  ///
  /// Contract requirements:
  /// - Must query actual system permission state
  /// - Must not trigger permission dialogs
  /// - Must update internal permissionStatus
  /// - Must handle all permission states correctly
  Future<PermissionStatus> checkPermissionStatus();

  /// Open app settings for manual permission management
  ///
  /// Contract requirements:
  /// - Must navigate to app-specific settings screen
  /// - Must return true if settings opened successfully
  /// - Must return false if navigation failed
  /// - Must handle platform differences (Android/iOS)
  Future<bool> openAppSettings();

  /// Check if permission rationale should be shown
  ///
  /// Contract requirements:
  /// - Must return true if user previously denied permission
  /// - Must return false if permission never requested
  /// - Must return false if permission permanently denied
  /// - Platform-specific behavior handling required
  Future<bool> shouldShowRationale();
}

/// Contract: Frame processing performance controller
abstract class ProcessingController {
  /// Current actual processing frequency
  ///
  /// Contract requirements:
  /// - Must reflect measured processing rate
  /// - Updated in real-time during processing
  /// - Value may differ from target due to throttling
  double get currentProcessingRate;

  /// Target desired processing frequency
  ///
  /// Contract requirements:
  /// - Must be the configured target rate
  /// - May be adjusted by adaptive throttling
  /// - Must be positive value
  double get targetProcessingRate;

  /// Whether adaptive throttling is currently active
  ///
  /// Contract requirements:
  /// - Must indicate if processing is being throttled
  /// - Updated when throttling state changes
  /// - True when currentRate < targetRate due to throttling
  bool get isThrottling;

  /// Adjust target processing frequency
  ///
  /// Contract requirements:
  /// - Must update targetProcessingRate immediately
  /// - Must validate rate is positive and reasonable
  /// - Must apply changes to active processing
  /// - Must not exceed camera frame rate
  Future<void> adjustProcessingRate(double targetRate);

  /// Enable or disable adaptive performance throttling
  ///
  /// Contract requirements:
  /// - Must immediately apply throttling setting
  /// - When disabled, must process at full target rate
  /// - When enabled, must monitor performance and adjust
  /// - Must handle device capability detection
  void enableThrottling(bool enabled);

  /// Get recommended processing rate for current device
  ///
  /// Contract requirements:
  /// - Must analyze device capabilities and current state
  /// - Must consider thermal conditions and battery level
  /// - Must return safe processing rate for sustained operation
  /// - Must not return rate higher than camera capabilities
  Future<double> getRecommendedRate();

  /// Monitor processing performance metrics
  ///
  /// Contract requirements:
  /// - Must track actual processing times and rates
  /// - Must monitor device thermal and battery state
  /// - Must update throttling decisions based on metrics
  /// - Must not impact processing performance significantly
  void monitorPerformance();
}

/// Contract: Camera preview display controller
abstract class CameraPreviewController {
  /// Current preview display size
  ///
  /// Contract requirements:
  /// - Must reflect actual preview widget dimensions
  /// - Must maintain camera aspect ratio when possible
  /// - Updated when preview size changes
  Size get previewSize;

  /// Camera preview aspect ratio
  ///
  /// Contract requirements:
  /// - Must match camera output aspect ratio
  /// - Used for preview layout calculations
  /// - Must be positive value
  double get aspectRatio;

  /// Whether processing result overlays are enabled
  ///
  /// Contract requirements:
  /// - Must indicate current overlay display state
  /// - Can be toggled during active processing
  bool get overlayEnabled;

  /// Update preview display dimensions
  ///
  /// Contract requirements:
  /// - Must adjust preview to new size constraints
  /// - Must maintain aspect ratio when possible
  /// - Must not interrupt camera processing
  /// - Must handle orientation changes gracefully
  Future<void> updatePreviewSize(Size newSize);

  /// Display ML processing results as overlay
  ///
  /// Contract requirements:
  /// - Must render results over camera preview
  /// - Must handle different result formats
  /// - Must not block camera processing
  /// - Must clear previous results before displaying new ones
  Future<void> showProcessingOverlay(Map<String, dynamic> results);

  /// Hide processing result overlays
  ///
  /// Contract requirements:
  /// - Must remove all overlay elements
  /// - Must not affect camera preview display
  /// - Must be safe to call when no overlays active
  Future<void> hideProcessingOverlay();

  /// Capture current preview frame as image
  ///
  /// Contract requirements:
  /// - Must capture actual preview content
  /// - Must return image in standard format
  /// - Must not interrupt ongoing processing
  /// - Must handle capture failures gracefully
  Future<Uint8List?> capturePreviewFrame();
}

/// Contract: Exception types for camera processing

/// Exception thrown when camera operations fail
abstract class CameraException implements Exception {
  String get message;
  CameraError get errorType;
}

/// Exception thrown when camera permissions are denied
abstract class PermissionException implements Exception {
  String get message;
  PermissionStatus get permissionStatus;
}

/// Exception thrown when frame processing fails
abstract class ProcessingException implements Exception {
  String get message;
  Map<String, dynamic>? get details;
}

/// Camera error types
enum CameraError {
  cameraNotAvailable,
  cameraAlreadyInUse,
  cameraInitializationFailed,
  unsupportedResolution,
  hardwareFailure
}

/// Permission status types
enum PermissionStatus {
  denied,
  granted,
  restricted,
  permanentlyDenied
}