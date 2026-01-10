import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/widgets.dart';

/// Camera operating mode
enum CameraMode {
  /// Real-time streaming - continuous frame capture for live inference
  live,

  /// Capture mode - preview with manual picture taking
  capture,
}

/// Base interface for camera controllers
/// Manages camera lifecycle independently of UI rebuilds
abstract class CameraController {
  /// Stream of camera frames as JPEG bytes (for live mode)
  Stream<Uint8List> get frameStream;

  /// Whether the camera is currently active
  bool get isActive;

  /// Current camera mode
  CameraMode get mode;

  /// Whether the camera preview is ready
  bool get isPreviewReady;

  /// Start the camera in the specified mode
  Future<void> start({CameraMode mode = CameraMode.live});

  /// Stop capturing frames (for live mode) or stop preview (for capture mode)
  Future<void> stop();

  /// Take a single picture (for capture mode)
  /// Returns JPEG bytes of the captured image
  Future<Uint8List?> takePicture();

  /// Build the camera preview widget
  Widget buildPreview();

  /// Dispose camera resources
  Future<void> dispose();
}
