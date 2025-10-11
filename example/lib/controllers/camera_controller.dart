import 'dart:async';
import 'dart:typed_data';

/// Base interface for camera controllers
/// Manages camera lifecycle independently of UI rebuilds
abstract class CameraController {
  /// Stream of camera frames as JPEG bytes
  Stream<Uint8List> get frameStream;

  /// Whether the camera is currently active
  bool get isActive;

  /// Start capturing frames from camera
  Future<void> start();

  /// Stop capturing frames
  Future<void> stop();

  /// Dispose camera resources
  Future<void> dispose();
}
