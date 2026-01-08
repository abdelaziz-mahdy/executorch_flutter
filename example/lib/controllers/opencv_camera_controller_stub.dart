/// Web stub for OpenCV camera controller
/// OpenCV is not available on web platform
library;

import 'dart:async';
import 'dart:typed_data';
import 'camera_controller.dart';

/// Stub implementation that throws UnsupportedError on web
class OpenCVCameraController implements CameraController {
  OpenCVCameraController({
    this.deviceId = 0,
    this.processingInterval = const Duration(milliseconds: 100),
  });

  final int deviceId;
  final Duration processingInterval;

  @override
  Stream<Uint8List> get frameStream =>
      throw UnsupportedError('OpenCV camera is not available on web.');

  @override
  bool get isActive => false;

  @override
  Future<void> start() {
    throw UnsupportedError(
      'OpenCV camera is not available on web. '
      'Use platform camera instead.',
    );
  }

  @override
  Future<void> stop() async {}

  @override
  Future<void> dispose() async {}
}
