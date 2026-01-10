/// Web stub for OpenCV camera controller
/// OpenCV is not available on web platform
library;

import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/widgets.dart';
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
  CameraMode get mode => CameraMode.capture;

  @override
  bool get isPreviewReady => false;

  @override
  Future<void> start({CameraMode mode = CameraMode.live}) {
    throw UnsupportedError(
      'OpenCV camera is not available on web. '
      'Use platform camera instead.',
    );
  }

  @override
  Future<Uint8List?> takePicture() {
    throw UnsupportedError('OpenCV camera is not available on web.');
  }

  @override
  Widget buildPreview() {
    throw UnsupportedError('OpenCV camera is not available on web.');
  }

  @override
  Future<void> stop() async {}

  @override
  Future<void> dispose() async {}
}
