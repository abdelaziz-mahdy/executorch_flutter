/// Web stub for platform camera controller
/// Real-time camera streaming is not supported on web in the same way
library;

import 'dart:async';
import 'dart:typed_data';
import 'camera_controller.dart';
import '../processors/camera_image_converter_stub.dart';

/// Stub implementation for web platform
class PlatformCameraController implements CameraController {
  PlatformCameraController({
    required this.converter,
    this.processingInterval = const Duration(milliseconds: 100),
  });

  final CameraImageConverter converter;
  final Duration processingInterval;

  @override
  Stream<Uint8List> get frameStream =>
      throw UnsupportedError('Camera streaming is not supported on web.');

  @override
  bool get isActive => false;

  @override
  Future<void> start() {
    throw UnsupportedError(
      'Real-time camera streaming is not supported on web. '
      'Use image picker for static image inference instead.',
    );
  }

  @override
  Future<void> stop() async {}

  @override
  Future<void> dispose() async {}
}
