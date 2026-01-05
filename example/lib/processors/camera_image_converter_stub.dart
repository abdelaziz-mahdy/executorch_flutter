/// Web stub for camera image converters
/// Camera image conversion requires dart:io which is not available on web
library;

import 'dart:typed_data';
import 'package:camera/camera.dart';

/// Abstract interface for converting CameraImage to JPEG bytes
abstract class CameraImageConverter {
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
    int? sensorOrientation,
  });
}

/// Web stub for ImageLib camera converter
class ImageLibCameraConverter implements CameraImageConverter {
  @override
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
    int? sensorOrientation,
  }) {
    throw UnsupportedError(
      'Camera image conversion is not supported on web. '
      'Use image picker instead.',
    );
  }
}

/// Web stub for OpenCV camera converter
class OpenCVCameraConverter implements CameraImageConverter {
  @override
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
    int? sensorOrientation,
  }) {
    throw UnsupportedError(
      'OpenCV camera conversion is not available on web.',
    );
  }
}
