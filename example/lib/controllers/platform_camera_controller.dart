import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart' as camera_pkg;
import 'package:flutter/foundation.dart';
import 'camera_controller.dart';
import '../processors/camera_image_converter.dart';

/// Platform camera controller for mobile (Android/iOS) using camera package
class PlatformCameraController implements CameraController {
  PlatformCameraController({
    required this.converter,
    this.processingInterval = const Duration(milliseconds: 100),
  });

  final CameraImageConverter converter;
  final Duration processingInterval;

  camera_pkg.CameraController? _cameraController;
  final StreamController<Uint8List> _frameController =
      StreamController<Uint8List>.broadcast();

  bool _isActive = false;
  bool _isProcessing = false;
  DateTime? _lastProcessedTime;
  int _sensorOrientation = 0;

  @override
  Stream<Uint8List> get frameStream => _frameController.stream;

  @override
  bool get isActive => _isActive;

  @override
  Future<void> start() async {
    if (_isActive) return;

    try {
      debugPrint('📱 PlatformCameraController: Initializing camera');

      // Get available cameras
      final cameras = await camera_pkg.availableCameras();
      if (cameras.isEmpty) {
        throw Exception('No cameras available');
      }

      // Get sensor orientation for Android (iOS is always 0)
      final camera = cameras.first;
      _sensorOrientation = Platform.isAndroid ? camera.sensorOrientation : 0;
      debugPrint('📱 Camera sensor orientation: $_sensorOrientation');

      // Initialize camera controller
      // iOS requires bgra8888, Android uses yuv420
      _cameraController = camera_pkg.CameraController(
        camera,
        camera_pkg.ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? camera_pkg.ImageFormatGroup.yuv420
            : camera_pkg.ImageFormatGroup.bgra8888,
      );

      await _cameraController!.initialize();

      // Start image stream
      await _cameraController!.startImageStream(_onImage);

      _isActive = true;
      debugPrint('✅ PlatformCameraController: Camera started successfully');
    } catch (e) {
      debugPrint('❌ PlatformCameraController: Failed to start camera: $e');
      await stop();
      rethrow;
    }
  }

  Future<void> _onImage(camera_pkg.CameraImage image) async {
    // Throttle processing based on interval
    final now = DateTime.now();
    if (_lastProcessedTime != null &&
        now.difference(_lastProcessedTime!) < processingInterval) {
      return;
    }

    if (_isProcessing) return;

    _isProcessing = true;
    _lastProcessedTime = now;

    try {
      // Convert camera image to JPEG bytes with sensor orientation
      final jpegBytes = await converter.convertToJpeg(
        image,
        sensorOrientation: _sensorOrientation,
      );

      if (jpegBytes.isNotEmpty) {
        _frameController.add(jpegBytes);
      }
    } catch (e) {
      debugPrint('❌ PlatformCameraController: Frame processing error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  @override
  Future<void> stop() async {
    if (!_isActive) return;

    debugPrint('🛑 PlatformCameraController: Stopping camera');

    try {
      await _cameraController?.stopImageStream();
    } catch (e) {
      debugPrint('⚠️ Error stopping image stream: $e');
    }

    _isActive = false;
  }

  @override
  Future<void> dispose() async {
    debugPrint('🧹 PlatformCameraController: Disposing');

    await stop();

    await _frameController.close();
    await _cameraController?.dispose();

    debugPrint('✅ PlatformCameraController: Disposed successfully');
  }
}
