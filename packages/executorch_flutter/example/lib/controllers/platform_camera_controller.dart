import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart' as camera_pkg;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'camera_controller.dart';
import '../processors/camera_image_converter.dart';

/// Platform camera controller using the camera package.
/// Supports both live streaming and capture modes on all platforms.
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
  bool _isPreviewReady = false;
  DateTime? _lastProcessedTime;
  int _sensorOrientation = 0;
  CameraMode _mode = CameraMode.live;

  @override
  Stream<Uint8List> get frameStream => _frameController.stream;

  @override
  bool get isActive => _isActive;

  @override
  CameraMode get mode => _mode;

  @override
  bool get isPreviewReady => _isPreviewReady;

  @override
  Future<void> start({CameraMode mode = CameraMode.live}) async {
    if (_isActive) return;

    _mode = mode;

    try {
      debugPrint(
        '📱 PlatformCameraController: Initializing camera (mode: $mode)',
      );

      // Get available cameras
      final cameras = await camera_pkg.availableCameras();
      if (cameras.isEmpty) {
        throw Exception('No cameras available');
      }

      // Get sensor orientation for Android (iOS is always 0)
      final camera = cameras.first;
      _sensorOrientation = !kIsWeb && Platform.isAndroid
          ? camera.sensorOrientation
          : 0;
      debugPrint('📱 Camera sensor orientation: $_sensorOrientation');

      // Determine image format based on platform
      camera_pkg.ImageFormatGroup? imageFormatGroup;
      if (!kIsWeb) {
        imageFormatGroup = Platform.isAndroid
            ? camera_pkg.ImageFormatGroup.yuv420
            : camera_pkg.ImageFormatGroup.bgra8888;
      }

      // Initialize camera controller
      _cameraController = camera_pkg.CameraController(
        camera,
        camera_pkg.ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: imageFormatGroup,
      );

      await _cameraController!.initialize();
      _isPreviewReady = true;

      // Start image stream only in live mode
      if (mode == CameraMode.live) {
        await _cameraController!.startImageStream(_onImage);
      }

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
  Future<Uint8List?> takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      debugPrint('❌ PlatformCameraController: Camera not initialized');
      return null;
    }

    try {
      debugPrint('📸 PlatformCameraController: Taking picture');
      final xFile = await _cameraController!.takePicture();
      final bytes = await xFile.readAsBytes();
      debugPrint(
        '✅ PlatformCameraController: Picture taken (${bytes.length} bytes)',
      );
      return bytes;
    } catch (e) {
      debugPrint('❌ PlatformCameraController: Failed to take picture: $e');
      return null;
    }
  }

  @override
  Widget buildPreview() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return camera_pkg.CameraPreview(_cameraController!);
  }

  @override
  Future<void> stop() async {
    if (!_isActive) return;

    debugPrint('🛑 PlatformCameraController: Stopping camera');

    try {
      if (_mode == CameraMode.live) {
        await _cameraController?.stopImageStream();
      }
    } catch (e) {
      debugPrint('⚠️ Error stopping image stream: $e');
    }

    _isActive = false;
    _isPreviewReady = false;
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
