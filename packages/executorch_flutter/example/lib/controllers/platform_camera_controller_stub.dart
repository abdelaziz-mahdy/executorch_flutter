/// Web implementation for platform camera controller
/// Supports capture mode on web, live streaming requires native platforms
library;

import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart' as camera_pkg;
import 'package:flutter/material.dart';
import 'camera_controller.dart';
import '../processors/camera_image_converter_stub.dart';

/// Web implementation using camera package's takePicture() functionality
/// Live streaming is not supported on web, but capture mode works
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
  bool _isPreviewReady = false;
  CameraMode _mode = CameraMode.capture;

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

    // Web only supports capture mode, not live streaming
    if (mode == CameraMode.live) {
      throw UnsupportedError(
        'Live camera streaming is not yet supported on web. '
        'Use capture mode instead.',
      );
    }

    _mode = mode;

    try {
      debugPrint('🌐 WebCameraController: Initializing camera (mode: $mode)');

      // Get available cameras
      final cameras = await camera_pkg.availableCameras();
      if (cameras.isEmpty) {
        throw Exception('No cameras available');
      }

      // Initialize camera controller
      _cameraController = camera_pkg.CameraController(
        cameras.first,
        camera_pkg.ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      _isPreviewReady = true;
      _isActive = true;

      debugPrint('✅ WebCameraController: Camera started successfully');
    } catch (e) {
      debugPrint('❌ WebCameraController: Failed to start camera: $e');
      await stop();
      rethrow;
    }
  }

  @override
  Future<Uint8List?> takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      debugPrint('❌ WebCameraController: Camera not initialized');
      return null;
    }

    try {
      debugPrint('📸 WebCameraController: Taking picture');
      final xFile = await _cameraController!.takePicture();
      final bytes = await xFile.readAsBytes();
      debugPrint('✅ WebCameraController: Picture taken (${bytes.length} bytes)');
      return bytes;
    } catch (e) {
      debugPrint('❌ WebCameraController: Failed to take picture: $e');
      return null;
    }
  }

  @override
  Widget buildPreview() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    return camera_pkg.CameraPreview(_cameraController!);
  }

  @override
  Future<void> stop() async {
    if (!_isActive) return;

    debugPrint('🛑 WebCameraController: Stopping camera');
    _isActive = false;
    _isPreviewReady = false;
  }

  @override
  Future<void> dispose() async {
    debugPrint('🧹 WebCameraController: Disposing');

    await stop();
    await _frameController.close();
    await _cameraController?.dispose();

    debugPrint('✅ WebCameraController: Disposed successfully');
  }
}
