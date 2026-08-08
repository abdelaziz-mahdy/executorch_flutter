import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:dartcv4/dartcv.dart' as cv;
import 'package:permission_handler/permission_handler.dart';
import 'package:universal_platform/universal_platform.dart';
import 'camera_controller.dart';

/// OpenCV-based camera controller for desktop platforms (macOS, Windows, Linux)
/// Supports both live streaming and capture modes
class OpenCVCameraController implements CameraController {
  OpenCVCameraController({
    this.deviceId = 0,
    this.processingInterval = const Duration(milliseconds: 100),
  });

  final int deviceId;
  final Duration processingInterval;

  cv.VideoCapture? _capture;
  Timer? _frameTimer;
  final StreamController<Uint8List> _frameController =
      StreamController<Uint8List>.broadcast();

  bool _isActive = false;
  bool _isProcessing = false;
  bool _isPreviewReady = false;
  cv.Mat? _currentFrame;
  CameraMode _mode = CameraMode.live;

  /// Returns the appropriate OpenCV API preference for the current platform
  int _getPlatformApiPreference() {
    // Let OpenCV auto-detect the best backend for the platform
    // Specifying explicit backends (CAP_ANDROID, CAP_AVFOUNDATION, etc.)
    // can cause format compatibility issues on some devices
    return cv.CAP_ANY;
  }

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
        '🎥 OpenCVCameraController: Initializing camera (mode: $mode)',
      );

      // Request camera permission on mobile platforms
      if (UniversalPlatform.isAndroid || UniversalPlatform.isIOS) {
        debugPrint(
          '🔐 OpenCVCameraController: Requesting camera permission...',
        );
        final status = await Permission.camera.request();
        if (!status.isGranted) {
          throw Exception(
            'Camera permission denied. Please grant camera permission in app settings.',
          );
        }
        debugPrint('✅ OpenCVCameraController: Camera permission granted');
      }

      // Initialize VideoCapture with platform-appropriate API
      final int apiPreference = _getPlatformApiPreference();
      debugPrint(
        '🎥 OpenCVCameraController: Using API preference: $apiPreference',
      );

      _capture = cv.VideoCapture.fromDevice(
        deviceId,
        apiPreference: apiPreference,
      );

      if (_capture == null || !_capture!.isOpened) {
        throw Exception('Failed to open camera device $deviceId');
      }

      // Set camera properties
      _capture!.set(cv.CAP_PROP_FRAME_WIDTH, 640);
      _capture!.set(cv.CAP_PROP_FRAME_HEIGHT, 480);
      _capture!.set(cv.CAP_PROP_FPS, 30);

      _isPreviewReady = true;

      // Start frame capture timer only in live mode
      if (mode == CameraMode.live) {
        debugPrint(
          '⏰ OpenCVCameraController: Starting timer (${processingInterval.inMilliseconds}ms)',
        );
        _frameTimer = Timer.periodic(
          processingInterval,
          (_) => _captureFrame(),
        );
      }

      _isActive = true;
      debugPrint('✅ OpenCVCameraController: Camera started successfully');
    } catch (e) {
      debugPrint('❌ OpenCVCameraController: Failed to start camera: $e');
      await stop();
      rethrow;
    }
  }

  Future<void> _captureFrame() async {
    if (_isProcessing || _capture == null || !_capture!.isOpened) {
      return;
    }

    _isProcessing = true;

    try {
      // Read frame
      final (success, frame) = _capture!.read();

      if (!success || frame.isEmpty) {
        debugPrint('⚠️ OpenCVCameraController: Failed to read frame');
        _isProcessing = false;
        return;
      }

      // Store for display
      _currentFrame?.dispose();
      _currentFrame = frame.clone();

      // Encode to JPEG
      final (encodeSuccess, jpegBytes) = await cv.imencodeAsync('.jpg', frame);

      if (encodeSuccess && jpegBytes.isNotEmpty) {
        // Emit to stream (non-blocking)
        _frameController.add(jpegBytes);
      }

      // Cleanup
      frame.dispose();
    } catch (e) {
      debugPrint('❌ OpenCVCameraController: Frame capture error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  @override
  Future<Uint8List?> takePicture() async {
    if (_capture == null || !_capture!.isOpened) {
      debugPrint('❌ OpenCVCameraController: Camera not initialized');
      return null;
    }

    try {
      debugPrint('📸 OpenCVCameraController: Taking picture');

      // Read frame
      final (success, frame) = _capture!.read();

      if (!success || frame.isEmpty) {
        debugPrint('❌ OpenCVCameraController: Failed to read frame');
        return null;
      }

      // Encode to JPEG
      final (encodeSuccess, jpegBytes) = await cv.imencodeAsync('.jpg', frame);
      frame.dispose();

      if (encodeSuccess && jpegBytes.isNotEmpty) {
        debugPrint(
          '✅ OpenCVCameraController: Picture taken (${jpegBytes.length} bytes)',
        );
        return jpegBytes;
      }

      return null;
    } catch (e) {
      debugPrint('❌ OpenCVCameraController: Failed to take picture: $e');
      return null;
    }
  }

  @override
  Widget buildPreview() {
    // OpenCV doesn't have a native preview widget, so we show the latest frame
    // In a real implementation, you might use a custom painter or texture
    if (_currentFrame == null || !_isPreviewReady) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('OpenCV Camera Preview'),
          ],
        ),
      );
    }

    // For now, return a placeholder since OpenCV doesn't have Flutter widget integration
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.videocam, size: 64),
          SizedBox(height: 16),
          Text('OpenCV Camera Active'),
          Text('(Preview not available)', style: TextStyle(fontSize: 12)),
        ],
      ),
    );
  }

  @override
  Future<void> stop() async {
    if (!_isActive) return;

    debugPrint('🛑 OpenCVCameraController: Stopping camera');

    _frameTimer?.cancel();
    _frameTimer = null;

    _isActive = false;
    _isPreviewReady = false;
  }

  @override
  Future<void> dispose() async {
    debugPrint('🧹 OpenCVCameraController: Disposing');

    await stop();

    await _frameController.close();
    _currentFrame?.dispose();
    _capture?.release();
    _capture?.dispose();

    debugPrint('✅ OpenCVCameraController: Disposed successfully');
  }
}
