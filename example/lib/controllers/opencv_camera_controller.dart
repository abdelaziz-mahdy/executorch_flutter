import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'camera_controller.dart';

/// OpenCV-based camera controller for desktop platforms (macOS, Windows, Linux)
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
  cv.Mat? _currentFrame;

  @override
  Stream<Uint8List> get frameStream => _frameController.stream;

  @override
  bool get isActive => _isActive;

  @override
  Future<void> start() async {
    if (_isActive) return;

    try {
      debugPrint('🎥 OpenCVCameraController: Initializing camera');

      // Initialize VideoCapture
      _capture = cv.VideoCapture.fromDevice(
        deviceId,
        apiPreference: cv.CAP_AVFOUNDATION,
      );

      if (_capture == null || !_capture!.isOpened) {
        throw Exception('Failed to open camera device $deviceId');
      }

      // Set camera properties
      _capture!.set(cv.CAP_PROP_FRAME_WIDTH, 640);
      _capture!.set(cv.CAP_PROP_FRAME_HEIGHT, 480);
      _capture!.set(cv.CAP_PROP_FPS, 30);

      // Start frame capture timer
      debugPrint('⏰ OpenCVCameraController: Starting timer (${processingInterval.inMilliseconds}ms)');
      _frameTimer = Timer.periodic(processingInterval, (_) => _captureFrame());

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
  Future<void> stop() async {
    if (!_isActive) return;

    debugPrint('🛑 OpenCVCameraController: Stopping camera');

    _frameTimer?.cancel();
    _frameTimer = null;

    _isActive = false;
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
