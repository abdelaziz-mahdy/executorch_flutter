import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// OpenCV-based camera stream processor for desktop platforms (macOS)
/// Uses OpenCV VideoCapture for camera access
class OpenCVCameraStreamProcessor extends StatefulWidget {
  const OpenCVCameraStreamProcessor({
    super.key,
    required this.onFrameProcessed,
    this.processingInterval = const Duration(milliseconds: 100),
    this.deviceId = 0,
  });

  /// Callback when a frame is converted to image bytes
  final Future<void> Function(Uint8List imageBytes) onFrameProcessed;

  /// Minimum interval between processing frames (to control FPS)
  final Duration processingInterval;

  /// Camera device ID (default 0 for primary camera)
  final int deviceId;

  @override
  State<OpenCVCameraStreamProcessor> createState() =>
      _OpenCVCameraStreamProcessorState();
}

class _OpenCVCameraStreamProcessorState
    extends State<OpenCVCameraStreamProcessor> {
  cv.VideoCapture? _capture;
  Timer? _frameTimer;
  bool _isProcessing = false;
  String? errorMessage;
  cv.Mat? _currentFrame;

  @override
  void initState() {
    super.initState();
    // Initialize camera after the first frame to avoid blocking the UI
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeCamera();
    });
  }

  Future<void> _initializeCamera() async {
    try {
      // Run camera initialization in a separate isolate/compute to avoid blocking UI
      await Future.delayed(Duration.zero); // Allow UI to breathe

      // Initialize VideoCapture with device ID
      // Use CAP_AVFOUNDATION for macOS
      _capture = cv.VideoCapture.fromDevice(
        widget.deviceId,
        apiPreference: cv.CAP_AVFOUNDATION,
      );

      if (!_capture!.isOpened) {
        if (mounted) {
          setState(() {
            errorMessage =
                'Failed to open camera. Please check camera permissions in System Settings > Privacy & Security > Camera';
          });
        }
        return;
      }

      // Set camera properties for better performance
      _capture!.set(cv.CAP_PROP_FRAME_WIDTH, 640);
      _capture!.set(cv.CAP_PROP_FRAME_HEIGHT, 480);
      _capture!.set(cv.CAP_PROP_FPS, 30);

      // Start frame capture timer
      debugPrint(
        '⏰ Creating timer with interval: ${widget.processingInterval.inMilliseconds}ms',
      );
      _frameTimer = Timer.periodic(widget.processingInterval, (_) {
        debugPrint('⏰ Timer tick - calling _captureAndProcessFrame');
        _captureAndProcessFrame();
      });
      debugPrint('✅ Timer created successfully');

      if (mounted) {
        setState(() {});
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          errorMessage =
              'Failed to initialize camera: $e\n\nPlease check camera permissions in System Settings.';
        });
      }
    }
  }

  Future<void> _captureAndProcessFrame() async {
    debugPrint(
      '🎥 _captureAndProcessFrame called, _isProcessing=$_isProcessing, capture=${_capture != null}, opened=${_capture?.isOpened}',
    );

    if (_isProcessing || _capture == null || !_capture!.isOpened) {
      debugPrint(
        '⏸️ Skipping frame capture: isProcessing=$_isProcessing, capture=${_capture != null}, opened=${_capture?.isOpened}',
      );
      return;
    }

    _isProcessing = true;
    debugPrint('🎬 Starting frame capture');

    try {
      // Read frame from camera
      final (success, frame) = _capture!.read();
      debugPrint('📸 Frame read: success=$success, isEmpty=${frame.isEmpty}');

      if (!success || frame.isEmpty) {
        debugPrint('Failed to read frame from camera');
        _isProcessing = false;
        return;
      }

      // Update current frame for preview
      _currentFrame?.dispose();
      _currentFrame = frame.clone();

      // Convert frame to JPEG bytes
      final (encodeSuccess, jpegBytes) = await cv.imencodeAsync('.jpg', frame);
      debugPrint(
        '🖼️ Encoded to JPEG: success=$encodeSuccess, bytes=${jpegBytes.length}',
      );

      if (!encodeSuccess) {
        debugPrint('Failed to encode frame to JPEG');
        frame.dispose();
        _isProcessing = false;
        return;
      }

      // Process the frame asynchronously (don't wait for it)
      // The playground has its own throttling with _isProcessingFrame
      debugPrint('🚀 Sending frame to processing (async)');
      widget.onFrameProcessed(jpegBytes).catchError((e) {
        debugPrint('Frame processing error: $e');
      });

      // Clean up
      frame.dispose();
      debugPrint('🧹 Frame disposed');

      // Update UI with new frame
      if (mounted) {
        setState(() {});
      }
      debugPrint(
        '✅ Frame capture complete, _isProcessing will be set to false',
      );
    } catch (e) {
      debugPrint('Frame capture error: $e');
    } finally {
      _isProcessing = false;
      debugPrint('🔓 Camera capture unlocked (_isProcessing=false)');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red, fontSize: 16),
              ),
            ],
          ),
        ),
      );
    }

    if (_capture == null || !_capture!.isOpened) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_currentFrame == null || _currentFrame!.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }

    // Convert current frame to displayable image
    return FutureBuilder<Uint8List>(
      future: _convertFrameToImage(_currentFrame!),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        return Image.memory(
          snapshot.data!,
          fit: BoxFit.contain,
          gaplessPlayback: true,
        );
      },
    );
  }

  Future<Uint8List> _convertFrameToImage(cv.Mat frame) async {
    final (success, bytes) = await cv.imencodeAsync('.jpg', frame);
    if (!success) {
      throw Exception('Failed to encode frame for display');
    }
    return bytes;
  }

  @override
  void dispose() {
    debugPrint('🧹 OpenCVCameraStreamProcessor disposing - cancelling timer');
    _frameTimer?.cancel();
    _currentFrame?.dispose();
    _capture?.release();
    _capture?.dispose();
    super.dispose();
  }
}
