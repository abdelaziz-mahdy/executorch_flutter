import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../processors/camera_image_converter.dart';

/// Generic camera stream processor that converts camera frames to image bytes
/// Delegates actual processing to the provided callback
class CameraStreamProcessor extends StatefulWidget {
  const CameraStreamProcessor({
    super.key,
    required this.onFrameProcessed,
    this.processingInterval = const Duration(milliseconds: 100),
    this.converter,
  });

  /// Callback when a frame is converted to image bytes
  final Future<void> Function(Uint8List imageBytes) onFrameProcessed;

  /// Minimum interval between processing frames (to control FPS)
  final Duration processingInterval;

  /// Image converter to use (defaults to ImageLibCameraConverter)
  final CameraImageConverter? converter;

  @override
  State<CameraStreamProcessor> createState() => _CameraStreamProcessorState();
}

class _CameraStreamProcessorState extends State<CameraStreamProcessor>
    with WidgetsBindingObserver {
  List<CameraDescription>? cameras;
  CameraController? cameraController;
  bool _isProcessing = false;
  DateTime? _lastProcessTime;
  String? errorMessage;
  late CameraImageConverter _converter;

  @override
  void initState() {
    super.initState();
    _converter = widget.converter ?? ImageLibCameraConverter();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      cameras = await availableCameras();

      if (cameras == null || cameras!.isEmpty) {
        setState(() {
          errorMessage = 'No cameras available';
        });
        return;
      }

      // Prefer back camera
      var idx = cameras!.indexWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
      );
      if (idx < 0) idx = 0;

      cameraController = CameraController(
        cameras![idx],
        ResolutionPreset.medium,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
        enableAudio: false,
      );

      await cameraController!.initialize();
      await cameraController!.startImageStream(_onLatestImageAvailable);

      if (mounted) {
        setState(() {});
      }
    } on CameraException catch (e) {
      setState(() {
        errorMessage = _getCameraErrorMessage(e);
      });
    } catch (e) {
      setState(() {
        errorMessage = 'Failed to initialize camera: $e';
      });
    }
  }

  String _getCameraErrorMessage(CameraException e) {
    switch (e.code) {
      case 'CameraAccessDenied':
        return 'Camera access denied';
      case 'CameraAccessDeniedWithoutPrompt':
        return 'Please enable camera access in Settings';
      case 'CameraAccessRestricted':
        return 'Camera access is restricted';
      default:
        return 'Camera error: ${e.description}';
    }
  }

  Future<void> _onLatestImageAvailable(CameraImage image) async {
    if (!mounted) return;
    if (_isProcessing) return;

    // Throttle processing based on interval
    final now = DateTime.now();
    if (_lastProcessTime != null &&
        now.difference(_lastProcessTime!) < widget.processingInterval) {
      return;
    }

    _isProcessing = true;
    _lastProcessTime = now;

    try {
      final imageBytes = await _converter.convertToJpeg(image);
      await widget.onFrameProcessed(imageBytes);
    } catch (e) {
      debugPrint('Frame processing error: $e');
    } finally {
      _isProcessing = false;
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

    if (cameraController == null || !cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return CameraPreview(cameraController!);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return;
    }

    switch (state) {
      case AppLifecycleState.paused:
        cameraController?.stopImageStream();
        break;
      case AppLifecycleState.resumed:
        if (!cameraController!.value.isStreamingImages) {
          cameraController?.startImageStream(_onLatestImageAvailable);
        }
        break;
      default:
        break;
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cameraController?.dispose();
    super.dispose();
  }
}
