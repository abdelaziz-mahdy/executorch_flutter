import 'dart:developer';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:image/image.dart' as img;

import 'camera_view_singleton.dart';
import '../processors/yolo_processor.dart';

/// Callback to pass detection results after inference
typedef DetectionCallback =
    void Function(List<DetectedObject> detections, Duration inferenceTime);

/// [CameraView] sends each frame for inference
class CameraView extends StatefulWidget {
  /// Callback to pass results after inference
  final DetectionCallback resultsCallback;

  /// Model path for YOLO model
  final String modelAssetPath;

  /// Labels asset path
  final String labelsAssetPath;

  /// Confidence threshold for detections
  final double confidenceThreshold;

  /// IoU threshold for NMS
  final double iouThreshold;

  /// Constructor
  const CameraView({
    super.key,
    required this.resultsCallback,
    required this.modelAssetPath,
    required this.labelsAssetPath,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.45,
  });

  @override
  State<CameraView> createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> with WidgetsBindingObserver {
  /// List of available cameras
  late List<CameraDescription> cameras;

  /// Controller
  CameraController? cameraController;

  /// true when inference is ongoing
  bool predicting = false;

  /// ExecuTorch model and processor
  ExecuTorchModel? _model;
  YoloProcessor? _processor;
  List<String>? _classLabels;

  int _camFrameRotation = 0;
  String errorMessage = "";

  @override
  void initState() {
    super.initState();
    initStateAsync();
  }

  /// Load YOLO model and labels
  Future<void> loadModel() async {
    try {
      // Load model asset
      final byteData = await rootBundle.load(widget.modelAssetPath);
      final directory = await getApplicationCacheDirectory();
      final fileName = widget.modelAssetPath.split('/').last;
      final file = File('${directory.path}/$fileName');
      await file.writeAsBytes(byteData.buffer.asUint8List());

      // Load model
      _model = await ExecuTorchModel.load(file.path);

      // Load class labels
      final labelsData = await rootBundle.loadString(widget.labelsAssetPath);
      _classLabels = labelsData
          .split('\n')
          .where((label) => label.trim().isNotEmpty)
          .toList();

      // Create YOLO processor
      _processor = YoloProcessor(
        preprocessConfig: const YoloPreprocessConfig(
          targetWidth: 640,
          targetHeight: 640,
        ),
        classLabels: _classLabels!,
        confidenceThreshold: widget.confidenceThreshold,
        iouThreshold: widget.iouThreshold,
      );

      debugPrint('✅ YOLO model loaded successfully');
    } catch (e) {
      debugPrint('❌ Failed to load YOLO model: $e');
      if (e is PlatformException) {
        debugPrint('Platform specific error: ${e.message}');
      }
    }
  }

  void initStateAsync() async {
    WidgetsBinding.instance.addObserver(this);
    await loadModel();

    // Camera initialization
    try {
      initializeCamera();
    } on CameraException catch (e) {
      switch (e.code) {
        case 'CameraAccessDenied':
          errorMessage = ('You have denied camera access.');
          break;
        case 'CameraAccessDeniedWithoutPrompt':
          // iOS only
          errorMessage = ('Please go to Settings app to enable camera access.');
          break;
        case 'CameraAccessRestricted':
          // iOS only
          errorMessage = ('Camera access is restricted.');
          break;
        case 'AudioAccessDenied':
          errorMessage = ('You have denied audio access.');
          break;
        case 'AudioAccessDeniedWithoutPrompt':
          // iOS only
          errorMessage = ('Please go to Settings app to enable audio access.');
          break;
        case 'AudioAccessRestricted':
          // iOS only
          errorMessage = ('Audio access is restricted.');
          break;
        default:
          errorMessage = (e.toString());
          break;
      }
      setState(() {});
    }

    // Initially predicting = false
    setState(() {
      predicting = false;
    });
  }

  /// Initializes the camera by setting [cameraController]
  void initializeCamera() async {
    cameras = await availableCameras();

    var idx = cameras.indexWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
    );
    if (idx < 0) {
      log("No Back camera found - using front camera");
      idx = 0;
    }

    var desc = cameras[idx];
    _camFrameRotation = Platform.isAndroid ? desc.sensorOrientation : 0;

    // cameras[0] for rear-camera
    cameraController = CameraController(
      desc,
      ResolutionPreset.medium,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.yuv420
          : ImageFormatGroup.bgra8888,
      enableAudio: false,
    );

    cameraController?.initialize().then((_) async {
      // Stream of image passed to [onLatestImageAvailable] callback
      await cameraController?.startImageStream(onLatestImageAvailable);

      /// previewSize is size of each image frame captured by controller
      Size? previewSize = cameraController?.value.previewSize;

      /// previewSize is size of raw input image to the model
      CameraViewSingleton.inputImageSize = previewSize!;

      // the display width of image on screen is
      // same as screenWidth while maintaining the aspectRatio
      Size screenSize = MediaQuery.of(context).size;
      CameraViewSingleton.screenSize = screenSize;
      CameraViewSingleton.ratio = cameraController!.value.aspectRatio;

      setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) {
    // Show error message if camera initialization failed
    if (errorMessage.isNotEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(
            errorMessage,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.red, fontSize: 16),
          ),
        ),
      );
    }

    // Return empty container while the camera is not initialized
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return CameraPreview(cameraController!);
  }

  Future<void> runObjectDetection(CameraImage cameraImage) async {
    if (predicting || _model == null || _processor == null) {
      return;
    }
    if (!mounted) {
      return;
    }

    setState(() {
      predicting = true;
    });

    try {
      // Start the stopwatch
      Stopwatch stopwatch = Stopwatch()..start();

      // Convert CameraImage to Uint8List (JPEG)
      final imageBytes = await _convertCameraImageToJpeg(cameraImage);

      // Use YOLO processor to process the image
      final result = await _processor!.process(imageBytes, _model!);

      // Stop the stopwatch
      stopwatch.stop();

      widget.resultsCallback(result.detectedObjects, stopwatch.elapsed);
    } catch (e) {
      debugPrint('❌ Object detection failed: $e');
    }

    if (!mounted) {
      return;
    }

    setState(() {
      predicting = false;
    });
  }

  /// Convert CameraImage to JPEG bytes
  Future<Uint8List> _convertCameraImageToJpeg(CameraImage cameraImage) async {
    try {
      img.Image? convertedImage;

      if (Platform.isAndroid) {
        // Android uses YUV420
        convertedImage = _convertYUV420ToImage(cameraImage);
      } else {
        // iOS uses BGRA8888
        convertedImage = _convertBGRA8888ToImage(cameraImage);
      }

      if (convertedImage == null) {
        throw Exception('Failed to convert camera image');
      }

      // Encode to JPEG
      return Uint8List.fromList(img.encodeJpg(convertedImage, quality: 85));
    } catch (e) {
      debugPrint('❌ Image conversion failed: $e');
      rethrow;
    }
  }

  /// Convert YUV420 (Android) to Image
  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 0;

    final image = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * width + x;

        final yp = cameraImage.planes[0].bytes[index];
        final up = cameraImage.planes[1].bytes[uvIndex];
        final vp = cameraImage.planes[2].bytes[uvIndex];

        // Convert YUV to RGB
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

        image.setPixelRgb(x, y, r, g, b);
      }
    }

    return image;
  }

  /// Convert BGRA8888 (iOS) to Image
  img.Image? _convertBGRA8888ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final img.Image image = img.Image(width: width, height: height);
    final Uint8List bytes = cameraImage.planes[0].bytes;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixelIndex = (y * width + x) * 4;
        final int b = bytes[pixelIndex];
        final int g = bytes[pixelIndex + 1];
        final int r = bytes[pixelIndex + 2];
        final int a = bytes[pixelIndex + 3];

        image.setPixelRgba(x, y, r, g, b, a);
      }
    }

    return image;
  }

  /// Callback to receive each frame [CameraImage] perform inference on it
  onLatestImageAvailable(CameraImage cameraImage) async {
    // Make sure we are still mounted
    if (!mounted) {
      return;
    }

    runObjectDetection(cameraImage);

    // Make sure we are still mounted
    if (!mounted) {
      return;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    if (!mounted) {
      return;
    }
    switch (state) {
      case AppLifecycleState.paused:
        cameraController?.stopImageStream();
        break;
      case AppLifecycleState.resumed:
        if (cameraController != null &&
            !cameraController!.value.isStreamingImages) {
          await cameraController?.startImageStream(onLatestImageAvailable);
        }
        break;
      default:
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cameraController?.dispose();
    _model?.dispose();
    super.dispose();
  }
}
