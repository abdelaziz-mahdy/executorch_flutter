import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../yolo_processor.dart';

/// GPU-accelerated YOLO preprocessor using Flutter Fragment Shaders
///
/// Uses Flutter's native image decoder and GPU shaders for fast preprocessing:
/// - Hardware-accelerated image decoding (decodeImageFromList)
/// - GPU-based letterbox resize to 640x640 (maintains aspect ratio)
/// - Gray padding (114, 114, 114)
/// - Normalization to [0, 1] range
/// - Optimized single-loop tensor conversion
class GpuYoloPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  GpuYoloPreprocessor({required this.config});

  final YoloPreprocessConfig config;
  ui.FragmentProgram? _program;
  bool _isInitialized = false;

  @override
  String get inputTypeName => 'Image (Uint8List) [GPU]';

  /// Initialize the fragment shader
  Future<void> _initializeShader() async {
    if (_isInitialized) return;

    try {
      _program = await ui.FragmentProgram.fromAsset(
        'shaders/yolo_preprocess.frag',
      );
      _isInitialized = true;
      debugPrint('✅ GPU shader initialized successfully');
    } catch (e) {
      debugPrint('⚠️ Failed to load GPU shader: $e');
      throw PreprocessingException('Failed to initialize GPU shader: $e', e);
    }
  }

  @override
  bool validateInput(Uint8List input) {
    return input.isNotEmpty;
  }

  @override
  Future<List<TensorData>> preprocess(Uint8List input) async {
    try {
      // Initialize shader on first use
      await _initializeShader();

      // Use Flutter's native image decoder (hardware accelerated)
      final ui.Image image = await _decodeImageNative(input);

      // Process on GPU
      final processedImage = await _processOnGpu(image);

      // Convert to tensor
      final tensorData = await _imageToTensor(processedImage);

      // Cleanup
      image.dispose();
      processedImage.dispose();

      return [tensorData];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('GPU YOLO preprocessing failed: $e', e);
    }
  }

  /// Decode image using Flutter's native decoder (hardware accelerated)
  Future<ui.Image> _decodeImageNative(Uint8List bytes) async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (ui.Image image) {
      completer.complete(image);
    });
    return completer.future;
  }

  /// Process image on GPU using Fragment Shader
  Future<ui.Image> _processOnGpu(ui.Image inputImage) async {
    if (_program == null) {
      throw PreprocessingException('Shader not initialized');
    }

    final shader = _program!.fragmentShader();

    // Set uniforms
    // uInputSize (vec2)
    shader.setFloat(0, inputImage.width.toDouble());
    shader.setFloat(1, inputImage.height.toDouble());

    // uOutputSize (vec2)
    shader.setFloat(2, config.targetWidth.toDouble());
    shader.setFloat(3, config.targetHeight.toDouble());

    // uTexture (sampler2D) - set image sampler
    shader.setImageSampler(0, inputImage);

    // Create picture recorder
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // Draw using shader
    final paint = Paint()..shader = shader;
    canvas.drawRect(
      Rect.fromLTWH(
        0,
        0,
        config.targetWidth.toDouble(),
        config.targetHeight.toDouble(),
      ),
      paint,
    );

    // Convert to image
    final picture = recorder.endRecording();
    final outputImage = await picture.toImage(
      config.targetWidth,
      config.targetHeight,
    );

    // Cleanup
    shader.dispose();
    picture.dispose();

    return outputImage;
  }

  /// Convert ui.Image to TensorData
  Future<TensorData> _imageToTensor(ui.Image image) async {
    // Get raw bytes from image
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      throw PreprocessingException('Failed to get image bytes');
    }

    final pixels = byteData.buffer.asUint8List();

    // Create float32 tensor in NCHW format
    // Modern YOLO models (v8, v11, etc.) expect [0, 1] normalized inputs
    final totalPixels = config.targetWidth * config.targetHeight;
    final floats = Float32List(3 * totalPixels);

    // Optimized single-loop conversion: process all pixels once
    // This is faster than 3 separate loops due to better cache locality
    const scale = 1.0 / 255.0;
    for (int i = 0; i < totalPixels; i++) {
      final pixelIndex = i * 4;
      floats[i] = pixels[pixelIndex] * scale; // R channel
      floats[i + totalPixels] = pixels[pixelIndex + 1] * scale; // G channel
      floats[i + totalPixels * 2] = pixels[pixelIndex + 2] * scale; // B channel
    }

    return TensorData(
      shape: [1, 3, config.targetHeight, config.targetWidth].cast<int?>(),
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'images',
    );
  }

  /// Dispose resources
  void dispose() {
    _program = null;
    _isInitialized = false;
  }
}
