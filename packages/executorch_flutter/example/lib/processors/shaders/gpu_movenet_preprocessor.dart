import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

import '../movenet_input_processor.dart';

/// GPU-accelerated MoveNet preprocessor using Flutter Fragment Shaders
///
/// Uses Flutter's native image decoder and GPU shaders for fast preprocessing:
/// - Hardware-accelerated image decoding (decodeImageFromList)
/// - GPU-based simple resize to target size (192 for Lightning, 256 for Thunder)
/// - Normalization to [0, 1] range
/// - Optimized single-loop tensor conversion to NHWC format
class GpuMoveNetPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  GpuMoveNetPreprocessor({required this.config});

  final MoveNetPreprocessConfig config;
  ui.FragmentProgram? _program;
  bool _isInitialized = false;

  @override
  String get inputTypeName => 'Image (Uint8List) [GPU]';

  /// Initialize the fragment shader
  Future<void> _initializeShader() async {
    if (_isInitialized) return;

    try {
      _program = await ui.FragmentProgram.fromAsset(
        'shaders/movenet_preprocess.frag',
      );
      _isInitialized = true;
      debugPrint('GpuMoveNetPreprocessor: GPU shader initialized successfully');
    } catch (e) {
      debugPrint('GpuMoveNetPreprocessor: Failed to load GPU shader: $e');
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

      // Convert to tensor in NHWC format
      final tensorData = await _imageToTensor(processedImage);

      // Cleanup
      image.dispose();
      processedImage.dispose();

      return [tensorData];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('GPU MoveNet preprocessing failed: $e', e);
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
    shader.setFloat(2, config.targetSize.toDouble());
    shader.setFloat(3, config.targetSize.toDouble());

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
        config.targetSize.toDouble(),
        config.targetSize.toDouble(),
      ),
      paint,
    );

    // Convert to image
    final picture = recorder.endRecording();
    final outputImage = await picture.toImage(
      config.targetSize,
      config.targetSize,
    );

    // Cleanup
    shader.dispose();
    picture.dispose();

    return outputImage;
  }

  /// Convert ui.Image to TensorData in NHWC format
  Future<TensorData> _imageToTensor(ui.Image image) async {
    // Get raw bytes from image
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      throw PreprocessingException('Failed to get image bytes');
    }

    final pixels = byteData.buffer.asUint8List();

    // Create float32 tensor in NHWC format [1, height, width, 3]
    // MoveNet uses NHWC format (batch, height, width, channels)
    final totalPixels = config.targetSize * config.targetSize;
    final floats = Float32List(3 * totalPixels);

    // NHWC conversion: interleaved RGB for each pixel
    // floats[i * 3] = R, floats[i * 3 + 1] = G, floats[i * 3 + 2] = B
    const scale = 1.0 / 255.0;
    for (int i = 0; i < totalPixels; i++) {
      final pixelIndex = i * 4; // RGBA format from image
      floats[i * 3] = pixels[pixelIndex] * scale; // R channel
      floats[i * 3 + 1] = pixels[pixelIndex + 1] * scale; // G channel
      floats[i * 3 + 2] = pixels[pixelIndex + 2] * scale; // B channel
    }

    debugPrint(
      'GpuMoveNetPreprocessor: Tensor shape: [1, ${config.targetSize}, ${config.targetSize}, 3] (NHWC)',
    );

    return TensorData(
      shape: [1, config.targetSize, config.targetSize, 3].cast<int?>(),
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
      name: 'input',
    );
  }

  /// Dispose resources
  void dispose() {
    _program = null;
    _isInitialized = false;
  }
}
