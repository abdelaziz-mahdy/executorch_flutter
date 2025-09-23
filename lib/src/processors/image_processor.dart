import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';

import '../generated/executorch_api.dart';
import 'base_processor.dart';

/// Configuration for image preprocessing
@immutable
class ImagePreprocessConfig {
  const ImagePreprocessConfig({
    this.targetWidth = 224,
    this.targetHeight = 224,
    this.normalizeToFloat = true,
    this.meanSubtraction = const [0.485, 0.456, 0.406],
    this.standardDeviation = const [0.229, 0.224, 0.225],
    this.cropMode = ImageCropMode.centerCrop,
  });

  /// Target width for resizing (default: 224 for ImageNet)
  final int targetWidth;

  /// Target height for resizing (default: 224 for ImageNet)
  final int targetHeight;

  /// Whether to normalize pixel values to float range [0,1]
  final bool normalizeToFloat;

  /// Mean values for normalization (RGB channels)
  /// Default: ImageNet means [0.485, 0.456, 0.406]
  final List<double> meanSubtraction;

  /// Standard deviation values for normalization (RGB channels)
  /// Default: ImageNet std devs [0.229, 0.224, 0.225]
  final List<double> standardDeviation;

  /// How to crop/resize the image to target dimensions
  final ImageCropMode cropMode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ImagePreprocessConfig &&
          runtimeType == other.runtimeType &&
          targetWidth == other.targetWidth &&
          targetHeight == other.targetHeight &&
          normalizeToFloat == other.normalizeToFloat &&
          _listEquals(meanSubtraction, other.meanSubtraction) &&
          _listEquals(standardDeviation, other.standardDeviation) &&
          cropMode == other.cropMode;

  @override
  int get hashCode =>
      targetWidth.hashCode ^
      targetHeight.hashCode ^
      normalizeToFloat.hashCode ^
      meanSubtraction.hashCode ^
      standardDeviation.hashCode ^
      cropMode.hashCode;

  bool _listEquals(List<double> a, List<double> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Modes for cropping/resizing images
enum ImageCropMode {
  /// Resize to exact dimensions (may distort aspect ratio)
  stretch,
  /// Crop from center to maintain aspect ratio
  centerCrop,
  /// Fit image within dimensions, padding with zeros
  letterbox,
}

/// Result of image classification
@immutable
class ClassificationResult {
  const ClassificationResult({
    required this.className,
    required this.confidence,
    required this.classIndex,
    required this.allProbabilities,
  });

  /// The predicted class name/label
  final String className;

  /// Confidence score for the prediction (0.0 to 1.0)
  final double confidence;

  /// Index of the predicted class
  final int classIndex;

  /// All class probabilities (softmax outputs)
  final List<double> allProbabilities;

  @override
  String toString() =>
      'ClassificationResult(class: $className, confidence: ${(confidence * 100).toStringAsFixed(1)}%, index: $classIndex)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ClassificationResult &&
          runtimeType == other.runtimeType &&
          className == other.className &&
          confidence == other.confidence &&
          classIndex == other.classIndex;

  @override
  int get hashCode =>
      className.hashCode ^ confidence.hashCode ^ classIndex.hashCode;
}

/// Preprocessor for image data to tensor conversion
class ImageNetPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  ImageNetPreprocessor({
    required this.config,
  });

  final ImagePreprocessConfig config;

  @override
  String get inputTypeName => 'Image (Uint8List)';

  @override
  bool validateInput(Uint8List input) {
    if (input.isEmpty) return false;

    try {
      final image = img.decodeImage(input);
      return image != null && image.width > 0 && image.height > 0;
    } catch (e) {
      return false;
    }
  }

  @override
  Future<List<TensorData>> preprocess(Uint8List input, {ModelMetadata? metadata}) async {
    try {
      // Decode image
      final image = img.decodeImage(input);
      if (image == null) {
        throw PreprocessingException('Failed to decode image');
      }

      // Resize/crop image
      final processedImage = _resizeImage(image);

      // Ensure we have RGB format (3 channels)
      final rgbImage = processedImage.numChannels >= 3
          ? processedImage
          : processedImage; // Keep as is for now - in practice would convert grayscale to RGB

      // Convert to tensor data
      final tensorData = _imageToTensor(rgbImage);

      final tensor = ProcessorTensorUtils.createTensor(
        shape: [1, 3, config.targetHeight, config.targetWidth], // NCHW format
        dataType: TensorType.float32,
        data: tensorData,
        name: 'input',
      );

      return [tensor];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('Image preprocessing failed: $e', e);
    }
  }

  img.Image _resizeImage(img.Image image) {
    switch (config.cropMode) {
      case ImageCropMode.stretch:
        return img.copyResize(
          image,
          width: config.targetWidth,
          height: config.targetHeight,
        );

      case ImageCropMode.centerCrop:
        // Calculate crop dimensions to maintain aspect ratio
        final aspectRatio = config.targetWidth / config.targetHeight;
        final imageAspectRatio = image.width / image.height;

        late img.Image croppedImage;
        if (imageAspectRatio > aspectRatio) {
          // Image is wider, crop horizontally
          final newWidth = (image.height * aspectRatio).round();
          final cropX = (image.width - newWidth) ~/ 2;
          croppedImage = img.copyCrop(
            image,
            x: cropX,
            y: 0,
            width: newWidth,
            height: image.height,
          );
        } else {
          // Image is taller, crop vertically
          final newHeight = (image.width / aspectRatio).round();
          final cropY = (image.height - newHeight) ~/ 2;
          croppedImage = img.copyCrop(
            image,
            x: 0,
            y: cropY,
            width: image.width,
            height: newHeight,
          );
        }

        return img.copyResize(
          croppedImage,
          width: config.targetWidth,
          height: config.targetHeight,
        );

      case ImageCropMode.letterbox:
        // Scale to fit within dimensions, then pad
        final scaleW = config.targetWidth / image.width;
        final scaleH = config.targetHeight / image.height;
        final scale = math.min(scaleW, scaleH);

        final newWidth = (image.width * scale).round();
        final newHeight = (image.height * scale).round();

        final resized = img.copyResize(image, width: newWidth, height: newHeight);

        // Create target image with padding
        final target = img.Image(
          width: config.targetWidth,
          height: config.targetHeight,
          numChannels: image.numChannels,
        );
        img.fill(target, color: img.ColorRgb8(0, 0, 0)); // Black padding

        final offsetX = (config.targetWidth - newWidth) ~/ 2;
        final offsetY = (config.targetHeight - newHeight) ~/ 2;

        img.compositeImage(target, resized, dstX: offsetX, dstY: offsetY);
        return target;
    }
  }

  List<double> _imageToTensor(img.Image image) {
    final data = <double>[];

    // Convert to NCHW format (batch=1, channels=3, height, width)
    // Channel order: R, G, B

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
          final pixel = image.getPixel(x, y);
          late double value;

          switch (c) {
            case 0: // Red channel
              value = pixel.r.toDouble();
              break;
            case 1: // Green channel
              value = pixel.g.toDouble();
              break;
            case 2: // Blue channel
              value = pixel.b.toDouble();
              break;
          }

          // Normalize to [0, 1] if enabled
          if (config.normalizeToFloat) {
            value /= 255.0;
          }

          // Apply mean subtraction and standard deviation
          if (config.meanSubtraction.isNotEmpty &&
              config.standardDeviation.isNotEmpty &&
              c < config.meanSubtraction.length &&
              c < config.standardDeviation.length) {
            value = (value - config.meanSubtraction[c]) / config.standardDeviation[c];
          }

          data.add(value);
        }
      }
    }

    return data;
  }
}

/// Postprocessor for classification results
class ImageNetPostprocessor extends ExecuTorchPostprocessor<ClassificationResult> {
  ImageNetPostprocessor({
    required this.classLabels,
  });

  final List<String> classLabels;

  @override
  String get outputTypeName => 'Classification Result';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    if (outputs.isEmpty) return false;

    final output = outputs.first;
    if (output.dataType != TensorType.float32) return false;

    // Check if shape represents logits/probabilities
    final shape = output.shape?.where((dim) => dim != null).toList() ?? [];
    if (shape.isEmpty) return false;

    // Should have at least as many outputs as we have labels
    final outputSize = shape.last!;
    return outputSize >= classLabels.length;
  }

  @override
  Future<ClassificationResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
    try {
      if (outputs.isEmpty) {
        throw PostprocessingException('No output tensors provided');
      }

      final output = outputs.first;
      final logits = ProcessorTensorUtils.extractFloat32Data(output);

      // Apply softmax to get probabilities
      final probabilities = _applySoftmax(logits);

      // Find the class with highest probability
      double maxProb = 0.0;
      int maxIndex = 0;

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      // Get class name
      String className;
      if (maxIndex < classLabels.length) {
        className = classLabels[maxIndex];
      } else {
        className = 'Unknown Class $maxIndex';
      }

      // Validate confidence range
      if (maxProb < 0.0 || maxProb > 1.0) {
        throw PostprocessingException(
          'Invalid confidence value: $maxProb (should be between 0.0 and 1.0)'
        );
      }

      return ClassificationResult(
        className: className,
        confidence: maxProb,
        classIndex: maxIndex,
        allProbabilities: probabilities,
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException('Classification postprocessing failed: $e', e);
    }
  }

  List<double> _applySoftmax(Float32List logits) {
    // Find max value for numerical stability
    double maxLogit = logits.reduce(math.max);

    // Compute exp(x - max) for each element
    final expValues = logits.map((x) => math.exp(x - maxLogit)).toList();

    // Compute sum of exponentials
    final sumExp = expValues.reduce((a, b) => a + b);

    // Normalize to get probabilities
    return expValues.map((x) => x / sumExp).toList();
  }
}

/// Complete ImageNet classification processor
class ImageNetProcessor extends ExecuTorchProcessor<Uint8List, ClassificationResult> {
  ImageNetProcessor({
    required this.preprocessConfig,
    required this.classLabels,
  }) : _preprocessor = ImageNetPreprocessor(config: preprocessConfig),
       _postprocessor = ImageNetPostprocessor(classLabels: classLabels);

  final ImagePreprocessConfig preprocessConfig;
  final List<String> classLabels;
  final ImageNetPreprocessor _preprocessor;
  final ImageNetPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<Uint8List> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<ClassificationResult> get postprocessor => _postprocessor;
}