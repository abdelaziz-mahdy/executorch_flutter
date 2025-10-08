import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';

import 'package:executorch_flutter/executorch_flutter.dart';

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
    this.classLabels = const [],
  });

  /// The predicted class name/label
  final String className;

  /// Confidence score for the prediction (0.0 to 1.0)
  final double confidence;

  /// Index of the predicted class
  final int classIndex;

  /// All class probabilities (softmax outputs)
  final List<double> allProbabilities;

  /// All class labels for mapping indices to names
  final List<String> classLabels;

  /// Get top K classification results (including this one as the first result)
  List<({String className, double confidence, int classIndex})> get topK {
    // Return top 5 results from allProbabilities
    final indexed = <({int index, double probability})>[];
    for (int i = 0; i < allProbabilities.length; i++) {
      indexed.add((index: i, probability: allProbabilities[i]));
    }

    // Sort by probability descending
    indexed.sort((a, b) => b.probability.compareTo(a.probability));

    // Take top 5 and return with actual class names
    return indexed.take(5).map((item) {
      String label;
      if (classLabels.isNotEmpty && item.index < classLabels.length) {
        label = classLabels[item.index];
      } else {
        label = 'Class ${item.index}';
      }
      return (
        className: label,
        confidence: item.probability,
        classIndex: item.index,
      );
    }).toList();
  }

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
  ImageNetPreprocessor({required this.config});

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
  Future<List<TensorData>> preprocess(Uint8List input) async {
    try {
      // Decode image - exactly like working code
      final decodedImage = img.decodeImage(input);
      if (decodedImage == null) {
        throw PreprocessingException('Failed to decode image');
      }

      // Resize to model input size (224x224 for most image models) - exactly like working code
      final resized = img.copyResize(
        decodedImage,
        width: config.targetWidth,
        height: config.targetHeight,
      );

      // Convert to RGB if needed - exactly like working code
      final rgbImage = resized.convert(numChannels: 3);

      // ImageNet normalization constants - exactly like working code
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];

      // Create float32 tensor in NCHW format - exactly like working code
      final floats = Float32List(
        1 * 3 * config.targetHeight * config.targetWidth,
      );

      // Fill tensor in NCHW format: [batch, channel, height, width] - exactly like working code
      int index = 0;

      // Channel 0 (Red) - exactly like working code
      for (int y = 0; y < config.targetHeight; y++) {
        for (int x = 0; x < config.targetWidth; x++) {
          final pixel = rgbImage.getPixel(x, y);
          final normalizedValue = (pixel.r / 255.0 - mean[0]) / std[0];
          floats[index++] = normalizedValue;
        }
      }

      // Channel 1 (Green) - exactly like working code
      for (int y = 0; y < config.targetHeight; y++) {
        for (int x = 0; x < config.targetWidth; x++) {
          final pixel = rgbImage.getPixel(x, y);
          final normalizedValue = (pixel.g / 255.0 - mean[1]) / std[1];
          floats[index++] = normalizedValue;
        }
      }

      // Channel 2 (Blue) - exactly like working code
      for (int y = 0; y < config.targetHeight; y++) {
        for (int x = 0; x < config.targetWidth; x++) {
          final pixel = rgbImage.getPixel(x, y);
          final normalizedValue = (pixel.b / 255.0 - mean[2]) / std[2];
          floats[index++] = normalizedValue;
        }
      }

      // Create tensor data - exactly like working code
      final tensorData = TensorData(
        shape: [
          1,
          3,
          config.targetHeight,
          config.targetWidth,
        ].cast<int?>(), // NCHW format
        dataType: TensorType.float32,
        data: floats.buffer.asUint8List(),
        name: 'input',
      );

      return [tensorData];
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

        final resized = img.copyResize(
          image,
          width: newWidth,
          height: newHeight,
        );

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

    // Convert to RGB if needed - ensure we have exactly 3 channels
    final rgbImage = image.convert(numChannels: 3);

    // Convert to NCHW format (batch=1, channels=3, height, width)
    // Channel order: R, G, B
    // This matches the working implementation exactly

    // Channel 0 (Red)
    for (int y = 0; y < rgbImage.height; y++) {
      for (int x = 0; x < rgbImage.width; x++) {
        final pixel = rgbImage.getPixel(x, y);
        double value = pixel.r.toDouble();

        // Normalize to [0, 1] if enabled
        if (config.normalizeToFloat) {
          value /= 255.0;
        }

        // Apply mean subtraction and standard deviation for red channel
        if (config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty &&
            config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty) {
          value =
              (value - config.meanSubtraction[0]) / config.standardDeviation[0];
        }

        data.add(value);
      }
    }

    // Channel 1 (Green)
    for (int y = 0; y < rgbImage.height; y++) {
      for (int x = 0; x < rgbImage.width; x++) {
        final pixel = rgbImage.getPixel(x, y);
        double value = pixel.g.toDouble();

        // Normalize to [0, 1] if enabled
        if (config.normalizeToFloat) {
          value /= 255.0;
        }

        // Apply mean subtraction and standard deviation for green channel
        if (config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty &&
            config.meanSubtraction.length > 1 &&
            config.standardDeviation.length > 1) {
          value =
              (value - config.meanSubtraction[1]) / config.standardDeviation[1];
        }

        data.add(value);
      }
    }

    // Channel 2 (Blue)
    for (int y = 0; y < rgbImage.height; y++) {
      for (int x = 0; x < rgbImage.width; x++) {
        final pixel = rgbImage.getPixel(x, y);
        double value = pixel.b.toDouble();

        // Normalize to [0, 1] if enabled
        if (config.normalizeToFloat) {
          value /= 255.0;
        }

        // Apply mean subtraction and standard deviation for blue channel
        if (config.meanSubtraction.isNotEmpty &&
            config.standardDeviation.isNotEmpty &&
            config.meanSubtraction.length > 2 &&
            config.standardDeviation.length > 2) {
          value =
              (value - config.meanSubtraction[2]) / config.standardDeviation[2];
        }

        data.add(value);
      }
    }

    return data;
  }
}

/// Postprocessor for classification results
class ImageNetPostprocessor
    extends ExecuTorchPostprocessor<ClassificationResult> {
  ImageNetPostprocessor({required this.classLabels});

  final List<String> classLabels;

  @override
  String get outputTypeName => 'Classification Result';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    if (outputs.isEmpty) return false;

    final output = outputs.first;
    if (output.dataType != TensorType.float32) return false;

    // Check if shape represents logits/probabilities
    final shape = output.shape.where((dim) => dim != null).toList();
    if (shape.isEmpty) return false;

    // Should have reasonable number of outputs (at least 100 classes, max 100k)
    final outputSize = shape.last!;
    return outputSize >= 100 && outputSize <= 100000;
  }

  @override
  Future<ClassificationResult> postprocess(List<TensorData> outputs) async {
    try {
      if (outputs.isEmpty) {
        throw PostprocessingException('No output tensors provided');
      }

      final output = outputs.first;

      // For classification models, find the class with highest probability - exactly like working code
      if (output.dataType == TensorType.float32) {
        final byteData = ByteData.sublistView(output.data);
        final logits = <double>[];

        for (int i = 0; i < output.data.length ~/ 4; i++) {
          logits.add(byteData.getFloat32(i * 4, Endian.host));
        }

        // Apply softmax to convert logits to probabilities - exactly like working code
        final probabilities = _applySoftmax(logits);

        // Find top-5 predictions - exactly like working code
        final indexed = List.generate(
          probabilities.length,
          (i) => MapEntry(i, probabilities[i]),
        );
        indexed.sort((a, b) => b.value.compareTo(a.value));

        final top5 = indexed.take(5).toList();

        // Get top result
        final topResult = top5.first;
        final maxIndex = topResult.key;
        final maxProb = topResult.value;

        // Get class name
        String className;
        if (maxIndex < classLabels.length) {
          className = classLabels[maxIndex];
        } else {
          className = 'Unknown';
        }

        return ClassificationResult(
          className: className,
          confidence: maxProb,
          classIndex: maxIndex,
          allProbabilities: probabilities,
          classLabels: classLabels,
        );
      }

      throw PostprocessingException(
        'Unsupported output data type: ${output.dataType}',
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException(
        'Classification postprocessing failed: $e',
        e,
      );
    }
  }

  List<double> _applySoftmax(List<double> logits) {
    // Find max for numerical stability - exactly like working code
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);

    // Compute exp(logit - max) for each logit - exactly like working code
    final expValues = logits
        .map((logit) => math.exp(logit - maxLogit))
        .toList();

    // Compute sum of all exp values - exactly like working code
    final sumExp = expValues.reduce((a, b) => a + b);

    // Normalize to get probabilities - exactly like working code
    return expValues.map((exp) => exp / sumExp).toList();
  }
}

/// Complete ImageNet classification processor
class ImageNetProcessor
    extends ExecuTorchProcessor<Uint8List, ClassificationResult> {
  ImageNetProcessor({required this.preprocessConfig, required this.classLabels})
    : _preprocessor = ImageNetPreprocessor(config: preprocessConfig),
      _postprocessor = ImageNetPostprocessor(classLabels: classLabels);

  final ImagePreprocessConfig preprocessConfig;
  final List<String> classLabels;
  final ImageNetPreprocessor _preprocessor;
  final ImageNetPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<Uint8List> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<ClassificationResult> get postprocessor =>
      _postprocessor;
}
