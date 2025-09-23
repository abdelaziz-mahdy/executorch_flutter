/// API Contract: ImageNet Processor Implementation
///
/// This file defines the contracts for ImageNet-specific processor implementation.
/// These contracts demonstrate how to implement model-specific processors.

import 'dart:typed_data';
import 'processor_interfaces.dart';

/// Contract: ImageNet preprocessing configuration
abstract class ImagePreprocessConfig {
  /// Target width for image resizing
  ///
  /// Contract requirements:
  /// - Must be positive integer
  /// - Typically 224 for ImageNet models
  int get targetWidth;

  /// Target height for image resizing
  ///
  /// Contract requirements:
  /// - Must be positive integer
  /// - Typically 224 for ImageNet models
  int get targetHeight;

  /// Whether to normalize pixel values to float range [0.0, 1.0]
  ///
  /// Contract requirements:
  /// - Must be boolean value
  /// - True for float32 models, false for uint8 models
  bool get normalizeToFloat;

  /// Mean values for normalization (per channel)
  ///
  /// Contract requirements:
  /// - Must have 3 values for RGB channels
  /// - ImageNet standard: [0.485, 0.456, 0.406]
  List<double> get meanSubtraction;

  /// Standard deviation values for normalization (per channel)
  ///
  /// Contract requirements:
  /// - Must have 3 values for RGB channels
  /// - ImageNet standard: [0.229, 0.224, 0.225]
  List<double> get standardDeviation;

  /// Channel ordering for color channels
  ///
  /// Contract requirements:
  /// - Must be ChannelOrder.rgb or ChannelOrder.bgr
  /// - ImageNet typically uses RGB
  ChannelOrder get channelOrder;

  /// Data layout for tensor dimensions
  ///
  /// Contract requirements:
  /// - Must be DataLayout.nchw or DataLayout.nhwc
  /// - NCHW: [batch, channels, height, width]
  /// - NHWC: [batch, height, width, channels]
  DataLayout get dataLayout;
}

/// Contract: Channel ordering options
enum ChannelOrder { rgb, bgr }

/// Contract: Data layout options
enum DataLayout { nchw, nhwc }

/// Contract: ImageNet classification result
abstract class ClassificationResult {
  /// The predicted class name
  ///
  /// Contract requirements:
  /// - Must be non-empty string
  /// - Should correspond to ImageNet class label
  String get className;

  /// Confidence score for the prediction
  ///
  /// Contract requirements:
  /// - Must be between 0.0 and 1.0
  /// - Higher values indicate higher confidence
  double get confidence;

  /// Index of the predicted class
  ///
  /// Contract requirements:
  /// - Must be non-negative integer
  /// - Should correspond to model output class index
  int get classIndex;

  /// All class probabilities
  ///
  /// Contract requirements:
  /// - Must have 1000 entries for ImageNet
  /// - Each value must be between 0.0 and 1.0
  /// - Values should sum to approximately 1.0
  List<double> get allProbabilities;
}

/// Contract: ImageNet image preprocessor
abstract class ImageNetPreprocessor extends ExecuTorchPreprocessor<Uint8List> {
  /// Configuration for preprocessing operations
  ///
  /// Contract requirements:
  /// - Must return valid ImagePreprocessConfig instance
  /// - Must be consistent across calls
  ImagePreprocessConfig get config;

  /// Preprocess image bytes for ImageNet inference
  ///
  /// Contract requirements:
  /// - Must resize image to config.targetWidth x config.targetHeight
  /// - Must apply normalization based on config.normalizeToFloat
  /// - Must apply mean subtraction and standard deviation normalization
  /// - Must handle channel ordering (RGB/BGR) based on config.channelOrder
  /// - Must arrange data according to config.dataLayout (NCHW/NHWC)
  /// - Must return single TensorData with shape [1, 3, 224, 224] for NCHW
  /// - Must validate input using validateInput() first
  @override
  Future<List<TensorData>> preprocess(Uint8List input, {ModelMetadata? metadata});

  /// Validate image input data
  ///
  /// Contract requirements:
  /// - Must return false for empty input
  /// - Must return true for non-empty Uint8List
  /// - Should validate basic image format if possible
  @override
  bool validateInput(Uint8List input);

  /// Input type identifier
  ///
  /// Contract requirements:
  /// - Must return "Image Bytes (Uint8List)"
  @override
  String get inputTypeName;
}

/// Contract: ImageNet classification postprocessor
abstract class ImageNetPostprocessor extends ExecuTorchPostprocessor<ClassificationResult> {
  /// ImageNet class labels
  ///
  /// Contract requirements:
  /// - Must have exactly 1000 labels
  /// - Must be in correct ImageNet order
  /// - Each label must be non-empty string
  List<String> get classLabels;

  /// Postprocess model outputs to classification result
  ///
  /// Contract requirements:
  /// - Must validate outputs using validateOutputs() first
  /// - Must extract probabilities from first output tensor
  /// - Must apply softmax if outputs are logits
  /// - Must find class with highest probability
  /// - Must create ClassificationResult with all required fields
  /// - Must handle edge case of all-zero outputs
  @override
  Future<ClassificationResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata});

  /// Validate model output tensors
  ///
  /// Contract requirements:
  /// - Must return false for empty outputs list
  /// - Must return false if first tensor is not float32
  /// - Must return false if tensor doesn't have 1000 elements
  /// - Must return true for valid ImageNet output format
  @override
  bool validateOutputs(List<TensorData> outputs);

  /// Output type identifier
  ///
  /// Contract requirements:
  /// - Must return "Classification Result"
  @override
  String get outputTypeName;
}

/// Contract: Complete ImageNet processor
abstract class ImageNetProcessor extends ExecuTorchProcessor<Uint8List, ClassificationResult> {
  /// ImageNet-specific preprocessor
  ///
  /// Contract requirements:
  /// - Must return ImageNetPreprocessor instance
  /// - Must be configured for ImageNet requirements
  @override
  ImageNetPreprocessor get preprocessor;

  /// ImageNet-specific postprocessor
  ///
  /// Contract requirements:
  /// - Must return ImageNetPostprocessor instance
  /// - Must have complete ImageNet class labels
  @override
  ImageNetPostprocessor get postprocessor;

  /// Process image through complete ImageNet pipeline
  ///
  /// Contract requirements:
  /// - Must handle complete pipeline from image bytes to classification
  /// - Must validate image input before processing
  /// - Must handle inference errors gracefully
  /// - Must return valid ClassificationResult for successful inference
  @override
  Future<ClassificationResult> process(Uint8List input, dynamic model, {ModelMetadata? metadata});
}