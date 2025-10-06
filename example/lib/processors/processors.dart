/// ExecuTorch Flutter Example Processors
///
/// This library contains reference implementations of processors for
/// image classification and object detection tasks.
///
/// ## Available Processors
///
/// ### Image Classification
/// - [ImageNetProcessor]: Complete ImageNet classification pipeline (MobileNet)
/// - [ImagePreprocessConfig]: Configuration for image preprocessing
/// - [ClassificationResult]: Image classification results
///
/// ### Object Detection
/// - [YoloProcessor]: Complete YOLO object detection pipeline
/// - [YoloPreprocessConfig]: Configuration for YOLO preprocessing
/// - [ObjectDetectionResult]: Object detection results with bounding boxes
/// - [DetectedObject]: Individual detected object
/// - [BoundingBox]: Bounding box coordinates
///
/// ## Usage
///
/// ```dart
/// // Image Classification
/// final imageProcessor = ImageNetProcessor(
///   preprocessConfig: ImagePreprocessConfig(
///     targetWidth: 224,
///     targetHeight: 224,
///     normalizeToFloat: true,
///     meanSubtraction: [0.485, 0.456, 0.406],
///     standardDeviation: [0.229, 0.224, 0.225],
///   ),
///   classLabels: imageNetLabels,
/// );
/// final result = await imageProcessor.process(imageBytes, model);
///
/// // Object Detection
/// final yoloProcessor = YoloProcessor(
///   preprocessConfig: YoloPreprocessConfig(
///     targetWidth: 640,
///     targetHeight: 640,
///   ),
///   classLabels: cocoLabels,
///   confidenceThreshold: 0.25,
///   iouThreshold: 0.45,
/// );
/// final detections = await yoloProcessor.process(imageBytes, model);
/// ```
library example_processors;

// Image classification
export 'image_processor.dart';

// Object detection (YOLO)
export 'yolo_processor.dart';