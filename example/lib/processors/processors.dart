/// ExecuTorch Flutter Example Processors
///
/// This library contains reference implementations of processors for common
/// machine learning tasks. These serve as examples showing how to build
/// custom processors using the base classes from the main ExecuTorch Flutter library.
///
/// ## Available Reference Processors
///
/// ### Image Processing
/// - [ImageNetProcessor]: Complete ImageNet classification pipeline
/// - [ImageNetPreprocessor]: Image preprocessing with normalization
/// - [ImageNetPostprocessor]: Classification result postprocessing
/// - [ImagePreprocessConfig]: Configuration for image preprocessing
/// - [ClassificationResult]: Image classification results
///
/// ### Object Detection
/// - [ObjectDetectionProcessor]: Complete object detection pipeline
/// - [ObjectDetectionPreprocessor]: Object detection preprocessing
/// - [ObjectDetectionPostprocessor]: Bounding box and NMS postprocessing
/// - [ObjectDetectionPreprocessConfig]: Configuration for object detection
/// - [ObjectDetectionResult]: Object detection results with bounding boxes
/// - [DetectedObject]: Individual detected object
/// - [BoundingBox]: Bounding box coordinates
///
/// ### Text Processing
/// - [TextClassificationProcessor]: Complete text classification pipeline
/// - [TextClassificationPreprocessor]: Text tokenization and preprocessing
/// - [TextClassificationPostprocessor]: Text classification postprocessing
/// - [SentimentAnalysisProcessor]: Specialized sentiment analysis
/// - [TopicClassificationProcessor]: Specialized topic classification
/// - [TextClassificationResult]: Text classification results
/// - [SimpleTokenizer]: Simple word-based tokenizer
/// - [BPETokenizer]: Byte Pair Encoding tokenizer
///
/// ### Audio Processing
/// - [AudioClassificationProcessor]: Complete audio classification pipeline
/// - [AudioClassificationPreprocessor]: Audio preprocessing and feature extraction
/// - [AudioClassificationPostprocessor]: Audio classification postprocessing
/// - [SpeechCommandProcessor]: Specialized speech command recognition
/// - [MusicGenreProcessor]: Specialized music genre classification
/// - [EnvironmentalSoundProcessor]: Environmental sound classification
/// - [AudioClassificationResult]: Audio classification results
/// - [AudioPreprocessConfig]: Configuration for audio preprocessing
///
/// ## Usage
///
/// These processors are reference implementations. You can:
/// 1. Use them directly for compatible models
/// 2. Modify them for your specific requirements
/// 3. Use them as templates for building your own processors
///
/// ### Example Usage
/// ```dart
/// import 'processors/processors.dart';
///
/// final processor = ImageNetProcessor(
///   preprocessConfig: const ImagePreprocessConfig(
///     targetWidth: 224,
///     targetHeight: 224,
///   ),
///   classLabels: myClassLabels,
/// );
///
/// final result = await processor.process(imageBytes, model);
/// ```
library example_processors;

// Image processing
export 'image_processor.dart';

// Object detection
export 'object_detection_processor.dart';

// Text processing
export 'text_processor.dart';

// Audio processing
export 'audio_processor.dart';