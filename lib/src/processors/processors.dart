/// ExecuTorch Flutter Processors
///
/// This library provides preprocessing and postprocessing classes for common
/// machine learning tasks. These processors convert domain-specific input
/// (images, text, audio) into tensors for ExecuTorch model inference and
/// convert model outputs back into meaningful results.
///
/// ## Core Architecture
///
/// The processor architecture is based on three main abstract classes:
///
/// - [ExecuTorchPreprocessor]: Converts input data to tensors
/// - [ExecuTorchPostprocessor]: Converts output tensors to results
/// - [ExecuTorchProcessor]: Combines preprocessing and postprocessing
///
/// ## Available Processors
///
/// ### Image Classification
/// - [ImageNetProcessor]: Complete ImageNet classification pipeline
/// - [ImageNetPreprocessor]: Image preprocessing with normalization
/// - [ImageNetPostprocessor]: Classification result postprocessing
/// - [ImagePreprocessConfig]: Configuration for image preprocessing
/// - [ClassificationResult]: Image classification results
///
/// ### Text Classification
/// - [TextClassificationProcessor]: Complete text classification pipeline
/// - [TextClassificationPreprocessor]: Text tokenization and preprocessing
/// - [TextClassificationPostprocessor]: Text classification postprocessing
/// - [SentimentAnalysisProcessor]: Specialized sentiment analysis
/// - [TopicClassificationProcessor]: Specialized topic classification
/// - [TextClassificationResult]: Text classification results
/// - [SimpleTokenizer]: Simple word-based tokenizer
/// - [BPETokenizer]: Byte Pair Encoding tokenizer
///
/// ### Audio Classification
/// - [AudioClassificationProcessor]: Complete audio classification pipeline
/// - [AudioClassificationPreprocessor]: Audio preprocessing and feature extraction
/// - [AudioClassificationPostprocessor]: Audio classification postprocessing
/// - [SpeechCommandProcessor]: Specialized speech command recognition
/// - [MusicGenreProcessor]: Specialized music genre classification
/// - [EnvironmentalSoundProcessor]: Environmental sound classification
/// - [AudioClassificationResult]: Audio classification results
/// - [AudioPreprocessConfig]: Configuration for audio preprocessing
///
/// ## Utilities
/// - [ProcessorTensorUtils]: Tensor creation and manipulation utilities
/// - [ProcessorException]: Base exception class for processor errors
/// - [PreprocessingException]: Preprocessing-specific exceptions
/// - [PostprocessingException]: Postprocessing-specific exceptions
/// - [InvalidInputException]: Invalid input exceptions
/// - [InvalidOutputException]: Invalid output exceptions
///
/// ## Usage Examples
///
/// ### Image Classification
/// ```dart
/// final processor = ImageNetProcessor(
///   preprocessConfig: const ImagePreprocessConfig(
///     targetWidth: 224,
///     targetHeight: 224,
///     normalizeToFloat: true,
///     meanSubtraction: [0.485, 0.456, 0.406],
///     standardDeviation: [0.229, 0.224, 0.225],
///   ),
///   classLabels: imageNetLabels,
/// );
///
/// final result = await processor.process(imageBytes, model);
/// print('Predicted: ${result.className} (${result.confidence})');
/// ```
///
/// ### Text Classification
/// ```dart
/// final processor = SentimentAnalysisProcessor(
///   tokenizer: SimpleTokenizer(
///     vocabulary: vocabulary,
///     maxLength: 128,
///   ),
/// );
///
/// final result = await processor.process('This movie is great!', model);
/// print('Sentiment: ${result.className}');
/// ```
///
/// ### Audio Classification
/// ```dart
/// final processor = SpeechCommandProcessor(
///   sampleRate: 16000,
///   windowSize: 1024,
///   commands: ['yes', 'no', 'stop', 'go'],
/// );
///
/// final result = await processor.process(audioSamples, model);
/// print('Command: ${result.className}');
/// ```
library processors;

// Base processor classes and utilities
export 'base_processor.dart';

// Image processing
export 'image_processor.dart';

// Text processing
export 'text_processor.dart';

// Audio processing
export 'audio_processor.dart';