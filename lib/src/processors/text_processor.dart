import 'dart:math' as math;
import 'dart:typed_data';

import 'package:meta/meta.dart';

import '../generated/executorch_api.dart';
import 'base_processor.dart';

/// Simple tokenizer interface for text processing
abstract class TextTokenizer {
  /// Tokenizes text into a list of token IDs
  List<int> tokenize(String text);

  /// Maximum sequence length supported
  int get maxLength;

  /// Vocabulary size
  int get vocabularySize;

  /// Padding token ID
  int get padTokenId;

  /// Unknown token ID for out-of-vocabulary words
  int get unkTokenId;
}

/// Simple word-based tokenizer implementation
class SimpleTokenizer implements TextTokenizer {
  SimpleTokenizer({
    required this.vocabulary,
    required this.maxLength,
    this.padTokenId = 0,
    this.unkTokenId = 1,
  });

  final Map<String, int> vocabulary;

  @override
  final int maxLength;

  @override
  final int padTokenId;

  @override
  final int unkTokenId;

  @override
  int get vocabularySize => vocabulary.length;

  @override
  List<int> tokenize(String text) {
    // Simple whitespace tokenization and lowercasing
    final words = text.toLowerCase().split(RegExp(r'\s+'));
    final tokens = <int>[];

    for (final word in words) {
      if (word.isNotEmpty) {
        final tokenId = vocabulary[word] ?? unkTokenId;
        tokens.add(tokenId);
      }
    }

    // Truncate if too long
    if (tokens.length > maxLength) {
      tokens.removeRange(maxLength, tokens.length);
    }

    // Pad if too short
    while (tokens.length < maxLength) {
      tokens.add(padTokenId);
    }

    return tokens;
  }
}

/// BPE (Byte Pair Encoding) tokenizer for more advanced text processing
class BPETokenizer implements TextTokenizer {
  BPETokenizer({
    required this.vocabulary,
    required this.merges,
    required this.maxLength,
    this.padTokenId = 0,
    this.unkTokenId = 1,
  });

  final Map<String, int> vocabulary;
  final List<List<String>> merges;

  @override
  final int maxLength;

  @override
  final int padTokenId;

  @override
  final int unkTokenId;

  @override
  int get vocabularySize => vocabulary.length;

  @override
  List<int> tokenize(String text) {
    // This is a simplified BPE implementation
    // In practice, you'd use a proper BPE tokenizer like tiktoken or Hugging Face tokenizers

    var tokens = text.toLowerCase().split('').map((char) => char).toList();

    // Apply BPE merges
    for (final merge in merges) {
      if (merge.length != 2) continue;

      final first = merge[0];
      final second = merge[1];
      final combined = first + second;

      final newTokens = <String>[];
      int i = 0;
      while (i < tokens.length) {
        if (i < tokens.length - 1 &&
            tokens[i] == first &&
            tokens[i + 1] == second) {
          newTokens.add(combined);
          i += 2;
        } else {
          newTokens.add(tokens[i]);
          i++;
        }
      }
      tokens = newTokens;
    }

    // Convert to token IDs
    final tokenIds = tokens.map((token) => vocabulary[token] ?? unkTokenId).toList();

    // Truncate and pad
    if (tokenIds.length > maxLength) {
      tokenIds.removeRange(maxLength, tokenIds.length);
    }

    while (tokenIds.length < maxLength) {
      tokenIds.add(padTokenId);
    }

    return tokenIds;
  }
}

/// Result of text classification
@immutable
class TextClassificationResult {
  const TextClassificationResult({
    required this.className,
    required this.confidence,
    required this.classIndex,
    required this.allProbabilities,
    this.tokenCount,
  });

  /// The predicted class name/label
  final String className;

  /// Confidence score for the prediction (0.0 to 1.0)
  final double confidence;

  /// Index of the predicted class
  final int classIndex;

  /// All class probabilities (softmax outputs)
  final List<double> allProbabilities;

  /// Number of tokens in the input text (optional)
  final int? tokenCount;

  @override
  String toString() =>
      'TextClassificationResult(class: $className, confidence: ${(confidence * 100).toStringAsFixed(1)}%, tokens: $tokenCount)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TextClassificationResult &&
          runtimeType == other.runtimeType &&
          className == other.className &&
          confidence == other.confidence &&
          classIndex == other.classIndex;

  @override
  int get hashCode =>
      className.hashCode ^ confidence.hashCode ^ classIndex.hashCode;
}

/// Preprocessor for text data to tensor conversion
class TextClassificationPreprocessor extends ExecuTorchPreprocessor<String> {
  TextClassificationPreprocessor({
    required this.tokenizer,
  });

  final TextTokenizer tokenizer;

  @override
  String get inputTypeName => 'Text (String)';

  @override
  bool validateInput(String input) {
    return input.isNotEmpty && input.trim().isNotEmpty;
  }

  @override
  Future<List<TensorData>> preprocess(String input, {ModelMetadata? metadata}) async {
    try {
      // Tokenize the input text
      final tokenIds = tokenizer.tokenize(input);

      // Create input_ids tensor
      final inputIdsTensor = ProcessorTensorUtils.createTensor(
        shape: [1, tokenIds.length], // [batch_size, sequence_length]
        dataType: TensorType.int32,
        data: tokenIds,
        name: 'input_ids',
      );

      // Create attention_mask tensor (1 for real tokens, 0 for padding)
      final attentionMask = tokenIds
          .map((tokenId) => tokenId != tokenizer.padTokenId ? 1 : 0)
          .toList();

      final attentionMaskTensor = ProcessorTensorUtils.createTensor(
        shape: [1, attentionMask.length],
        dataType: TensorType.int32,
        data: attentionMask,
        name: 'attention_mask',
      );

      return [inputIdsTensor, attentionMaskTensor];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('Text preprocessing failed: $e', e);
    }
  }
}

/// Postprocessor for text classification results
class TextClassificationPostprocessor extends ExecuTorchPostprocessor<TextClassificationResult> {
  TextClassificationPostprocessor({
    required this.classLabels,
  });

  final List<String> classLabels;

  @override
  String get outputTypeName => 'Text Classification Result';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    if (outputs.isEmpty) return false;

    final output = outputs.first;
    if (output.dataType != TensorType.float32) return false;

    // Check if shape represents logits/probabilities
    final shape = output.shape?.where((dim) => dim != null).toList() ?? [];
    if (shape.isEmpty) return false;

    // Should have correct number of classes
    final outputSize = shape.last!;
    return outputSize == classLabels.length;
  }

  @override
  Future<TextClassificationResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
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

      return TextClassificationResult(
        className: className,
        confidence: maxProb,
        classIndex: maxIndex,
        allProbabilities: probabilities,
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException('Text classification postprocessing failed: $e', e);
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

/// Complete text classification processor
class TextClassificationProcessor extends ExecuTorchProcessor<String, TextClassificationResult> {
  TextClassificationProcessor({
    required this.tokenizer,
    required this.classLabels,
  }) : _preprocessor = TextClassificationPreprocessor(tokenizer: tokenizer),
       _postprocessor = TextClassificationPostprocessor(classLabels: classLabels);

  final TextTokenizer tokenizer;
  final List<String> classLabels;
  final TextClassificationPreprocessor _preprocessor;
  final TextClassificationPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<String> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<TextClassificationResult> get postprocessor => _postprocessor;
}

/// Sentiment analysis processor (specialized text classification)
class SentimentAnalysisProcessor extends TextClassificationProcessor {
  SentimentAnalysisProcessor({
    required TextTokenizer tokenizer,
  }) : super(
    tokenizer: tokenizer,
    classLabels: const ['negative', 'neutral', 'positive'],
  );
}

/// Topic classification processor (specialized text classification)
class TopicClassificationProcessor extends TextClassificationProcessor {
  TopicClassificationProcessor({
    required TextTokenizer tokenizer,
    required List<String> topics,
  }) : super(
    tokenizer: tokenizer,
    classLabels: topics,
  );
}