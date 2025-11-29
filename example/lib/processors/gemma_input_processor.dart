import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_input.dart';
import 'base_processor.dart';

/// Input processor for Gemma text generation model
/// Converts text input to token IDs tensor
class GemmaInputProcessor extends InputProcessor<TextPromptInput> {
  final int maxLength;
  final Map<String, int> vocabulary;
  final int padTokenId;
  final int bosTokenId;
  final int eosTokenId;

  const GemmaInputProcessor({
    required this.maxLength,
    required this.vocabulary,
    required this.padTokenId,
    required this.bosTokenId,
    required this.eosTokenId,
  });

  @override
  Future<List<TensorData>> process(TextPromptInput input) async {
    // Tokenize text (simple word-level tokenization as placeholder)
    // In production, use a proper SentencePiece tokenizer
    final tokens = _tokenize(input.text);

    // Add BOS token at start
    final List<int> tokenIds = [bosTokenId, ...tokens];

    // Trim or pad to maxLength
    if (tokenIds.length > maxLength) {
      tokenIds.removeRange(maxLength, tokenIds.length);
    } else {
      while (tokenIds.length < maxLength) {
        tokenIds.add(padTokenId);
      }
    }

    // Convert to Int32List (required by ExecuTorch)
    final int32Data = Int32List.fromList(tokenIds);

    // Convert to Uint8List (bytes)
    final bytes = Uint8List.view(int32Data.buffer);

    // Create tensor data
    final tensorData = TensorData(
      shape: [1, maxLength], // [batch_size=1, sequence_length]
      dataType: TensorType.int32,
      data: bytes,
      name: 'input_ids',
    );

    return [tensorData];
  }

  /// Simple tokenization (word-level)
  /// Note: In production, replace with proper SentencePiece tokenizer
  List<int> _tokenize(String text) {
    final words = text.toLowerCase().split(RegExp(r'\s+'));
    final tokens = <int>[];

    for (final word in words) {
      // Check if word exists in vocabulary
      if (vocabulary.containsKey(word)) {
        tokens.add(vocabulary[word]!);
      } else {
        // Use a default unknown token ID (typically 0 or 1)
        tokens.add(vocabulary['<unk>'] ?? 0);
      }
    }

    return tokens;
  }
}

/// Simple vocabulary loader for Gemma tokenizer
/// In production, use a proper SentencePiece tokenizer library
class GemmaVocabularyLoader {
  /// Load vocabulary from a simple text file (word per line)
  /// Format: `word\tid`
  static Future<Map<String, int>> loadFromFile(String path) async {
    // Placeholder implementation
    // In production, load from the actual tokenizer vocabulary file
    return {
      '<pad>': 0,
      '<unk>': 1,
      '<bos>': 2,
      '<eos>': 3,
      // Add more tokens...
    };
  }

  /// Create a simple character-level vocabulary
  static Map<String, int> createCharacterVocabulary() {
    final vocab = <String, int>{'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3};

    // Add ASCII printable characters
    int id = 4;
    for (int i = 32; i < 127; i++) {
      vocab[String.fromCharCode(i)] = id++;
    }

    return vocab;
  }
}
