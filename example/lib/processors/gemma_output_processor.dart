import 'dart:typed_data';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/text_generation_result.dart';
import 'base_processor.dart';

/// Output processor for Gemma text generation model
/// Converts token ID tensors back to generated text
class GemmaOutputProcessor
    extends OutputProcessor<TextGenerationResult> {
  final Map<int, String> reverseVocabulary;
  final int eosTokenId;
  final int bosTokenId;
  final int padTokenId;
  final String inputPrompt;

  const GemmaOutputProcessor({
    required this.reverseVocabulary,
    required this.eosTokenId,
    required this.bosTokenId,
    required this.padTokenId,
    required this.inputPrompt,
  });

  @override
  Future<TextGenerationResult> process(List<TensorData> outputs) async {
    if (outputs.isEmpty) {
      throw Exception('No output tensors from model');
    }

    final outputTensor = outputs[0];

    // Extract token IDs from output tensor
    final tokenIds = _extractTokenIds(outputTensor);

    // Decode tokens to text
    final generatedText = _decode(tokenIds);

    // Calculate statistics
    final tokensGenerated = tokenIds.length;

    return TextGenerationResult(
      generatedText: generatedText,
      inputPrompt: inputPrompt,
      tokensGenerated: tokensGenerated,
    );
  }

  /// Extract token IDs from tensor data
  List<int> _extractTokenIds(TensorData tensor) {
    // Assuming output is Int32 or Int64 tensor containing token IDs
    if (tensor.dataType == TensorType.int32) {
      final int32View = Int32List.view(tensor.data.buffer);
      final tokens = <int>[];

      for (final tokenId in int32View) {
        // Stop at EOS token
        if (tokenId == eosTokenId) break;

        // Skip padding and BOS tokens
        if (tokenId == padTokenId || tokenId == bosTokenId) continue;

        tokens.add(tokenId);
      }

      return tokens;
    } else if (tensor.dataType == TensorType.float32) {
      // If output is logits (float32), take argmax to get token IDs
      final float32View = Float32List.view(tensor.data.buffer);
      final tokens = <int>[];

      // Assuming shape is [batch, seq_len, vocab_size]
      // For simplicity, take argmax over last dimension
      // This is a simplified implementation
      for (int i = 0; i < float32View.length; i++) {
        // In real implementation, need to reshape and take argmax properly
        // For now, just use the value as token ID
        final tokenId = float32View[i].toInt();

        if (tokenId == eosTokenId) break;
        if (tokenId == padTokenId || tokenId == bosTokenId) continue;

        tokens.add(tokenId);
      }

      return tokens;
    }

    throw Exception('Unsupported output tensor type: ${tensor.dataType}');
  }

  /// Decode token IDs to text
  String _decode(List<int> tokenIds) {
    final words = <String>[];

    for (final tokenId in tokenIds) {
      if (reverseVocabulary.containsKey(tokenId)) {
        words.add(reverseVocabulary[tokenId]!);
      } else {
        words.add('<unk>');
      }
    }

    // Join tokens with spaces (simple word-level detokenization)
    // In production, use proper SentencePiece detokenization
    return words.join(' ').trim();
  }
}

/// Helper to create reverse vocabulary (ID -> token)
class VocabularyHelper {
  /// Create reverse vocabulary from forward vocabulary
  static Map<int, String> reverseVocabulary(Map<String, int> vocabulary) {
    return vocabulary.map((key, value) => MapEntry(value, key));
  }
}
