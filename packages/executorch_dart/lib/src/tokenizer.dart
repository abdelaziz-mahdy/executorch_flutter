// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Standalone tokenizer: text to token ids, and back.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'executorch_errors.dart';
import 'ffi/native_tokenizer.dart';

/// Tokenizer formats the native library can read.
enum TokenizerFormat {
  /// HuggingFace `tokenizer.json`. **BPE models only** — see
  /// [Tokenizer.load] for what that excludes.
  huggingFace('hf_json'),

  /// TikToken (OpenAI) vocabularies.
  tikToken('tiktoken'),

  /// SentencePiece `.model` files.
  sentencePiece('sentencepiece'),

  /// The llama2.c binary tokenizer format.
  llama2c('llama2c'),

  /// Reported when the native side names a format this enum predates.
  unknown('unknown');

  const TokenizerFormat(this.nativeName);

  /// Identifier used by the native layer.
  final String nativeName;

  /// Map a native format identifier onto this enum, falling back to
  /// [TokenizerFormat.unknown] for names added natively since this was
  /// written.
  static TokenizerFormat fromNative(String name) => values.firstWhere(
    (f) => f.nativeName == name,
    orElse: () => TokenizerFormat.unknown,
  );
}

/// Converts text to token ids and back, independently of any model.
///
/// This is the counterpart to `ExecuTorchModel` for text inputs. Encoder
/// models — embeddings, classification, retrieval — take token ids as an
/// input tensor and produce a vector in one `forward()` call, with no
/// generation loop. Tokenize here, build your own `TensorData`, run the
/// model:
///
/// ```dart
/// final tokenizer = await Tokenizer.load('/path/to/tokenizer.json');
/// final ids = tokenizer.encode('some text');
///
/// final input = TensorData(
///   shape: [1, ids.length],
///   dataType: TensorType.int64,
///   data: Int64List.fromList(ids).buffer.asUint8List(),
/// );
/// final outputs = await model.forward([input]);
///
/// tokenizer.dispose();
/// ```
///
/// For generative models use `ExecuTorchLLM` instead — it owns a tokenizer
/// internally and streams text.
class Tokenizer {
  Tokenizer._(this._native);

  final NativeTokenizer _native;

  /// Load a tokenizer file, auto-detecting the format.
  ///
  /// **Supported:** HuggingFace `tokenizer.json` built on **BPE** (GPT-2,
  /// Llama, Gemma, Qwen, Mistral), SentencePiece `.model` files, TikToken,
  /// and the llama2.c binary format.
  ///
  /// **Not supported:** WordPiece / BERT-family tokenizers. ExecuTorch's
  /// HuggingFace reader derives from a BPE base and its normalizer implements
  /// only `Replace`, `Prepend`, `Sequence` and `NFC`, so a `BertNormalizer`
  /// is rejected. That rules out BERT, DistilBERT, MiniLM and most
  /// sentence-transformers models. When one of those is passed, the thrown
  /// exception says so by name rather than reporting a generic parse failure.
  ///
  /// Throws [ExecuTorchIOException] when the file does not exist, and
  /// [ExecuTorchModelException] when no reader accepts it.
  static Future<Tokenizer> load(String path) async {
    final file = File(path);
    if (!file.existsSync()) {
      throw ExecuTorchIOException('Tokenizer file not found: $path');
    }

    try {
      return Tokenizer._(NativeTokenizer.create(path));
    } on ExecuTorchException catch (e) {
      // The native side can only report "nothing recognized this file". Look
      // at the file ourselves so the caller learns WHY, instead of guessing.
      final diagnosis = await _diagnose(file);
      if (diagnosis == null) rethrow;
      throw ExecuTorchModelException(
        '$diagnosis\n\nUnderlying error: ${e.message}',
      );
    }
  }

  /// Explain an unsupported tokenizer file, or null when there is nothing
  /// more useful to say than the original error.
  ///
  /// Runs only after a failed load, so the happy path never pays for it.
  static Future<String?> _diagnose(File file) async {
    try {
      // Anything this large is not a tokenizer.json we can help with.
      if (await file.length() > 64 * 1024 * 1024) return null;
      return diagnoseTokenizerJson(await file.readAsString());
    } catch (_) {
      return null; // Unreadable — nothing to add beyond the original error.
    }
  }

  /// Which reader claimed the file.
  TokenizerFormat get format => TokenizerFormat.fromNative(_native.format);

  /// Size of the vocabulary.
  int get vocabSize => _native.vocabSize;

  /// Beginning-of-sequence token id.
  int get bosId => _native.bosId;

  /// End-of-sequence token id.
  int get eosId => _native.eosId;

  /// Whether [dispose] has been called.
  bool get isDisposed => _native.isDisposed;

  /// Encode [text] into token ids.
  ///
  /// [bosCount] and [eosCount] are how many BOS/EOS tokens to add — counts,
  /// not flags, because some models want two. Whether they are honoured
  /// depends on the tokenizer: readers with no BOS/EOS configured ignore
  /// them.
  ///
  /// Multi-byte input is fine; the readers work on bytes, not code points.
  Uint64List encode(String text, {int bosCount = 0, int eosCount = 0}) =>
      _native.encode(text, bosCount: bosCount, eosCount: eosCount);

  /// Decode token ids back into text.
  ///
  /// Set [skipSpecialTokens] to leave control tokens out of the output.
  String decode(List<int> ids, {bool skipSpecialTokens = false}) =>
      _native.decode(ids, skipSpecialTokens: skipSpecialTokens);

  /// Free the native tokenizer. Idempotent.
  ///
  /// As everywhere else in this package, lifetime is yours to manage —
  /// nothing is released automatically.
  void dispose() => _native.dispose();
}

/// Explain why a `tokenizer.json` is unsupported, or null when nothing useful
/// can be said.
///
/// Split out from [Tokenizer.load] because it is pure text analysis: the
/// native library only ever reports "no reader accepted this file", which
/// tells a user nothing about what to do next. Keeping it separate also means
/// it can be tested without a native build.
///
/// Returns null for input that is not JSON at all — the binary formats are
/// legitimate inputs and a failure there is not diagnosable this way.
String? diagnoseTokenizerJson(String contents) {
  Object? decoded;
  try {
    decoded = json.decode(contents);
  } catch (_) {
    return null;
  }
  if (decoded is! Map<String, dynamic>) return null;

  // Read defensively. Hand-edited and generator-produced configs both turn up
  // in the wild, and a diagnostic that throws on a malformed file would
  // replace a useful error with a confusing one.
  String? typeOf(Object? node) {
    if (node is! Map) return null;
    final type = node['type'];
    return type is String ? type : null;
  }

  final modelType = typeOf(decoded['model']);
  final normalizerType = typeOf(decoded['normalizer']);

  const supported =
      'Supported here: HuggingFace tokenizer.json built on '
      'BPE, SentencePiece, TikToken, llama2.c.';

  if (modelType == 'WordPiece' || normalizerType == 'BertNormalizer') {
    final modelName = modelType ?? 'unknown';
    final normName = normalizerType ?? 'none';
    return 'This is a WordPiece/BERT-family tokenizer '
        '(model: $modelName, normalizer: $normName), which the ExecuTorch '
        'tokenizer library cannot read: its HuggingFace reader supports BPE '
        'only, and its normalizer has no BertNormalizer. This affects BERT, '
        'DistilBERT, MiniLM and most sentence-transformers models.\n'
        '$supported';
  }

  if (modelType != null && modelType != 'BPE') {
    return 'This tokenizer uses the $modelType model, which the ExecuTorch '
        'HuggingFace reader does not implement — it supports BPE only.\n'
        '$supported';
  }

  // Normalizers ExecuTorch implements; anything else throws inside the
  // native reader.
  const knownNormalizers = {'Replace', 'Prepend', 'Sequence', 'NFC'};
  if (normalizerType != null && !knownNormalizers.contains(normalizerType)) {
    return 'This tokenizer uses the $normalizerType normalizer, which '
        'ExecuTorch does not implement. Supported normalizers are '
        '${knownNormalizers.join(", ")}.';
  }

  return null;
}
