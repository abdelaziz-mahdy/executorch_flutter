// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Tests for tokenizer format diagnosis and metadata.
///
/// These cover the parts that need no native library, so they run today.
/// Encode/decode round-trips are exercised natively — see the tokenizer
/// harness in the `executorch_native` repo, which is where a mismatch would
/// actually show up.
library;

import 'dart:convert';
import 'dart:io';

import 'package:executorch_dart/executorch_dart.dart';
import 'package:executorch_dart/src/tokenizer.dart' show diagnoseTokenizerJson;
import 'package:test/test.dart';

/// Shaped like the real thing: field names and values are copied from
/// `sentence-transformers/all-MiniLM-L6-v2`, minus the 30k-entry vocabulary.
String _bertTokenizerJson() => json.encode({
  'version': '1.0',
  'normalizer': {
    'type': 'BertNormalizer',
    'clean_text': true,
    'handle_chinese_chars': true,
    'strip_accents': null,
    'lowercase': true,
  },
  'pre_tokenizer': {'type': 'BertPreTokenizer'},
  'model': {
    'type': 'WordPiece',
    'unk_token': '[UNK]',
    'continuing_subword_prefix': '##',
    'max_input_chars_per_word': 100,
    'vocab': {'[PAD]': 0, '[UNK]': 100, 'hello': 7592},
  },
});

String _bpeTokenizerJson() => json.encode({
  'version': '1.0',
  'pre_tokenizer': {'type': 'ByteLevel'},
  'model': {
    'type': 'BPE',
    'vocab': {'hello': 31373},
    'merges': <String>[],
  },
});

void main() {
  group('diagnoseTokenizerJson', () {
    test('names WordPiece and BertNormalizer explicitly', () {
      final message = diagnoseTokenizerJson(_bertTokenizerJson());

      expect(message, isNotNull);
      // The whole point is telling the user WHICH parts are unsupported, so
      // assert on the specifics rather than just "an error happened".
      expect(message, contains('WordPiece'));
      expect(message, contains('BertNormalizer'));
      // And that they learn what would work instead.
      expect(message, contains('SentencePiece'));
      expect(message, contains('BPE'));
    });

    test('mentions the model families a user would recognise', () {
      final message = diagnoseTokenizerJson(_bertTokenizerJson())!;
      expect(message, contains('sentence-transformers'));
      expect(message, contains('BERT'));
    });

    test('stays silent for a supported BPE tokenizer', () {
      // A BPE file that fails to load failed for some other reason; inventing
      // a format complaint would send the user down the wrong path.
      expect(diagnoseTokenizerJson(_bpeTokenizerJson()), isNull);
    });

    test('flags a Unigram model even without a BERT normalizer', () {
      final message = diagnoseTokenizerJson(
        json.encode({
          'model': {'type': 'Unigram', 'vocab': <String, int>{}},
        }),
      );
      expect(message, isNotNull);
      expect(message, contains('Unigram'));
    });

    test('flags an unimplemented normalizer on an otherwise fine model', () {
      final message = diagnoseTokenizerJson(
        json.encode({
          'normalizer': {'type': 'Precompiled'},
          'model': {'type': 'BPE', 'vocab': <String, int>{}},
        }),
      );
      expect(message, isNotNull);
      expect(message, contains('Precompiled'));
      expect(message, contains('Replace'));
    });

    test('accepts every normalizer ExecuTorch implements', () {
      for (final type in ['Replace', 'Prepend', 'Sequence', 'NFC']) {
        final message = diagnoseTokenizerJson(
          json.encode({
            'normalizer': {'type': type},
            'model': {'type': 'BPE', 'vocab': <String, int>{}},
          }),
        );
        expect(message, isNull, reason: '$type should be accepted');
      }
    });

    test('returns null for non-JSON input', () {
      // SentencePiece and llama2.c files are binary and perfectly valid —
      // failing to parse one is not evidence of anything.
      expect(diagnoseTokenizerJson('\x00\x01binary garbage'), isNull);
      expect(diagnoseTokenizerJson(''), isNull);
    });

    test('returns null for JSON that is not an object', () {
      expect(diagnoseTokenizerJson('[1, 2, 3]'), isNull);
      expect(diagnoseTokenizerJson('"a string"'), isNull);
    });

    test('survives a tokenizer.json with unexpected field types', () {
      // Hand-edited configs turn up in the wild; the diagnostic must not be
      // the thing that crashes.
      expect(diagnoseTokenizerJson('{"model": "not an object"}'), isNull);
      expect(diagnoseTokenizerJson('{"model": {"type": 42}}'), isNull);
      expect(diagnoseTokenizerJson('{"model": {"type": null}}'), isNull);
      expect(diagnoseTokenizerJson('{"normalizer": []}'), isNull);
      expect(diagnoseTokenizerJson('{}'), isNull);
    });
  });

  group('TokenizerFormat', () {
    test('maps every native identifier', () {
      expect(
        TokenizerFormat.fromNative('hf_json'),
        TokenizerFormat.huggingFace,
      );
      expect(TokenizerFormat.fromNative('tiktoken'), TokenizerFormat.tikToken);
      expect(
        TokenizerFormat.fromNative('sentencepiece'),
        TokenizerFormat.sentencePiece,
      );
      expect(TokenizerFormat.fromNative('llama2c'), TokenizerFormat.llama2c);
    });

    test('falls back to unknown rather than throwing', () {
      // A newer native library may report a format this enum predates; that
      // must not crash a caller who only wanted vocabSize.
      expect(
        TokenizerFormat.fromNative('some_future_format'),
        TokenizerFormat.unknown,
      );
      expect(TokenizerFormat.fromNative(''), TokenizerFormat.unknown);
    });
  });

  group('Tokenizer.load', () {
    test('reports a missing file as an IO error, not a format error', () async {
      await expectLater(
        () => Tokenizer.load('/no/such/tokenizer.json'),
        throwsA(isA<ExecuTorchIOException>()),
      );
    });

    test('missing-file error names the path', () async {
      try {
        await Tokenizer.load('/no/such/tokenizer.json');
        fail('expected an exception');
      } on ExecuTorchIOException catch (e) {
        expect(e.message, contains('/no/such/tokenizer.json'));
      }
    });

    test('a directory is rejected, not treated as a tokenizer', () async {
      final dir = Directory.systemTemp.createTempSync('tokenizer_test');
      addTearDown(() => dir.deleteSync(recursive: true));
      // File.existsSync() is false for a directory, so this takes the IO path.
      await expectLater(
        () => Tokenizer.load(dir.path),
        throwsA(isA<ExecuTorchIOException>()),
      );
    });
  });
}
