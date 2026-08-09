// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Runtime tests for [Tokenizer] against the real native library.
///
/// Unlike `tokenizer_diagnosis_test.dart`, these actually load a tokenizer and
/// call encode/decode through FFI, so they exercise the C API, the memory
/// ownership rules across the boundary, and the UTF-8 handling.
///
/// The fixture is generated here rather than downloaded. A real
/// `tokenizer.json` is ~500 KB to 5 MB and would make the test suite depend on
/// the network; a synthetic BPE vocabulary is a few hundred bytes, is
/// deterministic, and exercises the same reader.
///
/// ## Both build configurations matter here
///
/// The tokenizer ships in the base library, so it must work with **`llm:
/// false`** — that is the entire point of the feature, since encoder models
/// have no use for the generation runner.
///
/// It is easy to test only the wrong one. With `llm: true` the LLM runner
/// pulls in `regex_lookahead` transitively, so lookahead pre-tokenizer
/// patterns load that would fail on the base variant. A tokenizer bug
/// reported against the base build was invisible under `llm: true` for
/// exactly this reason (executorch_flutter#45).
///
/// Both are covered today, but by two different jobs rather than by anything
/// in this file:
///
/// - **`llm: false`** — the pure-Dart CI job, which stages this package
///   outside the workspace. `packages/executorch_dart/pubspec.yaml` sets no
///   `llm` key, so it defaults off.
/// - **`llm: true`** — the Flutter jobs, which inherit the workspace root
///   pubspec.
///
/// Adding `llm: true` to this package's own pubspec would silently delete the
/// base-variant coverage.
library;

import 'dart:convert';
import 'dart:io';

import 'package:executorch_dart/executorch_dart.dart';
import 'package:test/test.dart';

/// GPT-2's `bytes_to_unicode`: every byte mapped to a printable character, so
/// a byte-level vocabulary can spell any input.
///
/// Printable ASCII and two Latin-1 runs map to themselves; everything else is
/// shifted into the private-use area starting at U+0100.
List<String> _byteAlphabet() {
  final direct = <int>[
    ...List.generate(0x7E - 0x21 + 1, (i) => 0x21 + i),
    ...List.generate(0xAC - 0xA1 + 1, (i) => 0xA1 + i),
    ...List.generate(0xFF - 0xAE + 1, (i) => 0xAE + i),
  ];
  final mapped = <String>[];
  var shifted = 0;
  for (var b = 0; b < 256; b++) {
    mapped.add(
      direct.contains(b)
          ? String.fromCharCode(b)
          : String.fromCharCode(256 + shifted++),
    );
  }
  return mapped;
}

/// A minimal but genuine HuggingFace BPE `tokenizer.json`.
///
/// ByteLevel pre-tokenizer, explicit merges, and the `Ġ` word-boundary marker
/// that byte-level BPE uses for a leading space — the same shape GPT-2 has,
/// just tiny.
String _syntheticBpeTokenizer() {
  final vocab = <String, int>{};
  var next = 0;

  // The full 256-entry byte alphabet, in GPT-2's bytes-to-unicode mapping.
  // Without it the vocabulary cannot represent bytes outside its handful of
  // literal tokens, and any non-ASCII input encodes to nothing at all — which
  // is a property of the fixture, not of the tokenizer.
  for (final ch in _byteAlphabet()) {
    vocab.putIfAbsent(ch, () => next++);
  }

  for (final token in [
    'Ġ',
    'Ġh',
    'he',
    'hel',
    'lo',
    'hello',
    'Ġworld',
    'wor',
    'ld',
    '<unk>',
  ]) {
    vocab.putIfAbsent(token, () => next++);
  }

  return json.encode({
    'version': '1.0',
    'truncation': null,
    'padding': null,
    'added_tokens': <Object>[],
    'normalizer': null,
    'pre_tokenizer': {
      'type': 'ByteLevel',
      'add_prefix_space': false,
      'trim_offsets': true,
      'use_regex': true,
    },
    'post_processor': null,
    'decoder': {
      'type': 'ByteLevel',
      'add_prefix_space': true,
      'trim_offsets': true,
    },
    'model': {
      'type': 'BPE',
      'dropout': null,
      'unk_token': null,
      'continuing_subword_prefix': null,
      'end_of_word_suffix': null,
      'fuse_unk': false,
      'byte_fallback': false,
      'vocab': vocab,
      'merges': ['h e', 'he l', 'l o', 'hel lo', 'w o', 'wo r', 'r l', 'l d'],
    },
  });
}

void main() {
  late Directory tempDir;
  late String tokenizerPath;

  setUpAll(() {
    tempDir = Directory.systemTemp.createTempSync('executorch_tokenizer');
    tokenizerPath = '${tempDir.path}/tokenizer.json';
    File(tokenizerPath).writeAsStringSync(_syntheticBpeTokenizer());
  });

  tearDownAll(() => tempDir.deleteSync(recursive: true));

  group('Tokenizer (native)', () {
    test('loads a HuggingFace BPE tokenizer and reports its format', () async {
      final tokenizer = await Tokenizer.load(tokenizerPath);
      addTearDown(tokenizer.dispose);

      expect(tokenizer.format, TokenizerFormat.huggingFace);
      expect(tokenizer.vocabSize, greaterThan(0));
      expect(tokenizer.isDisposed, isFalse);
    });

    test('encodes text to ids and decodes them back', () async {
      final tokenizer = await Tokenizer.load(tokenizerPath);
      addTearDown(tokenizer.dispose);

      final ids = tokenizer.encode('hello');
      expect(ids, isNotEmpty);
      // Every id must be a real vocabulary entry, not garbage read off the
      // end of the native buffer.
      for (final id in ids) {
        expect(id, lessThan(tokenizer.vocabSize));
      }

      expect(tokenizer.decode(ids), contains('hello'));
    });

    test('round-trips multi-byte UTF-8', () async {
      // The C API takes bytes, so a multi-byte string is where a length-vs-
      // codepoint mistake on either side of the boundary would show up.
      final tokenizer = await Tokenizer.load(tokenizerPath);
      addTearDown(tokenizer.dispose);

      for (final text in ['héllo', 'naïve', 'ありがとう']) {
        final ids = tokenizer.encode(text);
        expect(ids, isNotEmpty, reason: 'no ids for "$text"');
        // Byte-level BPE can represent any input, so decoding must return the
        // original text rather than a replacement character.
        expect(tokenizer.decode(ids), text, reason: 'round trip failed: $text');
      }
    });

    test('encoding is stable across calls and instances', () async {
      final first = await Tokenizer.load(tokenizerPath);
      addTearDown(first.dispose);
      final second = await Tokenizer.load(tokenizerPath);
      addTearDown(second.dispose);

      expect(first.encode('hello world'), second.encode('hello world'));
      expect(first.encode('hello'), first.encode('hello'));
    });

    test('empty input yields no ids, and no ids yield empty text', () async {
      final tokenizer = await Tokenizer.load(tokenizerPath);
      addTearDown(tokenizer.dispose);

      expect(tokenizer.encode(''), isEmpty);
      // Empty, not null — the C API allocates a NUL string for this case.
      expect(tokenizer.decode(const []), isEmpty);
    });

    test('exposes bos and eos ids without throwing', () async {
      final tokenizer = await Tokenizer.load(tokenizerPath);
      addTearDown(tokenizer.dispose);

      // This fixture configures neither, so the values are whatever the reader
      // defaults to. The contract under test is that reading them is safe.
      expect(tokenizer.bosId, isA<int>());
      expect(tokenizer.eosId, isA<int>());
    });

    test('a large input does not truncate or overrun', () async {
      final tokenizer = await Tokenizer.load(tokenizerPath);
      addTearDown(tokenizer.dispose);

      // Exercises the heap-allocated id array across the FFI boundary at a
      // size well past any small-buffer optimisation.
      final long = List.filled(2000, 'hello').join(' ');
      final ids = tokenizer.encode(long);
      expect(ids.length, greaterThan(2000));
      expect(tokenizer.decode(ids), contains('hello'));
    });

    test('dispose is idempotent and use-after-dispose throws', () async {
      final tokenizer = await Tokenizer.load(tokenizerPath);

      tokenizer.dispose();
      expect(tokenizer.isDisposed, isTrue);
      // Double dispose must not double-free.
      tokenizer.dispose();

      expect(
        () => tokenizer.encode('hello'),
        throwsA(isA<ExecuTorchException>()),
      );
      expect(() => tokenizer.vocabSize, throwsA(isA<ExecuTorchException>()));
    });

    test('a malformed tokenizer file is rejected, not loaded', () async {
      final bad = '${tempDir.path}/broken.json';
      File(bad).writeAsStringSync('{"model": {"type": "BPE"}');
      addTearDown(() => File(bad).deleteSync());

      await expectLater(
        () => Tokenizer.load(bad),
        throwsA(isA<ExecuTorchException>()),
      );
    });
  });
}
