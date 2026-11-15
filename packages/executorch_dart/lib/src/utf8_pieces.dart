// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Turns an LLM's token bytes into text pieces.
library;

import 'dart:convert';

/// Decodes a generation's token bytes into text pieces.
///
/// A token can end partway through a multi-byte UTF-8 character (byte-level
/// BPE tokenizers split emoji and many non-Latin characters), so bytes are
/// held back until the character is complete. Invalid bytes become U+FFFD,
/// and a byte order mark the model generates is kept (Dart's decoder would
/// drop one at the start).
class Utf8Pieces {
  /// Sends every decoded, non-empty piece to [onPiece].
  Utf8Pieces(void Function(String piece) onPiece) : _onPiece = onPiece {
    _sink = const Utf8Decoder(
      allowMalformed: true,
    ).startChunkedConversion(_PieceSink(onPiece));
  }

  static const _bom = [0xEF, 0xBB, 0xBF];

  final void Function(String piece) _onPiece;
  late final ByteConversionSink _sink;

  /// Bytes seen before the decoder starts, while they could still be a BOM.
  List<int>? _head = [];

  /// Adds the bytes of one token.
  void add(List<int> bytes) {
    final head = _head;
    if (head == null) {
      _sink.add(bytes);
      return;
    }
    head.addAll(bytes);
    var start = 0;
    while (head.length - start >= 3 &&
        head[start] == _bom[0] &&
        head[start + 1] == _bom[1] &&
        head[start + 2] == _bom[2]) {
      _onPiece('﻿');
      start += 3;
    }
    final rest = head.sublist(start);
    if (_isBomPrefix(rest)) {
      _head = rest; // wait for more bytes
      return;
    }
    _head = null;
    if (rest.isNotEmpty) _sink.add(rest);
  }

  /// Emits whatever is still held back; incomplete bytes become U+FFFD.
  void close() {
    final head = _head;
    _head = null;
    if (head != null && head.isNotEmpty) _sink.add(head);
    _sink.close();
  }

  static bool _isBomPrefix(List<int> bytes) {
    if (bytes.length >= 3) return false;
    for (var i = 0; i < bytes.length; i++) {
      if (bytes[i] != _bom[i]) return false;
    }
    return true;
  }
}

class _PieceSink implements Sink<String> {
  _PieceSink(this._onPiece);

  final void Function(String piece) _onPiece;

  @override
  void add(String data) {
    if (data.isNotEmpty) _onPiece(data);
  }

  @override
  void close() {}
}
