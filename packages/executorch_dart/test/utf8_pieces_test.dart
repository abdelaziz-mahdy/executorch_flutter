import 'dart:convert';

import 'package:executorch_dart/src/utf8_pieces.dart';
import 'package:test/test.dart';

List<String> decode(List<List<int>> tokens) {
  final out = <String>[];
  final pieces = Utf8Pieces(out.add);
  tokens.forEach(pieces.add);
  pieces.close();
  return out;
}

void main() {
  test('a character split across tokens arrives whole', () {
    final pizza = utf8.encode('🍕'); // F0 9F 8D 95
    expect(
      decode([
        utf8.encode('a'),
        pizza.sublist(0, 2),
        pizza.sublist(2),
        utf8.encode('b'),
      ]),
      ['a', '🍕', 'b'],
    );
  });

  test('an incomplete character before ASCII becomes U+FFFD, ASCII kept', () {
    expect(
      decode([
        [0xF0, 0x9F],
        utf8.encode(' the'),
      ]).join(),
      '� the',
    );
  });

  test('bytes still held back at the end become U+FFFD', () {
    expect(
      decode([
        utf8.encode('x'),
        [0xF0, 0x9F, 0x98],
      ]).join(),
      'x�',
    );
  });

  test('a byte order mark the model generates is kept', () {
    expect(
      decode([
        [0xEF, 0xBB],
        [0xBF, 0x41],
      ]).join(),
      '﻿A',
    );
    expect(
      decode([
        utf8.encode('A'),
        [0xEF, 0xBB, 0xBF],
      ]).join(),
      'A﻿',
    );
  });

  test('a lone BOM prefix at the end is not lost', () {
    expect(
      decode([
        [0xEF, 0xBB],
      ]).join(),
      '�',
    );
  });

  test('plain text passes through piece by piece', () {
    expect(decode([utf8.encode('Hel'), utf8.encode('lo')]), ['Hel', 'lo']);
  });
}
