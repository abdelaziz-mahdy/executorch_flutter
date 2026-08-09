// ignore_for_file: avoid_print

// Tokenize text with executorch_dart, without loading a model.
//
// Encoder models — embeddings, classification, retrieval — take token ids as
// an input tensor and produce a vector in a single forward() call. They have
// no generation loop to borrow a tokenizer from, so the tokenizer is exposed
// on its own.
//
//   dart run bin/tokenize.dart /path/to/tokenizer.json "some text"

import 'dart:io';

import 'package:executorch_dart/executorch_dart.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln('usage: tokenize <tokenizer.json> [text]');
    exitCode = 2;
    return;
  }

  final text = args.length > 1 ? args[1] : 'hello world';

  Tokenizer? tokenizer;
  try {
    tokenizer = await Tokenizer.load(args[0]);

    print('format     : ${tokenizer.format.nativeName}');
    print('vocab size : ${tokenizer.vocabSize}');
    print('bos / eos  : ${tokenizer.bosId} / ${tokenizer.eosId}');

    final ids = tokenizer.encode(text);
    print('');
    print('text   : $text');
    print('ids    : $ids');
    print('decoded: ${tokenizer.decode(ids)}');

    // Feed the ids straight into a model with ExecuTorchModel.forward:
    //
    //   final input = TensorData(
    //     shape: [1, ids.length],
    //     dataType: TensorType.int64,
    //     data: Int64List.fromList(ids).buffer.asUint8List(),
    //   );
  } on ExecuTorchException catch (e) {
    // An unsupported tokenizer reports why — a WordPiece model or a
    // BertNormalizer, for instance — rather than a generic parse failure.
    stderr.writeln('Error: ${e.message}');
    exitCode = 1;
  } finally {
    // Nothing is released automatically; the lifetime is yours.
    tokenizer?.dispose();
  }
}
