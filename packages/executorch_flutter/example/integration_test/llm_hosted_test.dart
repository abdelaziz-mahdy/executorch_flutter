// Hosted LLM test: every model in the catalog that runs on this device is
// downloaded through the example's own download code (streamed to disk, hash
// checked), loaded, and asked a question with its catalog chat format.
//
//   flutter test integration_test/llm_hosted_test.dart -d macos
//
// The first run downloads about 0.5 GB per model into the app cache; later
// runs reuse it. Pass --dart-define=LLM_CATALOG_URL=... to test a catalog
// before it is merged into the models repo.
// ignore_for_file: avoid_print

@Timeout(Duration(minutes: 30))
library;

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter_example/llm/chat_format.dart';
import 'package:executorch_flutter_example/llm/llm_catalog.dart';
import 'package:executorch_flutter_example/llm/llm_files_io.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('hosted LLMs download, load and answer', (tester) async {
    final catalog = await fetchLlmCatalog();
    final here = catalog.where((e) => backendRunsHere(e.backend)).toList();
    print('Catalog: ${catalog.length} LLMs, ${here.length} run here');
    expect(here, isNotEmpty, reason: 'no hosted LLM runs on this device');

    for (final entry in here) {
      print('== ${entry.id} (${entry.backendLabel})');
      var lastPct = -10;
      final paths = await downloadLlm(
        entry,
        onProgress: (received, total) {
          final pct = received * 100 ~/ total;
          if (pct >= lastPct + 10) {
            lastPct = pct;
            print('   download $pct%');
          }
        },
      );
      expect(await cachedFiles(entry), isNotNull);

      final llm = await ExecuTorchLLM.load(
        modelPath: paths[entry.model.name]!,
        tokenizerPath: paths[entry.tokenizer.name]!,
        mlxMetallibPath: entry.metallib == null
            ? null
            : paths[entry.metallib!.name],
      );
      try {
        final prompt = entry.format.build(const [
          ChatTurn(fromUser: true, text: 'What is the capital of France?'),
        ]);
        final answer = StringBuffer();
        await for (final piece in llm.generate(
          prompt,
          config: const GenConfig(maxNewTokens: 40, temperature: 0),
        )) {
          final cleaned = entry.format.clean(piece);
          answer.write(cleaned.text);
          if (cleaned.ended) llm.stop();
        }
        print('   answer: ${answer.toString().trim()}');
        expect(answer.toString(), contains('Paris'));
        expect(answer.toString(), isNot(contains('<|im_end|>')));
      } finally {
        await llm.dispose();
      }
    }
  });
}
