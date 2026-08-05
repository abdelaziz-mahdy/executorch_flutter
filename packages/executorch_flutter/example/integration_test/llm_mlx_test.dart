// MLX Gemma 4 LLM integration test.
//
// Unlike the C++ harness (which hand-places mlx.metallib next to the dylib and
// thus bypasses real app bundling), this test runs inside the Flutter app:
// the dylib loads from its generated .framework, the sandbox is active, and the
// metallib must be delivered the way a shipped app delivers it (data asset ->
// ExecuTorchLLM materializes it -> ET_MLX_METALLIB_PATH). So it actually
// exercises the MLX backend init that fails with 0x23 when bundling is wrong.
//
// The model is large (~6.4 GB) and loaded by path, so it is NOT bundled. Provide
// the paths via --dart-define; the test is skipped if they are absent/unreadable
// so CI without the model still passes:
//
//   flutter test integration_test/llm_mlx_test.dart -d macos \
//     --dart-define=MLX_MODEL=/abs/path/gemma-4-E2B-it_mlx.pte \
//     --dart-define=MLX_TOKENIZER=/abs/path/gemma-4-E2B-it_tokenizer.json
//
// Defaults point at the repo's models/gemma4 copies.
// ignore_for_file: avoid_print

@Timeout(Duration(minutes: 10))
library;

import 'dart:io';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

// Paths are resolved at runtime so the test is portable (no machine-specific
// absolute paths). Override any of them with --dart-define if your files live
// elsewhere. The model + metallib are large, locally-exported artifacts and are
// NOT committed, so the test skips when they're absent.
const _modelOverride = String.fromEnvironment('MLX_MODEL');
const _tokenizerOverride = String.fromEnvironment('MLX_TOKENIZER');
const _metallibOverride = String.fromEnvironment('MLX_METALLIB');

/// Find `<repo>/models/gemma4/<name>` relative to the current working directory,
/// which may be the example dir, the package root, or the repo root depending on
/// how `flutter test` was invoked. Returns the first candidate that exists, or
/// the example-relative path as a non-existent fallback (so the test reports it).
String _resolveModelFile(String name, String override) {
  if (override.isNotEmpty) return override;
  final cwd = Directory.current.path;
  final candidates = <String>[
    '$cwd/models/gemma4/$name', // cwd = repo root
    '$cwd/../models/gemma4/$name', // cwd = example/
    '$cwd/../../models/gemma4/$name', // cwd = example/<sub>
  ];
  return candidates.firstWhere(
    (p) => File(p).existsSync(),
    orElse: () => candidates[1],
  );
}

final _modelPath = _resolveModelFile('gemma-4-E2B-it_mlx.pte', _modelOverride);
final _tokenizerPath =
    _resolveModelFile('gemma-4-E2B-it_tokenizer.json', _tokenizerOverride);
final _metallibPath = _resolveModelFile('mlx.metallib', _metallibOverride);

/// Gemma 4 chat template. Turn markers are the literal special tokens `<|turn>`
/// (start) and `<turn|>` (end) — matching the example app's _buildGemmaPrompt.
String _gemmaPrompt(String message) =>
    '<bos><|turn>user\n$message<turn|>\n<|turn>model\n';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('MLX Gemma 4 LLM', () {
    final hasModel =
        File(_modelPath).existsSync() && File(_tokenizerPath).existsSync();

    testWidgets(
      'loads MLX model and generates a coherent answer (no 0x23)',
      (tester) async {
        print('Model: $_modelPath');
        print('Tokenizer: $_tokenizerPath');

        final metallib = File(_metallibPath).existsSync() ? _metallibPath : null;
        print('Metallib: ${metallib ?? "(none)"}');
        final llm = await ExecuTorchLLM.load(
          modelPath: _modelPath,
          tokenizerPath: _tokenizerPath,
          mlxMetallibPath: metallib,
        );
        addTearDown(llm.dispose);

        final buffer = StringBuffer();
        // Greedy (temperature 0) for a deterministic answer.
        await for (final piece in llm.generate(
          _gemmaPrompt('What is the capital of France?'),
          config: const GenConfig(maxNewTokens: 64, temperature: 0),
        )) {
          buffer.write(piece);
        }

        final output = buffer.toString();
        print('Generated: $output');

        // Reaching here without an ExecuTorchException means MLX initialized
        // (metallib found) — i.e. the 0x23 bundling bug is fixed.
        expect(output.trim(), isNotEmpty,
            reason: 'model produced no output');
        expect(output.toLowerCase(), contains('paris'),
            reason: 'expected the capital of France in the answer');
      },
      // Skipped when the (large, unbundled) model isn't present. Pass
      // --dart-define=MLX_MODEL=... and MLX_TOKENIZER=... to run it.
      skip: !hasModel,
    );
  });
}
