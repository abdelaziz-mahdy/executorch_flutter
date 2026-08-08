import 'dart:io';

import 'package:test/test.dart';

/// Guards the split between `executorch_dart.dart` and
/// `executorch_dart_shared.dart`.
///
/// `package:executorch_flutter` re-exports the shared library on web, because
/// `export ... hide` still compiles the library it hides names from and the
/// full core reaches dart:ffi. An ffi-free export added to
/// `executorch_dart.dart` instead of `executorch_dart_shared.dart` therefore
/// appears on native and silently vanishes on web — a failure no VM test and
/// no `dart analyze` run can see. Reading the file as text catches it here.
void main() {
  test('executorch_dart.dart exports only shared plus ffi lines', () {
    final source = File('lib/executorch_dart.dart').readAsStringSync();
    final exports = source
        .split('\n')
        .where((line) => line.startsWith('export '))
        .map((line) => line.trim())
        .toList();

    expect(
      exports,
      equals(const [
        "export 'executorch_dart_shared.dart';",
        "export 'src/executorch_llm.dart' show ExecuTorchLLM, GenConfig;",
        "export 'src/ffi/backend_query.dart' show BackendQuery;",
        "export 'src/ffi/native_logging.dart' show setNativeDebugLogging;",
        "export 'src/ffi/version.dart' show ExecuTorchVersion;",
        "export 'src/tokenizer.dart' show Tokenizer, TokenizerFormat;",
      ]),
      reason:
          'Add ffi-free exports to lib/executorch_dart_shared.dart, not '
          'lib/executorch_dart.dart — the latter is not reachable on web. If '
          'you added a genuinely ffi-backed export, add it to this list and to '
          "the Flutter wrapper's hide/route block, which must stay in lockstep.",
    );
  });
}
