import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Guards the `hide` list and the routed `show` names in
/// `lib/executorch_flutter.dart` staying in lockstep.
///
/// The blanket export re-exports the core, `hide`-ing every name that the
/// platform-routing block below it re-exports to a native or web
/// implementation. `flutter analyze` only ever resolves the native branch,
/// so a name added to one list and not the other compiles fine there and
/// passes every VM test — it only breaks the web build, where the hidden
/// name has nothing routing it back in and silently vanishes from the
/// public API. Reading the file as text catches it here, on every platform.
void main() {
  test('hide list and routed show names are the same set', () {
    final code = File('lib/executorch_flutter.dart')
        .readAsLinesSync()
        .where((line) => !line.trim().startsWith('//'))
        .join('\n');

    final hideClause = RegExp(r'hide\s+([^;]+);').firstMatch(code)?.group(1);
    expect(
      hideClause,
      isNotNull,
      reason:
          'no hide clause found in '
          'lib/executorch_flutter.dart — did the export shape change?',
    );
    final hidden = _names(hideClause!);

    final routed = RegExp(
      r'show\s+([^;]+);',
    ).allMatches(code).expand((match) => _names(match.group(1)!)).toSet();

    expect(
      routed,
      equals(hidden),
      reason:
          'A name hidden from the blanket export must be routed back by '
          'exactly one show clause below it, and vice versa — otherwise it '
          'compiles fine natively and silently vanishes on web. Keep the '
          'hide list and the routed show names in lockstep.',
    );
  });
}

/// Splits a comma-separated combinator clause into trimmed identifiers.
Set<String> _names(String clause) => clause
    .split(',')
    .map((name) => name.trim())
    .where((name) => name.isNotEmpty)
    .toSet();
