// Copyright (c) 2024 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

import 'package:hooks/hooks.dart';

/// Keys that configured the native build before it moved to executorch_dart.
const _legacyKeys = <String>[
  'build_mode',
  'backends',
  'llm',
  'debug',
  'local_lib_dir',
  'executorch_source',
  'prebuilt_version',
];

void main(List<String> args) async {
  await build(args, (input, output) async {
    final stale = _legacyKeys
        .where((key) => input.userDefines[key] != null)
        .toList(growable: false);
    if (stale.isEmpty) return;
    throw BuildError(
      message: 'executorch_flutter no longer owns the native build.\n'
          'Rename this key in your pubspec.yaml:\n'
          '  hooks: user_defines: executorch_flutter:  ->  executorch_dart:\n'
          'Found stale keys: ${stale.join(', ')}',
    );
  });
}
