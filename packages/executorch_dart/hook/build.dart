// Copyright (c) 2024 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

import 'package:executorch_dart/src/build/run_build.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    await runBuild(input, output);
  });
}
