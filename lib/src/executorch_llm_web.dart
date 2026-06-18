// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Web stub for `ExecuTorchLLM`. The ExecuTorch LLM runner is not available on
/// web (no WASM runner build yet), so every operation throws
/// [UnsupportedError].
library;

import 'dart:async';

import 'llm_types.dart';

export 'llm_types.dart' show GenConfig;

/// Web stub — not supported. See the native [ExecuTorchLLM] for the real API.
class ExecuTorchLLM {
  ExecuTorchLLM._();

  static const String _unsupported =
      'ExecuTorchLLM (on-device LLM) is not supported on web.';

  /// Always throws [UnsupportedError] on web.
  static Future<ExecuTorchLLM> load({
    required String modelPath,
    required String tokenizerPath,
    String? dataPath,
  }) {
    throw UnsupportedError(_unsupported);
  }

  /// Always false on web.
  bool get isLoaded => false;

  /// Always true on web.
  bool get isDisposed => true;

  /// Always throws [UnsupportedError] on web.
  Stream<String> generate(
    String prompt, {
    GenConfig config = const GenConfig(),
  }) {
    throw UnsupportedError(_unsupported);
  }

  /// No-op on web.
  void stop() {}

  /// No-op on web.
  void reset() {}

  /// No-op on web.
  Future<void> dispose() async {}
}
