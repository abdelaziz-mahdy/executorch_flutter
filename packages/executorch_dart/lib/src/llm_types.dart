// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Types for the LLM (text generation) API. Pure Dart — safe to import on any
/// platform, so the native entry here and the web entry in
/// `package:executorch_flutter` both share it.
library;

/// Generation parameters for `ExecuTorchLLM.generate`.
///
/// Fields mirror ExecuTorch's `extension/llm/runner` `GenerationConfig` exactly.
/// Sampling is temperature-only — there is intentionally no top-p / top-k,
/// because the upstream runner does not support them.
class GenConfig {
  /// Creates a generation config. Defaults match the upstream runner defaults.
  const GenConfig({
    this.maxNewTokens = -1,
    this.seqLen = -1,
    this.temperature = 0.8,
    this.echo = false,
    this.ignoreEos = false,
    this.numBos = 0,
    this.numEos = 0,
  });

  /// Maximum number of new tokens to generate.
  ///
  /// `-1` derives the limit from the model's `max_context_len` metadata.
  final int maxNewTokens;

  /// Maximum number of total tokens (prompt + generated).
  ///
  /// `-1` uses the model's `max_context_len` metadata directly.
  final int seqLen;

  /// Sampling temperature in `[0, 1]` (higher = more random). `0` is greedy
  /// (argmax); values outside the range fail the generation.
  final double temperature;

  /// Whether to echo the input prompt back at the start of the output stream.
  final bool echo;

  /// Whether to keep generating past the EOS token up to the token limit.
  final bool ignoreEos;

  /// Number of BOS tokens to prepend to the prompt.
  final int numBos;

  /// Number of EOS tokens to append to the prompt.
  final int numEos;

  /// Returns a copy with the given fields replaced.
  GenConfig copyWith({
    int? maxNewTokens,
    int? seqLen,
    double? temperature,
    bool? echo,
    bool? ignoreEos,
    int? numBos,
    int? numEos,
  }) => GenConfig(
    maxNewTokens: maxNewTokens ?? this.maxNewTokens,
    seqLen: seqLen ?? this.seqLen,
    temperature: temperature ?? this.temperature,
    echo: echo ?? this.echo,
    ignoreEos: ignoreEos ?? this.ignoreEos,
    numBos: numBos ?? this.numBos,
    numEos: numEos ?? this.numEos,
  );
}

/// Why [config] can't be used, or null if it can. The runner rejects these
/// too, but only after prefill; checking first fails fast, before any output,
/// the same way on every platform.
String? validateGenConfig(GenConfig config) {
  final t = config.temperature;
  if (!(t >= 0 && t <= 1)) {
    return 'Temperature must be in [0, 1], got $t';
  }
  const int32Max = 0x7FFFFFFF;
  if (config.maxNewTokens == 0 ||
      config.maxNewTokens < -1 ||
      config.maxNewTokens > int32Max) {
    return 'maxNewTokens must be -1 (fill the context) or between 1 and '
        '$int32Max, got ${config.maxNewTokens}';
  }
  if (config.seqLen == 0 || config.seqLen < -1 || config.seqLen > int32Max) {
    return "seqLen must be -1 (the model's context) or between 1 and "
        '$int32Max, got ${config.seqLen}';
  }
  // The tokenizer takes these counts as 8-bit integers.
  for (final (name, count) in [
    ('numBos', config.numBos),
    ('numEos', config.numEos),
  ]) {
    if (count < 0 || count > 127) {
      return '$name must be between 0 and 127, got $count';
    }
  }
  return null;
}
