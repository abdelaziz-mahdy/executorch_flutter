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

  /// Sampling temperature (higher = more random). `<= 0` is greedy (argmax).
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
