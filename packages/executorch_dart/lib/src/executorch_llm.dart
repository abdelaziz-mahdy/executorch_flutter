// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Public API for on-device LLM (text generation) inference.
library;

import 'dart:async';

import 'ffi/native_llm.dart';
import 'llm_types.dart';

export 'llm_types.dart' show GenConfig;

/// On-device large language model for streaming text generation.
///
/// Unlike `ExecuTorchModel` (single-shot tensor inference), an LLM is driven by
/// a stateful, autoregressive decode loop with a KV cache and a real tokenizer.
///
/// Weights are large (often 1+ GB), so the model is loaded from a **file path**
/// and memory-mapped — not copied into a Dart byte buffer.
///
/// ```dart
/// final llm = await ExecuTorchLLM.load(
///   modelPath: '/path/gemma4_e2b_xnnpack.pte',
///   tokenizerPath: '/path/tokenizer.json',
/// );
/// await for (final piece in llm.generate('Explain Flutter in one line.')) {
///   stdout.write(piece);
/// }
/// await llm.dispose();
/// ```
class ExecuTorchLLM {
  ExecuTorchLLM._(this._runner);

  /// Load an LLM from a model file and tokenizer file.
  ///
  /// [modelPath] is the `.pte` model. [tokenizerPath] is the tokenizer file
  /// (HF `tokenizer.json`, SentencePiece, or TikToken — auto-detected).
  /// [dataPath] is an optional `.ptd` weight blob produced by some exports.
  ///
  /// [mlxMetallibPath] is REQUIRED for **MLX** (Apple-GPU) models and ignored
  /// by every other backend. The MLX backend loads its Metal kernels from a
  /// `mlx.metallib` file at runtime, but a sandboxed app can't reach the copy
  /// next to the native library, so the app must point at a readable copy here
  /// (a file the user picked, or one bundled as a Flutter asset and copied to a
  /// writable directory). Without it, MLX model load fails with `Error 0x23`
  /// (`MLXBackend` could not load its metallib). See the README / example for
  /// how to obtain and ship the metallib.
  static Future<ExecuTorchLLM> load({
    required String modelPath,
    required String tokenizerPath,
    String? dataPath,
    String? mlxMetallibPath,
  }) async {
    // Point the MLX Metal-kernel loader at the metallib BEFORE creating the
    // runner (the device initializes on first GPU op). No-op for non-MLX.
    if (mlxMetallibPath != null && mlxMetallibPath.isNotEmpty) {
      NativeLLMRunner.setMetallibPath(mlxMetallibPath);
    }

    final runner = NativeLLMRunner.create(
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
      dataPath: dataPath,
    );
    return ExecuTorchLLM._(runner);
  }

  final NativeLLMRunner _runner;

  /// Whether the underlying model has finished loading.
  bool get isLoaded => _runner.isLoaded;

  /// Whether this instance has been disposed.
  bool get isDisposed => _runner.isDisposed;

  /// Generate text from [prompt], streaming each decoded piece as produced.
  ///
  /// Only one generation runs at a time per instance. Cancelling the returned
  /// stream's subscription cooperatively stops generation.
  Stream<String> generate(
    String prompt, {
    GenConfig config = const GenConfig(),
  }) => _runner.generate(prompt, config: config);

  /// Cooperatively stop an in-flight generation.
  void stop() => _runner.stop();

  /// Clear the conversation/KV cache and start fresh.
  void reset() => _runner.reset();

  /// Stop any in-flight generation, then release native resources.
  Future<void> dispose() => _runner.disposeAsync();
}
