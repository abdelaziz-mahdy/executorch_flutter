// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native LLM runner wrapper for the FFI layer.
library;

import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
import '../generated/executorch_ffi.g.dart' show ETStatus;
import '../generated/executorch_llm_ffi.g.dart';
import '../llm_types.dart';
import '../utf8_pieces.dart';
import 'native_status.dart';

/// Thin wrapper around the native `ETLLMRunner` with streaming generation.
///
/// Generation runs on a native worker thread; tokens are marshalled to the
/// Dart isolate via `NativeCallable.listener` and surfaced as a
/// `Stream<String>`.
class NativeLLMRunner implements ffi.Finalizable {
  NativeLLMRunner._(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Create a runner from a model file and tokenizer file.
  ///
  /// [dataPath] is an optional `.ptd` weight blob (pass null when weights are
  /// embedded in the `.pte`).
  factory NativeLLMRunner.create({
    required String modelPath,
    required String tokenizerPath,
    String? dataPath,
  }) {
    final modelPtr = modelPath.toNativeUtf8().cast<ffi.Char>();
    final tokenizerPtr = tokenizerPath.toNativeUtf8().cast<ffi.Char>();
    final dataPtr = dataPath != null
        ? dataPath.toNativeUtf8().cast<ffi.Char>()
        : ffi.nullptr;
    final outPtr = calloc<ffi.Pointer<ETLLMRunner>>();
    try {
      final status = et_llm_runner_create(
        modelPtr,
        tokenizerPtr,
        dataPtr,
        outPtr,
      );
      checkStatus(status); // throws on error, frees status
      return NativeLLMRunner._(outPtr.value);
    } finally {
      calloc.free(outPtr);
      malloc
        ..free(modelPtr)
        ..free(tokenizerPtr);
      if (dataPtr != ffi.nullptr) malloc.free(dataPtr);
    }
  }

  /// Loads the model's methods (for XNNPACK, compiling the delegate graphs),
  /// which the runner otherwise does on the first generate. Runs on a helper
  /// isolate: it takes seconds for a large model and would block the caller.
  Future<void> loadAsync() async {
    _checkDisposed();
    final address = _ptr.address;
    final statusAddress = await Isolate.run(
      () => et_llm_runner_load(
        ffi.Pointer<ETLLMRunner>.fromAddress(address),
      ).address,
    );
    checkStatus(ffi.Pointer<ETStatus>.fromAddress(statusAddress));
  }

  /// Point the MLX backend at its Metal kernel library (`mlx.metallib`).
  ///
  /// Must be called BEFORE [NativeLLMRunner.create] for MLX models. No-op for
  /// other backends.
  /// See `et_llm_set_metallib_path` in the native header for why this is needed
  /// (sandboxed Flutter apps bundle the metallib as a data asset, not next to
  /// the dylib where MLX would otherwise look).
  static void setMetallibPath(String path) {
    final p = path.toNativeUtf8().cast<ffi.Char>();
    try {
      et_llm_set_metallib_path(p);
    } finally {
      malloc.free(p);
    }
  }

  final ffi.Pointer<ETLLMRunner> _ptr;
  bool _disposed = false;

  /// Tracks an in-flight generation so [disposeAsync] never frees the runner
  /// mid-generation (which would crash deep in the kernels, use-after-free).
  Completer<void>? _generation;

  /// Whether stop() was asked for during the current generation.
  bool _stopRequested = false;

  /// Whether reset() was asked for during the current generation; it applies
  /// when the generation ends.
  bool _resetPending = false;

  static final _finalizer = ffi.NativeFinalizer(
    addresses.et_llm_runner_free.cast(),
  );

  /// Whether the runner has been disposed.
  bool get isDisposed => _disposed;

  /// Whether the runner has loaded its model.
  bool get isLoaded => !_disposed && et_llm_is_loaded(_ptr) == 1;

  /// Stream generated text token-by-token for [prompt].
  ///
  /// Only one generation may run at a time per runner. Cancelling the returned
  /// stream's subscription cooperatively stops generation.
  Stream<String> generate(
    String prompt, {
    GenConfig config = const GenConfig(),
  }) {
    _checkDisposed();
    if (_generation != null) {
      throw const ExecuTorchInferenceException(
        'A generation is already in progress on this runner',
      );
    }

    final invalid = validateGenConfig(config);
    if (invalid != null) {
      return Stream.error(ExecuTorchValidationException(invalid));
    }

    final controller = StreamController<String>();
    final generation = Completer<void>();
    _generation = generation;
    _stopRequested = false;

    // A token's bytes can end partway through a multi-byte character (byte-
    // level BPE splits emoji and many non-Latin characters), so pieces are
    // decoded with the partial bytes held back until the next token.
    final text = Utf8Pieces((piece) {
      if (!controller.isClosed) controller.add(piece);
    });

    final promptPtr = prompt.toNativeUtf8().cast<ffi.Char>();
    final cfgPtr = calloc<ETGenConfig>();
    cfgPtr.ref
      ..max_new_tokens = config.maxNewTokens
      ..seq_len = config.seqLen
      ..temperature = config.temperature
      ..echo = config.echo ? 1 : 0
      ..ignore_eos = config.ignoreEos ? 1 : 0
      ..num_bos = config.numBos
      ..num_eos = config.numEos;

    late final ffi.NativeCallable<
      ffi.Void Function(ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Void>)
    >
    tokenCallable;
    late final ffi.NativeCallable<ffi.Void Function(ffi.Pointer<ffi.Void>)>
    doneCallable;

    var cleaned = false;
    void cleanup() {
      if (cleaned) return;
      cleaned = true;
      tokenCallable.close();
      doneCallable.close();
      calloc.free(cfgPtr);
      malloc.free(promptPtr);
      _generation = null;
      // The generation has released the runner, so this doesn't block.
      if (_resetPending && !_disposed) {
        _resetPending = false;
        et_llm_reset(_ptr);
      }
      if (!generation.isCompleted) generation.complete();
    }

    void onToken(ffi.Pointer<ffi.Char> token, ffi.Pointer<ffi.Void> _) {
      if (token == ffi.nullptr) return;
      // The runner's token loop clears its stop flag when it starts, after
      // prefill, so a stop asked for during prefill is lost. Ask again: the
      // loop is running now.
      if (_stopRequested && !_disposed) et_llm_stop(_ptr);
      try {
        final length = token.cast<Utf8>().length;
        // Copied before the native string is freed below.
        text.add(
          Uint8List.fromList(token.cast<ffi.Uint8>().asTypedList(length)),
        );
      } finally {
        // We own the heap string handed to us by the native side.
        et_llm_string_free(token);
      }
    }

    void onDone(ffi.Pointer<ffi.Void> statusVoid) {
      text.close(); // emits any bytes still held back, before the outcome
      try {
        checkStatus(statusVoid.cast<ETStatus>());
      } catch (e, st) {
        if (!controller.isClosed) controller.addError(e, st);
      } finally {
        cleanup();
        if (!controller.isClosed) controller.close();
      }
    }

    tokenCallable = ffi.NativeCallable.listener(onToken);
    doneCallable = ffi.NativeCallable.listener(onDone);

    // Cooperatively stop generation if the consumer cancels. The native done
    // callback still fires afterwards and runs cleanup(). A subscription can
    // be cancelled after its generation ended and another began; leave that
    // one alone.
    controller.onCancel = () {
      if (identical(_generation, generation)) stop();
    };

    et_llm_generate_async(
      _ptr,
      promptPtr,
      cfgPtr,
      tokenCallable.nativeFunction,
      ffi.nullptr,
      doneCallable.nativeFunction,
    );

    return controller.stream;
  }

  /// Cooperatively stop an in-flight generation.
  void stop() {
    if (_disposed || _generation == null) return;
    _stopRequested = true;
    et_llm_stop(_ptr);
  }

  /// Clear the KV cache and start a fresh conversation.
  ///
  /// During a generation the reset waits for it to end: the runner is busy,
  /// and waiting here would block the calling isolate until then.
  void reset() {
    _checkDisposed();
    if (_generation != null) {
      _resetPending = true;
      return;
    }
    et_llm_reset(_ptr);
  }

  /// Stop any in-flight generation, wait for it to finish, then free the
  /// runner.
  Future<void> disposeAsync() async {
    if (_disposed) return;
    final generation = _generation;
    if (generation != null) {
      stop(); // also re-sent once tokens flow, if this lands in prefill
      await generation.future.catchError((_) {});
    }
    dispose();
  }

  /// Free the runner immediately. Unsafe to call during an in-flight
  /// generation — prefer [disposeAsync]. Safe to call multiple times.
  void dispose() {
    if (_disposed) return;
    _finalizer.detach(this);
    et_llm_runner_free(_ptr);
    _disposed = true;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw const ExecuTorchModelException('LLM runner has been disposed');
    }
  }
}
