// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// `ExecuTorchLLM` on web: the ExecuTorch LLM runner compiled to WebAssembly,
/// running in a Web Worker (`web/js/executorch_llm_worker.js` +
/// `web/wasm/executorch_llm.{js,wasm}`).
library;

import 'dart:async';
import 'dart:js_interop';

import 'package:executorch_dart/executorch_dart_shared.dart'
    show ExecuTorchInferenceException, ExecuTorchModelException, GenConfig;

export 'package:executorch_dart/executorch_dart_shared.dart' show GenConfig;

@JS('Worker')
extension type _Worker._(JSObject _) implements JSObject {
  external factory _Worker(String url);
  external void postMessage(JSAny? message);
  external void terminate();
  external JSFunction? get onmessage;
  external set onmessage(JSFunction? handler);
  external JSFunction? get onerror;
  external set onerror(JSFunction? handler);
}

extension type _MessageEvent._(JSObject _) implements JSObject {
  external JSAny? get data;
}

extension type _ErrorEvent._(JSObject _) implements JSObject {
  external String? get message;
}

/// On-device LLM text generation in the browser.
///
/// Same API as on the other platforms, with two differences:
///
/// - [load]'s `modelPath` and `tokenizerPath` are **URLs**. They are read from
///   the origin private file system (OPFS) when stored there — directory
///   `executorch-llm`, file name `Uri.encodeComponent(url)` — so an app can
///   download them once and keep them across reloads, and fetched otherwise.
///   A cross-origin URL must send CORS headers.
/// - Only XNNPACK (CPU) exports run, on a single thread, so expect a fraction
///   of native speed.
///
/// The app must serve `js/executorch_llm_worker.js` and
/// `wasm/executorch_llm.{js,wasm}` from its `web/` directory (copy them from
/// this package's `web/` folder).
class ExecuTorchLLM {
  ExecuTorchLLM._(this._worker) {
    _worker.onmessage = _onMessage.toJS;
    _worker.onerror = _onError.toJS;
  }

  /// Where the worker script is served from, relative to the page.
  static const String workerUrl = 'js/executorch_llm_worker.js';

  /// Load an LLM from a model URL and a tokenizer URL.
  ///
  /// [dataPath] (a separate `.ptd` weight file) is not supported on web yet,
  /// and [mlxMetallibPath] is ignored: there is no MLX on web.
  static Future<ExecuTorchLLM> load({
    required String modelPath,
    required String tokenizerPath,
    String? dataPath,
    String? mlxMetallibPath,
  }) async {
    if (dataPath != null && dataPath.isNotEmpty) {
      throw const ExecuTorchModelException(
        'Separate .ptd weight files are not supported on web yet',
      );
    }
    final llm = ExecuTorchLLM._(_Worker(workerUrl));
    try {
      await llm._request({
        'type': 'load',
        'modelUrl': Uri.base.resolve(modelPath).toString(),
        'tokenizerUrl': Uri.base.resolve(tokenizerPath).toString(),
      });
    } catch (e) {
      llm._terminate();
      throw ExecuTorchModelException('Could not load the LLM: $e');
    }
    llm._loaded = true;
    return llm;
  }

  final _Worker _worker;
  final Map<int, Completer<void>> _pending = {};
  var _nextId = 0;
  var _loaded = false;
  var _disposed = false;
  int? _generationId;
  StreamController<String>? _generation;

  /// Whether the model has finished loading.
  bool get isLoaded => _loaded && !_disposed;

  /// Whether this instance has been disposed.
  bool get isDisposed => _disposed;

  /// Generate text from [prompt], streaming each decoded piece as produced.
  ///
  /// Only one generation runs at a time per instance. Cancelling the returned
  /// stream's subscription stops generation.
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
    final id = _nextId++;
    // Closed by _endGeneration when the worker reports done or an error.
    // ignore: close_sinks
    final controller = StreamController<String>()..onCancel = stop;
    _generation = controller;
    _generationId = id;
    _worker.postMessage(
      {
        'id': id,
        'type': 'generate',
        'prompt': prompt,
        'config': {
          'maxNewTokens': config.maxNewTokens,
          'seqLen': config.seqLen,
          'temperature': config.temperature,
          'echo': config.echo,
          'ignoreEos': config.ignoreEos,
          'numBos': config.numBos,
          'numEos': config.numEos,
        },
      }.jsify(),
    );
    return controller.stream;
  }

  /// Stop an in-flight generation after the token being computed.
  void stop() {
    if (_disposed || _generation == null) return;
    _worker.postMessage({'type': 'stop'}.jsify());
  }

  /// Clear the conversation (KV cache position) and start fresh.
  void reset() {
    _checkDisposed();
    unawaited(_request({'type': 'reset'}).catchError((Object _) {}));
  }

  /// Stop any in-flight generation, then release the worker and its memory.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _terminate();
  }

  Future<void> _request(Map<String, Object?> message) {
    final id = _nextId++;
    final completer = Completer<void>();
    _pending[id] = completer;
    _worker.postMessage({...message, 'id': id}.jsify());
    return completer.future;
  }

  void _onMessage(_MessageEvent event) {
    final data = event.data.dartify();
    if (data is! Map) return;
    final id = (data['id'] as num?)?.toInt();
    final type = data['type'] as String?;

    if (id != null && id == _generationId) {
      // ignore: close_sinks, closed by _endGeneration
      final controller = _generation!;
      switch (type) {
        case 'piece':
          controller.add(data['piece'] as String);
        case 'error':
          controller.addError(
            ExecuTorchInferenceException(data['message'] as String),
          );
          _endGeneration();
        case 'done':
          _endGeneration();
      }
      return;
    }

    final completer = id == null ? null : _pending.remove(id);
    if (completer == null) return;
    if (type == 'error') {
      completer.completeError(Exception(data['message']));
    } else {
      completer.complete();
    }
  }

  void _onError(_ErrorEvent event) {
    // The worker script failed (not found, or the Wasm module could not
    // start). Fail everything that is waiting.
    final message = 'LLM worker failed: ${event.message ?? 'unknown error'}';
    for (final completer in _pending.values) {
      completer.completeError(Exception(message));
    }
    _pending.clear();
    _generation?.addError(ExecuTorchInferenceException(message));
    _endGeneration();
  }

  void _endGeneration() {
    final controller = _generation;
    _generation = null;
    _generationId = null;
    controller?.close();
  }

  void _terminate() {
    _worker.terminate();
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(
          const ExecuTorchModelException('LLM runner has been disposed'),
        );
      }
    }
    _pending.clear();
    _endGeneration();
  }

  void _checkDisposed() {
    if (_disposed) {
      throw const ExecuTorchModelException('LLM runner has been disposed');
    }
  }
}
