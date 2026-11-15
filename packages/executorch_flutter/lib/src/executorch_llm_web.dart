// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// `ExecuTorchLLM` on web: the ExecuTorch LLM runner compiled to WebAssembly,
/// running in a Web Worker (`web/js/executorch_llm_worker.js` +
/// `web/wasm/executorch_llm.{js,wasm}`).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';

import 'package:executorch_dart/executorch_dart_shared.dart'
    show
        ExecuTorchException,
        ExecuTorchInferenceException,
        ExecuTorchMemoryException,
        ExecuTorchModelException,
        ExecuTorchPlatformException,
        ExecuTorchValidationException,
        GenConfig,
        validateGenConfig;

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

/// The page's base URL: `<base href>` when set, as fetch, workers and the
/// vision path resolve against. Dart's `Uri.base` is the current location,
/// which differs on deep links under a path URL strategy.
@JS('document.baseURI')
external String get _documentBase;

Uri _resolve(String url) => Uri.parse(_documentBase).resolve(url);

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
///   `executorch-llm`, file name `Uri.encodeComponent` of the absolute URL,
///   resolved against `document.baseURI` — so an app can download them once
///   and keep them across reloads, and fetched otherwise. A URL that fails
///   (404, or 403 from object stores) may be served split, as
///   `<url>.parts.json` (`{"size": bytes, "parts": [names relative to it]}`),
///   for static hosts that cap file size. A cross-origin URL must send CORS
///   headers. A stored file is used as is: store only complete, verified
///   files, and delete one when the file at its URL changes.
/// - Only XNNPACK (CPU) exports with a KV cache run, on a single thread, so
///   expect a fraction of native speed.
///
/// The app must serve `js/executorch_llm_worker.js` and
/// `wasm/executorch_llm.{js,wasm}` from its `web/` directory (copy them from
/// this package's `web/` folder).
class ExecuTorchLLM {
  ExecuTorchLLM._(this._worker) {
    // The handlers hold this object weakly: a strong reference from the
    // worker would keep it alive forever, and the finalizer could never end
    // a worker whose ExecuTorchLLM was dropped without dispose(). While a
    // reply is awaited, _inFlight holds it instead (see _track).
    final self = WeakReference(this);
    _worker
      ..onmessage = ((_MessageEvent event) => self.target?._onMessage(
        event,
      )).toJS
      ..onerror = ((_ErrorEvent event) => self.target?._onError(event)).toJS;
    _finalizer.attach(this, _worker, detach: this);
  }

  /// Ends the worker (and frees its Wasm heap) of an instance that was
  /// garbage-collected without dispose(), as native frees its runner.
  static final Finalizer<_Worker> _finalizer = Finalizer(
    (worker) => worker.terminate(),
  );

  /// Instances waiting for a reply from their worker. Only the weak handler
  /// leads back to an instance whose caller is awaiting a future, so without
  /// this it could be collected mid-request and the reply lost.
  static final Set<ExecuTorchLLM> _inFlight = {};

  void _track() {
    if (_pending.isNotEmpty || _generation != null) {
      _inFlight.add(this);
    } else {
      _inFlight.remove(this);
    }
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
    // Same checks, and messages, as the native runner's create.
    if (modelPath.isEmpty) {
      throw const ExecuTorchValidationException('model_path is null/empty');
    }
    if (tokenizerPath.isEmpty) {
      throw const ExecuTorchValidationException('tokenizer_path is null/empty');
    }
    if (dataPath != null && dataPath.isNotEmpty) {
      throw const ExecuTorchModelException(
        'Separate .ptd weight files are not supported on web yet',
      );
    }
    // Before the worker exists, so a malformed URL leaves nothing running.
    final String modelUrl;
    final String tokenizerUrl;
    try {
      modelUrl = _resolve(modelPath).toString();
      tokenizerUrl = _resolve(tokenizerPath).toString();
    } on FormatException catch (e) {
      throw ExecuTorchValidationException('Not a valid URL: ${e.message}');
    }
    final ExecuTorchLLM llm;
    try {
      llm = ExecuTorchLLM._(_Worker(workerUrl));
    } catch (e) {
      // The script could not be started at all (blocked by a content security
      // policy, or a page opened from file://).
      throw ExecuTorchModelException('Could not start the LLM worker: $e');
    }
    try {
      await llm._request({
        'type': 'load',
        'modelUrl': modelUrl,
        'tokenizerUrl': tokenizerUrl,
      });
    } on ExecuTorchMemoryException {
      llm._terminate();
      rethrow;
    } on ExecuTorchException catch (e) {
      llm._terminate();
      throw ExecuTorchModelException('Could not load the LLM: ${e.message}');
    }
    llm._loaded = true;
    return llm;
  }

  /// The exception the native runner raises for the same class of failure.
  static ExecuTorchException _exceptionFor(String? kind, String message) =>
      switch (kind) {
        'validation' => ExecuTorchValidationException(message),
        'memory' => ExecuTorchMemoryException(message),
        'state' => ExecuTorchModelException(message),
        'unsupported' => ExecuTorchPlatformException(message),
        _ => ExecuTorchInferenceException(message),
      };

  final _Worker _worker;
  final Map<int, Completer<void>> _pending = {};
  var _nextId = 0;
  var _loaded = false;
  var _disposed = false;
  int? _generationId;
  StreamController<String>? _generation;
  Completer<void>? _generationEnded;
  Future<void>? _disposal;

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
    final invalid = validateGenConfig(config);
    if (invalid != null) {
      return Stream.error(ExecuTorchValidationException(invalid));
    }
    final id = _nextId++;
    // Closed by _endGeneration when the worker reports done or an error. A
    // subscription can be cancelled after its generation ended and another
    // began; leave that one alone.
    // ignore: close_sinks
    final controller = StreamController<String>()
      ..onCancel = () {
        if (_generationId == id) stop();
      };
    _generation = controller;
    _generationId = id;
    _generationEnded = Completer<void>();
    _track();
    _worker.postMessage(
      {
        'id': id,
        'type': 'generate',
        // As native's UTF-8 conversion does: a lone UTF-16 surrogate becomes
        // U+FFFD. Emscripten's conversion would corrupt or cut the prompt.
        'prompt': utf8.decode(utf8.encode(prompt)),
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
  ///
  /// Like native, a running generation is stopped and drained first, so its
  /// stream gets the pieces already produced before it closes.
  Future<void> dispose() => _disposal ??= _dispose();

  Future<void> _dispose() async {
    final ended = _generationEnded;
    if (_generation != null && ended != null) {
      stop();
      await ended.future;
    }
    _disposed = true;
    _terminate();
  }

  Future<void> _request(Map<String, Object?> message) {
    final id = _nextId++;
    final completer = Completer<void>();
    _pending[id] = completer;
    _track();
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
            _exceptionFor(data['kind'] as String?, data['message'] as String),
          );
          _endGeneration();
        case 'done':
          _endGeneration();
      }
      return;
    }

    final completer = id == null ? null : _pending.remove(id);
    if (completer == null) return;
    _track();
    if (type == 'error') {
      completer.completeError(
        _exceptionFor(data['kind'] as String?, data['message'] as String),
      );
    } else {
      completer.complete();
    }
  }

  void _onError(_ErrorEvent event) {
    // The worker script failed (not found, or the Wasm module could not
    // start). Fail everything that is waiting.
    final message = 'LLM worker failed: ${event.message ?? 'unknown error'}';
    for (final completer in _pending.values) {
      completer.completeError(ExecuTorchModelException(message));
    }
    _pending.clear();
    _generation?.addError(ExecuTorchInferenceException(message));
    _endGeneration();
  }

  void _endGeneration() {
    final controller = _generation;
    final ended = _generationEnded;
    _generation = null;
    _generationId = null;
    _generationEnded = null;
    _track();
    controller?.close();
    if (ended != null && !ended.isCompleted) ended.complete();
  }

  void _terminate() {
    _finalizer.detach(this);
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
    // From the moment dispose() starts: listeners still run while it drains
    // the last generation, and must not start another.
    if (_disposed || _disposal != null) {
      throw const ExecuTorchModelException('LLM runner has been disposed');
    }
  }
}
