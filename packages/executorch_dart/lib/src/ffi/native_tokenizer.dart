// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native tokenizer wrapper for the FFI layer.
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
// ETStatus comes through the tokenizer bindings, which re-export it as a
// typedef — importing it from the core bindings as well would be ambiguous.
import '../generated/executorch_tokenizer_ffi.g.dart';
import 'native_status.dart';

/// Thin wrapper around the native `ETTokenizer`.
///
/// Encoding and decoding are synchronous: tokenization is a short,
/// CPU-bound transform on text the caller already has in memory, so the
/// isolate hop an async API would need costs more than the work itself.
class NativeTokenizer implements ffi.Finalizable {
  NativeTokenizer._(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Load a tokenizer file, auto-detecting the format.
  factory NativeTokenizer.create(String tokenizerPath) {
    final pathPtr = tokenizerPath.toNativeUtf8().cast<ffi.Char>();
    final outPtr = calloc<ffi.Pointer<ETTokenizer>>();
    try {
      final status = et_tokenizer_create(pathPtr, outPtr);
      checkStatus(status); // throws on error, frees status
      return NativeTokenizer._(outPtr.value);
    } finally {
      calloc.free(outPtr);
      malloc.free(pathPtr);
    }
  }

  final ffi.Pointer<ETTokenizer> _ptr;
  bool _disposed = false;

  static final _finalizer = ffi.NativeFinalizer(
    addresses.et_tokenizer_free.cast(),
  );

  /// Whether the tokenizer has been disposed.
  bool get isDisposed => _disposed;

  /// Whether the tokenizer finished loading.
  bool get isLoaded => !_disposed && et_tokenizer_is_loaded(_ptr) == 1;

  /// Size of the vocabulary.
  int get vocabSize {
    _checkDisposed();
    return et_tokenizer_vocab_size(_ptr);
  }

  /// Beginning-of-sequence token id.
  int get bosId {
    _checkDisposed();
    return et_tokenizer_bos_id(_ptr);
  }

  /// End-of-sequence token id.
  int get eosId {
    _checkDisposed();
    return et_tokenizer_eos_id(_ptr);
  }

  /// Which reader claimed the file: `hf_json`, `tiktoken`, `sentencepiece`
  /// or `llama2c`.
  String get format {
    _checkDisposed();
    final ptr = et_tokenizer_format(_ptr);
    if (ptr == ffi.nullptr) return 'unknown';
    // A static literal owned by the native side — do not free it.
    return ptr.cast<Utf8>().toDartString();
  }

  /// Encode [text] into token ids.
  Uint64List encode(String text, {int bosCount = 0, int eosCount = 0}) {
    _checkDisposed();
    final textPtr = text.toNativeUtf8().cast<ffi.Char>();
    final idsPtr = calloc<ffi.Pointer<ffi.Uint64>>();
    final countPtr = calloc<ffi.Size>();
    try {
      final status = et_tokenizer_encode(
        _ptr,
        textPtr,
        bosCount,
        eosCount,
        idsPtr,
        countPtr,
      );
      checkStatus(status);

      final count = countPtr.value;
      if (count == 0 || idsPtr.value == ffi.nullptr) return Uint64List(0);
      // Copy out of native memory before freeing it — asTypedList would alias
      // a buffer we are about to release.
      final result = Uint64List.fromList(idsPtr.value.asTypedList(count));
      et_tokenizer_ids_free(idsPtr.value);
      return result;
    } finally {
      calloc
        ..free(idsPtr)
        ..free(countPtr);
      malloc.free(textPtr);
    }
  }

  /// Decode token ids back into text.
  String decode(List<int> ids, {bool skipSpecialTokens = false}) {
    _checkDisposed();
    final idsPtr = ids.isEmpty ? ffi.nullptr : calloc<ffi.Uint64>(ids.length);
    final outPtr = calloc<ffi.Pointer<ffi.Char>>();
    try {
      for (var i = 0; i < ids.length; i++) {
        idsPtr[i] = ids[i];
      }
      final status = et_tokenizer_decode(
        _ptr,
        idsPtr,
        ids.length,
        skipSpecialTokens ? 1 : 0,
        outPtr,
      );
      checkStatus(status);

      if (outPtr.value == ffi.nullptr) return '';
      final text = outPtr.value.cast<Utf8>().toDartString();
      et_tokenizer_string_free(outPtr.value);
      return text;
    } finally {
      calloc.free(outPtr);
      if (idsPtr != ffi.nullptr) calloc.free(idsPtr);
    }
  }

  /// Free the native tokenizer. Idempotent.
  void dispose() {
    if (_disposed) return;
    _finalizer.detach(this);
    et_tokenizer_free(_ptr);
    _disposed = true;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw const ExecuTorchModelException('Tokenizer has been disposed');
    }
  }
}
