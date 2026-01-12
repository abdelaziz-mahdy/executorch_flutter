// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Native status handling and error mapping for FFI layer.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
import '../generated/executorch_ffi.g.dart';

/// Check status pointer and throw appropriate exception if error.
///
/// This function takes ownership of the status pointer and frees it
/// after checking, regardless of success or failure.
///
/// Returns void on success, throws [ExecuTorchException] subclass on error.
void checkStatus(ffi.Pointer<ETStatus> status) {
  if (status == ffi.nullptr) {
    throw const ExecuTorchPlatformException(
      'Null status returned from native code',
    );
  }

  try {
    final code = status.ref.code;
    if (code == ETErrorCode.ET_OK.value) {
      return; // Success
    }

    // Extract error message and location
    final message = status.ref.message != ffi.nullptr
        ? status.ref.message.cast<Utf8>().toDartString()
        : 'Unknown error';
    final location = status.ref.location != ffi.nullptr
        ? status.ref.location.cast<Utf8>().toDartString()
        : null;

    // Map error code to exception type and throw
    throw _mapErrorCode(code, message, location);
  } finally {
    // Always free the status
    et_status_free(status);
  }
}

/// Check status and return a value on success.
///
/// Useful for functions that return a status and an output value.
T checkStatusWithResult<T>(
  ffi.Pointer<ETStatus> status,
  T Function() getValue,
) {
  checkStatus(status);
  return getValue();
}

/// Map native error code to Dart exception type.
ExecuTorchException _mapErrorCode(int code, String message, String? location) {
  final details = location != null ? 'Location: $location' : null;

  final errorCode = ETErrorCode.fromValue(code);
  return switch (errorCode) {
    ETErrorCode.ET_OK => throw StateError('Should not map ET_OK to exception'),
    ETErrorCode.ET_INVALID_ARGUMENT =>
      ExecuTorchValidationException(message, details),
    ETErrorCode.ET_OUT_OF_MEMORY => ExecuTorchMemoryException(message, details),
    ETErrorCode.ET_MODEL_LOAD_FAILED =>
      ExecuTorchModelException(message, details),
    ETErrorCode.ET_INFERENCE_FAILED =>
      ExecuTorchInferenceException(message, details),
    ETErrorCode.ET_INVALID_STATE => ExecuTorchModelException(message, details),
    ETErrorCode.ET_UNSUPPORTED =>
      ExecuTorchPlatformException(message, details),
    ETErrorCode.ET_IO_ERROR => ExecuTorchIOException(message, details),
    ETErrorCode.ET_INTERNAL => ExecuTorchPlatformException(message, details),
  };
}
