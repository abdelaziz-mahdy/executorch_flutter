/// Error conversion utilities for FFI bridge
library;

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

import '../executorch_errors.dart';
import '../generated/executorch_ffi_bindings.dart';

/// Converts C error to Dart exception
///
/// Throws appropriate exception type based on error code:
/// - ET_FLUTTER_ERROR_MODEL_LOAD → ExecuTorchModelException
/// - ET_FLUTTER_ERROR_INFERENCE → ExecuTorchInferenceException
/// - ET_FLUTTER_ERROR_VALIDATION → ExecuTorchValidationException
/// - ET_FLUTTER_ERROR_MEMORY → ExecuTorchMemoryException
/// - ET_FLUTTER_ERROR_IO → ExecuTorchIOException
/// - ET_FLUTTER_ERROR_PLATFORM → ExecuTorchPlatformException
/// - ET_FLUTTER_ERROR_INVALID_HANDLE → ExecuTorchModelException
/// - ET_FLUTTER_ERROR_INVALID_ARGUMENT → ExecuTorchValidationException
///
/// Does nothing if error code is ET_FLUTTER_SUCCESS
void throwIfError(ETFlutterError error) {
  // Convert int code to ETFlutterErrorCode enum
  final errorCode = ETFlutterErrorCode.fromValue(error.code);

  if (errorCode == ETFlutterErrorCode.ET_FLUTTER_SUCCESS) {
    return; // No error
  }

  // Extract error message from C string
  final message = _extractErrorMessage(error);

  // Map error code to Dart exception type
  switch (errorCode) {
    case ETFlutterErrorCode.ET_FLUTTER_ERROR_MODEL_LOAD:
    case ETFlutterErrorCode.ET_FLUTTER_ERROR_INVALID_HANDLE:
      throw ExecuTorchModelException(message);

    case ETFlutterErrorCode.ET_FLUTTER_ERROR_INFERENCE:
      throw ExecuTorchInferenceException(message);

    case ETFlutterErrorCode.ET_FLUTTER_ERROR_VALIDATION:
    case ETFlutterErrorCode.ET_FLUTTER_ERROR_INVALID_ARGUMENT:
      throw ExecuTorchValidationException(message);

    case ETFlutterErrorCode.ET_FLUTTER_ERROR_MEMORY:
      throw ExecuTorchMemoryException(message);

    case ETFlutterErrorCode.ET_FLUTTER_ERROR_IO:
      throw ExecuTorchIOException(message);

    case ETFlutterErrorCode.ET_FLUTTER_ERROR_PLATFORM:
      throw ExecuTorchPlatformException(message);

    case ETFlutterErrorCode.ET_FLUTTER_SUCCESS:
      // Already handled above, should not reach here
      return;
  }
}

/// Extracts error message from ETFlutterError struct
///
/// Converts the fixed-size char array to a Dart String.
/// Returns empty string if message is null/empty.
String _extractErrorMessage(ETFlutterError error) {
  try {
    // The message is a fixed-size array in the struct
    // Convert each char to a list until we hit null terminator
    final chars = <int>[];
    for (int i = 0; i < 256; i++) {
      final char = error.message[i];
      if (char == 0) break; // Null terminator
      chars.add(char);
    }

    if (chars.isEmpty) {
      return 'Unknown error';
    }

    // Convert to string
    return String.fromCharCodes(chars);
  } catch (e) {
    // Fallback if message extraction fails
    return 'Error code ${error.code}';
  }
}

/// Helper to get error code name as string (for debugging)
String getErrorCodeName(
  ExecutorchFfiBindings bindings,
  ETFlutterErrorCode code,
) {
  final namePtr = bindings.et_flutter_error_code_name(code);
  if (namePtr.address == 0) {
    return 'UNKNOWN_ERROR_CODE';
  }

  try {
    return namePtr.cast<Utf8>().toDartString();
  } catch (e) {
    return 'ERROR_NAME_UNAVAILABLE';
  }
}
