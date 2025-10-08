/// ExecuTorch Error Handling - Cross-Platform Error Mapping
///
/// This file provides standardized error handling across Android and iOS platforms,
/// ensuring consistent error reporting and debugging capabilities for ExecuTorch operations.
library;

import 'generated/executorch_api.dart';

/// Base exception class for all ExecuTorch-related errors
class ExecuTorchException implements Exception {
  const ExecuTorchException(this.message, [this.details]);

  final String message;
  final String? details;

  @override
  String toString() => details != null
    ? 'ExecuTorchException: $message\nDetails: $details'
    : 'ExecuTorchException: $message';
}

/// Model loading and lifecycle errors
class ExecuTorchModelException extends ExecuTorchException {
  const ExecuTorchModelException(super.message, [super.details]);
}

/// Inference execution errors
class ExecuTorchInferenceException extends ExecuTorchException {
  const ExecuTorchInferenceException(super.message, [super.details]);
}

/// Tensor validation and data errors
class ExecuTorchValidationException extends ExecuTorchException {
  const ExecuTorchValidationException(super.message, [super.details]);
}

/// Memory and resource errors
class ExecuTorchMemoryException extends ExecuTorchException {
  const ExecuTorchMemoryException(super.message, [super.details]);
}

/// Network and file I/O errors
class ExecuTorchIOException extends ExecuTorchException {
  const ExecuTorchIOException(super.message, [super.details]);
}

/// Platform-specific integration errors
class ExecuTorchPlatformException extends ExecuTorchException {
  const ExecuTorchPlatformException(super.message, [super.details]);
}

/// Error mapping utilities for cross-platform consistency
class ExecuTorchErrorMapper {

  /// Map platform-specific error messages to standardized exceptions
  static ExecuTorchException mapPlatformError(String platformError, [String? details]) {
    final lowerError = platformError.toLowerCase();

    // Model loading errors
    if (lowerError.contains('model not found') ||
        lowerError.contains('file not found') ||
        lowerError.contains('failed to load')) {
      return ExecuTorchModelException(platformError, details);
    }

    // Inference errors
    if (lowerError.contains('inference failed') ||
        lowerError.contains('forward failed') ||
        lowerError.contains('execution failed')) {
      return ExecuTorchInferenceException(platformError, details);
    }

    // Validation errors
    if (lowerError.contains('validation') ||
        lowerError.contains('invalid tensor') ||
        lowerError.contains('shape mismatch')) {
      return ExecuTorchValidationException(platformError, details);
    }

    // Memory errors
    if (lowerError.contains('memory') ||
        lowerError.contains('out of memory') ||
        lowerError.contains('allocation failed')) {
      return ExecuTorchMemoryException(platformError, details);
    }

    // I/O errors
    if (lowerError.contains('network') ||
        lowerError.contains('download') ||
        lowerError.contains('connection')) {
      return ExecuTorchIOException(platformError, details);
    }

    // Default to platform exception
    return ExecuTorchPlatformException(platformError, details);
  }

}