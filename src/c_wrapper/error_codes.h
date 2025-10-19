/**
 * @file error_codes.h
 * @brief Error code definitions for ExecuTorch Flutter C wrapper
 *
 * This header defines all error codes used by the C wrapper API.
 * These codes map to Dart exception types in the existing exception hierarchy.
 *
 * @date 2025-10-18
 * @version 1.0
 */

#ifndef EXECUTORCH_FLUTTER_ERROR_CODES_H
#define EXECUTORCH_FLUTTER_ERROR_CODES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error codes returned by C wrapper functions
 *
 * These error codes map to the following Dart exceptions:
 * - ET_FLUTTER_SUCCESS → (no exception)
 * - ET_FLUTTER_ERROR_MODEL_LOAD → ExecuTorchModelException
 * - ET_FLUTTER_ERROR_INFERENCE → ExecuTorchInferenceException
 * - ET_FLUTTER_ERROR_VALIDATION → ExecuTorchValidationException
 * - ET_FLUTTER_ERROR_MEMORY → ExecuTorchMemoryException
 * - ET_FLUTTER_ERROR_IO → ExecuTorchIOException
 * - ET_FLUTTER_ERROR_PLATFORM → ExecuTorchPlatformException
 * - ET_FLUTTER_ERROR_INVALID_HANDLE → ExecuTorchModelException
 * - ET_FLUTTER_ERROR_INVALID_ARGUMENT → ExecuTorchValidationException
 */
typedef enum {
    /** Operation completed successfully */
    ET_FLUTTER_SUCCESS = 0,

    /** Model loading failed (file not found, invalid format, memory mapping error, etc.) */
    ET_FLUTTER_ERROR_MODEL_LOAD = 1,

    /** Inference execution failed (invalid inputs, runtime error, backend error, etc.) */
    ET_FLUTTER_ERROR_INFERENCE = 2,

    /** Tensor validation failed (wrong shape, incompatible type, dimension mismatch, etc.) */
    ET_FLUTTER_ERROR_VALIDATION = 3,

    /** Memory allocation failed or out of memory */
    ET_FLUTTER_ERROR_MEMORY = 4,

    /** File I/O operation failed (read error, write error, permission denied, etc.) */
    ET_FLUTTER_ERROR_IO = 5,

    /** Platform-specific error (JNI error, framework error, unsupported platform, etc.) */
    ET_FLUTTER_ERROR_PLATFORM = 6,

    /** Invalid model handle (null pointer, disposed model, corrupted handle, etc.) */
    ET_FLUTTER_ERROR_INVALID_HANDLE = 7,

    /** Invalid argument passed to function (null pointer, out of range, etc.) */
    ET_FLUTTER_ERROR_INVALID_ARGUMENT = 8,

} ETFlutterErrorCode;

/**
 * @brief Get human-readable error code name
 *
 * @param code Error code
 * @return Constant string describing the error code (e.g., "ET_FLUTTER_SUCCESS")
 *         Returns "UNKNOWN_ERROR_CODE" for invalid codes
 *
 * @note The returned string is constant and should not be freed
 */
const char* et_flutter_error_code_name(ETFlutterErrorCode code);

#ifdef __cplusplus
}
#endif

#endif /* EXECUTORCH_FLUTTER_ERROR_CODES_H */
