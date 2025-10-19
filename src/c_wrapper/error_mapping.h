/**
 * @file error_mapping.h
 * @brief Error handling and mapping utilities
 *
 * Provides utilities for setting error messages and mapping ExecuTorch
 * errors to ETFlutterError structures.
 *
 * @date 2025-10-18
 * @version 1.0
 */

#ifndef EXECUTORCH_FLUTTER_ERROR_MAPPING_H
#define EXECUTORCH_FLUTTER_ERROR_MAPPING_H

#include "executorch_flutter_wrapper.h"
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set error with formatted message
 *
 * @param error Output error structure
 * @param code Error code
 * @param format Printf-style format string
 * @param ... Variable arguments for formatting
 */
void et_flutter_set_error(
    ETFlutterError* error,
    ETFlutterErrorCode code,
    const char* format,
    ...
);

/**
 * @brief Set error with formatted message (va_list version)
 *
 * @param error Output error structure
 * @param code Error code
 * @param format Printf-style format string
 * @param args Variable argument list
 */
void et_flutter_set_error_v(
    ETFlutterError* error,
    ETFlutterErrorCode code,
    const char* format,
    va_list args
);

/**
 * @brief Clear error structure (set to success)
 *
 * @param error Error structure to clear
 */
void et_flutter_clear_error(ETFlutterError* error);

#ifdef __cplusplus
}
#endif

#endif /* EXECUTORCH_FLUTTER_ERROR_MAPPING_H */
