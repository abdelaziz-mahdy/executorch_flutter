/**
 * @file error_mapping.c
 * @brief Error handling and mapping utility implementations
 */

#include "error_mapping.h"
#include <stdio.h>
#include <string.h>

void et_flutter_set_error(
    ETFlutterError* error,
    ETFlutterErrorCode code,
    const char* format,
    ...
) {
    if (!error) {
        return;
    }

    error->code = code;

    va_list args;
    va_start(args, format);
    vsnprintf(error->message, ET_FLUTTER_ERROR_MESSAGE_MAX_LEN, format, args);
    va_end(args);

    // Ensure null termination
    error->message[ET_FLUTTER_ERROR_MESSAGE_MAX_LEN - 1] = '\0';
}

void et_flutter_set_error_v(
    ETFlutterError* error,
    ETFlutterErrorCode code,
    const char* format,
    va_list args
) {
    if (!error) {
        return;
    }

    error->code = code;
    vsnprintf(error->message, ET_FLUTTER_ERROR_MESSAGE_MAX_LEN, format, args);

    // Ensure null termination
    error->message[ET_FLUTTER_ERROR_MESSAGE_MAX_LEN - 1] = '\0';
}

void et_flutter_clear_error(ETFlutterError* error) {
    if (!error) {
        return;
    }

    error->code = ET_FLUTTER_SUCCESS;
    error->message[0] = '\0';
}
