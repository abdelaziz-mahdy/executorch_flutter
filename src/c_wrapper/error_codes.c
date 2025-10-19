/**
 * @file error_codes.c
 * @brief Error code utility implementations
 */

#include "error_codes.h"

const char* et_flutter_error_code_name(ETFlutterErrorCode code) {
    switch (code) {
        case ET_FLUTTER_SUCCESS:
            return "ET_FLUTTER_SUCCESS";
        case ET_FLUTTER_ERROR_MODEL_LOAD:
            return "ET_FLUTTER_ERROR_MODEL_LOAD";
        case ET_FLUTTER_ERROR_INFERENCE:
            return "ET_FLUTTER_ERROR_INFERENCE";
        case ET_FLUTTER_ERROR_VALIDATION:
            return "ET_FLUTTER_ERROR_VALIDATION";
        case ET_FLUTTER_ERROR_MEMORY:
            return "ET_FLUTTER_ERROR_MEMORY";
        case ET_FLUTTER_ERROR_IO:
            return "ET_FLUTTER_ERROR_IO";
        case ET_FLUTTER_ERROR_PLATFORM:
            return "ET_FLUTTER_ERROR_PLATFORM";
        case ET_FLUTTER_ERROR_INVALID_HANDLE:
            return "ET_FLUTTER_ERROR_INVALID_HANDLE";
        case ET_FLUTTER_ERROR_INVALID_ARGUMENT:
            return "ET_FLUTTER_ERROR_INVALID_ARGUMENT";
        default:
            return "UNKNOWN_ERROR_CODE";
    }
}
