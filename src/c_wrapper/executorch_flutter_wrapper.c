/**
 * @file executorch_flutter_wrapper.c
 * @brief Main C wrapper implementation for ExecuTorch Flutter
 *
 * NOTE: This is a stub implementation that provides the C API interface.
 * The actual ExecuTorch integration requires C++ and will be implemented
 * in executorch_flutter_wrapper.cpp which will include ExecuTorch headers.
 *
 * This file provides utility functions that don't require ExecuTorch.
 */

#include "executorch_flutter_wrapper.h"
#include "error_mapping.h"
#include "tensor_utils.h"
#include <string.h>

// Utility functions implementation

size_t et_flutter_dtype_size(ETFlutterDataType dtype) {
    switch (dtype) {
        case ET_FLUTTER_DTYPE_FLOAT32:
            return 4;
        case ET_FLUTTER_DTYPE_INT32:
            return 4;
        case ET_FLUTTER_DTYPE_INT8:
            return 1;
        case ET_FLUTTER_DTYPE_UINT8:
            return 1;
        default:
            return 0;
    }
}

const char* et_flutter_dtype_name(ETFlutterDataType dtype) {
    switch (dtype) {
        case ET_FLUTTER_DTYPE_FLOAT32:
            return "float32";
        case ET_FLUTTER_DTYPE_INT32:
            return "int32";
        case ET_FLUTTER_DTYPE_INT8:
            return "int8";
        case ET_FLUTTER_DTYPE_UINT8:
            return "uint8";
        default:
            return "unknown";
    }
}

size_t et_flutter_shape_element_count(const ETFlutterTensorShape* shape) {
    if (!shape || shape->num_dims <= 0) {
        return 0;
    }

    size_t count = 1;
    for (int i = 0; i < shape->num_dims; i++) {
        count *= (size_t)shape->dims[i];
    }

    return count;
}

ETFlutterErrorCode et_flutter_validate_tensor(
    const ETFlutterTensorData* tensor,
    ETFlutterError* error
) {
    return et_flutter_tensor_validate(tensor, error);
}

const char* et_flutter_version(void) {
    return "0.0.2-ffi";
}

/*
 * NOTE: The following functions require ExecuTorch integration and will be
 * implemented in executorch_flutter_wrapper.cpp (C++ file) which can include
 * ExecuTorch C++ headers:
 *
 * - ETFlutterLoadResult et_flutter_load_model(const char* file_path)
 * - ETFlutterForwardOutput et_flutter_forward(ETFlutterModelHandle, const ETFlutterForwardInput*)
 * - void et_flutter_dispose_model(ETFlutterModelHandle)
 * - void et_flutter_free_forward_output(ETFlutterForwardOutput*)
 *
 * These will be implemented in a separate .cpp file that can link with
 * ExecuTorch C++ libraries.
 */
