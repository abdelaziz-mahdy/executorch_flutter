/**
 * @file tensor_utils.c
 * @brief Tensor conversion utility implementations
 */

#include "tensor_utils.h"
#include "error_mapping.h"
#include <string.h>

ETFlutterErrorCode et_flutter_tensor_validate(
    const ETFlutterTensorData* tensor,
    ETFlutterError* error
) {
    if (!tensor) {
        et_flutter_set_error(error, ET_FLUTTER_ERROR_INVALID_ARGUMENT,
                            "Tensor pointer is NULL");
        return ET_FLUTTER_ERROR_INVALID_ARGUMENT;
    }

    // Validate number of dimensions
    if (tensor->shape.num_dims < 1 || tensor->shape.num_dims > ET_FLUTTER_MAX_TENSOR_DIMS) {
        et_flutter_set_error(error, ET_FLUTTER_ERROR_VALIDATION,
                            "Invalid number of dimensions: %d (must be 1-%d)",
                            tensor->shape.num_dims, ET_FLUTTER_MAX_TENSOR_DIMS);
        return ET_FLUTTER_ERROR_VALIDATION;
    }

    // Validate all dimension sizes are positive
    for (int i = 0; i < tensor->shape.num_dims; i++) {
        if (tensor->shape.dims[i] <= 0) {
            et_flutter_set_error(error, ET_FLUTTER_ERROR_VALIDATION,
                                "Invalid dimension size at index %d: %lld (must be > 0)",
                                i, (long long)tensor->shape.dims[i]);
            return ET_FLUTTER_ERROR_VALIDATION;
        }
    }

    // Validate data pointer
    if (!tensor->data) {
        et_flutter_set_error(error, ET_FLUTTER_ERROR_INVALID_ARGUMENT,
                            "Tensor data pointer is NULL");
        return ET_FLUTTER_ERROR_INVALID_ARGUMENT;
    }

    // Validate data size matches expected size
    size_t expected_size = et_flutter_tensor_data_size(tensor);
    if (tensor->data_size != expected_size) {
        et_flutter_set_error(error, ET_FLUTTER_ERROR_VALIDATION,
                            "Tensor data size mismatch: got %zu bytes, expected %zu bytes",
                            tensor->data_size, expected_size);
        return ET_FLUTTER_ERROR_VALIDATION;
    }

    et_flutter_clear_error(error);
    return ET_FLUTTER_SUCCESS;
}

size_t et_flutter_tensor_data_size(const ETFlutterTensorData* tensor) {
    if (!tensor) {
        return 0;
    }

    // Calculate total number of elements
    size_t total_elements = 1;
    for (int i = 0; i < tensor->shape.num_dims; i++) {
        total_elements *= (size_t)tensor->shape.dims[i];
    }

    // Get element size for dtype
    size_t element_size = et_flutter_dtype_size(tensor->dtype);

    return total_elements * element_size;
}

ETFlutterErrorCode et_flutter_tensor_copy(
    const ETFlutterTensorData* src,
    ETFlutterTensorData* dst,
    ETFlutterError* error
) {
    if (!src || !dst) {
        et_flutter_set_error(error, ET_FLUTTER_ERROR_INVALID_ARGUMENT,
                            "Source or destination tensor is NULL");
        return ET_FLUTTER_ERROR_INVALID_ARGUMENT;
    }

    // Validate source tensor
    ETFlutterErrorCode result = et_flutter_tensor_validate(src, error);
    if (result != ET_FLUTTER_SUCCESS) {
        return result;
    }

    // Copy shape
    dst->shape = src->shape;

    // Copy dtype
    dst->dtype = src->dtype;

    // Copy data size
    dst->data_size = src->data_size;

    // Copy data (assumes dst->data is already allocated)
    if (dst->data && src->data) {
        memcpy(dst->data, src->data, src->data_size);
    }

    // Copy name
    if (src->name[0] != '\0') {
        strncpy(dst->name, src->name, ET_FLUTTER_TENSOR_NAME_MAX_LEN - 1);
        dst->name[ET_FLUTTER_TENSOR_NAME_MAX_LEN - 1] = '\0';
    } else {
        dst->name[0] = '\0';
    }

    et_flutter_clear_error(error);
    return ET_FLUTTER_SUCCESS;
}
