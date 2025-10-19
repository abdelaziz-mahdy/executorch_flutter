/**
 * @file tensor_utils.h
 * @brief Tensor conversion utilities for ExecuTorch Flutter C wrapper
 *
 * Provides utilities for converting between Dart tensor representations
 * (ETFlutterTensorData) and ExecuTorch native tensor formats.
 *
 * @date 2025-10-18
 * @version 1.0
 */

#ifndef EXECUTORCH_FLUTTER_TENSOR_UTILS_H
#define EXECUTORCH_FLUTTER_TENSOR_UTILS_H

#include "executorch_flutter_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate tensor data structure
 *
 * Checks that tensor has valid shape, dtype, data pointer, and size
 *
 * @param tensor Pointer to tensor to validate
 * @param error Output error structure (populated on failure)
 * @return ET_FLUTTER_SUCCESS on success, error code on failure
 */
ETFlutterErrorCode et_flutter_tensor_validate(
    const ETFlutterTensorData* tensor,
    ETFlutterError* error
);

/**
 * @brief Calculate total bytes needed for tensor data
 *
 * @param tensor Pointer to tensor
 * @return Total bytes required (shape elements * dtype size)
 */
size_t et_flutter_tensor_data_size(const ETFlutterTensorData* tensor);

/**
 * @brief Copy tensor data with validation
 *
 * @param src Source tensor
 * @param dst Destination tensor (data buffer must be pre-allocated)
 * @param error Output error structure
 * @return ET_FLUTTER_SUCCESS on success, error code on failure
 */
ETFlutterErrorCode et_flutter_tensor_copy(
    const ETFlutterTensorData* src,
    ETFlutterTensorData* dst,
    ETFlutterError* error
);

#ifdef __cplusplus
}
#endif

#endif /* EXECUTORCH_FLUTTER_TENSOR_UTILS_H */
