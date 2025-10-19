/**
 * @file c_wrapper_api.h
 * @brief Public C API for ExecuTorch Flutter plugin
 *
 * This header defines the complete public C API for the ExecuTorch Flutter plugin.
 * It is designed to be FFI-friendly (pure C, no C++ features) and will be used by
 * ffigen to auto-generate Dart FFI bindings.
 *
 * @date 2025-10-18
 * @version 1.0
 *
 * @section API Overview
 *
 * The API provides three core operations:
 * 1. Model Loading: et_flutter_load_model()
 * 2. Inference: et_flutter_forward()
 * 3. Disposal: et_flutter_dispose_model()
 *
 * @section Memory Ownership
 *
 * - Input tensors: Dart allocates, C reads (zero-copy), Dart owns
 * - Output tensors: C allocates, Dart copies, C frees via et_flutter_free_forward_output()
 * - Model handles: C allocates, Dart controls lifetime via dispose or NativeFinalizer
 * - Error messages: Fixed-size buffers (no heap allocation), copied to Dart
 *
 * @section Threading
 *
 * - All functions are synchronous (from C perspective)
 * - Dart wraps calls in Isolate.run() for async execution
 * - Model handles are NOT thread-safe (single-threaded use only)
 * - Multiple model handles can coexist (independent models)
 */

#ifndef EXECUTORCH_FLUTTER_WRAPPER_API_H
#define EXECUTORCH_FLUTTER_WRAPPER_API_H

#include <stddef.h>
#include <stdint.h>

#include "error_codes.h"

// Symbol visibility for shared libraries
#if defined(__APPLE__)
  #define ET_FLUTTER_EXPORT __attribute__((visibility("default")))
#elif defined(_WIN32)
  #define ET_FLUTTER_EXPORT __declspec(dllexport)
#else
  #define ET_FLUTTER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ========================================================================= */

/** Maximum number of tensor dimensions */
#define ET_FLUTTER_MAX_TENSOR_DIMS 8

/** Maximum number of input tensors for forward pass */
#define ET_FLUTTER_MAX_INPUTS 16

/** Maximum number of output tensors from forward pass */
#define ET_FLUTTER_MAX_OUTPUTS 16

/** Maximum length for error messages (including null terminator) */
#define ET_FLUTTER_ERROR_MESSAGE_MAX_LEN 256

/** Maximum length for tensor names (including null terminator) */
#define ET_FLUTTER_TENSOR_NAME_MAX_LEN 64

/** Maximum length for file paths (including null terminator) */
#define ET_FLUTTER_FILE_PATH_MAX_LEN 512

/* ============================================================================
 * Type Definitions
 * ========================================================================= */

/**
 * @brief Opaque handle to a loaded ExecuTorch model
 *
 * This is a pointer to internal model data managed by the C wrapper.
 * Dart code should treat this as an opaque Pointer<Void>.
 */
typedef void* ETFlutterModelHandle;

/**
 * @brief Tensor element data types
 */
typedef enum {
    /** 32-bit floating point (IEEE 754 single precision) */
    ET_FLUTTER_DTYPE_FLOAT32 = 0,

    /** 32-bit signed integer */
    ET_FLUTTER_DTYPE_INT32 = 1,

    /** 8-bit signed integer */
    ET_FLUTTER_DTYPE_INT8 = 2,

    /** 8-bit unsigned integer */
    ET_FLUTTER_DTYPE_UINT8 = 3,

} ETFlutterDataType;

/**
 * @brief Error structure containing code and message
 */
typedef struct {
    /** Error code (ET_FLUTTER_SUCCESS on success) */
    ETFlutterErrorCode code;

    /** Human-readable error message (UTF-8 encoded, null-terminated) */
    char message[ET_FLUTTER_ERROR_MESSAGE_MAX_LEN];

} ETFlutterError;

/**
 * @brief Tensor shape information
 */
typedef struct {
    /** Number of dimensions (1-8) */
    int32_t num_dims;

    /** Size of each dimension (e.g., [1, 3, 224, 224] for NCHW image) */
    int64_t dims[ET_FLUTTER_MAX_TENSOR_DIMS];

} ETFlutterTensorShape;

/**
 * @brief Tensor data passed between Dart and C
 *
 * Memory ownership depends on context:
 * - For inputs (Dart → C): Dart owns data, C reads only
 * - For outputs (C → Dart): C allocates, Dart copies, C frees
 */
typedef struct {
    /** Tensor shape (dimensions) */
    ETFlutterTensorShape shape;

    /** Element data type */
    ETFlutterDataType dtype;

    /** Pointer to tensor data (ownership depends on context) */
    void* data;

    /** Size of data buffer in bytes */
    size_t data_size;

    /** Optional tensor name (null-terminated UTF-8, can be NULL) */
    char name[ET_FLUTTER_TENSOR_NAME_MAX_LEN];

} ETFlutterTensorData;

/**
 * @brief Result of model loading operation
 */
typedef struct {
    /** Error information (code == ET_FLUTTER_SUCCESS on success) */
    ETFlutterError error;

    /** Opaque handle to loaded model (NULL on error) */
    ETFlutterModelHandle model_handle;

    /** Model file path (copy of input path, for debugging) */
    char file_path[ET_FLUTTER_FILE_PATH_MAX_LEN];

} ETFlutterLoadResult;

/**
 * @brief Input tensors for forward pass
 */
typedef struct {
    /** Number of input tensors (1-16) */
    int32_t num_inputs;

    /** Array of input tensor pointers (Dart owns data, C reads only) */
    ETFlutterTensorData* inputs[ET_FLUTTER_MAX_INPUTS];

} ETFlutterForwardInput;

/**
 * @brief Output tensors from forward pass
 */
typedef struct {
    /** Error information (code == ET_FLUTTER_SUCCESS on success) */
    ETFlutterError error;

    /** Number of output tensors (0 on error) */
    int32_t num_outputs;

    /** Array of output tensor pointers (C allocates, Dart must free) */
    ETFlutterTensorData* outputs[ET_FLUTTER_MAX_OUTPUTS];

} ETFlutterForwardOutput;

/* ============================================================================
 * Core API Functions
 * ========================================================================= */

/**
 * @brief Load an ExecuTorch model from a file path
 *
 * This function loads a .pte model file, validates it, and returns a handle
 * that can be used for inference operations. The model remains loaded until
 * explicitly disposed via et_flutter_dispose_model() or automatically freed
 * by Dart's NativeFinalizer.
 *
 * @param file_path Absolute path to a .pte model file (null-terminated UTF-8)
 * @return Load result containing model handle and error information
 *
 * @note The returned model_handle must be freed via et_flutter_dispose_model()
 * @note If error.code != ET_FLUTTER_SUCCESS, model_handle will be NULL
 *
 * @par Thread Safety
 * This function is not thread-safe. Call from a single thread only.
 *
 * @par Example
 * @code
 * ETFlutterLoadResult result = et_flutter_load_model("/path/to/model.pte");
 * if (result.error.code != ET_FLUTTER_SUCCESS) {
 *     fprintf(stderr, "Load failed: %s\n", result.error.message);
 *     return;
 * }
 * // Use result.model_handle for inference...
 * et_flutter_dispose_model(result.model_handle);
 * @endcode
 */
ET_FLUTTER_EXPORT ETFlutterLoadResult et_flutter_load_model(const char* file_path);

/**
 * @brief Run inference on a loaded model
 *
 * This function executes the model's forward pass with the provided input tensors
 * and returns output tensors. The input tensor data is read (zero-copy) but not
 * modified or freed. Output tensor data is allocated by this function and must be
 * freed by calling et_flutter_free_forward_output().
 *
 * @param model_handle Handle to a loaded model (from et_flutter_load_model)
 * @param input Input tensor data (Dart owns, C reads only)
 * @return Forward output containing result tensors and error information
 *
 * @note Caller must call et_flutter_free_forward_output() to free output memory
 * @note If error.code != ET_FLUTTER_SUCCESS, num_outputs will be 0
 *
 * @par Thread Safety
 * This function is not thread-safe. Do not call simultaneously with the same model_handle.
 *
 * @par Example
 * @code
 * ETFlutterForwardInput input = { .num_inputs = 1, .inputs = {&tensor} };
 * ETFlutterForwardOutput output = et_flutter_forward(model_handle, &input);
 * if (output.error.code != ET_FLUTTER_SUCCESS) {
 *     fprintf(stderr, "Inference failed: %s\n", output.error.message);
 *     return;
 * }
 * // Use output.outputs[0]->data...
 * et_flutter_free_forward_output(&output);
 * @endcode
 */
ET_FLUTTER_EXPORT ETFlutterForwardOutput et_flutter_forward(
    ETFlutterModelHandle model_handle,
    const ETFlutterForwardInput* input
);

/**
 * @brief Dispose a loaded model and free associated resources
 *
 * This function frees all memory associated with the model handle, including
 * internal ExecuTorch module data. After calling this function, the model_handle
 * becomes invalid and must not be used again.
 *
 * @param model_handle Handle to dispose (from et_flutter_load_model)
 *
 * @note It is safe to pass NULL (no-op)
 * @note Calling with an already-disposed handle is undefined behavior
 * @note This function does not return an error (best-effort cleanup)
 *
 * @par Thread Safety
 * This function is not thread-safe. Do not call simultaneously with the same model_handle.
 *
 * @par Example
 * @code
 * et_flutter_dispose_model(model_handle);
 * model_handle = NULL;  // Prevent accidental reuse
 * @endcode
 */
ET_FLUTTER_EXPORT void et_flutter_dispose_model(ETFlutterModelHandle model_handle);

/**
 * @brief Free output tensors allocated by et_flutter_forward()
 *
 * This function frees all memory allocated for output tensors, including
 * tensor data buffers and metadata structures. After calling this function,
 * the output pointer and all nested pointers become invalid.
 *
 * @param output Pointer to output structure (from et_flutter_forward)
 *
 * @note It is safe to pass NULL (no-op)
 * @note Calling with already-freed output is undefined behavior
 * @note This function does not return an error (best-effort cleanup)
 *
 * @par Example
 * @code
 * ETFlutterForwardOutput output = et_flutter_forward(model_handle, &input);
 * // Copy output data to Dart...
 * et_flutter_free_forward_output(&output);
 * @endcode
 */
ET_FLUTTER_EXPORT void et_flutter_free_forward_output(ETFlutterForwardOutput* output);

/* ============================================================================
 * Utility Functions
 * ========================================================================= */

/**
 * @brief Get the size in bytes of a tensor data type
 *
 * @param dtype Tensor data type
 * @return Size in bytes (e.g., 4 for FLOAT32, 1 for UINT8)
 *         Returns 0 for invalid types
 *
 * @par Example
 * @code
 * size_t element_size = et_flutter_dtype_size(ET_FLUTTER_DTYPE_FLOAT32);  // 4
 * @endcode
 */
ET_FLUTTER_EXPORT size_t et_flutter_dtype_size(ETFlutterDataType dtype);

/**
 * @brief Get human-readable name of a tensor data type
 *
 * @param dtype Tensor data type
 * @return Constant string (e.g., "float32", "int8")
 *         Returns "unknown" for invalid types
 *
 * @note The returned string is constant and should not be freed
 */
ET_FLUTTER_EXPORT const char* et_flutter_dtype_name(ETFlutterDataType dtype);

/**
 * @brief Calculate total number of elements in a tensor shape
 *
 * @param shape Pointer to tensor shape
 * @return Total element count (product of all dimensions)
 *         Returns 0 if shape is NULL or invalid
 *
 * @par Example
 * @code
 * ETFlutterTensorShape shape = { .num_dims = 4, .dims = {1, 3, 224, 224} };
 * size_t count = et_flutter_shape_element_count(&shape);  // 150528
 * @endcode
 */
ET_FLUTTER_EXPORT size_t et_flutter_shape_element_count(const ETFlutterTensorShape* shape);

/**
 * @brief Validate tensor data for correctness
 *
 * Validates that:
 * - Shape has valid dimensions (1-8 dims, all positive)
 * - Data type is valid
 * - Data pointer is not NULL
 * - Data size matches shape and dtype
 *
 * @param tensor Pointer to tensor data
 * @param error Output error structure (populated on failure)
 * @return ET_FLUTTER_SUCCESS on success, error code on failure
 *
 * @par Example
 * @code
 * ETFlutterError error;
 * if (et_flutter_validate_tensor(&tensor, &error) != ET_FLUTTER_SUCCESS) {
 *     fprintf(stderr, "Invalid tensor: %s\n", error.message);
 * }
 * @endcode
 */
ET_FLUTTER_EXPORT ETFlutterErrorCode et_flutter_validate_tensor(
    const ETFlutterTensorData* tensor,
    ETFlutterError* error
);

/**
 * @brief Get library version string
 *
 * @return Constant version string (e.g., "1.0.0")
 *
 * @note The returned string is constant and should not be freed
 */
ET_FLUTTER_EXPORT const char* et_flutter_version(void);

#ifdef __cplusplus
}
#endif

#endif /* EXECUTORCH_FLUTTER_WRAPPER_API_H */
