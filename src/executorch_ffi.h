/**
 * @file executorch_ffi.h
 * @brief C interface for ExecuTorch Flutter FFI bindings
 *
 * This header defines the C API for executorch_flutter, enabling
 * cross-platform native bindings via dart:ffi.
 *
 * Design Principles:
 * 1. Opaque Pointers: Hide implementation details
 * 2. Status Returns: Rich error information
 * 3. Memory Ownership: Clear caller/callee semantics
 * 4. Thread Safety: Document guarantees
 */

#ifndef EXECUTORCH_FFI_H
#define EXECUTORCH_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Export Macros
 * ============================================================================ */

#if defined(_WIN32) || defined(_WIN64)
    #ifdef EXECUTORCH_FFI_EXPORTS
        #define ET_API __declspec(dllexport)
    #else
        #define ET_API __declspec(dllimport)
    #endif
#else
    #define ET_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/**
 * Error codes returned by FFI functions.
 */
typedef enum {
    ET_OK = 0,                    /**< Success */
    ET_INVALID_ARGUMENT = 1,      /**< Invalid function argument */
    ET_OUT_OF_MEMORY = 2,         /**< Memory allocation failed */
    ET_MODEL_LOAD_FAILED = 3,     /**< Model loading failed */
    ET_INFERENCE_FAILED = 4,      /**< Forward pass failed */
    ET_INVALID_STATE = 5,         /**< Invalid object state */
    ET_UNSUPPORTED = 6,           /**< Unsupported operation */
    ET_IO_ERROR = 7,              /**< I/O error */
    ET_INTERNAL = 99              /**< Internal error */
} ETErrorCode;

/**
 * Status structure for error handling.
 *
 * When code is ET_OK, message and location are NULL.
 * When code is non-zero, message contains error description.
 * Caller must free with et_status_free().
 */
typedef struct ETStatus {
    int32_t code;           /**< Error code (0 = success) */
    char* message;          /**< Error message (heap allocated, may be NULL) */
    char* location;         /**< Source location "file:line:func" (may be NULL) */
} ETStatus;

/**
 * Free status structure and its strings.
 * Safe to call with NULL.
 */
ET_API void et_status_free(ETStatus* status);

/* ============================================================================
 * Tensor Types
 * ============================================================================ */

/**
 * Tensor data types.
 */
typedef enum {
    ET_DTYPE_FLOAT32 = 0,
    ET_DTYPE_FLOAT64 = 1,
    ET_DTYPE_INT64 = 2,
    ET_DTYPE_INT32 = 3,
    ET_DTYPE_INT16 = 4,
    ET_DTYPE_INT8 = 5,
    ET_DTYPE_UINT8 = 6,
    ET_DTYPE_BOOL = 7
} ETDType;

/**
 * Opaque tensor handle.
 */
typedef struct ETTensor ETTensor;

/**
 * Create a tensor from data.
 *
 * @param data      Pointer to tensor data (copied)
 * @param data_size Size of data in bytes
 * @param shape     Array of dimension sizes
 * @param rank      Number of dimensions
 * @param dtype     Data type
 * @param out       Output tensor handle
 * @return Status (caller must free)
 *
 * Memory: data is copied, caller retains ownership of original
 */
ET_API ETStatus* et_tensor_create(
    const void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out
);

/**
 * Get tensor data type.
 */
ET_API ETDType et_tensor_dtype(const ETTensor* tensor);

/**
 * Get tensor rank (number of dimensions).
 */
ET_API int32_t et_tensor_rank(const ETTensor* tensor);

/**
 * Get tensor shape array.
 *
 * @return Pointer to internal shape array (do not free)
 */
ET_API const int64_t* et_tensor_shape(const ETTensor* tensor);

/**
 * Get tensor data size in bytes.
 */
ET_API size_t et_tensor_data_size(const ETTensor* tensor);

/**
 * Get tensor data pointer.
 *
 * @return Pointer to internal data (do not free, valid until tensor freed)
 */
ET_API const void* et_tensor_data(const ETTensor* tensor);

/**
 * Free tensor handle.
 * Safe to call with NULL.
 */
ET_API void et_tensor_free(ETTensor* tensor);

/* ============================================================================
 * Module (Model) API
 * ============================================================================ */

/**
 * Opaque module handle.
 */
typedef struct ETModule ETModule;

/**
 * Load model from memory buffer.
 *
 * @param data       Model data (.pte format)
 * @param data_size  Size of model data
 * @param out        Output module handle
 * @return Status (caller must free)
 *
 * Memory: data is not retained after function returns
 * Thread Safety: Function is thread-safe
 */
ET_API ETStatus* et_module_load(
    const uint8_t* data,
    size_t data_size,
    ETModule** out
);

/**
 * Load model from file path.
 *
 * @param path   Path to .pte model file
 * @param out    Output module handle
 * @return Status (caller must free)
 */
ET_API ETStatus* et_module_load_file(
    const char* path,
    ETModule** out
);

/**
 * Get number of model inputs.
 */
ET_API int32_t et_module_input_count(const ETModule* module);

/**
 * Get number of model outputs.
 */
ET_API int32_t et_module_output_count(const ETModule* module);

/**
 * Run forward pass (inference).
 *
 * @param module       Module handle
 * @param inputs       Array of input tensor handles
 * @param input_count  Number of inputs
 * @param outputs      Output array of tensor handles (caller must free array and tensors)
 * @param output_count Output number of outputs
 * @return Status (caller must free)
 *
 * Memory: outputs array and tensors allocated by callee, caller must free
 * Thread Safety: Not thread-safe for same module, use mutex if concurrent
 */
ET_API ETStatus* et_module_forward(
    ETModule* module,
    ETTensor** inputs,
    int32_t input_count,
    ETTensor*** outputs,
    int32_t* output_count
);

/**
 * Free module handle.
 * Safe to call with NULL.
 */
ET_API void et_module_free(ETModule* module);

/* ============================================================================
 * Backend Query API
 * ============================================================================ */

/**
 * Backend identifiers.
 */
typedef enum {
    ET_BACKEND_XNNPACK = 0,
    ET_BACKEND_COREML = 1,
    ET_BACKEND_MPS = 2,
    ET_BACKEND_VULKAN = 3,
    ET_BACKEND_QNN = 4
} ETBackend;

/**
 * Check if backend is available (compiled in).
 *
 * @param backend  Backend to check
 * @return 1 if available, 0 if not
 */
ET_API int32_t et_backend_available(ETBackend backend);

/**
 * Get list of available backends.
 *
 * @param out        Output array of backends (caller allocates, max 16 elements)
 * @param max_count  Maximum number of backends to return
 * @return Number of available backends
 */
ET_API int32_t et_backend_list(ETBackend* out, int32_t max_count);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get library version string.
 *
 * @return Version string (do not free)
 */
ET_API const char* et_version(void);

/**
 * Get linked ExecuTorch version string.
 *
 * @return Version string (do not free)
 */
ET_API const char* et_executorch_version(void);

/**
 * Free an array of tensors.
 *
 * @param tensors  Array of tensor handles
 * @param count    Number of tensors
 */
ET_API void et_tensor_array_free(ETTensor** tensors, int32_t count);

/**
 * Free a string allocated by this library.
 */
ET_API void et_string_free(char* str);

/**
 * Set debug logging level.
 *
 * @param level  0=off, 1=error, 2=warn, 3=info, 4=debug
 */
ET_API void et_set_log_level(int32_t level);

#ifdef __cplusplus
}
#endif

#endif /* EXECUTORCH_FFI_H */
