/**
 * @file executorch_ffi.cpp
 * @brief C++ implementation of the ExecuTorch FFI interface
 *
 * This file implements the C API defined in executorch_ffi.h,
 * wrapping the ExecuTorch C++ API for use via dart:ffi.
 */

#include "executorch_ffi.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <memory>
#include <mutex>

// ExecuTorch headers
#include <executorch/extension/module/module.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

using namespace executorch::extension;
using namespace executorch::runtime;

/* ============================================================================
 * Library Version Info
 * ============================================================================ */

#define EXECUTORCH_FFI_VERSION "2.0.0"
#define EXECUTORCH_VERSION "1.0.1"

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct ETTensor {
    ETDType dtype;
    int32_t rank;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

struct ETModule {
    std::unique_ptr<Module> module;
    std::vector<uint8_t> model_buffer;  // Keep buffer alive for BufferDataLoader
    bool loaded;
    int32_t input_count;
    int32_t output_count;
    std::mutex mutex;  // Thread safety
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static ETStatus* create_status(ETErrorCode code, const char* message, const char* location) {
    ETStatus* status = static_cast<ETStatus*>(malloc(sizeof(ETStatus)));
    if (!status) return nullptr;

    status->code = code;
    status->message = message ? strdup(message) : nullptr;
    status->location = location ? strdup(location) : nullptr;

    return status;
}

static ETStatus* create_ok_status() {
    return create_status(ET_OK, nullptr, nullptr);
}

static size_t dtype_size(ETDType dtype) {
    switch (dtype) {
        case ET_DTYPE_FLOAT32: return 4;
        case ET_DTYPE_FLOAT64: return 8;
        case ET_DTYPE_INT64: return 8;
        case ET_DTYPE_INT32: return 4;
        case ET_DTYPE_INT16: return 2;
        case ET_DTYPE_INT8: return 1;
        case ET_DTYPE_UINT8: return 1;
        case ET_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

// Convert ETDType to ExecuTorch ScalarType
static executorch::aten::ScalarType to_scalar_type(ETDType dtype) {
    switch (dtype) {
        case ET_DTYPE_FLOAT32: return executorch::aten::ScalarType::Float;
        case ET_DTYPE_FLOAT64: return executorch::aten::ScalarType::Double;
        case ET_DTYPE_INT64: return executorch::aten::ScalarType::Long;
        case ET_DTYPE_INT32: return executorch::aten::ScalarType::Int;
        case ET_DTYPE_INT16: return executorch::aten::ScalarType::Short;
        case ET_DTYPE_INT8: return executorch::aten::ScalarType::Char;
        case ET_DTYPE_UINT8: return executorch::aten::ScalarType::Byte;
        case ET_DTYPE_BOOL: return executorch::aten::ScalarType::Bool;
        default: return executorch::aten::ScalarType::Float;
    }
}

// Convert ExecuTorch ScalarType to ETDType
static ETDType from_scalar_type(executorch::aten::ScalarType scalar_type) {
    switch (scalar_type) {
        case executorch::aten::ScalarType::Float: return ET_DTYPE_FLOAT32;
        case executorch::aten::ScalarType::Double: return ET_DTYPE_FLOAT64;
        case executorch::aten::ScalarType::Long: return ET_DTYPE_INT64;
        case executorch::aten::ScalarType::Int: return ET_DTYPE_INT32;
        case executorch::aten::ScalarType::Short: return ET_DTYPE_INT16;
        case executorch::aten::ScalarType::Char: return ET_DTYPE_INT8;
        case executorch::aten::ScalarType::Byte: return ET_DTYPE_UINT8;
        case executorch::aten::ScalarType::Bool: return ET_DTYPE_BOOL;
        default: return ET_DTYPE_FLOAT32;
    }
}

// Convert ETTensor to EValue
static EValue tensor_to_evalue(const ETTensor* tensor) {
    if (!tensor) {
        return EValue();
    }

    // Create sizes array
    std::vector<executorch::aten::SizesType> sizes(tensor->rank);
    for (int32_t i = 0; i < tensor->rank; i++) {
        sizes[i] = static_cast<executorch::aten::SizesType>(tensor->shape[i]);
    }

    // Create TensorImpl
    auto scalar_type = to_scalar_type(tensor->dtype);
    auto* impl = new executorch::runtime::etensor::TensorImpl(
        scalar_type,
        tensor->rank,
        sizes.data(),
        const_cast<void*>(static_cast<const void*>(tensor->data.data()))
    );

    return EValue(executorch::aten::Tensor(impl));
}

// Convert EValue tensor to ETTensor
static ETTensor* evalue_to_tensor(const EValue& evalue) {
    if (!evalue.isTensor()) {
        return nullptr;
    }

    const auto& tensor = evalue.toTensor();
    auto sizes = tensor.sizes();
    auto scalar_type = tensor.scalar_type();

    ETTensor* result = new (std::nothrow) ETTensor();
    if (!result) return nullptr;

    result->dtype = from_scalar_type(scalar_type);
    result->rank = static_cast<int32_t>(sizes.size());
    result->shape.resize(result->rank);

    size_t numel = 1;
    for (size_t i = 0; i < sizes.size(); i++) {
        result->shape[i] = sizes[i];
        numel *= sizes[i];
    }

    size_t data_size = numel * dtype_size(result->dtype);
    result->data.resize(data_size);

    const void* src = tensor.const_data_ptr();
    if (src) {
        memcpy(result->data.data(), src, data_size);
    }

    return result;
}

/* ============================================================================
 * Status Functions
 * ============================================================================ */

ET_API void et_status_free(ETStatus* status) {
    if (!status) return;

    if (status->message) free(status->message);
    if (status->location) free(status->location);
    free(status);
}

/* ============================================================================
 * Tensor Functions
 * ============================================================================ */

ET_API ETStatus* et_tensor_create(
    const void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out
) {
    if (!out) {
        return create_status(ET_INVALID_ARGUMENT, "out pointer is null", __func__);
    }

    if (!shape || rank <= 0) {
        return create_status(ET_INVALID_ARGUMENT, "invalid shape or rank", __func__);
    }

    // Calculate expected size
    size_t element_count = 1;
    for (int32_t i = 0; i < rank; i++) {
        if (shape[i] <= 0) {
            return create_status(ET_INVALID_ARGUMENT, "shape dimensions must be positive", __func__);
        }
        element_count *= static_cast<size_t>(shape[i]);
    }

    size_t expected_size = element_count * dtype_size(dtype);
    if (data_size != expected_size) {
        char msg[256];
        snprintf(msg, sizeof(msg), "data size mismatch: expected %zu, got %zu", expected_size, data_size);
        return create_status(ET_INVALID_ARGUMENT, msg, __func__);
    }

    // Allocate tensor
    ETTensor* tensor = new (std::nothrow) ETTensor();
    if (!tensor) {
        return create_status(ET_OUT_OF_MEMORY, "failed to allocate tensor", __func__);
    }

    tensor->dtype = dtype;
    tensor->rank = rank;
    tensor->shape.assign(shape, shape + rank);

    if (data && data_size > 0) {
        tensor->data.resize(data_size);
        memcpy(tensor->data.data(), data, data_size);
    }

    *out = tensor;
    return create_ok_status();
}

ET_API ETDType et_tensor_dtype(const ETTensor* tensor) {
    if (!tensor) return ET_DTYPE_FLOAT32;
    return tensor->dtype;
}

ET_API int32_t et_tensor_rank(const ETTensor* tensor) {
    if (!tensor) return 0;
    return tensor->rank;
}

ET_API const int64_t* et_tensor_shape(const ETTensor* tensor) {
    if (!tensor || tensor->shape.empty()) return nullptr;
    return tensor->shape.data();
}

ET_API size_t et_tensor_data_size(const ETTensor* tensor) {
    if (!tensor) return 0;
    return tensor->data.size();
}

ET_API const void* et_tensor_data(const ETTensor* tensor) {
    if (!tensor || tensor->data.empty()) return nullptr;
    return tensor->data.data();
}

ET_API void et_tensor_free(ETTensor* tensor) {
    if (tensor) {
        delete tensor;
    }
}

ET_API void et_tensor_array_free(ETTensor** tensors, int32_t count) {
    if (!tensors) return;

    for (int32_t i = 0; i < count; i++) {
        et_tensor_free(tensors[i]);
    }
    free(tensors);
}

/* ============================================================================
 * Module Functions
 * ============================================================================ */

ET_API ETStatus* et_module_load(
    const uint8_t* data,
    size_t data_size,
    ETModule** out
) {
    if (!out) {
        return create_status(ET_INVALID_ARGUMENT, "out pointer is null", __func__);
    }

    if (!data || data_size == 0) {
        return create_status(ET_INVALID_ARGUMENT, "invalid model data", __func__);
    }

    // Allocate module
    ETModule* module = new (std::nothrow) ETModule();
    if (!module) {
        return create_status(ET_OUT_OF_MEMORY, "failed to allocate module", __func__);
    }

    // Copy model data to keep it alive for BufferDataLoader
    module->model_buffer.assign(data, data + data_size);

    // Create BufferDataLoader
    auto data_loader = std::make_unique<BufferDataLoader>(
        module->model_buffer.data(),
        module->model_buffer.size()
    );

    // Create Module with the data loader
    module->module = std::make_unique<Module>(std::move(data_loader));

    // Load the program
    auto load_error = module->module->load();
    if (load_error != Error::Ok) {
        delete module;
        return create_status(ET_MODEL_LOAD_FAILED, "failed to load ExecuTorch program", __func__);
    }

    // Load the forward method
    auto forward_error = module->module->load_forward();
    if (forward_error != Error::Ok) {
        delete module;
        return create_status(ET_MODEL_LOAD_FAILED, "failed to load forward method", __func__);
    }

    // Get method metadata
    auto method_meta_result = module->module->method_meta("forward");
    if (method_meta_result.ok()) {
        auto& meta = method_meta_result.get();
        module->input_count = static_cast<int32_t>(meta.num_inputs());
        module->output_count = static_cast<int32_t>(meta.num_outputs());
    } else {
        module->input_count = 1;
        module->output_count = 1;
    }

    module->loaded = true;
    *out = module;
    return create_ok_status();
}

ET_API ETStatus* et_module_load_file(
    const char* path,
    ETModule** out
) {
    if (!out) {
        return create_status(ET_INVALID_ARGUMENT, "out pointer is null", __func__);
    }

    if (!path) {
        return create_status(ET_INVALID_ARGUMENT, "path is null", __func__);
    }

    // Allocate module
    ETModule* module = new (std::nothrow) ETModule();
    if (!module) {
        return create_status(ET_OUT_OF_MEMORY, "failed to allocate module", __func__);
    }

    // Create Module directly from file path
    module->module = std::make_unique<Module>(
        std::string(path),
        Module::LoadMode::MmapUseMlockIgnoreErrors
    );

    // Load the program
    auto load_error = module->module->load();
    if (load_error != Error::Ok) {
        delete module;
        char msg[512];
        snprintf(msg, sizeof(msg), "failed to load program from: %s", path);
        return create_status(ET_MODEL_LOAD_FAILED, msg, __func__);
    }

    // Load the forward method
    auto forward_error = module->module->load_forward();
    if (forward_error != Error::Ok) {
        delete module;
        return create_status(ET_MODEL_LOAD_FAILED, "failed to load forward method", __func__);
    }

    // Get method metadata
    auto method_meta_result = module->module->method_meta("forward");
    if (method_meta_result.ok()) {
        auto& meta = method_meta_result.get();
        module->input_count = static_cast<int32_t>(meta.num_inputs());
        module->output_count = static_cast<int32_t>(meta.num_outputs());
    } else {
        module->input_count = 1;
        module->output_count = 1;
    }

    module->loaded = true;
    *out = module;
    return create_ok_status();
}

ET_API int32_t et_module_input_count(const ETModule* module) {
    if (!module || !module->loaded) return 0;
    return module->input_count;
}

ET_API int32_t et_module_output_count(const ETModule* module) {
    if (!module || !module->loaded) return 0;
    return module->output_count;
}

ET_API ETStatus* et_module_forward(
    ETModule* module,
    ETTensor** inputs,
    int32_t input_count,
    ETTensor*** outputs,
    int32_t* output_count
) {
    if (!module || !module->loaded) {
        return create_status(ET_INVALID_STATE, "module not loaded", __func__);
    }

    if (!outputs || !output_count) {
        return create_status(ET_INVALID_ARGUMENT, "invalid output pointers", __func__);
    }

    if (input_count > 0 && !inputs) {
        return create_status(ET_INVALID_ARGUMENT, "inputs is null", __func__);
    }

    std::lock_guard<std::mutex> lock(module->mutex);

    // Convert input tensors to EValues
    std::vector<EValue> input_evalues;
    input_evalues.reserve(input_count);

    for (int32_t i = 0; i < input_count; i++) {
        if (!inputs[i]) {
            return create_status(ET_INVALID_ARGUMENT, "input tensor is null", __func__);
        }
        input_evalues.push_back(tensor_to_evalue(inputs[i]));
    }

    // Execute forward
    auto result = module->module->forward(input_evalues);
    if (!result.ok()) {
        return create_status(ET_INFERENCE_FAILED, "forward execution failed", __func__);
    }

    auto& output_evalues = result.get();
    *output_count = static_cast<int32_t>(output_evalues.size());

    // Allocate output array
    *outputs = static_cast<ETTensor**>(malloc(sizeof(ETTensor*) * (*output_count)));
    if (!*outputs) {
        return create_status(ET_OUT_OF_MEMORY, "failed to allocate outputs array", __func__);
    }

    // Convert output EValues to ETTensors
    for (int32_t i = 0; i < *output_count; i++) {
        ETTensor* out_tensor = evalue_to_tensor(output_evalues[i]);
        if (!out_tensor) {
            // Clean up
            for (int32_t j = 0; j < i; j++) {
                delete (*outputs)[j];
            }
            free(*outputs);
            *outputs = nullptr;
            *output_count = 0;
            return create_status(ET_INFERENCE_FAILED, "failed to convert output tensor", __func__);
        }
        (*outputs)[i] = out_tensor;
    }

    return create_ok_status();
}

ET_API void et_module_free(ETModule* module) {
    if (module) {
        delete module;
    }
}

/* ============================================================================
 * Backend Query Functions
 * ============================================================================ */

#ifndef ET_BUILD_XNNPACK
    #define ET_BUILD_XNNPACK 1
#endif

#ifndef ET_BUILD_COREML
    #if defined(__APPLE__)
        #define ET_BUILD_COREML 1
    #else
        #define ET_BUILD_COREML 0
    #endif
#endif

#ifndef ET_BUILD_MPS
    #if defined(__APPLE__) && defined(__aarch64__)
        #define ET_BUILD_MPS 1
    #else
        #define ET_BUILD_MPS 0
    #endif
#endif

#ifndef ET_BUILD_VULKAN
    #define ET_BUILD_VULKAN 0
#endif

#ifndef ET_BUILD_QNN
    #define ET_BUILD_QNN 0
#endif

ET_API int32_t et_backend_available(ETBackend backend) {
    switch (backend) {
        case ET_BACKEND_XNNPACK: return ET_BUILD_XNNPACK;
        case ET_BACKEND_COREML: return ET_BUILD_COREML;
        case ET_BACKEND_MPS: return ET_BUILD_MPS;
        case ET_BACKEND_VULKAN: return ET_BUILD_VULKAN;
        case ET_BACKEND_QNN: return ET_BUILD_QNN;
        default: return 0;
    }
}

ET_API int32_t et_backend_list(ETBackend* out, int32_t max_count) {
    if (!out || max_count <= 0) return 0;

    int32_t count = 0;

    #if ET_BUILD_XNNPACK
        if (count < max_count) out[count++] = ET_BACKEND_XNNPACK;
    #endif
    #if ET_BUILD_COREML
        if (count < max_count) out[count++] = ET_BACKEND_COREML;
    #endif
    #if ET_BUILD_MPS
        if (count < max_count) out[count++] = ET_BACKEND_MPS;
    #endif
    #if ET_BUILD_VULKAN
        if (count < max_count) out[count++] = ET_BACKEND_VULKAN;
    #endif
    #if ET_BUILD_QNN
        if (count < max_count) out[count++] = ET_BACKEND_QNN;
    #endif

    return count;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

ET_API const char* et_version(void) {
    return EXECUTORCH_FFI_VERSION;
}

ET_API const char* et_executorch_version(void) {
    return EXECUTORCH_VERSION;
}

ET_API void et_string_free(char* str) {
    if (str) free(str);
}

static int32_t g_log_level = 1;

ET_API void et_set_log_level(int32_t level) {
    g_log_level = level;
}
