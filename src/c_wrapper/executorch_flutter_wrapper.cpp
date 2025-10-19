/**
 * @file executorch_flutter_wrapper.cpp
 * @brief C++ implementation wrapping ExecuTorch Module API
 *
 * This file implements the core model loading, inference, and disposal
 * functions by interfacing with the ExecuTorch C++ API.
 */

#include "executorch_flutter_wrapper.h"
#include "error_mapping.h"
#include "tensor_utils.h"
#include "platform/platform_utils.h"

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/portable_type/tensor.h>

#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

using namespace executorch::extension::module;
using namespace executorch::runtime;

// Internal model data structure
struct ETFlutterModelData {
    std::unique_ptr<Module> module;
    std::string file_path;
};

//==============================================================================
// Core API Implementation
//==============================================================================

extern "C" {

ETFlutterLoadResult et_flutter_load_model(const char* file_path) {
    ETFlutterLoadResult result;
    memset(&result, 0, sizeof(result));
    et_flutter_clear_error(&result.error);

    // Validate input
    if (!file_path) {
        et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_INVALID_ARGUMENT,
                            "file_path is NULL");
        return result;
    }

    // Check file exists
    if (!et_flutter_platform_file_exists(file_path)) {
        et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_IO,
                            "Model file not found: %s", file_path);
        return result;
    }

    // Check file is readable
    if (!et_flutter_platform_file_readable(file_path)) {
        et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_IO,
                            "Model file not readable: %s", file_path);
        return result;
    }

    try {
        // Allocate model data
        ETFlutterModelData* model_data = new ETFlutterModelData();

        // Load ExecuTorch module (use Mmap for better performance)
        model_data->module = std::make_unique<Module>(
            std::string(file_path),
            Module::LoadMode::Mmap
        );
        model_data->file_path = file_path;

        // Load the program
        Error load_error = model_data->module->load();
        if (load_error != Error::Ok) {
            et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_MODEL_LOAD,
                                "Failed to load ExecuTorch module: error code %d",
                                static_cast<int>(load_error));
            delete model_data;
            return result;
        }

        // Load the forward method
        Error method_error = model_data->module->load_forward();
        if (method_error != Error::Ok) {
            et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_MODEL_LOAD,
                                "Failed to load forward method: error code %d",
                                static_cast<int>(method_error));
            delete model_data;
            return result;
        }

        // Success
        result.model_handle = static_cast<ETFlutterModelHandle>(model_data);
        strncpy(result.file_path, file_path, ET_FLUTTER_FILE_PATH_MAX_LEN - 1);
        result.file_path[ET_FLUTTER_FILE_PATH_MAX_LEN - 1] = '\0';

    } catch (const std::exception& e) {
        et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_MODEL_LOAD,
                            "Exception during model load: %s", e.what());
    } catch (...) {
        et_flutter_set_error(&result.error, ET_FLUTTER_ERROR_MODEL_LOAD,
                            "Unknown exception during model load");
    }

    return result;
}

// Helper function to convert ETFlutterTensorData to ExecuTorch Tensor
static Result<Tensor> convert_to_executorch_tensor(const ETFlutterTensorData* tensor) {
    if (!tensor || !tensor->data) {
        return Error::InvalidArgument;
    }

    // Map Flutter dtype to ExecuTorch ScalarType
    ScalarType scalar_type;
    switch (tensor->dtype) {
        case ET_FLUTTER_DTYPE_FLOAT32:
            scalar_type = ScalarType::Float;
            break;
        case ET_FLUTTER_DTYPE_INT32:
            scalar_type = ScalarType::Int;
            break;
        case ET_FLUTTER_DTYPE_INT8:
            scalar_type = ScalarType::Char;
            break;
        case ET_FLUTTER_DTYPE_UINT8:
            scalar_type = ScalarType::Byte;
            break;
        default:
            return Error::InvalidArgument;
    }

    // Convert dims to exec_aten::SizesType
    std::vector<exec_aten::SizesType> sizes;
    for (int i = 0; i < tensor->shape.num_dims; i++) {
        sizes.push_back(static_cast<exec_aten::SizesType>(tensor->shape.dims[i]));
    }

    // Create TensorImpl (shallow - shares data pointer with Dart)
    auto tensor_impl = TensorImpl(
        scalar_type,
        sizes.size(),
        sizes.data(),
        tensor->data,  // Share Dart's data pointer
        nullptr,       // No dim_order (default)
        nullptr,       // No strides (contiguous)
        TensorShapeDynamism::STATIC
    );

    return Tensor(&tensor_impl);
}

// Helper function to convert ExecuTorch Tensor to ETFlutterTensorData
static ETFlutterTensorData* convert_from_executorch_tensor(const Tensor& tensor) {
    ETFlutterTensorData* result = (ETFlutterTensorData*)malloc(sizeof(ETFlutterTensorData));
    if (!result) {
        return nullptr;
    }

    memset(result, 0, sizeof(ETFlutterTensorData));

    // Map ExecuTorch ScalarType to Flutter dtype
    switch (tensor.scalar_type()) {
        case ScalarType::Float:
            result->dtype = ET_FLUTTER_DTYPE_FLOAT32;
            break;
        case ScalarType::Int:
            result->dtype = ET_FLUTTER_DTYPE_INT32;
            break;
        case ScalarType::Char:
            result->dtype = ET_FLUTTER_DTYPE_INT8;
            break;
        case ScalarType::Byte:
            result->dtype = ET_FLUTTER_DTYPE_UINT8;
            break;
        default:
            free(result);
            return nullptr;
    }

    // Copy shape
    result->shape.num_dims = static_cast<int32_t>(tensor.dim());
    for (size_t i = 0; i < tensor.dim() && i < ET_FLUTTER_MAX_TENSOR_DIMS; i++) {
        result->shape.dims[i] = static_cast<int64_t>(tensor.size(i));
    }

    // Calculate data size
    result->data_size = et_flutter_tensor_data_size(result);

    // Allocate and copy data (deep copy for output tensors)
    result->data = malloc(result->data_size);
    if (!result->data) {
        free(result);
        return nullptr;
    }

    memcpy(result->data, tensor.const_data_ptr(), result->data_size);

    // Clear name (optional)
    result->name[0] = '\0';

    return result;
}

ETFlutterForwardOutput et_flutter_forward(
    ETFlutterModelHandle model_handle,
    const ETFlutterForwardInput* input
) {
    ETFlutterForwardOutput output;
    memset(&output, 0, sizeof(output));
    et_flutter_clear_error(&output.error);

    // Validate inputs
    if (!model_handle) {
        et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_INVALID_HANDLE,
                            "model_handle is NULL");
        return output;
    }

    if (!input || input->num_inputs <= 0) {
        et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_INVALID_ARGUMENT,
                            "input is NULL or has no tensors");
        return output;
    }

    ETFlutterModelData* model_data = static_cast<ETFlutterModelData*>(model_handle);

    try {
        // Convert input tensors to ExecuTorch EValues
        std::vector<EValue> inputs;
        for (int i = 0; i < input->num_inputs; i++) {
            auto tensor_result = convert_to_executorch_tensor(input->inputs[i]);
            if (!tensor_result.ok()) {
                et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_VALIDATION,
                                    "Failed to convert input tensor %d", i);
                return output;
            }
            inputs.push_back(EValue(tensor_result.get()));
        }

        // Execute forward pass
        auto forward_result = model_data->module->forward(inputs);
        if (!forward_result.ok()) {
            et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_INFERENCE,
                                "Forward pass failed: error code %d",
                                static_cast<int>(forward_result.error()));
            return output;
        }

        // Convert output EValues to C tensors
        const auto& outputs = forward_result.get();
        output.num_outputs = static_cast<int32_t>(outputs.size());

        for (size_t i = 0; i < outputs.size() && i < ET_FLUTTER_MAX_OUTPUTS; i++) {
            if (outputs[i].isTensor()) {
                output.outputs[i] = convert_from_executorch_tensor(outputs[i].toTensor());
                if (!output.outputs[i]) {
                    et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_MEMORY,
                                        "Failed to allocate output tensor %zu", i);
                    // Free already-allocated outputs
                    for (size_t j = 0; j < i; j++) {
                        if (output.outputs[j]) {
                            free(output.outputs[j]->data);
                            free(output.outputs[j]);
                        }
                    }
                    output.num_outputs = 0;
                    return output;
                }
            }
        }

    } catch (const std::exception& e) {
        et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_INFERENCE,
                            "Exception during forward pass: %s", e.what());
        output.num_outputs = 0;
    } catch (...) {
        et_flutter_set_error(&output.error, ET_FLUTTER_ERROR_INFERENCE,
                            "Unknown exception during forward pass");
        output.num_outputs = 0;
    }

    return output;
}

void et_flutter_dispose_model(ETFlutterModelHandle model_handle) {
    if (!model_handle) {
        return;  // Safe no-op
    }

    ETFlutterModelData* model_data = static_cast<ETFlutterModelData*>(model_handle);
    delete model_data;  // Unique_ptr will clean up Module
}

void et_flutter_free_forward_output(ETFlutterForwardOutput* output) {
    if (!output) {
        return;  // Safe no-op
    }

    // Free each output tensor
    for (int i = 0; i < output->num_outputs; i++) {
        if (output->outputs[i]) {
            if (output->outputs[i]->data) {
                free(output->outputs[i]->data);
            }
            free(output->outputs[i]);
        }
    }

    // Clear output structure
    output->num_outputs = 0;
}

} // extern "C"
