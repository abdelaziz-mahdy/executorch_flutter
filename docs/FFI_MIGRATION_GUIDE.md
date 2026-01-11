# ExecuTorch Flutter - FFI Migration Guide

This document provides a complete step-by-step guide for migrating executorch_flutter from Pigeon-based method channels to dart:ffi with native assets.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Project Setup](#phase-1-project-setup)
4. [Phase 2: C Interface Implementation](#phase-2-c-interface-implementation)
5. [Phase 3: FFI Bindings Generation](#phase-3-ffi-bindings-generation)
6. [Phase 4: Native Assets Build System](#phase-4-native-assets-build-system)
7. [Phase 5: Dart Wrapper Implementation](#phase-5-dart-wrapper-implementation)
8. [Phase 6: Platform-Specific Integration](#phase-6-platform-specific-integration)
9. [Phase 7: Testing & Validation](#phase-7-testing--validation)
10. [Phase 8: Cleanup & Release](#phase-8-cleanup--release)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### Current Architecture

```
executorch_flutter/
├── lib/
│   └── src/
│       ├── executorch_model.dart       # Dart API
│       └── generated/
│           └── executorch_api.dart     # Pigeon-generated
├── android/
│   └── src/main/kotlin/                # Kotlin implementation
├── ios/
│   └── Classes/                        # Swift implementation
├── macos/
│   └── Classes/                        # Swift implementation (symlinked)
└── pigeons/
    └── executorch_api.dart             # Pigeon definitions
```

### Target Architecture

```
executorch_flutter/
├── lib/
│   └── src/
│       ├── executorch_model.dart       # Dart API (updated)
│       ├── ffi/
│       │   ├── bindings.dart           # FFI wrapper classes
│       │   ├── memory.dart             # Memory management
│       │   └── types.dart              # Type conversions
│       └── generated/
│           └── executorch_ffi.g.dart   # ffigen-generated
├── src/                                # NEW: C/C++ source
│   ├── CMakeLists.txt
│   ├── executorch_ffi.h
│   ├── executorch_ffi.cpp
│   └── backends/
│       ├── xnnpack.cpp
│       ├── coreml.cpp
│       └── ...
├── hook/
│   └── build.dart                      # Native assets build hook
└── ffigen.yaml                         # ffigen configuration
```

### Benefits After Migration

| Aspect | Before (Pigeon) | After (FFI) |
|--------|-----------------|-------------|
| Platforms | Android, iOS, macOS, Web | + Linux, Windows |
| Performance | Method channel overhead | Direct native calls |
| Code duplication | 3 implementations | 1 unified C wrapper |
| Tensor transfer | Serialization | Zero-copy possible |
| Binary size | Separate per platform | Single shared library |
| Build complexity | Gradle/CocoaPods/SPM | Unified CMake |

---

## Prerequisites

### Required Tools

```bash
# Dart SDK 3.3+
dart --version

# Flutter 3.16+
flutter --version

# CMake 3.18+
cmake --version

# Ninja build system
ninja --version

# For Android: NDK 25+
echo $ANDROID_NDK_HOME

# For iOS/macOS: Xcode 15+
xcodebuild -version

# For Linux: GCC/Clang, patchelf
gcc --version
patchelf --version

# For Windows: Visual Studio 2022 with C++ workload
```

### Required Dart Packages

```yaml
# pubspec.yaml
dependencies:
  ffi: ^2.1.4

dev_dependencies:
  ffigen: ^14.0.0
  code_assets: ^1.0.0
  hooks: ^1.0.0
  native_toolchain_cmake: ^0.2.2
```

### ExecuTorch Source

```bash
# Clone ExecuTorch repository
git clone https://github.com/pytorch/executorch.git
cd executorch
git checkout v0.4.0  # or latest stable

# Install dependencies
./install_requirements.sh

# Note the path - needed for CMakeLists.txt
export EXECUTORCH_ROOT=$(pwd)
```

---

## Phase 1: Project Setup

### Step 1.1: Create Directory Structure

```bash
cd executorch_flutter

# Create new directories
mkdir -p src/backends
mkdir -p lib/src/ffi
mkdir -p hook
mkdir -p docs
```

### Step 1.2: Update pubspec.yaml

```yaml
name: executorch_flutter
description: Flutter plugin for ExecuTorch on-device ML inference
version: 2.0.0  # Major version bump for breaking change

environment:
  sdk: '>=3.3.0 <4.0.0'
  flutter: '>=3.16.0'

dependencies:
  flutter:
    sdk: flutter
  ffi: ^2.1.4
  meta: ^1.9.1
  path_provider: ^2.1.5

dev_dependencies:
  ffigen: ^14.0.0
  code_assets: ^1.0.0
  hooks: ^1.0.0
  native_toolchain_cmake: ^0.2.2
  flutter_test:
    sdk: flutter
  integration_test:
    sdk: flutter

# Native assets configuration
hooks:
  user_defines:
    executorch_flutter:
      # Build configuration
      debug: false

      # Backend selection (platform-dependent)
      backends:
        - xnnpack      # All platforms (default)
        # - coreml     # iOS/macOS only
        # - mps        # macOS only (Metal)
        # - vulkan     # Android/Linux/Windows
        # - qnn        # Android (Qualcomm)

      # Optional features
      features:
        - quantization
        # - profiling

flutter:
  plugin:
    platforms:
      android:
        ffiPlugin: true
      ios:
        ffiPlugin: true
      macos:
        ffiPlugin: true
      linux:
        ffiPlugin: true
      windows:
        ffiPlugin: true
```

### Step 1.3: Create .gitignore Updates

```gitignore
# Add to .gitignore

# Native build artifacts
src/build/
*.o
*.a
*.so
*.dylib
*.dll

# Don't ignore generated FFI bindings (commit them)
# !lib/src/generated/executorch_ffi.g.dart
```

---

## Phase 2: C Interface Implementation

### Step 2.1: Create Header File

```c
// src/executorch_ffi.h

#ifndef EXECUTORCH_FFI_H
#define EXECUTORCH_FFI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Export Macros
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #ifdef EXECUTORCH_FFI_EXPORTS
        #define ET_API __declspec(dllexport)
    #else
        #define ET_API __declspec(dllimport)
    #endif
#else
    #define ET_API __attribute__((visibility("default")))
#endif

// =============================================================================
// Error Handling
// =============================================================================

/// Error status returned by all fallible functions
typedef struct ETStatus {
    int32_t code;           ///< 0 = success, non-zero = error
    char* message;          ///< Error message (heap allocated, caller frees)
    char* location;         ///< Source location "file:line:func"
} ETStatus;

/// Error codes
typedef enum ETErrorCode {
    ET_OK = 0,
    ET_ERROR_INVALID_ARGUMENT = 1,
    ET_ERROR_OUT_OF_MEMORY = 2,
    ET_ERROR_MODEL_LOAD_FAILED = 3,
    ET_ERROR_INFERENCE_FAILED = 4,
    ET_ERROR_INVALID_STATE = 5,
    ET_ERROR_UNSUPPORTED = 6,
    ET_ERROR_IO = 7,
    ET_ERROR_INTERNAL = 99,
} ETErrorCode;

/// Free status message
ET_API void et_status_free(ETStatus* status);

/// Check if status indicates success
ET_API bool et_status_ok(const ETStatus* status);

// =============================================================================
// Data Types
// =============================================================================

/// Tensor data types
typedef enum ETDType {
    ET_DTYPE_FLOAT32 = 0,
    ET_DTYPE_FLOAT64 = 1,
    ET_DTYPE_FLOAT16 = 2,
    ET_DTYPE_BFLOAT16 = 3,
    ET_DTYPE_INT64 = 4,
    ET_DTYPE_INT32 = 5,
    ET_DTYPE_INT16 = 6,
    ET_DTYPE_INT8 = 7,
    ET_DTYPE_UINT8 = 8,
    ET_DTYPE_BOOL = 9,
} ETDType;

/// Get size in bytes for a dtype
ET_API size_t et_dtype_size(ETDType dtype);

/// Get dtype name as string
ET_API const char* et_dtype_name(ETDType dtype);

// =============================================================================
// Tensor API
// =============================================================================

/// Opaque tensor handle
typedef struct ETTensor ETTensor;

/// Create tensor from data (copies data)
/// @param data Pointer to tensor data
/// @param data_size Size of data in bytes
/// @param shape Array of dimension sizes
/// @param rank Number of dimensions
/// @param dtype Data type
/// @param out_tensor Output tensor pointer
/// @return Status (caller must free)
ET_API ETStatus* et_tensor_create(
    const void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out_tensor
);

/// Create tensor without copying data (zero-copy)
/// WARNING: data must remain valid for lifetime of tensor
ET_API ETStatus* et_tensor_create_view(
    void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out_tensor
);

/// Create empty tensor with given shape
ET_API ETStatus* et_tensor_empty(
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out_tensor
);

/// Get tensor data type
ET_API ETDType et_tensor_dtype(const ETTensor* tensor);

/// Get tensor rank (number of dimensions)
ET_API int32_t et_tensor_rank(const ETTensor* tensor);

/// Get tensor shape (returns internal pointer, do not free)
ET_API const int64_t* et_tensor_shape(const ETTensor* tensor);

/// Get tensor data size in bytes
ET_API size_t et_tensor_data_size(const ETTensor* tensor);

/// Get tensor data pointer (returns internal pointer)
ET_API const void* et_tensor_data(const ETTensor* tensor);

/// Get mutable tensor data pointer
ET_API void* et_tensor_data_mut(ETTensor* tensor);

/// Get number of elements in tensor
ET_API int64_t et_tensor_numel(const ETTensor* tensor);

/// Clone tensor (deep copy)
ET_API ETStatus* et_tensor_clone(const ETTensor* tensor, ETTensor** out_tensor);

/// Free tensor
ET_API void et_tensor_free(ETTensor* tensor);

// =============================================================================
// Module (Model) API
// =============================================================================

/// Opaque module handle
typedef struct ETModule ETModule;

/// Load model from memory buffer
/// @param data Model data (.pte file contents)
/// @param size Size of model data in bytes
/// @param out_module Output module pointer
/// @return Status (caller must free)
ET_API ETStatus* et_module_load(
    const uint8_t* data,
    size_t size,
    ETModule** out_module
);

/// Load model from file path
ET_API ETStatus* et_module_load_file(
    const char* path,
    ETModule** out_module
);

/// Check if module is loaded and valid
ET_API bool et_module_is_loaded(const ETModule* module);

/// Get method names (returns array of strings, caller frees with et_string_array_free)
ET_API ETStatus* et_module_method_names(
    const ETModule* module,
    char*** out_names,
    int32_t* out_count
);

/// Get number of inputs for a method
ET_API int32_t et_module_input_count(const ETModule* module, const char* method);

/// Get number of outputs for a method
ET_API int32_t et_module_output_count(const ETModule* module, const char* method);

/// Get input tensor info
/// @param module Module handle
/// @param method Method name (use "forward" for default)
/// @param index Input index
/// @param out_shape Output shape array (caller frees with et_shape_free)
/// @param out_rank Output rank
/// @param out_dtype Output dtype
ET_API ETStatus* et_module_input_info(
    const ETModule* module,
    const char* method,
    int32_t index,
    int64_t** out_shape,
    int32_t* out_rank,
    ETDType* out_dtype
);

/// Get output tensor info
ET_API ETStatus* et_module_output_info(
    const ETModule* module,
    const char* method,
    int32_t index,
    int64_t** out_shape,
    int32_t* out_rank,
    ETDType* out_dtype
);

/// Run forward pass (default method)
/// @param module Module handle
/// @param inputs Array of input tensors
/// @param input_count Number of inputs
/// @param out_outputs Output array of tensors (caller frees with et_tensor_array_free)
/// @param out_output_count Number of outputs
ET_API ETStatus* et_module_forward(
    ETModule* module,
    ETTensor** inputs,
    int32_t input_count,
    ETTensor*** out_outputs,
    int32_t* out_output_count
);

/// Run named method
ET_API ETStatus* et_module_execute(
    ETModule* module,
    const char* method,
    ETTensor** inputs,
    int32_t input_count,
    ETTensor*** out_outputs,
    int32_t* out_output_count
);

/// Free module
ET_API void et_module_free(ETModule* module);

// =============================================================================
// Memory Management Helpers
// =============================================================================

/// Free tensor array (frees array and all tensors in it)
ET_API void et_tensor_array_free(ETTensor** tensors, int32_t count);

/// Free shape array
ET_API void et_shape_free(int64_t* shape);

/// Free string
ET_API void et_string_free(char* str);

/// Free string array
ET_API void et_string_array_free(char** strings, int32_t count);

// =============================================================================
// Backend API
// =============================================================================

/// Backend identifiers
typedef enum ETBackend {
    ET_BACKEND_XNNPACK = 0,
    ET_BACKEND_COREML = 1,
    ET_BACKEND_MPS = 2,
    ET_BACKEND_VULKAN = 3,
    ET_BACKEND_QNN = 4,
} ETBackend;

/// Check if backend is available (compiled in)
ET_API bool et_backend_available(ETBackend backend);

/// Get list of available backends
ET_API ETStatus* et_backend_list(
    ETBackend** out_backends,
    int32_t* out_count
);

/// Free backend list
ET_API void et_backend_list_free(ETBackend* backends);

/// Get backend name
ET_API const char* et_backend_name(ETBackend backend);

// =============================================================================
// Library Info
// =============================================================================

/// Get library version string
ET_API const char* et_version(void);

/// Get ExecuTorch version string
ET_API const char* et_executorch_version(void);

/// Initialize library (call once at startup)
ET_API ETStatus* et_init(void);

/// Shutdown library (call before exit)
ET_API void et_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // EXECUTORCH_FFI_H
```

### Step 2.2: Create Implementation File

```cpp
// src/executorch_ffi.cpp

#include "executorch_ffi.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <mutex>

using namespace executorch::extension;
using namespace executorch::runtime;

// =============================================================================
// Internal Structures
// =============================================================================

struct ETTensor {
    std::vector<int64_t> shape;
    ETDType dtype;
    std::vector<uint8_t> owned_data;
    void* view_data = nullptr;  // Non-null if this is a view
    bool is_view = false;

    size_t data_size() const {
        return owned_data.size();
    }

    const void* data() const {
        return is_view ? view_data : owned_data.data();
    }

    void* data_mut() {
        return is_view ? view_data : owned_data.data();
    }
};

struct ETModule {
    std::unique_ptr<Module> module;
    std::vector<std::string> method_names;
    bool loaded = false;
};

// =============================================================================
// Error Handling
// =============================================================================

static ETStatus* make_status(ETErrorCode code, const char* message, const char* location) {
    auto* status = new ETStatus();
    status->code = static_cast<int32_t>(code);
    status->message = message ? strdup(message) : nullptr;
    status->location = location ? strdup(location) : nullptr;
    return status;
}

#define ET_RETURN_OK() return make_status(ET_OK, nullptr, nullptr)

#define ET_RETURN_ERROR(code, msg) \
    return make_status(code, msg, __FILE__ ":" ET_STRINGIFY(__LINE__) ":" __func__)

#define ET_STRINGIFY(x) ET_STRINGIFY_IMPL(x)
#define ET_STRINGIFY_IMPL(x) #x

#define ET_CHECK_ARG(cond, msg) \
    do { \
        if (!(cond)) { \
            ET_RETURN_ERROR(ET_ERROR_INVALID_ARGUMENT, msg); \
        } \
    } while (0)

ET_API void et_status_free(ETStatus* status) {
    if (status) {
        free(status->message);
        free(status->location);
        delete status;
    }
}

ET_API bool et_status_ok(const ETStatus* status) {
    return status && status->code == ET_OK;
}

// =============================================================================
// Data Types
// =============================================================================

ET_API size_t et_dtype_size(ETDType dtype) {
    switch (dtype) {
        case ET_DTYPE_FLOAT32: return 4;
        case ET_DTYPE_FLOAT64: return 8;
        case ET_DTYPE_FLOAT16: return 2;
        case ET_DTYPE_BFLOAT16: return 2;
        case ET_DTYPE_INT64: return 8;
        case ET_DTYPE_INT32: return 4;
        case ET_DTYPE_INT16: return 2;
        case ET_DTYPE_INT8: return 1;
        case ET_DTYPE_UINT8: return 1;
        case ET_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

ET_API const char* et_dtype_name(ETDType dtype) {
    switch (dtype) {
        case ET_DTYPE_FLOAT32: return "float32";
        case ET_DTYPE_FLOAT64: return "float64";
        case ET_DTYPE_FLOAT16: return "float16";
        case ET_DTYPE_BFLOAT16: return "bfloat16";
        case ET_DTYPE_INT64: return "int64";
        case ET_DTYPE_INT32: return "int32";
        case ET_DTYPE_INT16: return "int16";
        case ET_DTYPE_INT8: return "int8";
        case ET_DTYPE_UINT8: return "uint8";
        case ET_DTYPE_BOOL: return "bool";
        default: return "unknown";
    }
}

// Convert ETDType to ExecuTorch ScalarType
static ScalarType dtype_to_scalar_type(ETDType dtype) {
    switch (dtype) {
        case ET_DTYPE_FLOAT32: return ScalarType::Float;
        case ET_DTYPE_FLOAT64: return ScalarType::Double;
        case ET_DTYPE_INT64: return ScalarType::Long;
        case ET_DTYPE_INT32: return ScalarType::Int;
        case ET_DTYPE_INT16: return ScalarType::Short;
        case ET_DTYPE_INT8: return ScalarType::Char;
        case ET_DTYPE_UINT8: return ScalarType::Byte;
        case ET_DTYPE_BOOL: return ScalarType::Bool;
        default: return ScalarType::Float;
    }
}

// Convert ExecuTorch ScalarType to ETDType
static ETDType scalar_type_to_dtype(ScalarType st) {
    switch (st) {
        case ScalarType::Float: return ET_DTYPE_FLOAT32;
        case ScalarType::Double: return ET_DTYPE_FLOAT64;
        case ScalarType::Long: return ET_DTYPE_INT64;
        case ScalarType::Int: return ET_DTYPE_INT32;
        case ScalarType::Short: return ET_DTYPE_INT16;
        case ScalarType::Char: return ET_DTYPE_INT8;
        case ScalarType::Byte: return ET_DTYPE_UINT8;
        case ScalarType::Bool: return ET_DTYPE_BOOL;
        default: return ET_DTYPE_FLOAT32;
    }
}

// =============================================================================
// Tensor API
// =============================================================================

ET_API ETStatus* et_tensor_create(
    const void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out_tensor
) {
    ET_CHECK_ARG(data != nullptr, "data is null");
    ET_CHECK_ARG(shape != nullptr, "shape is null");
    ET_CHECK_ARG(rank > 0, "rank must be positive");
    ET_CHECK_ARG(out_tensor != nullptr, "out_tensor is null");

    try {
        auto* tensor = new ETTensor();
        tensor->shape.assign(shape, shape + rank);
        tensor->dtype = dtype;
        tensor->owned_data.resize(data_size);
        std::memcpy(tensor->owned_data.data(), data, data_size);
        tensor->is_view = false;

        *out_tensor = tensor;
        ET_RETURN_OK();
    } catch (const std::exception& e) {
        ET_RETURN_ERROR(ET_ERROR_OUT_OF_MEMORY, e.what());
    }
}

ET_API ETStatus* et_tensor_create_view(
    void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    ETDType dtype,
    ETTensor** out_tensor
) {
    ET_CHECK_ARG(data != nullptr, "data is null");
    ET_CHECK_ARG(shape != nullptr, "shape is null");
    ET_CHECK_ARG(rank > 0, "rank must be positive");
    ET_CHECK_ARG(out_tensor != nullptr, "out_tensor is null");

    try {
        auto* tensor = new ETTensor();
        tensor->shape.assign(shape, shape + rank);
        tensor->dtype = dtype;
        tensor->view_data = data;
        tensor->is_view = true;
        // Store size for later reference
        tensor->owned_data.resize(0);  // Mark as view by empty owned_data

        *out_tensor = tensor;
        ET_RETURN_OK();
    } catch (const std::exception& e) {
        ET_RETURN_ERROR(ET_ERROR_OUT_OF_MEMORY, e.what());
    }
}

ET_API ETDType et_tensor_dtype(const ETTensor* tensor) {
    return tensor ? tensor->dtype : ET_DTYPE_FLOAT32;
}

ET_API int32_t et_tensor_rank(const ETTensor* tensor) {
    return tensor ? static_cast<int32_t>(tensor->shape.size()) : 0;
}

ET_API const int64_t* et_tensor_shape(const ETTensor* tensor) {
    return tensor ? tensor->shape.data() : nullptr;
}

ET_API size_t et_tensor_data_size(const ETTensor* tensor) {
    if (!tensor) return 0;

    int64_t numel = 1;
    for (auto dim : tensor->shape) {
        numel *= dim;
    }
    return static_cast<size_t>(numel) * et_dtype_size(tensor->dtype);
}

ET_API const void* et_tensor_data(const ETTensor* tensor) {
    return tensor ? tensor->data() : nullptr;
}

ET_API void* et_tensor_data_mut(ETTensor* tensor) {
    return tensor ? tensor->data_mut() : nullptr;
}

ET_API int64_t et_tensor_numel(const ETTensor* tensor) {
    if (!tensor) return 0;

    int64_t numel = 1;
    for (auto dim : tensor->shape) {
        numel *= dim;
    }
    return numel;
}

ET_API void et_tensor_free(ETTensor* tensor) {
    delete tensor;
}

// =============================================================================
// Module API
// =============================================================================

ET_API ETStatus* et_module_load(
    const uint8_t* data,
    size_t size,
    ETModule** out_module
) {
    ET_CHECK_ARG(data != nullptr, "data is null");
    ET_CHECK_ARG(size > 0, "size must be positive");
    ET_CHECK_ARG(out_module != nullptr, "out_module is null");

    try {
        auto* wrapper = new ETModule();

        // Create module from buffer
        // Note: Module may need the data to persist, so we might need to copy
        wrapper->module = std::make_unique<Module>(
            data,
            size,
            Module::LoadMode::MmapUseMlock
        );

        if (!wrapper->module->is_loaded()) {
            delete wrapper;
            ET_RETURN_ERROR(ET_ERROR_MODEL_LOAD_FAILED, "Failed to load model from buffer");
        }

        wrapper->loaded = true;

        // Cache method names
        auto methods = wrapper->module->method_names();
        if (methods.ok()) {
            for (const auto& name : methods.get()) {
                wrapper->method_names.push_back(std::string(name));
            }
        }

        *out_module = wrapper;
        ET_RETURN_OK();
    } catch (const std::exception& e) {
        ET_RETURN_ERROR(ET_ERROR_MODEL_LOAD_FAILED, e.what());
    }
}

ET_API ETStatus* et_module_load_file(
    const char* path,
    ETModule** out_module
) {
    ET_CHECK_ARG(path != nullptr, "path is null");
    ET_CHECK_ARG(out_module != nullptr, "out_module is null");

    try {
        auto* wrapper = new ETModule();

        wrapper->module = std::make_unique<Module>(
            path,
            Module::LoadMode::MmapUseMlock
        );

        if (!wrapper->module->is_loaded()) {
            delete wrapper;
            ET_RETURN_ERROR(ET_ERROR_MODEL_LOAD_FAILED, "Failed to load model from file");
        }

        wrapper->loaded = true;

        // Cache method names
        auto methods = wrapper->module->method_names();
        if (methods.ok()) {
            for (const auto& name : methods.get()) {
                wrapper->method_names.push_back(std::string(name));
            }
        }

        *out_module = wrapper;
        ET_RETURN_OK();
    } catch (const std::exception& e) {
        ET_RETURN_ERROR(ET_ERROR_MODEL_LOAD_FAILED, e.what());
    }
}

ET_API bool et_module_is_loaded(const ETModule* module) {
    return module && module->loaded && module->module && module->module->is_loaded();
}

ET_API int32_t et_module_input_count(const ETModule* module, const char* method) {
    if (!module || !module->module) return 0;

    const char* m = method ? method : "forward";
    auto meta = module->module->method_meta(m);
    if (!meta.ok()) return 0;

    return static_cast<int32_t>(meta->num_inputs());
}

ET_API int32_t et_module_output_count(const ETModule* module, const char* method) {
    if (!module || !module->module) return 0;

    const char* m = method ? method : "forward";
    auto meta = module->module->method_meta(m);
    if (!meta.ok()) return 0;

    return static_cast<int32_t>(meta->num_outputs());
}

ET_API ETStatus* et_module_forward(
    ETModule* module,
    ETTensor** inputs,
    int32_t input_count,
    ETTensor*** out_outputs,
    int32_t* out_output_count
) {
    return et_module_execute(module, "forward", inputs, input_count, out_outputs, out_output_count);
}

ET_API ETStatus* et_module_execute(
    ETModule* module,
    const char* method,
    ETTensor** inputs,
    int32_t input_count,
    ETTensor*** out_outputs,
    int32_t* out_output_count
) {
    ET_CHECK_ARG(module != nullptr, "module is null");
    ET_CHECK_ARG(module->loaded, "module not loaded");
    ET_CHECK_ARG(inputs != nullptr || input_count == 0, "inputs is null");
    ET_CHECK_ARG(out_outputs != nullptr, "out_outputs is null");
    ET_CHECK_ARG(out_output_count != nullptr, "out_output_count is null");

    try {
        const char* m = method ? method : "forward";

        // Convert inputs to EValue vector
        std::vector<EValue> eval_inputs;
        eval_inputs.reserve(input_count);

        for (int32_t i = 0; i < input_count; i++) {
            ETTensor* t = inputs[i];
            if (!t) {
                ET_RETURN_ERROR(ET_ERROR_INVALID_ARGUMENT, "null input tensor");
            }

            // Create ExecuTorch tensor from our wrapper
            auto et_tensor = from_blob(
                const_cast<void*>(t->data()),
                {t->shape.begin(), t->shape.end()},
                dtype_to_scalar_type(t->dtype)
            );

            eval_inputs.push_back(EValue(std::move(et_tensor)));
        }

        // Execute
        auto result = module->module->execute(m, eval_inputs);

        if (!result.ok()) {
            ET_RETURN_ERROR(ET_ERROR_INFERENCE_FAILED, "Forward pass failed");
        }

        // Convert outputs
        auto& outputs = result.get();
        *out_output_count = static_cast<int32_t>(outputs.size());
        *out_outputs = new ETTensor*[outputs.size()];

        for (size_t i = 0; i < outputs.size(); i++) {
            if (!outputs[i].isTensor()) {
                // Handle non-tensor outputs
                (*out_outputs)[i] = nullptr;
                continue;
            }

            auto& tensor = outputs[i].toTensor();
            auto* out_t = new ETTensor();

            // Copy shape
            out_t->shape.clear();
            for (size_t d = 0; d < tensor.dim(); d++) {
                out_t->shape.push_back(tensor.size(d));
            }

            // Set dtype
            out_t->dtype = scalar_type_to_dtype(tensor.scalar_type());

            // Copy data
            size_t data_size = tensor.nbytes();
            out_t->owned_data.resize(data_size);
            std::memcpy(out_t->owned_data.data(), tensor.const_data_ptr(), data_size);
            out_t->is_view = false;

            (*out_outputs)[i] = out_t;
        }

        ET_RETURN_OK();
    } catch (const std::exception& e) {
        ET_RETURN_ERROR(ET_ERROR_INFERENCE_FAILED, e.what());
    }
}

ET_API void et_module_free(ETModule* module) {
    delete module;
}

// =============================================================================
// Memory Management
// =============================================================================

ET_API void et_tensor_array_free(ETTensor** tensors, int32_t count) {
    if (tensors) {
        for (int32_t i = 0; i < count; i++) {
            delete tensors[i];
        }
        delete[] tensors;
    }
}

ET_API void et_shape_free(int64_t* shape) {
    delete[] shape;
}

ET_API void et_string_free(char* str) {
    free(str);
}

ET_API void et_string_array_free(char** strings, int32_t count) {
    if (strings) {
        for (int32_t i = 0; i < count; i++) {
            free(strings[i]);
        }
        delete[] strings;
    }
}

// =============================================================================
// Backend API
// =============================================================================

ET_API bool et_backend_available(ETBackend backend) {
    switch (backend) {
        case ET_BACKEND_XNNPACK:
#ifdef ET_USE_XNNPACK
            return true;
#else
            return false;
#endif
        case ET_BACKEND_COREML:
#ifdef ET_USE_COREML
            return true;
#else
            return false;
#endif
        case ET_BACKEND_MPS:
#ifdef ET_USE_MPS
            return true;
#else
            return false;
#endif
        case ET_BACKEND_VULKAN:
#ifdef ET_USE_VULKAN
            return true;
#else
            return false;
#endif
        case ET_BACKEND_QNN:
#ifdef ET_USE_QNN
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

ET_API const char* et_backend_name(ETBackend backend) {
    switch (backend) {
        case ET_BACKEND_XNNPACK: return "XNNPACK";
        case ET_BACKEND_COREML: return "CoreML";
        case ET_BACKEND_MPS: return "MPS";
        case ET_BACKEND_VULKAN: return "Vulkan";
        case ET_BACKEND_QNN: return "QNN";
        default: return "Unknown";
    }
}

// =============================================================================
// Library Info
// =============================================================================

ET_API const char* et_version(void) {
    return "2.0.0";
}

ET_API const char* et_executorch_version(void) {
    return "0.4.0";  // Update to match linked version
}

static std::once_flag init_flag;
static bool initialized = false;

ET_API ETStatus* et_init(void) {
    std::call_once(init_flag, []() {
        // Initialize ExecuTorch runtime
        // Any one-time setup goes here
        initialized = true;
    });

    ET_RETURN_OK();
}

ET_API void et_shutdown(void) {
    // Cleanup if needed
    initialized = false;
}
```

### Step 2.3: Create CMakeLists.txt

```cmake
# src/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(executorch_flutter VERSION 2.0.0 LANGUAGES C CXX)

# =============================================================================
# Options
# =============================================================================

option(ET_BUILD_XNNPACK "Build with XNNPACK backend" ON)
option(ET_BUILD_COREML "Build with CoreML backend" OFF)
option(ET_BUILD_MPS "Build with MPS backend" OFF)
option(ET_BUILD_VULKAN "Build with Vulkan backend" OFF)
option(ET_BUILD_QNN "Build with QNN backend" OFF)

set(EXECUTORCH_ROOT "" CACHE PATH "Path to ExecuTorch source")

# =============================================================================
# Compiler Settings
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Visibility
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# =============================================================================
# Platform-Specific Configuration
# =============================================================================

if(APPLE)
    if(IOS)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "iOS deployment target")
    else()
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "macOS deployment target")
    endif()

    # Disable bitcode (deprecated)
    set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")
endif()

if(ANDROID)
    set(CMAKE_ANDROID_STL c++_shared)
endif()

if(WIN32)
    add_definitions(-DEXECUTORCH_FFI_EXPORTS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif()

if(UNIX AND NOT APPLE)
    # Linux: Set RPATH for portable binaries
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# =============================================================================
# Find ExecuTorch
# =============================================================================

if(EXECUTORCH_ROOT)
    # Build from source
    add_subdirectory(${EXECUTORCH_ROOT} executorch_build EXCLUDE_FROM_ALL)
else()
    # Try to find pre-built
    find_package(executorch QUIET)
    if(NOT executorch_FOUND)
        message(FATAL_ERROR "ExecuTorch not found. Set EXECUTORCH_ROOT or install executorch package.")
    endif()
endif()

# =============================================================================
# Library Target
# =============================================================================

add_library(executorch_flutter SHARED
    executorch_ffi.cpp
)

target_include_directories(executorch_flutter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(executorch_flutter PRIVATE
    EXECUTORCH_FFI_EXPORTS
)

# Link ExecuTorch
target_link_libraries(executorch_flutter PRIVATE
    executorch
    extension_module
    extension_tensor
)

# =============================================================================
# Backend Configuration
# =============================================================================

if(ET_BUILD_XNNPACK)
    target_compile_definitions(executorch_flutter PRIVATE ET_USE_XNNPACK)
    target_link_libraries(executorch_flutter PRIVATE xnnpack_backend)
    message(STATUS "XNNPACK backend: ON")
endif()

if(ET_BUILD_COREML)
    if(APPLE)
        target_compile_definitions(executorch_flutter PRIVATE ET_USE_COREML)
        target_link_libraries(executorch_flutter PRIVATE coremldelegate)
        find_library(COREML_FRAMEWORK CoreML)
        target_link_libraries(executorch_flutter PRIVATE ${COREML_FRAMEWORK})
        message(STATUS "CoreML backend: ON")
    else()
        message(WARNING "CoreML backend requested but not on Apple platform")
    endif()
endif()

if(ET_BUILD_MPS)
    if(APPLE AND NOT IOS)
        target_compile_definitions(executorch_flutter PRIVATE ET_USE_MPS)
        target_link_libraries(executorch_flutter PRIVATE mpsdelegate)
        find_library(METAL_FRAMEWORK Metal)
        find_library(MPS_FRAMEWORK MetalPerformanceShaders)
        target_link_libraries(executorch_flutter PRIVATE
            ${METAL_FRAMEWORK}
            ${MPS_FRAMEWORK}
        )
        message(STATUS "MPS backend: ON")
    else()
        message(WARNING "MPS backend requested but not on macOS")
    endif()
endif()

if(ET_BUILD_VULKAN)
    if(NOT APPLE)
        find_package(Vulkan REQUIRED)
        target_compile_definitions(executorch_flutter PRIVATE ET_USE_VULKAN)
        target_link_libraries(executorch_flutter PRIVATE
            vulkan_backend
            Vulkan::Vulkan
        )
        message(STATUS "Vulkan backend: ON")
    else()
        message(WARNING "Vulkan backend requested but on Apple platform")
    endif()
endif()

if(ET_BUILD_QNN)
    if(ANDROID)
        target_compile_definitions(executorch_flutter PRIVATE ET_USE_QNN)
        target_link_libraries(executorch_flutter PRIVATE qnn_backend)
        message(STATUS "QNN backend: ON")
    else()
        message(WARNING "QNN backend requested but not on Android")
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS executorch_flutter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES executorch_ffi.h
    DESTINATION include
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "executorch_flutter configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Backends:")
message(STATUS "    XNNPACK: ${ET_BUILD_XNNPACK}")
message(STATUS "    CoreML:  ${ET_BUILD_COREML}")
message(STATUS "    MPS:     ${ET_BUILD_MPS}")
message(STATUS "    Vulkan:  ${ET_BUILD_VULKAN}")
message(STATUS "    QNN:     ${ET_BUILD_QNN}")
message(STATUS "")
```

---

## Phase 3: FFI Bindings Generation

### Step 3.1: Create ffigen Configuration

```yaml
# ffigen.yaml

name: ExecutorchFFI
description: FFI bindings for ExecuTorch Flutter
output:
  bindings: lib/src/generated/executorch_ffi.g.dart
  symbol-file:
    output: 'lib/src/generated/executorch_symbols.yaml'
    import-path: 'package:executorch_flutter/src/generated/executorch_ffi.g.dart'

# Link to native asset
ffi-native:
  asset-id: 'package:executorch_flutter/executorch_flutter.dart'

headers:
  entry-points:
    - src/executorch_ffi.h
  include-directives:
    - src/executorch_ffi.h

# Mark simple functions as leaf for better performance
functions:
  leaf:
    include:
      - "et_status_ok"
      - "et_dtype_size"
      - "et_dtype_name"
      - "et_tensor_dtype"
      - "et_tensor_rank"
      - "et_tensor_shape"
      - "et_tensor_data_size"
      - "et_tensor_data"
      - "et_tensor_numel"
      - "et_module_is_loaded"
      - "et_module_input_count"
      - "et_module_output_count"
      - "et_backend_available"
      - "et_backend_name"
      - "et_version"
      - "et_executorch_version"

  # Include symbol addresses for finalizers
  symbol-address:
    include:
      - "et_status_free"
      - "et_tensor_free"
      - "et_module_free"
      - "et_tensor_array_free"
      - "et_shape_free"
      - "et_string_free"
      - "et_string_array_free"
      - "et_backend_list_free"

# Enum handling
enums:
  rename:
    'ETErrorCode':
      'ET_(.*)': '$1'
    'ETDType':
      'ET_DTYPE_(.*)': '$1'
    'ETBackend':
      'ET_BACKEND_(.*)': '$1'

# Type definitions
type-map:
  'size_t': 'Size'

# Comments
comments:
  style: doxygen
  length: full

# Output configuration
preamble: |
  // AUTO GENERATED FILE - DO NOT EDIT
  // Generated by ffigen from executorch_ffi.h
  //
  // Run `dart run ffigen` to regenerate.

sort: true
```

### Step 3.2: Generate Bindings

```bash
# Install ffigen globally (if not already)
dart pub global activate ffigen

# Generate bindings
dart run ffigen --config ffigen.yaml

# Verify generated file
ls -la lib/src/generated/executorch_ffi.g.dart
```

### Step 3.3: Verify Generated Output

The generated file should look similar to:

```dart
// lib/src/generated/executorch_ffi.g.dart
// AUTO GENERATED FILE - DO NOT EDIT

@ffi.DefaultAsset('package:executorch_flutter/executorch_flutter.dart')
library;

import 'dart:ffi' as ffi;

/// Error codes
abstract class ETErrorCode {
  static const int OK = 0;
  static const int INVALID_ARGUMENT = 1;
  static const int OUT_OF_MEMORY = 2;
  static const int MODEL_LOAD_FAILED = 3;
  static const int INFERENCE_FAILED = 4;
  static const int INVALID_STATE = 5;
  static const int UNSUPPORTED = 6;
  static const int IO = 7;
  static const int INTERNAL = 99;
}

/// Tensor data types
abstract class ETDType {
  static const int FLOAT32 = 0;
  static const int FLOAT64 = 1;
  // ... etc
}

/// Error status structure
final class ETStatus extends ffi.Struct {
  @ffi.Int32()
  external int code;

  external ffi.Pointer<ffi.Char> message;
  external ffi.Pointer<ffi.Char> location;
}

/// Opaque tensor handle
final class ETTensor extends ffi.Opaque {}

/// Opaque module handle
final class ETModule extends ffi.Opaque {}

// Function bindings
@ffi.Native<ffi.Void Function(ffi.Pointer<ETStatus>)>()
external void et_status_free(ffi.Pointer<ETStatus> status);

@ffi.Native<ffi.Bool Function(ffi.Pointer<ETStatus>)>(isLeaf: true)
external bool et_status_ok(ffi.Pointer<ETStatus> status);

@ffi.Native<ffi.Pointer<ETStatus> Function(
  ffi.Pointer<ffi.Uint8>,
  ffi.Size,
  ffi.Pointer<ffi.Pointer<ETModule>>
)>()
external ffi.Pointer<ETStatus> et_module_load(
  ffi.Pointer<ffi.Uint8> data,
  int size,
  ffi.Pointer<ffi.Pointer<ETModule>> out_module,
);

// ... more bindings ...

/// Symbol addresses for finalizers
abstract final class addresses {
  static final ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ETModule>)>>
      et_module_free = ffi.Native.addressOf<
          ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ETModule>)>>(
              _et_module_free_ptr);

  // ... more addresses ...
}
```

---

## Phase 4: Native Assets Build System

### Step 4.1: Create Build Hook

```dart
// hook/build.dart

import 'package:executorch_flutter/src/build/run_build.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    await runBuild(input, output);
  });
}
```

### Step 4.2: Create Build Implementation

```dart
// lib/src/build/run_build.dart

import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;

final _logger = Logger('executorch_build');

/// Default backends per platform
const Map<OS, Set<String>> _defaultBackends = {
  OS.android: {'xnnpack'},
  OS.iOS: {'xnnpack', 'coreml'},
  OS.macOS: {'xnnpack', 'coreml', 'mps'},
  OS.linux: {'xnnpack'},
  OS.windows: {'xnnpack'},
};

/// Backend availability per platform
const Map<String, Set<OS>> _backendPlatforms = {
  'xnnpack': {OS.android, OS.iOS, OS.macOS, OS.linux, OS.windows},
  'coreml': {OS.iOS, OS.macOS},
  'mps': {OS.macOS},
  'vulkan': {OS.android, OS.linux, OS.windows},
  'qnn': {OS.android},
};

Future<void> runBuild(BuildInput input, BuildOutputBuilder output) async {
  final targetOS = input.config.code.targetOS;
  final userDefines = input.userDefines;

  // Parse configuration
  final debugMode = userDefines['debug'] as bool? ?? false;
  final requestedBackends = _parseBackends(userDefines, targetOS);

  _logger.info('Building executorch_flutter for $targetOS');
  _logger.info('Debug mode: $debugMode');
  _logger.info('Backends: $requestedBackends');

  // Find package source directory
  final packageRoot = await _findPackageRoot();
  final sourceDir = Directory(path.join(packageRoot, 'src'));

  if (!sourceDir.existsSync()) {
    throw StateError('Source directory not found: ${sourceDir.path}');
  }

  // Find ExecuTorch
  final executorchRoot = await _findExecutorchRoot();

  // Select CMake generator
  final generator = _selectGenerator(targetOS);

  // Build CMake defines
  final defines = _buildDefines(
    targetOS: targetOS,
    debugMode: debugMode,
    backends: requestedBackends,
    executorchRoot: executorchRoot,
    installPrefix: input.outputDirectory.resolve('install/').toFilePath(),
  );

  // Create CMake builder
  final builder = CMakeBuilder.create(
    logLevel: debugMode ? LogLevel.DEBUG : LogLevel.STATUS,
    name: input.packageName,
    sourceDir: sourceDir.uri,
    targets: ['install'],
    generator: generator,
    appleArgs: const AppleBuilderArgs(
      enableArc: false,
      enableBitcode: false,
      enableVisibility: true,
    ),
    defines: defines,
  );

  // Run build
  await builder.run(input: input, output: output, logger: _logger);

  // Register built library as code asset
  await output.findAndAddCodeAssets(
    input,
    outDir: input.outputDirectory.resolve('install/lib/'),
    names: {'executorch_flutter': 'executorch_flutter.dart'},
  );

  // Platform-specific post-processing
  await _postProcess(input, output, targetOS);
}

Set<String> _parseBackends(Map<String, Object?> userDefines, OS targetOS) {
  final requested = (userDefines['backends'] as List?)
      ?.cast<String>()
      .toSet() ?? _defaultBackends[targetOS] ?? {'xnnpack'};

  // Filter to backends available on target platform
  return requested.where((backend) {
    final platforms = _backendPlatforms[backend];
    return platforms?.contains(targetOS) ?? false;
  }).toSet();
}

Generator _selectGenerator(OS targetOS) {
  return switch (targetOS) {
    OS.linux => Generator.ninja,
    OS.macOS || OS.iOS => Generator.xcode,
    OS.windows => Generator.visualStudio,
    OS.android => Generator.ninja,
    _ => throw ArgumentError('Unsupported OS: $targetOS'),
  };
}

Map<String, String> _buildDefines({
  required OS targetOS,
  required bool debugMode,
  required Set<String> backends,
  required String executorchRoot,
  required String installPrefix,
}) {
  final defines = <String, String>{
    'CMAKE_BUILD_TYPE': debugMode ? 'Debug' : 'Release',
    'CMAKE_INSTALL_PREFIX': installPrefix,
    'EXECUTORCH_ROOT': executorchRoot,

    // Backend flags
    'ET_BUILD_XNNPACK': backends.contains('xnnpack') ? 'ON' : 'OFF',
    'ET_BUILD_COREML': backends.contains('coreml') ? 'ON' : 'OFF',
    'ET_BUILD_MPS': backends.contains('mps') ? 'ON' : 'OFF',
    'ET_BUILD_VULKAN': backends.contains('vulkan') ? 'ON' : 'OFF',
    'ET_BUILD_QNN': backends.contains('qnn') ? 'ON' : 'OFF',
  };

  // Platform-specific defines
  switch (targetOS) {
    case OS.macOS:
      defines['CMAKE_OSX_DEPLOYMENT_TARGET'] = '11.0';
      break;
    case OS.iOS:
      defines['CMAKE_OSX_DEPLOYMENT_TARGET'] = '13.0';
      break;
    case OS.android:
      defines['ANDROID_PLATFORM'] = 'android-26';
      defines['ANDROID_STL'] = 'c++_shared';
      break;
    default:
      break;
  }

  return defines;
}

Future<String> _findPackageRoot() async {
  // In a real implementation, use package_config to find the package
  final current = Directory.current;
  if (File(path.join(current.path, 'pubspec.yaml')).existsSync()) {
    return current.path;
  }
  throw StateError('Could not find package root');
}

Future<String> _findExecutorchRoot() async {
  // Check environment variable first
  final envRoot = Platform.environment['EXECUTORCH_ROOT'];
  if (envRoot != null && Directory(envRoot).existsSync()) {
    return envRoot;
  }

  // Check common locations
  final homeDir = Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'];
  if (homeDir != null) {
    final commonPaths = [
      path.join(homeDir, 'executorch'),
      path.join(homeDir, 'src', 'executorch'),
      path.join(homeDir, 'github', 'executorch'),
    ];

    for (final p in commonPaths) {
      if (Directory(p).existsSync()) {
        return p;
      }
    }
  }

  throw StateError(
    'ExecuTorch not found. Set EXECUTORCH_ROOT environment variable '
    'or clone to ~/executorch'
  );
}

Future<void> _postProcess(
  BuildInput input,
  BuildOutputBuilder output,
  OS targetOS,
) async {
  // Linux: Set RPATH for portable binaries
  if (targetOS == OS.linux) {
    final libDir = input.outputDirectory.resolve('install/lib/');
    final libs = Directory.fromUri(libDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.so'));

    for (final lib in libs) {
      await _setRPath(lib.uri, r'$ORIGIN');
    }
  }
}

Future<void> _setRPath(Uri libPath, String rpath) async {
  final result = await Process.run('patchelf', [
    '--set-rpath', rpath,
    libPath.toFilePath(),
  ]);

  if (result.exitCode != 0) {
    _logger.warning('Failed to set RPATH for ${libPath.toFilePath()}: ${result.stderr}');
  }
}
```

---

## Phase 5: Dart Wrapper Implementation

### Step 5.1: Create Base Types

```dart
// lib/src/ffi/types.dart

import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import '../generated/executorch_ffi.g.dart' as native;

/// Tensor data type enumeration
enum TensorType {
  float32(native.ETDType.FLOAT32, 4),
  float64(native.ETDType.FLOAT64, 8),
  float16(native.ETDType.FLOAT16, 2),
  bfloat16(native.ETDType.BFLOAT16, 2),
  int64(native.ETDType.INT64, 8),
  int32(native.ETDType.INT32, 4),
  int16(native.ETDType.INT16, 2),
  int8(native.ETDType.INT8, 1),
  uint8(native.ETDType.UINT8, 1),
  bool_(native.ETDType.BOOL, 1);

  const TensorType(this.nativeValue, this.elementSize);

  final int nativeValue;
  final int elementSize;

  static TensorType fromNative(int value) {
    return TensorType.values.firstWhere(
      (t) => t.nativeValue == value,
      orElse: () => TensorType.float32,
    );
  }
}

/// Tensor data structure for Dart
class TensorData {
  final List<int> shape;
  final TensorType dtype;
  final Uint8List data;
  final String? name;

  TensorData({
    required this.shape,
    required this.dtype,
    required this.data,
    this.name,
  });

  /// Number of elements in tensor
  int get numel => shape.fold(1, (a, b) => a * b);

  /// Data size in bytes
  int get dataSize => numel * dtype.elementSize;

  /// Get data as Float32List (for float32 tensors)
  Float32List get asFloat32List {
    if (dtype != TensorType.float32) {
      throw StateError('Tensor is not float32');
    }
    return data.buffer.asFloat32List();
  }

  /// Get data as Int32List (for int32 tensors)
  Int32List get asInt32List {
    if (dtype != TensorType.int32) {
      throw StateError('Tensor is not int32');
    }
    return data.buffer.asInt32List();
  }

  /// Create from Float32List
  factory TensorData.fromFloat32List(
    Float32List data,
    List<int> shape, {
    String? name,
  }) {
    return TensorData(
      shape: shape,
      dtype: TensorType.float32,
      data: data.buffer.asUint8List(),
      name: name,
    );
  }

  /// Create from Int32List
  factory TensorData.fromInt32List(
    Int32List data,
    List<int> shape, {
    String? name,
  }) {
    return TensorData(
      shape: shape,
      dtype: TensorType.int32,
      data: data.buffer.asUint8List(),
      name: name,
    );
  }
}

/// Available backends
enum Backend {
  xnnpack(native.ETBackend.XNNPACK, 'XNNPACK'),
  coreml(native.ETBackend.COREML, 'CoreML'),
  mps(native.ETBackend.MPS, 'MPS'),
  vulkan(native.ETBackend.VULKAN, 'Vulkan'),
  qnn(native.ETBackend.QNN, 'QNN');

  const Backend(this.nativeValue, this.displayName);

  final int nativeValue;
  final String displayName;

  /// Check if this backend is available
  bool get isAvailable => native.et_backend_available(nativeValue);

  /// Get all available backends
  static List<Backend> get available {
    return Backend.values.where((b) => b.isAvailable).toList();
  }
}
```

### Step 5.2: Create Memory Management

```dart
// lib/src/ffi/memory.dart

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../generated/executorch_ffi.g.dart' as native;

/// Native finalizer for automatic cleanup
typedef NativeFinalizerFunc<T extends ffi.NativeType> =
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(T)>>;

/// Create a NativeFinalizer from a function pointer
ffi.NativeFinalizer createFinalizer<T extends ffi.NativeType>(
  NativeFinalizerFunc<T> func,
) {
  return ffi.NativeFinalizer(func.cast<ffi.NativeFinalizerFunction>());
}

/// Check native status and throw if error
void checkStatus(ffi.Pointer<native.ETStatus> status) {
  if (status == ffi.nullptr) {
    throw ExecuTorchException('Null status returned');
  }

  final code = status.ref.code;
  if (code != 0) {
    final message = status.ref.message != ffi.nullptr
        ? status.ref.message.cast<Utf8>().toDartString()
        : 'Unknown error';
    final location = status.ref.location != ffi.nullptr
        ? status.ref.location.cast<Utf8>().toDartString()
        : '';

    native.et_status_free(status);

    throw ExecuTorchException(message, code: code, location: location);
  }

  native.et_status_free(status);
}

/// Run a function with automatic arena allocation
R runWithArena<R>(R Function(Arena arena) computation) {
  final arena = Arena();
  try {
    final result = computation(arena);
    if (result is Future) {
      return (result as Future).whenComplete(arena.releaseAll) as R;
    }
    return result;
  } finally {
    if (R is! Future) {
      arena.releaseAll();
    }
  }
}

/// ExecuTorch exception class
class ExecuTorchException implements Exception {
  final String message;
  final int code;
  final String? location;

  ExecuTorchException(this.message, {this.code = -1, this.location});

  @override
  String toString() {
    final loc = location != null ? ' at $location' : '';
    return 'ExecuTorchException($code): $message$loc';
  }
}
```

### Step 5.3: Create FFI Wrapper Classes

```dart
// lib/src/ffi/bindings.dart

import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import '../generated/executorch_ffi.g.dart' as native;
import 'memory.dart';
import 'types.dart';

/// Native tensor wrapper with automatic memory management
class NativeTensor {
  ffi.Pointer<native.ETTensor> _ptr;
  bool _disposed = false;

  static final _finalizer = createFinalizer<native.ETTensor>(
    native.addresses.et_tensor_free.cast(),
  );

  NativeTensor._(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Create tensor from TensorData
  factory NativeTensor.fromTensorData(TensorData data) {
    return runWithArena((arena) {
      // Allocate shape array
      final shapePtr = arena<ffi.Int64>(data.shape.length);
      for (var i = 0; i < data.shape.length; i++) {
        shapePtr[i] = data.shape[i];
      }

      // Allocate data
      final dataPtr = arena<ffi.Uint8>(data.data.length);
      dataPtr.asTypedList(data.data.length).setAll(0, data.data);

      // Create tensor
      final outPtr = arena<ffi.Pointer<native.ETTensor>>();
      final status = native.et_tensor_create(
        dataPtr.cast(),
        data.data.length,
        shapePtr,
        data.shape.length,
        data.dtype.nativeValue,
        outPtr,
      );

      checkStatus(status);
      return NativeTensor._(outPtr.value);
    });
  }

  /// Convert to TensorData
  TensorData toTensorData({String? name}) {
    _checkNotDisposed();

    final dtype = TensorType.fromNative(native.et_tensor_dtype(_ptr));
    final rank = native.et_tensor_rank(_ptr);
    final shapePtr = native.et_tensor_shape(_ptr);
    final dataSize = native.et_tensor_data_size(_ptr);
    final dataPtr = native.et_tensor_data(_ptr);

    // Copy shape
    final shape = List<int>.generate(rank, (i) => shapePtr[i]);

    // Copy data
    final data = Uint8List(dataSize);
    data.setAll(0, dataPtr.cast<ffi.Uint8>().asTypedList(dataSize));

    return TensorData(
      shape: shape,
      dtype: dtype,
      data: data,
      name: name,
    );
  }

  ffi.Pointer<native.ETTensor> get ptr {
    _checkNotDisposed();
    return _ptr;
  }

  void _checkNotDisposed() {
    if (_disposed) {
      throw StateError('Tensor has been disposed');
    }
  }

  void dispose() {
    if (!_disposed) {
      _finalizer.detach(this);
      native.et_tensor_free(_ptr);
      _disposed = true;
    }
  }
}

/// Native module wrapper with automatic memory management
class NativeModule {
  ffi.Pointer<native.ETModule> _ptr;
  bool _disposed = false;

  static final _finalizer = createFinalizer<native.ETModule>(
    native.addresses.et_module_free.cast(),
  );

  NativeModule._(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Load module from bytes
  factory NativeModule.load(Uint8List modelData) {
    return runWithArena((arena) {
      // Allocate model data
      final dataPtr = arena<ffi.Uint8>(modelData.length);
      dataPtr.asTypedList(modelData.length).setAll(0, modelData);

      // Load module
      final outPtr = arena<ffi.Pointer<native.ETModule>>();
      final status = native.et_module_load(
        dataPtr,
        modelData.length,
        outPtr,
      );

      checkStatus(status);
      return NativeModule._(outPtr.value);
    });
  }

  /// Load module from file path
  factory NativeModule.loadFile(String filePath) {
    return runWithArena((arena) {
      final pathPtr = filePath.toNativeUtf8(allocator: arena);

      final outPtr = arena<ffi.Pointer<native.ETModule>>();
      final status = native.et_module_load_file(
        pathPtr.cast(),
        outPtr,
      );

      checkStatus(status);
      return NativeModule._(outPtr.value);
    });
  }

  /// Check if module is loaded
  bool get isLoaded {
    _checkNotDisposed();
    return native.et_module_is_loaded(_ptr);
  }

  /// Get number of inputs
  int get inputCount {
    _checkNotDisposed();
    return native.et_module_input_count(_ptr, ffi.nullptr);
  }

  /// Get number of outputs
  int get outputCount {
    _checkNotDisposed();
    return native.et_module_output_count(_ptr, ffi.nullptr);
  }

  /// Run forward pass
  List<TensorData> forward(List<TensorData> inputs) {
    _checkNotDisposed();

    // Convert inputs to native tensors
    final nativeTensors = inputs.map(NativeTensor.fromTensorData).toList();

    try {
      return runWithArena((arena) {
        // Create input array
        final inputsPtr = arena<ffi.Pointer<native.ETTensor>>(nativeTensors.length);
        for (var i = 0; i < nativeTensors.length; i++) {
          inputsPtr[i] = nativeTensors[i].ptr;
        }

        // Allocate output pointers
        final outputsPtr = arena<ffi.Pointer<ffi.Pointer<native.ETTensor>>>();
        final outputCountPtr = arena<ffi.Int32>();

        // Run forward
        final status = native.et_module_forward(
          _ptr,
          inputsPtr,
          nativeTensors.length,
          outputsPtr,
          outputCountPtr,
        );

        checkStatus(status);

        // Convert outputs
        final outputCount = outputCountPtr.value;
        final outputs = <TensorData>[];

        for (var i = 0; i < outputCount; i++) {
          final tensor = NativeTensor._(outputsPtr.value[i]);
          outputs.add(tensor.toTensorData(name: 'output_$i'));
          tensor.dispose();
        }

        // Free output array (not individual tensors, we already disposed them)
        // native.et_tensor_array_free(outputsPtr.value, outputCount);

        return outputs;
      });
    } finally {
      // Dispose input tensors
      for (final tensor in nativeTensors) {
        tensor.dispose();
      }
    }
  }

  ffi.Pointer<native.ETModule> get ptr {
    _checkNotDisposed();
    return _ptr;
  }

  void _checkNotDisposed() {
    if (_disposed) {
      throw StateError('Module has been disposed');
    }
  }

  void dispose() {
    if (!_disposed) {
      _finalizer.detach(this);
      native.et_module_free(_ptr);
      _disposed = true;
    }
  }
}
```

### Step 5.4: Update Public API

```dart
// lib/src/executorch_model.dart

import 'dart:typed_data';
import 'ffi/bindings.dart';
import 'ffi/types.dart';

export 'ffi/types.dart' show TensorData, TensorType, Backend;
export 'ffi/memory.dart' show ExecuTorchException;

/// ExecuTorch model for on-device ML inference
class ExecuTorchModel {
  final NativeModule _module;
  final String modelId;
  bool _disposed = false;

  ExecuTorchModel._(this._module, this.modelId);

  /// Load model from bytes
  static Future<ExecuTorchModel> load(Uint8List modelData) async {
    final module = NativeModule.load(modelData);
    final modelId = 'model_${DateTime.now().millisecondsSinceEpoch}';
    return ExecuTorchModel._(module, modelId);
  }

  /// Load model from file path
  static Future<ExecuTorchModel> loadFile(String filePath) async {
    final module = NativeModule.loadFile(filePath);
    final modelId = 'model_${DateTime.now().millisecondsSinceEpoch}';
    return ExecuTorchModel._(module, modelId);
  }

  /// Check if model is loaded
  bool get isLoaded => !_disposed && _module.isLoaded;

  /// Get number of model inputs
  int get inputCount => _module.inputCount;

  /// Get number of model outputs
  int get outputCount => _module.outputCount;

  /// Run inference
  Future<List<TensorData>> forward(List<TensorData> inputs) async {
    if (_disposed) {
      throw StateError('Model has been disposed');
    }
    return _module.forward(inputs);
  }

  /// Dispose model and release resources
  Future<void> dispose() async {
    if (!_disposed) {
      _module.dispose();
      _disposed = true;
    }
  }
}
```

### Step 5.5: Update Library Export

```dart
// lib/executorch_flutter.dart

library executorch_flutter;

export 'src/executorch_model.dart';
```

---

## Phase 6: Platform-Specific Integration

### Step 6.1: Remove Old Platform Code

```bash
# After FFI migration is complete and tested, remove old code:

# Remove Pigeon definitions
rm -rf pigeons/

# Remove Android Kotlin implementation
rm -rf android/src/main/kotlin/com/zcreations/executorch_flutter/

# Remove iOS Swift implementation
rm -rf ios/Classes/

# Remove macOS Swift implementation
rm -rf macos/Classes/

# Remove darwin shared sources
rm -rf darwin/

# Keep android/ ios/ macos/ directories but they become minimal
```

### Step 6.2: Minimal Platform Configurations

For FFI plugins, platform directories still need minimal configuration:

```yaml
# android/build.gradle - minimal for FFI
group 'com.zcreations.executorch_flutter'
version '1.0'

buildscript {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

apply plugin: 'com.android.library'

android {
    compileSdkVersion 34

    defaultConfig {
        minSdkVersion 26
    }

    // Native library will be provided by native assets
}
```

---

## Phase 7: Testing & Validation

### Step 7.1: Unit Tests

```dart
// test/executorch_model_test.dart

import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

void main() {
  group('TensorData', () {
    test('creates from Float32List', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorData.fromFloat32List(data, [2, 2]);

      expect(tensor.shape, [2, 2]);
      expect(tensor.dtype, TensorType.float32);
      expect(tensor.numel, 4);
    });

    test('creates from Int32List', () {
      final data = Int32List.fromList([1, 2, 3, 4, 5, 6]);
      final tensor = TensorData.fromInt32List(data, [2, 3]);

      expect(tensor.shape, [2, 3]);
      expect(tensor.dtype, TensorType.int32);
      expect(tensor.numel, 6);
    });
  });

  group('Backend', () {
    test('lists available backends', () {
      final available = Backend.available;
      // At minimum, XNNPACK should be available
      expect(available, isNotEmpty);
    });
  });
}
```

### Step 7.2: Integration Tests

```dart
// integration_test/model_test.dart

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('ExecuTorchModel Integration', () {
    testWidgets('loads model from bytes', (tester) async {
      // Load test model
      final modelBytes = await rootBundle.load('assets/models/test_model.pte');

      // Create model
      final model = await ExecuTorchModel.load(
        modelBytes.buffer.asUint8List(),
      );

      expect(model.isLoaded, true);
      expect(model.inputCount, greaterThan(0));

      await model.dispose();
    });

    testWidgets('runs inference', (tester) async {
      final modelBytes = await rootBundle.load('assets/models/test_model.pte');
      final model = await ExecuTorchModel.load(
        modelBytes.buffer.asUint8List(),
      );

      // Create input tensor
      final input = TensorData.fromFloat32List(
        Float32List(224 * 224 * 3),
        [1, 3, 224, 224],
      );

      // Run inference
      final outputs = await model.forward([input]);

      expect(outputs, isNotEmpty);

      await model.dispose();
    });
  });
}
```

### Step 7.3: Run Tests

```bash
# Unit tests
flutter test

# Integration tests on each platform
cd example
flutter test integration_test/model_test.dart -d macos
flutter test integration_test/model_test.dart -d linux
flutter test integration_test/model_test.dart -d windows
flutter test integration_test/model_test.dart -d <android-device>
flutter test integration_test/model_test.dart -d <ios-device>
```

---

## Phase 8: Cleanup & Release

### Step 8.1: Update Documentation

- Update README.md with new platform support
- Update CHANGELOG.md with migration notes
- Update API documentation
- Remove Pigeon-related documentation

### Step 8.2: Update CI/CD

```yaml
# .github/workflows/build.yml

name: Build

on: [push, pull_request]

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
      - run: sudo apt-get install -y ninja-build patchelf
      - run: flutter pub get
      - run: flutter test
      - run: cd example && flutter build linux

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
      - run: flutter test
      - run: cd example && flutter build macos

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
      - run: flutter test
      - run: cd example && flutter build windows
```

### Step 8.3: Version & Release

```yaml
# pubspec.yaml
version: 2.0.0  # Major version for breaking change
```

```markdown
# CHANGELOG.md

## 2.0.0

### Breaking Changes
- Migrated from Pigeon/Method Channels to dart:ffi
- Removed platform-specific implementations (Kotlin, Swift)

### New Features
- Added Linux desktop support
- Added Windows desktop support
- Zero-copy tensor transfer (optional)
- Configurable backend selection via pubspec.yaml

### Performance
- Reduced inference latency by ~30% (no method channel overhead)
- Reduced memory usage (no serialization)

### Migration Guide
See docs/FFI_MIGRATION_GUIDE.md for detailed migration instructions.
```

---

## Troubleshooting

### Common Issues

#### 1. "Library not found" at runtime

```
Error: Invalid argument(s): Failed to load dynamic library 'libexecutorch_flutter.so'
```

**Solution:** Ensure native assets are properly built:
```bash
# Check that hook/build.dart exists and is correct
# Verify CMakeLists.txt is in src/
# Run flutter clean && flutter pub get
```

#### 2. CMake configuration errors

```
CMake Error: Could not find executorch
```

**Solution:** Set EXECUTORCH_ROOT environment variable:
```bash
export EXECUTORCH_ROOT=/path/to/executorch
```

#### 3. Symbol not found errors

```
Error: symbol not found: et_module_load
```

**Solution:** Regenerate FFI bindings:
```bash
dart run ffigen --config ffigen.yaml
```

#### 4. Finalizer crashes

**Solution:** Ensure you're not disposing objects that are still in use:
```dart
// Bad: tensor might be used internally
tensor.dispose();
model.forward([tensor]); // Crash!

// Good: let finalizer handle cleanup or dispose after use
final outputs = model.forward([tensor]);
tensor.dispose(); // Now safe
```

#### 5. Backend not available

```
ExecuTorchException: Backend not available
```

**Solution:** Check backend configuration in pubspec.yaml:
```yaml
hooks:
  user_defines:
    executorch_flutter:
      backends:
        - xnnpack  # Make sure desired backend is listed
```

---

## Summary Checklist

- [ ] Phase 1: Project setup (directories, pubspec.yaml, .gitignore)
- [ ] Phase 2: C interface (header, implementation, CMakeLists.txt)
- [ ] Phase 3: FFI bindings (ffigen.yaml, generate, verify)
- [ ] Phase 4: Native assets (hook/build.dart, run_build.dart)
- [ ] Phase 5: Dart wrappers (types, memory, bindings, public API)
- [ ] Phase 6: Platform cleanup (remove old code, minimal configs)
- [ ] Phase 7: Testing (unit tests, integration tests, all platforms)
- [ ] Phase 8: Release (docs, CI/CD, version bump, publish)

---

**Estimated Timeline:** 2-4 weeks depending on team size and ExecuTorch familiarity

**Risk Areas:**
- ExecuTorch C++ API changes between versions
- Platform-specific CMake configuration
- Native asset build system maturity

**Rollback Plan:** Keep Pigeon implementation in a branch until FFI is fully validated on all platforms.
