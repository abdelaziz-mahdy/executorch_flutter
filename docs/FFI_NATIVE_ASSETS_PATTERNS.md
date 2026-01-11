# FFI and Native Assets Patterns

This document outlines patterns for implementing cross-platform native bindings using Dart FFI and Flutter's native assets system. These patterns are derived from analyzing production-grade projects like opencv_dart and can be applied to executorch_flutter for unified platform support.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [C Interface Design](#c-interface-design)
3. [FFI Bindings with ffigen](#ffi-bindings-with-ffigen)
4. [Native Assets Build System](#native-assets-build-system)
5. [Memory Management](#memory-management)
6. [Backend Configuration](#backend-configuration)
7. [Platform-Specific Considerations](#platform-specific-considerations)
8. [Migration Strategy](#migration-strategy)

---

## Architecture Overview

### Current Architecture (Pigeon/Method Channels)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Dart API  │ ──▶ │    Pigeon    │ ──▶ │  Kotlin/Swift   │ ──▶ │ ExecuTorch   │
│             │     │ (generated)  │     │  (per platform) │     │   C++ API    │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
```

**Pros:**
- Type-safe generated code
- Platform-specific optimizations possible
- Familiar Flutter plugin pattern

**Cons:**
- Method channel serialization overhead
- Separate implementations per platform
- Limited to platforms with Pigeon support

### Target Architecture (FFI + Native Assets)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Dart API  │ ──▶ │   dart:ffi   │ ──▶ │   C Wrapper     │ ──▶ │ ExecuTorch   │
│             │     │ (generated)  │     │   (unified)     │     │   C++ API    │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
```

**Pros:**
- Direct native calls (no serialization)
- Single C implementation for all platforms
- Supports all FFI-capable platforms (Android, iOS, macOS, Linux, Windows)
- Zero-copy tensor transfer possible

**Cons:**
- Requires C wrapper layer
- More complex build system
- Manual memory management

---

## C Interface Design

### Design Principles

1. **Opaque Pointers**: Use opaque struct pointers to hide implementation details
2. **Error Handling**: Return status codes with error messages
3. **Memory Ownership**: Clear ownership semantics (caller vs callee allocated)
4. **Thread Safety**: Document thread-safety guarantees

### Status/Error Pattern

```c
// executorch_ffi.h

#ifndef EXECUTORCH_FFI_H
#define EXECUTORCH_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Export macro for shared library
#if defined(_WIN32) || defined(_WIN64)
    #define ET_EXPORT __declspec(dllexport)
#else
    #define ET_EXPORT __attribute__((visibility("default")))
#endif

// ============================================================================
// Error Handling
// ============================================================================

typedef struct {
    int code;           // 0 = success, non-zero = error
    const char* msg;    // Error message (caller must free with et_free_string)
    const char* file;   // Source file where error occurred
    const char* func;   // Function name
    int line;           // Line number
} ETStatus;

ET_EXPORT void et_status_free(ETStatus* status);
ET_EXPORT const char* et_status_message(ETStatus* status);

// ============================================================================
// Module (Model) API
// ============================================================================

// Opaque pointer to module
typedef struct ETModule ETModule;

// Load model from memory buffer
// Ownership: Caller owns modelData, callee copies if needed
ET_EXPORT ETStatus* et_module_load(
    const uint8_t* modelData,
    size_t modelSize,
    ETModule** outModule
);

// Load model from file path
ET_EXPORT ETStatus* et_module_load_file(
    const char* filePath,
    ETModule** outModule
);

// Get number of inputs/outputs
ET_EXPORT int et_module_input_count(ETModule* module);
ET_EXPORT int et_module_output_count(ETModule* module);

// Get input/output shapes (returns array, caller must free)
ET_EXPORT ETStatus* et_module_input_shape(
    ETModule* module,
    int index,
    int64_t** outShape,
    int* outRank
);

// Free module
ET_EXPORT void et_module_free(ETModule* module);

// ============================================================================
// Tensor API
// ============================================================================

typedef struct ETTensor ETTensor;

// Tensor data types
typedef enum {
    ET_DTYPE_FLOAT32 = 0,
    ET_DTYPE_FLOAT64 = 1,
    ET_DTYPE_INT32 = 2,
    ET_DTYPE_INT64 = 3,
    ET_DTYPE_INT8 = 4,
    ET_DTYPE_UINT8 = 5,
    ET_DTYPE_BOOL = 6,
} ETDType;

// Create tensor from data (copies data)
ET_EXPORT ETStatus* et_tensor_create(
    const void* data,
    size_t dataSize,
    const int64_t* shape,
    int rank,
    ETDType dtype,
    ETTensor** outTensor
);

// Create tensor with zero-copy (data must outlive tensor)
ET_EXPORT ETStatus* et_tensor_create_nocopy(
    void* data,
    size_t dataSize,
    const int64_t* shape,
    int rank,
    ETDType dtype,
    ETTensor** outTensor
);

// Get tensor properties
ET_EXPORT ETDType et_tensor_dtype(ETTensor* tensor);
ET_EXPORT int et_tensor_rank(ETTensor* tensor);
ET_EXPORT const int64_t* et_tensor_shape(ETTensor* tensor);
ET_EXPORT size_t et_tensor_data_size(ETTensor* tensor);
ET_EXPORT const void* et_tensor_data(ETTensor* tensor);

// Free tensor
ET_EXPORT void et_tensor_free(ETTensor* tensor);

// ============================================================================
// Inference API
// ============================================================================

// Run forward pass
// Ownership: Caller owns inputs, callee allocates outputs (caller must free)
ET_EXPORT ETStatus* et_module_forward(
    ETModule* module,
    ETTensor** inputs,
    int inputCount,
    ETTensor*** outOutputs,
    int* outOutputCount
);

// Free output tensor array (not individual tensors)
ET_EXPORT void et_tensor_array_free(ETTensor** tensors, int count);

// ============================================================================
// Utility
// ============================================================================

ET_EXPORT void et_free_string(char* str);
ET_EXPORT void et_free_shape(int64_t* shape);
ET_EXPORT const char* et_version(void);

#ifdef __cplusplus
}
#endif

#endif // EXECUTORCH_FFI_H
```

### C++ Implementation Pattern

```cpp
// executorch_ffi.cpp

#include "executorch_ffi.h"
#include <executorch/runtime/executor/program.h>
#include <executorch/extension/module/module.h>
#include <cstring>
#include <memory>

using namespace executorch::runtime;
using namespace executorch::extension;

// Internal module wrapper
struct ETModule {
    std::unique_ptr<Module> module;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
};

// Internal tensor wrapper
struct ETTensor {
    std::vector<int64_t> shape;
    ETDType dtype;
    std::vector<uint8_t> data;  // Owned data
    void* externalData;          // Non-owned data (for nocopy)
    bool ownsData;
};

// Helper: Create status
static ETStatus* make_status(int code, const char* msg,
                              const char* file, const char* func, int line) {
    auto* status = new ETStatus();
    status->code = code;
    status->msg = msg ? strdup(msg) : nullptr;
    status->file = file ? strdup(file) : nullptr;
    status->func = func ? strdup(func) : nullptr;
    status->line = line;
    return status;
}

#define ET_OK() make_status(0, nullptr, nullptr, nullptr, 0)
#define ET_ERROR(code, msg) make_status(code, msg, __FILE__, __func__, __LINE__)

// ============================================================================
// Implementation
// ============================================================================

ET_EXPORT void et_status_free(ETStatus* status) {
    if (status) {
        free((void*)status->msg);
        free((void*)status->file);
        free((void*)status->func);
        delete status;
    }
}

ET_EXPORT ETStatus* et_module_load(
    const uint8_t* modelData,
    size_t modelSize,
    ETModule** outModule
) {
    if (!modelData || !outModule) {
        return ET_ERROR(1, "Invalid arguments");
    }

    try {
        auto* wrapper = new ETModule();

        // Load module from buffer
        wrapper->module = std::make_unique<Module>(
            modelData, modelSize, Module::LoadMode::MmapUseMlock
        );

        if (!wrapper->module->is_loaded()) {
            delete wrapper;
            return ET_ERROR(2, "Failed to load model");
        }

        *outModule = wrapper;
        return ET_OK();
    } catch (const std::exception& e) {
        return ET_ERROR(3, e.what());
    }
}

ET_EXPORT ETStatus* et_module_forward(
    ETModule* module,
    ETTensor** inputs,
    int inputCount,
    ETTensor*** outOutputs,
    int* outOutputCount
) {
    if (!module || !inputs || !outOutputs || !outOutputCount) {
        return ET_ERROR(1, "Invalid arguments");
    }

    try {
        // Convert inputs to EValue
        std::vector<EValue> evalInputs;
        evalInputs.reserve(inputCount);

        for (int i = 0; i < inputCount; i++) {
            ETTensor* t = inputs[i];
            // Create ExecuTorch tensor from our wrapper
            // ... conversion logic
        }

        // Run inference
        auto result = module->module->forward(evalInputs);

        if (!result.ok()) {
            return ET_ERROR(4, "Forward pass failed");
        }

        // Convert outputs
        auto& outputs = result.get();
        *outOutputCount = outputs.size();
        *outOutputs = new ETTensor*[outputs.size()];

        for (size_t i = 0; i < outputs.size(); i++) {
            // Convert EValue to ETTensor
            // ... conversion logic
        }

        return ET_OK();
    } catch (const std::exception& e) {
        return ET_ERROR(5, e.what());
    }
}

ET_EXPORT void et_module_free(ETModule* module) {
    delete module;
}

ET_EXPORT void et_tensor_free(ETTensor* tensor) {
    delete tensor;
}
```

---

## FFI Bindings with ffigen

### ffigen Configuration

Create `ffigen.yaml` in the package root:

```yaml
name: ExecutorchNative
description: FFI bindings for ExecuTorch
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

# Performance optimization: mark simple functions as leaf
functions:
  leaf:
    include:
      - "et_tensor_dtype"
      - "et_tensor_rank"
      - "et_tensor_shape"
      - "et_tensor_data_size"
      - "et_tensor_data"
      - "et_module_input_count"
      - "et_module_output_count"
      - "et_version"

# Include symbol addresses for finalizers
  symbol-address:
    include:
      - "et_module_free"
      - "et_tensor_free"
      - "et_status_free"

# Type mappings
type-map:
  'size_t': 'Size'

preamble: |
  // AUTO GENERATED FILE, DO NOT EDIT.
  // Generated by ffigen.

comments:
  style: doxygen
  length: full
```

### Generated FFI Code Pattern

The generated code will look like:

```dart
// lib/src/generated/executorch_ffi.g.dart
// AUTO GENERATED FILE, DO NOT EDIT.

@ffi.DefaultAsset('package:executorch_flutter/executorch_flutter.dart')
library;

import 'dart:ffi' as ffi;

// Status struct
final class ETStatus extends ffi.Struct {
  @ffi.Int()
  external int code;

  external ffi.Pointer<ffi.Char> msg;
  external ffi.Pointer<ffi.Char> file;
  external ffi.Pointer<ffi.Char> func;

  @ffi.Int()
  external int line;
}

// Opaque types
final class ETModule extends ffi.Opaque {}
final class ETTensor extends ffi.Opaque {}

// Function bindings
@ffi.Native<ffi.Pointer<ETStatus> Function(
  ffi.Pointer<ffi.Uint8>,
  ffi.Size,
  ffi.Pointer<ffi.Pointer<ETModule>>
)>()
external ffi.Pointer<ETStatus> et_module_load(
  ffi.Pointer<ffi.Uint8> modelData,
  int modelSize,
  ffi.Pointer<ffi.Pointer<ETModule>> outModule,
);

// Leaf functions (more efficient, no callback support)
@ffi.Native<ffi.Int Function(ffi.Pointer<ETModule>)>(isLeaf: true)
external int et_module_input_count(ffi.Pointer<ETModule> module);

// Symbol addresses for finalizers
@ffi.Native<ffi.Void Function(ffi.Pointer<ETModule>)>()
external void et_module_free(ffi.Pointer<ETModule> module);

// Addresses class for finalizer setup
abstract final class addresses {
  static final et_module_free = ffi.Native.addressOf<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ETModule>)>
  >(executorch_ffi.et_module_free);

  static final et_tensor_free = ffi.Native.addressOf<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ETTensor>)>
  >(executorch_ffi.et_tensor_free);
}
```

### Running ffigen

Add to `pubspec.yaml`:

```yaml
dev_dependencies:
  ffigen: ^14.0.0
```

Run:
```bash
dart run ffigen --config ffigen.yaml
```

---

## Native Assets Build System

### Required Dependencies

```yaml
# pubspec.yaml
dependencies:
  ffi: ^2.1.4

dev_dependencies:
  code_assets: ^1.0.0
  hooks: ^1.0.0
  native_toolchain_cmake: ^0.2.2
```

### Hook Configuration

```yaml
# pubspec.yaml
hooks:
  user_defines:
    executorch_flutter:
      # Enable debug logging
      debug: false

      # Backend selection
      backends:
        - xnnpack       # CPU backend (all platforms)
        # - coreml      # iOS/macOS only
        # - mps         # macOS Metal only
        # - vulkan      # Android/Linux/Windows
        # - qnn         # Qualcomm NPU

      # Optional features
      features:
        - quantization  # INT8 quantized model support
        # - profiling   # Performance profiling
```

### Build Hook Implementation

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

```dart
// lib/src/build/run_build.dart
import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';
import 'package:logging/logging.dart';

final logger = Logger('executorch_build');

// Default backends per platform
const Map<OS, Set<String>> defaultBackends = {
  OS.android: {'xnnpack'},
  OS.iOS: {'xnnpack', 'coreml'},
  OS.macOS: {'xnnpack', 'coreml', 'mps'},
  OS.linux: {'xnnpack'},
  OS.windows: {'xnnpack'},
};

// Backend availability per platform
const Map<String, Set<OS>> backendPlatforms = {
  'xnnpack': {OS.android, OS.iOS, OS.macOS, OS.linux, OS.windows},
  'coreml': {OS.iOS, OS.macOS},
  'mps': {OS.macOS},
  'vulkan': {OS.android, OS.linux, OS.windows},
  'qnn': {OS.android},
};

Future<void> runBuild(BuildInput input, BuildOutputBuilder output) async {
  final packagePath = Directory(await getPackagePath('executorch_flutter'));
  final targetOS = input.config.code.targetOS;
  final userDefines = input.userDefines;

  // Parse user configuration
  final debugMode = userDefines['debug'] as bool? ?? false;
  final requestedBackends = (userDefines['backends'] as List?)
      ?.cast<String>()
      .toSet() ?? defaultBackends[targetOS] ?? {'xnnpack'};

  // Filter backends for target platform
  final enabledBackends = requestedBackends
      .where((b) => backendPlatforms[b]?.contains(targetOS) ?? false)
      .toSet();

  if (enabledBackends.isEmpty) {
    throw ArgumentError('No valid backends for $targetOS. '
        'Requested: $requestedBackends, '
        'Available: ${backendPlatforms.entries
            .where((e) => e.value.contains(targetOS))
            .map((e) => e.key)
            .toList()}');
  }

  logger.info('Building for $targetOS with backends: $enabledBackends');

  // Select CMake generator
  final generator = switch (targetOS) {
    OS.linux => Generator.ninja,
    OS.macOS || OS.iOS => Generator.xcode,
    OS.windows => Generator.visualStudio,
    OS.android => Generator.ninja,
    _ => throw ArgumentError('Unsupported OS: $targetOS'),
  };

  // Platform-specific CMake arguments
  final platformDefines = <String, String>{};

  if (targetOS == OS.macOS) {
    platformDefines['CMAKE_OSX_DEPLOYMENT_TARGET'] = '11.0';
  } else if (targetOS == OS.iOS) {
    platformDefines['CMAKE_OSX_DEPLOYMENT_TARGET'] = '13.0';
    platformDefines['CMAKE_OSX_SYSROOT'] = 'iphoneos';
  } else if (targetOS == OS.android) {
    platformDefines['ANDROID_PLATFORM'] = 'android-26';
    platformDefines['ANDROID_STL'] = 'c++_shared';
  }

  // Backend CMake options
  final backendDefines = {
    'ET_BUILD_XNNPACK': enabledBackends.contains('xnnpack') ? 'ON' : 'OFF',
    'ET_BUILD_COREML': enabledBackends.contains('coreml') ? 'ON' : 'OFF',
    'ET_BUILD_MPS': enabledBackends.contains('mps') ? 'ON' : 'OFF',
    'ET_BUILD_VULKAN': enabledBackends.contains('vulkan') ? 'ON' : 'OFF',
    'ET_BUILD_QNN': enabledBackends.contains('qnn') ? 'ON' : 'OFF',
  };

  // Build with CMake
  final builder = CMakeBuilder.create(
    logLevel: debugMode ? LogLevel.DEBUG : LogLevel.STATUS,
    name: input.packageName,
    sourceDir: packagePath.uri.resolve('src/'),
    targets: ['install'],
    generator: generator,
    appleArgs: AppleBuilderArgs(
      enableArc: false,
      enableBitcode: false,
      enableVisibility: true,
    ),
    defines: {
      'CMAKE_BUILD_TYPE': debugMode ? 'Debug' : 'Release',
      'CMAKE_INSTALL_PREFIX': input.outputDirectory.resolve('install/').toFilePath(),
      'BUILD_SHARED_LIBS': 'ON',
      ...platformDefines,
      ...backendDefines,
    },
  );

  await builder.run(input: input, output: output, logger: logger);

  // Register built libraries as code assets
  await output.findAndAddCodeAssets(
    input,
    outDir: input.outputDirectory.resolve('install/lib/'),
    names: {'executorch_flutter': 'executorch_flutter.dart'},
  );

  // On Linux, set RPATH for portable libraries
  if (targetOS == OS.linux) {
    final assets = await output.findAndAddCodeAssets(
      input,
      outDir: input.outputDirectory.resolve('install/lib/'),
      names: {'executorch_flutter': 'executorch_flutter.dart'},
    );
    for (final asset in assets) {
      if (asset.file != null) {
        await _setRPath(asset.file!, r'$ORIGIN');
      }
    }
  }
}

Future<void> _setRPath(Uri libPath, String rpath) async {
  final result = await Process.run('patchelf', [
    '--set-rpath', rpath,
    libPath.toFilePath(),
  ]);
  if (result.exitCode != 0) {
    logger.warning('Failed to set RPATH: ${result.stderr}');
  }
}

Future<String> getPackagePath(String packageName) async {
  // Implementation to find package path
  return Directory.current.path;
}
```

### CMakeLists.txt

```cmake
# src/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(executorch_flutter VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Backend options
option(ET_BUILD_XNNPACK "Build with XNNPACK backend" ON)
option(ET_BUILD_COREML "Build with CoreML backend" OFF)
option(ET_BUILD_MPS "Build with MPS backend" OFF)
option(ET_BUILD_VULKAN "Build with Vulkan backend" OFF)
option(ET_BUILD_QNN "Build with QNN backend" OFF)

# Find or fetch ExecuTorch
include(FetchContent)

# Option 1: Use pre-built ExecuTorch
# find_package(executorch REQUIRED)

# Option 2: Build from source (slower but more flexible)
FetchContent_Declare(
  executorch
  GIT_REPOSITORY https://github.com/pytorch/executorch.git
  GIT_TAG v0.4.0
)
FetchContent_MakeAvailable(executorch)

# Our FFI wrapper library
add_library(executorch_flutter SHARED
  executorch_ffi.cpp
)

target_include_directories(executorch_flutter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${executorch_SOURCE_DIR}/..
)

target_link_libraries(executorch_flutter PRIVATE
  executorch
  executorch_module
)

# Link backends based on options
if(ET_BUILD_XNNPACK)
  target_link_libraries(executorch_flutter PRIVATE xnnpack_backend)
  target_compile_definitions(executorch_flutter PRIVATE ET_USE_XNNPACK)
endif()

if(ET_BUILD_COREML)
  target_link_libraries(executorch_flutter PRIVATE coreml_backend)
  target_compile_definitions(executorch_flutter PRIVATE ET_USE_COREML)
endif()

if(ET_BUILD_MPS)
  target_link_libraries(executorch_flutter PRIVATE mps_backend)
  target_compile_definitions(executorch_flutter PRIVATE ET_USE_MPS)
endif()

# Installation
install(TARGETS executorch_flutter
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES executorch_ffi.h
  DESTINATION include
)
```

---

## Memory Management

### NativeFinalizer Pattern

```dart
// lib/src/core/base.dart
import 'dart:ffi' as ffi;
import 'package:executorch_flutter/src/generated/executorch_ffi.g.dart' as native;

/// Type alias for finalizer function
typedef NativeFinalizerFunc<T extends ffi.NativeType> =
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(T)>>;

/// Create a NativeFinalizer from function pointer
ffi.NativeFinalizer createFinalizer<T extends ffi.NativeType>(
    NativeFinalizerFunc<T> func) {
  return ffi.NativeFinalizer(func.cast<ffi.NativeFinalizerFunction>());
}

/// Base class for native-backed objects
abstract class NativeObject<T extends ffi.NativeType> {
  ffi.Pointer<T> _ptr;
  bool _disposed = false;

  NativeObject(this._ptr);

  ffi.Pointer<T> get ptr {
    if (_disposed) {
      throw StateError('Object has been disposed');
    }
    return _ptr;
  }

  bool get isDisposed => _disposed;

  /// Attach finalizer for automatic cleanup
  void attachFinalizer(ffi.NativeFinalizer finalizer, {int? externalSize}) {
    finalizer.attach(
      this,
      _ptr.cast(),
      detach: this,
      externalSize: externalSize,
    );
  }

  /// Manual disposal (detaches finalizer)
  void dispose();
}
```

### Module Wrapper with Finalizer

```dart
// lib/src/model.dart
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:executorch_flutter/src/generated/executorch_ffi.g.dart' as native;
import 'base.dart';

class ExecuTorchModel extends NativeObject<native.ETModule> {
  // Static finalizer shared by all instances
  static final _finalizer = createFinalizer<native.ETModule>(
    native.addresses.et_module_free,
  );

  ExecuTorchModel._(ffi.Pointer<native.ETModule> ptr) : super(ptr) {
    attachFinalizer(_finalizer);
  }

  /// Load model from bytes
  static Future<ExecuTorchModel> load(Uint8List modelData) async {
    return using((arena) {
      // Allocate native buffer
      final dataPtr = arena<ffi.Uint8>(modelData.length);
      final dataList = dataPtr.asTypedList(modelData.length);
      dataList.setAll(0, modelData);

      // Allocate output pointer
      final modulePtr = arena<ffi.Pointer<native.ETModule>>();

      // Call native function
      final status = native.et_module_load(
        dataPtr,
        modelData.length,
        modulePtr,
      );

      _checkStatus(status);

      return ExecuTorchModel._(modulePtr.value);
    });
  }

  /// Run inference
  Future<List<TensorData>> forward(List<TensorData> inputs) async {
    if (_disposed) {
      throw StateError('Model has been disposed');
    }

    return using((arena) {
      // Convert inputs to native tensors
      final nativeInputs = arena<ffi.Pointer<native.ETTensor>>(inputs.length);
      for (var i = 0; i < inputs.length; i++) {
        nativeInputs[i] = _createNativeTensor(inputs[i], arena);
      }

      // Allocate output pointers
      final outputsPtr = arena<ffi.Pointer<ffi.Pointer<native.ETTensor>>>();
      final outputCountPtr = arena<ffi.Int>();

      // Run forward pass
      final status = native.et_module_forward(
        ptr,
        nativeInputs,
        inputs.length,
        outputsPtr,
        outputCountPtr,
      );

      _checkStatus(status);

      // Convert outputs to Dart
      final outputCount = outputCountPtr.value;
      final outputs = <TensorData>[];

      for (var i = 0; i < outputCount; i++) {
        outputs.add(_convertNativeTensor(outputsPtr.value[i]));
      }

      // Free native output array
      native.et_tensor_array_free(outputsPtr.value, outputCount);

      return outputs;
    });
  }

  @override
  void dispose() {
    if (!_disposed) {
      _finalizer.detach(this);
      native.et_module_free(ptr);
      _disposed = true;
    }
  }

  static void _checkStatus(ffi.Pointer<native.ETStatus> status) {
    if (status.ref.code != 0) {
      final msg = status.ref.msg.cast<Utf8>().toDartString();
      native.et_status_free(status);
      throw ExecuTorchException(msg);
    }
    native.et_status_free(status);
  }
}
```

### Arena-Based Temporary Allocations

```dart
// lib/src/core/arena.dart
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

/// Run computation with automatic memory cleanup
R runWithArena<R>(R Function(Arena arena) computation) {
  final arena = Arena();
  try {
    final result = computation(arena);
    if (result is Future) {
      return result.whenComplete(arena.releaseAll) as R;
    }
    return result;
  } finally {
    if (R is! Future) {
      arena.releaseAll();
    }
  }
}

// Usage example:
Future<List<TensorData>> forward(List<TensorData> inputs) {
  return runWithArena((arena) async {
    // All allocations via arena are automatically freed
    final buffer = arena<ffi.Uint8>(1024);
    // ... use buffer
    return results;
  });
}
```

---

## Backend Configuration

### Runtime Backend Selection

```dart
// lib/src/backends.dart

/// Available ExecuTorch backends
enum ETBackend {
  /// XNNPACK CPU backend (all platforms)
  xnnpack('XNNPACK', 'Optimized CPU inference'),

  /// CoreML backend (iOS/macOS)
  coreml('CoreML', 'Apple Neural Engine acceleration'),

  /// MPS backend (macOS)
  mps('MPS', 'Metal Performance Shaders'),

  /// Vulkan backend (Android/Linux/Windows)
  vulkan('Vulkan', 'GPU compute via Vulkan'),

  /// Qualcomm QNN backend (Android)
  qnn('QNN', 'Qualcomm NPU acceleration');

  const ETBackend(this.displayName, this.description);

  final String displayName;
  final String description;

  /// Check if backend is available on current platform
  static Set<ETBackend> get available {
    // This would be determined at build time based on
    // which backends were compiled in
    return _availableBackends;
  }
}

// Set at build time via native function
late final Set<ETBackend> _availableBackends;

void _initBackends() {
  _availableBackends = {};

  // Query native library for available backends
  if (native.et_backend_available(native.ET_BACKEND_XNNPACK)) {
    _availableBackends.add(ETBackend.xnnpack);
  }
  if (native.et_backend_available(native.ET_BACKEND_COREML)) {
    _availableBackends.add(ETBackend.coreml);
  }
  // ... etc
}
```

### User Configuration in pubspec.yaml

```yaml
# App's pubspec.yaml
dependencies:
  executorch_flutter: ^1.0.0

# Customize native build
hooks:
  user_defines:
    executorch_flutter:
      # Only include needed backends to reduce binary size
      backends:
        - xnnpack
        - coreml  # Only on iOS/macOS

      # Enable additional features
      features:
        - quantization

      # Debug build for development
      debug: true
```

---

## Platform-Specific Considerations

### Android

```cmake
# Android-specific CMake
if(ANDROID)
  # Use NDK toolchain
  set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
  set(CMAKE_SYSTEM_NAME Android)
  set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
  set(CMAKE_ANDROID_STL c++_shared)

  # Link Android log library
  find_library(log-lib log)
  target_link_libraries(executorch_flutter PRIVATE ${log-lib})
endif()
```

### iOS

```cmake
# iOS-specific CMake
if(IOS)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
  set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "13.0")

  # Disable bitcode (deprecated)
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")

  # Framework search paths for CoreML
  if(ET_BUILD_COREML)
    find_library(COREML_FRAMEWORK CoreML)
    target_link_libraries(executorch_flutter PRIVATE ${COREML_FRAMEWORK})
  endif()
endif()
```

### macOS

```cmake
# macOS-specific CMake
if(APPLE AND NOT IOS)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")

  # Universal binary (Intel + Apple Silicon)
  set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")

  # Metal framework for MPS backend
  if(ET_BUILD_MPS)
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    target_link_libraries(executorch_flutter PRIVATE
      ${METAL_FRAMEWORK}
      ${MPS_FRAMEWORK}
    )
  endif()
endif()
```

### Linux

```cmake
# Linux-specific CMake
if(UNIX AND NOT APPLE)
  # Set RPATH for portable binaries
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

  # Vulkan support
  if(ET_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(executorch_flutter PRIVATE Vulkan::Vulkan)
  endif()
endif()
```

### Windows

```cmake
# Windows-specific CMake
if(WIN32)
  # Export all symbols
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

  # Use static CRT for standalone distribution
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  # Vulkan support via Vulkan SDK
  if(ET_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(executorch_flutter PRIVATE Vulkan::Vulkan)
  endif()
endif()
```

---

## Migration Strategy

### Phase 1: Add FFI Layer (Non-Breaking)

1. Create C wrapper (`src/executorch_ffi.h`, `src/executorch_ffi.cpp`)
2. Set up ffigen configuration
3. Implement native assets build hook
4. Create Dart FFI wrapper classes
5. Add FFI implementation alongside existing Pigeon implementation

```dart
// lib/src/executorch_model.dart
class ExecuTorchModel {
  // Keep both implementations during transition
  static bool useFFI = false;

  final _pigeonModel = useFFI ? null : PigeonExecuTorchModel();
  final _ffiModel = useFFI ? FFIExecuTorchModel() : null;

  Future<List<TensorData>> forward(List<TensorData> inputs) {
    if (useFFI) {
      return _ffiModel!.forward(inputs);
    } else {
      return _pigeonModel!.forward(inputs);
    }
  }
}
```

### Phase 2: Add Desktop Platforms

1. Test FFI implementation on Linux and Windows
2. Add platform-specific backends (Vulkan)
3. Update CI/CD for new platforms

### Phase 3: Migrate Mobile Platforms

1. Benchmark FFI vs Pigeon performance
2. If FFI is faster, migrate Android and iOS
3. Deprecate Pigeon implementation

### Phase 4: Remove Pigeon (Breaking Change)

1. Remove Pigeon dependency
2. Remove platform-specific Kotlin/Swift code
3. Simplify to single FFI implementation
4. Major version bump

---

## References

- [opencv_dart](https://github.com/rainyl/opencv_dart) - Production FFI patterns
- [dart:ffi documentation](https://dart.dev/guides/libraries/c-interop)
- [ffigen documentation](https://pub.dev/packages/ffigen)
- [Native Assets proposal](https://github.com/dart-lang/sdk/issues/50565)
- [ExecuTorch C++ API](https://pytorch.org/executorch/)

---

**Last Updated**: 2025-01-11
**Status**: Reference Documentation
**Applicability**: Future architecture consideration for executorch_flutter
