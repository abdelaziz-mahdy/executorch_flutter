# ExecuTorch Wasm Library Build for Flutter Web
# This CMakeLists.txt builds executorch_wasm as an embeddable library
# using Emscripten Embind for JavaScript interop.

cmake_minimum_required(VERSION 3.29)

project(executorch_flutter_wasm)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Emscripten is required to build this target")
endif()

# Source root directory for executorch
if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "EXECUTORCH_ROOT must be set")
endif()

set(CMAKE_CXX_STANDARD 17)

# Include ExecuTorch CMake utilities
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Build options
set(_common_compile_options
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall
    -Werror
    -Wno-deprecated-declarations
    -fPIC>
)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Library dependencies
set(link_libraries)
list(
  APPEND
  link_libraries
  embind
  executorch_core
  extension_data_loader
  extension_module_static
  extension_tensor
  extension_runner_util
)

# Create the wasm library target as an executable (Emscripten outputs .js/.wasm)
add_executable(executorch_wasm_lib)

# Add the wasm_bindings source
target_sources(executorch_wasm_lib PRIVATE
  ${EXECUTORCH_ROOT}/extension/wasm/wasm_bindings.cpp
)

target_compile_options(executorch_wasm_lib PRIVATE ${_common_compile_options})
target_include_directories(executorch_wasm_lib PRIVATE ${_common_include_directories})

# Link libraries
target_link_libraries(
  executorch_wasm_lib
  PRIVATE ${link_libraries}
  PRIVATE executorch_kernels
)

# Emscripten-specific link options for embeddable library
target_link_options(
  executorch_wasm_lib
  PRIVATE
  # Allow dynamic memory growth
  -sALLOW_MEMORY_GROWTH=1
  # Enable embind
  -lembind
  # Export necessary runtime methods
  -sEXPORTED_RUNTIME_METHODS=['FS','ccall','cwrap']
  # Modular output (can be imported as ES module)
  -sMODULARIZE=1
  # Export name for the module factory
  -sEXPORT_NAME=createExecuTorchModule
  # Environment support
  -sENVIRONMENT=web,worker
  # Enable async module instantiation
  -sASYNC=1
  # Don't exit runtime (keep WASM alive for further calls)
  -sNO_EXIT_RUNTIME=1
  # Assertions for debugging (can be disabled in production)
  -sASSERTIONS=1
  # Enable filesystem
  -sFORCE_FILESYSTEM=1
  # Initial memory size (64MB)
  -sINITIAL_MEMORY=67108864
  # Enable exception handling
  -fexceptions
  # Output ES6 module format for better integration
  -sEXPORT_ES6=0
)

# Set output name
set_target_properties(
  executorch_wasm_lib
  PROPERTIES
  OUTPUT_NAME "executorch"
  SUFFIX ".js"
)

message(STATUS "ExecuTorch Flutter Wasm library target configured")
