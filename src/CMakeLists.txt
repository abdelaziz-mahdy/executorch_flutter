cmake_minimum_required(VERSION 3.18)

project(executorch_ffi VERSION 2.0.0 LANGUAGES CXX C)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ExecuTorch version
# Note: 0.6+ removed buck2 dependency for CMake builds
if(NOT EXECUTORCH_VERSION)
    set(EXECUTORCH_VERSION "1.0.1")
endif()

message(STATUS "executorch_ffi Version: ${PROJECT_VERSION}")
message(STATUS "ExecuTorch Version: ${EXECUTORCH_VERSION}")

# Backend configuration options
option(ET_BUILD_XNNPACK "Build with XNNPACK backend" ON)
option(ET_BUILD_COREML "Build with CoreML backend" OFF)
option(ET_BUILD_MPS "Build with MPS backend" OFF)
option(ET_BUILD_VULKAN "Build with Vulkan backend" OFF)
option(ET_BUILD_QNN "Build with QNN backend" OFF)

# Platform-specific defaults
if(APPLE)
    set(ET_BUILD_COREML ON CACHE BOOL "Build with CoreML backend" FORCE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ET_BUILD_MPS ON CACHE BOOL "Build with MPS backend" FORCE)
    endif()
endif()

# ============================================================================
# Download and Build ExecuTorch
# ============================================================================

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# Silence FetchContent_Populate deprecation warning
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
    cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)

# ============================================================================
# Python Setup - Use Python executable passed from Dart build hook
# ============================================================================
# Python validation is done in Dart (run_build.dart) with better error messages.
# The PYTHON_EXECUTABLE is passed from the Dart build hook.
if(NOT PYTHON_EXECUTABLE)
    # Fallback: find Python if not passed from Dart
    set(Python3_FIND_FRAMEWORK NEVER)
    set(Python3_FIND_STRATEGY LOCATION)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE STRING "Python executable" FORCE)
endif()
message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")

# ============================================================================
# Caching Configuration
# ============================================================================
# Use EXECUTORCH_CACHE_DIR if set (for faster rebuilds)
if(DEFINED EXECUTORCH_CACHE_DIR AND NOT "${EXECUTORCH_CACHE_DIR}" STREQUAL "")
    if(NOT EXISTS "${EXECUTORCH_CACHE_DIR}")
        file(MAKE_DIRECTORY "${EXECUTORCH_CACHE_DIR}")
    endif()
    # Set FetchContent cache to persist across builds
    set(FETCHCONTENT_BASE_DIR
        "${EXECUTORCH_CACHE_DIR}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}"
        CACHE PATH "FetchContent cache directory" FORCE)
    message(STATUS "Using cache directory: ${FETCHCONTENT_BASE_DIR}")
endif()

# ExecuTorch build options (must be set before FetchContent_MakeAvailable)
# Enable host targets to build flatc and other tools
set(EXECUTORCH_BUILD_HOST_TARGETS ON CACHE BOOL "Build host targets" FORCE)
set(EXECUTORCH_BUILD_FLATC ON CACHE BOOL "Build flatc" FORCE)
# Extension modules and their dependencies (chain: module -> flat_tensor, named_data_map -> runner, data_loader)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "Build extension module" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "Build flat tensor extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "Build named data map extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON CACHE BOOL "Build runner util extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "Build data loader extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "Build tensor extension" FORCE)
set(EXECUTORCH_BUILD_KERNELS_PORTABLE ON CACHE BOOL "Build portable kernels" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF CACHE BOOL "Build optimized kernels" FORCE)
set(EXECUTORCH_BUILD_DEVTOOLS OFF CACHE BOOL "Build devtools" FORCE)
set(EXECUTORCH_BUILD_SDK OFF CACHE BOOL "Build SDK" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(EXECUTORCH_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "Build pybind" FORCE)

# Backend options
if(ET_BUILD_XNNPACK)
    set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "Build XNNPACK backend" FORCE)
else()
    set(EXECUTORCH_BUILD_XNNPACK OFF CACHE BOOL "Build XNNPACK backend" FORCE)
endif()

if(ET_BUILD_COREML AND APPLE)
    set(EXECUTORCH_BUILD_COREML ON CACHE BOOL "Build CoreML backend" FORCE)
else()
    set(EXECUTORCH_BUILD_COREML OFF CACHE BOOL "Build CoreML backend" FORCE)
endif()

if(ET_BUILD_MPS AND APPLE)
    set(EXECUTORCH_BUILD_MPS ON CACHE BOOL "Build MPS backend" FORCE)
else()
    set(EXECUTORCH_BUILD_MPS OFF CACHE BOOL "Build MPS backend" FORCE)
endif()

if(ET_BUILD_VULKAN)
    set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "Build Vulkan backend" FORCE)
else()
    set(EXECUTORCH_BUILD_VULKAN OFF CACHE BOOL "Build Vulkan backend" FORCE)
endif()

if(ET_BUILD_QNN)
    set(EXECUTORCH_BUILD_QNN ON CACHE BOOL "Build QNN backend" FORCE)
else()
    set(EXECUTORCH_BUILD_QNN OFF CACHE BOOL "Build QNN backend" FORCE)
endif()

# ExecuTorch source and binary directories
set(executorch_SOURCE_DIR ${CMAKE_BINARY_DIR}/executorch)
set(executorch_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/executorch_fetch-build)

# ============================================================================
# PYTHONPATH Setup for ExecuTorch codegen
# ============================================================================
# ExecuTorch's Python codegen scripts need to import from 'executorch' and 'torchgen'
# These modules are in the source directory and need PYTHONPATH to be set.
# We create a Python wrapper script that sets PYTHONPATH before calling Python.

# Save the original Python executable path
set(_original_python ${PYTHON_EXECUTABLE})

set(_python_wrapper_dir ${CMAKE_BINARY_DIR}/python_wrapper)
file(MAKE_DIRECTORY ${_python_wrapper_dir})

if(WIN32)
    set(_python_wrapper ${_python_wrapper_dir}/python_wrapper.bat)
    file(WRITE ${_python_wrapper}
"@echo off
set PYTHONPATH=${executorch_SOURCE_DIR};%PYTHONPATH%
\"${_original_python}\" %*
")
else()
    set(_python_wrapper ${_python_wrapper_dir}/python_wrapper.sh)
    file(WRITE ${_python_wrapper}
"#!/bin/bash
export PYTHONPATH=\"${executorch_SOURCE_DIR}:\$PYTHONPATH\"
exec \"${_original_python}\" \"$@\"
")
    # Make wrapper executable
    execute_process(COMMAND chmod +x ${_python_wrapper})
endif()

# Override PYTHON_EXECUTABLE to use our wrapper
set(PYTHON_EXECUTABLE ${_python_wrapper} CACHE STRING "Python wrapper with PYTHONPATH" FORCE)
message(STATUS "Using Python wrapper: ${PYTHON_EXECUTABLE}")
message(STATUS "  Wrapping: ${_original_python}")
message(STATUS "  PYTHONPATH includes: ${executorch_SOURCE_DIR}")

# Check if ExecuTorch is already present (manual clone or previous fetch)
if(EXISTS "${executorch_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "ExecuTorch v${EXECUTORCH_VERSION} already present at ${executorch_SOURCE_DIR}")
else()
    message(STATUS "Fetching ExecuTorch v${EXECUTORCH_VERSION}...")

    # Enable progress output for FetchContent
    set(FETCHCONTENT_QUIET FALSE)

    # Use FetchContent to download - only clone needed submodules
    FetchContent_Declare(
        executorch_fetch
        GIT_REPOSITORY https://github.com/pytorch/executorch.git
        GIT_TAG v${EXECUTORCH_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SOURCE_DIR ${executorch_SOURCE_DIR}
        # Only clone required submodules (reduces download significantly)
        GIT_SUBMODULES
            third-party/flatbuffers
            third-party/flatcc
            third-party/json
            third-party/gflags
            backends/xnnpack/third-party/XNNPACK
            backends/xnnpack/third-party/cpuinfo
            backends/xnnpack/third-party/pthreadpool
            backends/xnnpack/third-party/FP16
            backends/xnnpack/third-party/FXdiv
    )

    FetchContent_MakeAvailable(executorch_fetch)
    message(STATUS "ExecuTorch fetched successfully")
endif()

# Add ExecuTorch as subdirectory if not already added
if(NOT TARGET executorch)
    message(STATUS "Adding ExecuTorch as subdirectory...")
    add_subdirectory(${executorch_SOURCE_DIR} ${executorch_BINARY_DIR})
endif()

# ExecuTorch include directories
set(EXECUTORCH_INCLUDE_DIRS
    ${executorch_SOURCE_DIR}
    ${executorch_SOURCE_DIR}/runtime/core
    ${executorch_SOURCE_DIR}/runtime/core/exec_aten
    ${executorch_SOURCE_DIR}/runtime/executor
    ${executorch_SOURCE_DIR}/extension/module
    ${executorch_SOURCE_DIR}/extension/data_loader
    ${executorch_SOURCE_DIR}/extension/tensor
    ${executorch_BINARY_DIR}
)

# ============================================================================
# Build executorch_ffi Library
# ============================================================================

set(SOURCES
    executorch_ffi.cpp
)

set(HEADERS
    executorch_ffi.h
)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# Set export definition
target_compile_definitions(${PROJECT_NAME} PRIVATE EXECUTORCH_FFI_EXPORTS)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${EXECUTORCH_INCLUDE_DIRS}
)

# Link against ExecuTorch libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    executorch
    extension_module_static
    extension_data_loader
    extension_tensor
    portable_ops_lib
    portable_kernels
)

# Link backend libraries if available
if(ET_BUILD_XNNPACK AND TARGET xnnpack_backend)
    target_link_libraries(${PROJECT_NAME} PRIVATE xnnpack_backend)
endif()

if(ET_BUILD_COREML AND TARGET coreml_backend)
    target_link_libraries(${PROJECT_NAME} PRIVATE coreml_backend)
endif()

if(ET_BUILD_MPS AND TARGET mps_backend)
    target_link_libraries(${PROJECT_NAME} PRIVATE mps_backend)
endif()

if(ET_BUILD_VULKAN AND TARGET vulkan_backend)
    target_link_libraries(${PROJECT_NAME} PRIVATE vulkan_backend)
endif()

if(ET_BUILD_QNN AND TARGET qnn_backend)
    target_link_libraries(${PROJECT_NAME} PRIVATE qnn_backend)
endif()

# Backend compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ET_BUILD_XNNPACK=$<BOOL:${ET_BUILD_XNNPACK}>
    ET_BUILD_COREML=$<BOOL:${ET_BUILD_COREML}>
    ET_BUILD_MPS=$<BOOL:${ET_BUILD_MPS}>
    ET_BUILD_VULKAN=$<BOOL:${ET_BUILD_VULKAN}>
    ET_BUILD_QNN=$<BOOL:${ET_BUILD_QNN}>
)

# Platform-specific settings
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        FRAMEWORK FALSE
        MACOSX_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_NAME_DIR "@rpath"
    )
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_C_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

    # Apple frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${FOUNDATION_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
    )

elseif(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )

elseif(ANDROID)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_NAME "executorch_ffi"
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE log android)

elseif(UNIX AND NOT APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_NAME "executorch_ffi"
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread dl)
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include)
