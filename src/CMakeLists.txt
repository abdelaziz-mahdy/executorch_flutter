# ExecuTorch Flutter Plugin - Main CMakeLists.txt
# Self-Contained Build System using CMake FetchContent
#
# This file is the entry point for all platform builds:
# - Android: Called by android/build.gradle externalNativeBuild
# - iOS/macOS: Called by Package.swift build scripts
#
# Architecture:
# 1. Configure platform-specific options (executorch_options.cmake)
# 2. Download ExecuTorch source (executorch_fetch.cmake)
# 3. Build C++ wrapper linking ExecuTorch
# 4. Install to platform-specific location

cmake_minimum_required(VERSION 3.22)

# Project definition
project(executorch_flutter_wrapper LANGUAGES C CXX)

# Set C++17 standard (required by ExecuTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C11 standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

message(STATUS "==================================================")
message(STATUS "ExecuTorch Flutter Plugin Build")
message(STATUS "  CMake Version: ${CMAKE_VERSION}")
message(STATUS "  System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(ANDROID)
    message(STATUS "  Android ABI: ${ANDROID_ABI}")
endif()
message(STATUS "==================================================")

# ==============================================================================
# Step 1: Configure ExecuTorch Build Options
# ==============================================================================

# This MUST come before executorch_fetch.cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/build_config/executorch_options.cmake)

# ==============================================================================
# Step 2: Download and Build ExecuTorch
# ==============================================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/build_config/executorch_fetch.cmake)

# ==============================================================================
# Step 3: Build C++ Wrapper
# ==============================================================================

# Collect wrapper source files
set(WRAPPER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/error_codes.c
    ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/error_mapping.c
    ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/tensor_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/executorch_flutter_wrapper.cpp
)

# Add platform-specific source files
if(ANDROID)
    list(APPEND WRAPPER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/platform/platform_android.c
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(APPEND WRAPPER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/platform/platform_ios.c
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND WRAPPER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper/platform/platform_macos.c
    )
endif()

# Create shared library
add_library(executorch_flutter_wrapper SHARED ${WRAPPER_SOURCES})

# Include directories
target_include_directories(executorch_flutter_wrapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper
        ${executorch_SOURCE_DIR}  # ExecuTorch headers
)

# Link ExecuTorch and platform libraries
target_link_libraries(executorch_flutter_wrapper
    PUBLIC
        executorch  # Main ExecuTorch library
)

# Platform-specific linking
if(ANDROID)
    # Android logging library
    find_library(LOG_LIB log)
    target_link_libraries(executorch_flutter_wrapper PRIVATE ${LOG_LIB})

    message(STATUS "Android Build Configuration:")
    message(STATUS "  Wrapper Library: libexecutorch_flutter_wrapper.so")
    message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")

elseif(APPLE)
    # Apple frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(executorch_flutter_wrapper PRIVATE ${FOUNDATION_FRAMEWORK})

    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        message(STATUS "iOS Build Configuration:")
        message(STATUS "  Wrapper Framework: executorch_flutter_wrapper")
    else()
        message(STATUS "macOS Build Configuration:")
        message(STATUS "  Wrapper Framework: executorch_flutter_wrapper")
    endif()
endif()

# Compiler warnings
target_compile_options(executorch_flutter_wrapper PRIVATE
    -Wall
    -Wextra
    -Werror=return-type  # Error on missing return statements
)

# ==============================================================================
# Step 4: Installation
# ==============================================================================

if(ANDROID)
    # Install to jniLibs directory for Android
    install(TARGETS executorch_flutter_wrapper
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${ANDROID_ABI}
    )
else()
    # For iOS/macOS, installation handled by Package.swift
    install(TARGETS executorch_flutter_wrapper
        LIBRARY DESTINATION lib
        FRAMEWORK DESTINATION .
    )
endif()

message(STATUS "==================================================")
message(STATUS "ExecuTorch Flutter Plugin Configuration Complete")
message(STATUS "==================================================")
