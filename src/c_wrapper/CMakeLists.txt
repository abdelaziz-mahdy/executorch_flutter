# CMakeLists.txt for ExecuTorch Flutter C/C++ Wrapper
# Builds the C wrapper library that interfaces with ExecuTorch C++ API
#
# This file is used by:
# - Android NDK build (via android/CMakeLists.txt)
# - Potentially Linux build in the future
#
# Platform-specific configurations are handled in parent CMakeLists.txt files

cmake_minimum_required(VERSION 3.10)

project(executorch_flutter_wrapper VERSION 1.0.0 LANGUAGES C CXX)

# C/C++ Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror=implicit-function-declaration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17")

# Release optimizations
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")

# ==============================================================================
# Source Files
# ==============================================================================

# C utility sources
set(C_WRAPPER_SOURCES_C
    error_codes.c
    error_mapping.c
    tensor_utils.c
    executorch_flutter_wrapper.c
)

# C++ core wrapper (requires ExecuTorch headers)
set(C_WRAPPER_SOURCES_CXX
    executorch_flutter_wrapper.cpp
)

# Platform-specific sources
if(ANDROID)
    list(APPEND C_WRAPPER_SOURCES_C platform/platform_android.c)
elseif(IOS)
    list(APPEND C_WRAPPER_SOURCES_C platform/platform_ios.c)
elseif(APPLE)  # macOS
    list(APPEND C_WRAPPER_SOURCES_C platform/platform_macos.c)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# ==============================================================================
# ExecuTorch Dependency
# ==============================================================================

# ExecuTorch include directories
# These should be provided by parent CMakeLists.txt or as CMake variables:
# - EXECUTORCH_INCLUDE_DIR: Path to ExecuTorch headers
# - EXECUTORCH_LIBRARY_DIR: Path to ExecuTorch libraries (optional, for static linking)
#
# For Android: ExecuTorch AAR provides headers and .so files
# For iOS/macOS: ExecuTorch XCFrameworks provide headers via SPM

if(NOT DEFINED EXECUTORCH_INCLUDE_DIR)
    # Fallback to user's executorch fork location
    set(EXECUTORCH_INCLUDE_DIR "/Users/AbdelazizMahdy/flutter_projects/executorch_fork")
    message(WARNING "EXECUTORCH_INCLUDE_DIR not set, using default: ${EXECUTORCH_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${EXECUTORCH_INCLUDE_DIR}")
    message(FATAL_ERROR "ExecuTorch include directory not found: ${EXECUTORCH_INCLUDE_DIR}")
endif()

# ==============================================================================
# Library Target
# ==============================================================================

add_library(executorch_flutter_wrapper SHARED
    ${C_WRAPPER_SOURCES_C}
    ${C_WRAPPER_SOURCES_CXX}
)

# Include directories
target_include_directories(executorch_flutter_wrapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EXECUTORCH_INCLUDE_DIR}
)

# Link libraries
# ExecuTorch libraries will be linked by platform-specific configurations
# For Android: libexecutorch.so from AAR
# For iOS/macOS: Frameworks from XCFrameworks

if(ANDROID)
    # Android-specific linking will be handled in android/CMakeLists.txt
    # AAR provides libexecutorch.so which will be found via ANDROID_ABI path
endif()

# ==============================================================================
# Installation (for Android APK packaging)
# ==============================================================================

if(ANDROID)
    # Install to jniLibs for APK packaging
    install(TARGETS executorch_flutter_wrapper
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# ==============================================================================
# Properties
# ==============================================================================

# Set library output name
set_target_properties(executorch_flutter_wrapper PROPERTIES
    OUTPUT_NAME "executorch_flutter"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Position-independent code (required for shared libraries)
set_target_properties(executorch_flutter_wrapper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Platform-Specific Configurations
# ==============================================================================

if(ANDROID)
    # Android NDK build
    target_compile_definitions(executorch_flutter_wrapper PRIVATE __ANDROID__)

    # Find and link ExecuTorch library from AAR (extracted by Gradle task)
    # Library path: ${EXECUTORCH_LIBRARY_DIR}/${ANDROID_ABI}/
    set(EXECUTORCH_LIB_PATH "${EXECUTORCH_LIBRARY_DIR}/${ANDROID_ABI}")

    find_library(EXECUTORCH_LIB
        NAMES executorch libexecutorch
        PATHS ${EXECUTORCH_LIB_PATH}
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
    )

    if(EXECUTORCH_LIB)
        message(STATUS "Found ExecuTorch library: ${EXECUTORCH_LIB}")
        target_link_libraries(executorch_flutter_wrapper ${EXECUTORCH_LIB})
    else()
        message(WARNING "ExecuTorch library not found. Searched in: ${EXECUTORCH_LIB_PATH}")
        message(WARNING "Available files:")
        file(GLOB AVAILABLE_LIBS "${EXECUTORCH_LIB_PATH}/*")
        foreach(lib ${AVAILABLE_LIBS})
            message(STATUS "  - ${lib}")
        endforeach()
    endif()

    # Link Android log library
    find_library(LOG_LIB log)
    target_link_libraries(executorch_flutter_wrapper ${LOG_LIB})

elseif(IOS)
    # iOS build (would be integrated with Xcode project via podspec)
    target_compile_definitions(executorch_flutter_wrapper PRIVATE
        __APPLE__
        TARGET_OS_IPHONE=1
    )

elseif(APPLE)
    # macOS build (would be integrated with Xcode project via podspec)
    target_compile_definitions(executorch_flutter_wrapper PRIVATE
        __APPLE__
        TARGET_OS_OSX=1
    )
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "==================================")
message(STATUS "ExecuTorch Flutter C/C++ Wrapper")
message(STATUS "==================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Standard: C11")
message(STATUS "C++ Standard: C++17")
message(STATUS "ExecuTorch include: ${EXECUTORCH_INCLUDE_DIR}")
if(DEFINED EXECUTORCH_LIBRARY_DIR)
    message(STATUS "ExecuTorch library: ${EXECUTORCH_LIBRARY_DIR}")
endif()
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
if(ANDROID)
    message(STATUS "Android ABI: ${ANDROID_ABI}")
endif()
message(STATUS "==================================")
