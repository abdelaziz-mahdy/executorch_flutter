# ExecuTorch Platform-Specific Build Options
# Configures ExecuTorch build based on target platform
#
# Usage (BEFORE executorch_fetch.cmake):
#   include(${CMAKE_CURRENT_SOURCE_DIR}/build_config/executorch_options.cmake)
#   include(${CMAKE_CURRENT_SOURCE_DIR}/build_config/executorch_fetch.cmake)
#
# Environment Variable Overrides:
#   EXECUTORCH_BUILD_XNNPACK=ON/OFF - Enable/disable XNNPACK backend
#   EXECUTORCH_BUILD_COREML=ON/OFF - Enable/disable CoreML (iOS/macOS only)
#   EXECUTORCH_BUILD_MPS=ON/OFF - Enable/disable MPS (iOS/macOS only)
#   CMAKE_BUILD_TYPE=Debug/Release - Build mode
#
# Sets CMake CACHE variables that ExecuTorch's CMakeLists.txt will use

message(STATUS "Configuring ExecuTorch build options for ${CMAKE_SYSTEM_NAME}...")

# ==============================================================================
# Helper macro to allow environment variable overrides
# ==============================================================================

macro(set_option_with_env_override var_name default_value description)
    # Check if environment variable is set
    if(DEFINED ENV{${var_name}})
        set(${var_name} $ENV{${var_name}} CACHE BOOL "${description}" FORCE)
        message(STATUS "  [ENV] ${var_name} = $ENV{${var_name}} (from environment)")
    else()
        set(${var_name} ${default_value} CACHE BOOL "${description}" FORCE)
    endif()
endmacro()

# ==============================================================================
# Core Extensions (Required for all platforms)
# ==============================================================================

set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "Build Data Loader extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "Build Flat Tensor extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "Build Module extension" FORCE)
set(EXECUTORCH_BUILD_PORTABLE_OPS ON CACHE BOOL "Build portable ops library" FORCE)

# Module extension dependencies (auto-enabled)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "Build Named Data Map extension" FORCE)

# ==============================================================================
# Platform-Specific Configuration
# ==============================================================================

if(ANDROID)
  message(STATUS "  Platform: Android (${ANDROID_ABI})")

  # Android Platform Abstraction Layer
  set(EXECUTORCH_PAL_DEFAULT "android" CACHE STRING "Platform Abstraction Layer" FORCE)

  # Logging (disabled for release builds)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "Enable logging" FORCE)
    set(EXECUTORCH_LOG_LEVEL "Debug" CACHE STRING "Log level" FORCE)
  else()
    set(EXECUTORCH_ENABLE_LOGGING OFF CACHE BOOL "Enable logging" FORCE)
    set(EXECUTORCH_LOG_LEVEL "Error" CACHE STRING "Log level" FORCE)
  endif()

  # Backends (with environment variable overrides)
  set_option_with_env_override(EXECUTORCH_BUILD_XNNPACK ON "Build XNNPACK backend")
  set_option_with_env_override(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON "Build optimized kernels")

  # XNNPACK Optimizations
  set(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON CACHE BOOL "Share workspace across delegates" FORCE)
  set(EXECUTORCH_XNNPACK_ENABLE_KLEIDI ON CACHE BOOL "Enable Arm Kleidi kernels" FORCE)

  # Android-specific JNI (optional, not needed for FFI)
  set(EXECUTORCH_BUILD_ANDROID_JNI OFF CACHE BOOL "Build Android JNI" FORCE)

elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  message(STATUS "  Platform: iOS (arm64)")

  # POSIX Platform Abstraction Layer
  set(EXECUTORCH_PAL_DEFAULT "posix" CACHE STRING "Platform Abstraction Layer" FORCE)

  # Logging (disabled for release builds)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "Enable logging" FORCE)
    set(EXECUTORCH_LOG_LEVEL "Debug" CACHE STRING "Log level" FORCE)
  else()
    set(EXECUTORCH_ENABLE_LOGGING OFF CACHE BOOL "Enable logging" FORCE)
    set(EXECUTORCH_LOG_LEVEL "Error" CACHE STRING "Log level" FORCE)
  endif()

  # Apple Backends (with environment variable overrides)
  set_option_with_env_override(EXECUTORCH_BUILD_XNNPACK ON "Build XNNPACK backend")
  set_option_with_env_override(EXECUTORCH_BUILD_COREML ON "Build CoreML backend")
  set_option_with_env_override(EXECUTORCH_BUILD_MPS ON "Build MPS backend")
  set_option_with_env_override(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON "Build optimized kernels")

  # Apple Extensions
  set(EXECUTORCH_BUILD_EXTENSION_APPLE ON CACHE BOOL "Build Apple extension" FORCE)

  # XNNPACK Optimizations
  set(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON CACHE BOOL "Share workspace across delegates" FORCE)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "  Platform: macOS (arm64)")

  # POSIX Platform Abstraction Layer
  set(EXECUTORCH_PAL_DEFAULT "posix" CACHE STRING "Platform Abstraction Layer" FORCE)

  # Logging (disabled for release builds)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "Enable logging" FORCE)
    set(EXECUTORCH_LOG_LEVEL "Debug" CACHE STRING "Log level" FORCE)
  else()
    set(EXECUTORCH_ENABLE_LOGGING OFF CACHE BOOL "Enable logging" FORCE)
    set(EXECUTORCH_LOG_LEVEL "Error" CACHE STRING "Log level" FORCE)
  endif()

  # Apple Backends (same as iOS, with environment variable overrides)
  set_option_with_env_override(EXECUTORCH_BUILD_XNNPACK ON "Build XNNPACK backend")
  set_option_with_env_override(EXECUTORCH_BUILD_COREML ON "Build CoreML backend")
  set_option_with_env_override(EXECUTORCH_BUILD_MPS ON "Build MPS backend")
  set_option_with_env_override(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON "Build optimized kernels")

  # Apple Extensions
  set(EXECUTORCH_BUILD_EXTENSION_APPLE ON CACHE BOOL "Build Apple extension" FORCE)

  # XNNPACK Optimizations
  set(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON CACHE BOOL "Share workspace across delegates" FORCE)

else()
  message(FATAL_ERROR
    "Unsupported platform: ${CMAKE_SYSTEM_NAME}. "
    "Supported: Android, iOS, macOS (Darwin)"
  )
endif()

# ==============================================================================
# Build Mode Optimizations
# ==============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "  Build Mode: Release (optimized)")

  # Disable verification for smaller binary size (~20kB savings)
  set(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION OFF CACHE BOOL "Enable program verification" FORCE)

  # Use -O2 for performance (not -Os for size)
  set(EXECUTORCH_OPTIMIZE_SIZE OFF CACHE BOOL "Optimize for size" FORCE)

  # Disable event tracer
  set(EXECUTORCH_ENABLE_EVENT_TRACER OFF CACHE BOOL "Enable event tracer" FORCE)

else()
  message(STATUS "  Build Mode: Debug (with symbols and logging)")

  # Enable verification for debugging
  set(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION ON CACHE BOOL "Enable program verification" FORCE)

  # Use standard optimization
  set(EXECUTORCH_OPTIMIZE_SIZE OFF CACHE BOOL "Optimize for size" FORCE)

  # Enable event tracer for profiling
  set(EXECUTORCH_ENABLE_EVENT_TRACER ON CACHE BOOL "Enable event tracer" FORCE)
endif()

# ==============================================================================
# XNNPACK Dependencies (auto-enabled when XNNPACK is ON)
# ==============================================================================

if(EXECUTORCH_BUILD_XNNPACK)
  set(EXECUTORCH_BUILD_CPUINFO ON CACHE BOOL "Build cpuinfo library" FORCE)
  set(EXECUTORCH_BUILD_PTHREADPOOL ON CACHE BOOL "Build pthreadpool library" FORCE)
  message(STATUS "  XNNPACK: Enabled (with cpuinfo + pthreadpool)")
else()
  message(STATUS "  XNNPACK: Disabled")
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "ExecuTorch build configuration complete:")
message(STATUS "  - Data Loader: ${EXECUTORCH_BUILD_EXTENSION_DATA_LOADER}")
message(STATUS "  - Module API: ${EXECUTORCH_BUILD_EXTENSION_MODULE}")
message(STATUS "  - XNNPACK: ${EXECUTORCH_BUILD_XNNPACK}")
if(NOT ANDROID)
  message(STATUS "  - CoreML: ${EXECUTORCH_BUILD_COREML}")
  message(STATUS "  - MPS: ${EXECUTORCH_BUILD_MPS}")
endif()
message(STATUS "  - Optimized Kernels: ${EXECUTORCH_BUILD_KERNELS_OPTIMIZED}")
message(STATUS "  - Logging: ${EXECUTORCH_ENABLE_LOGGING}")
