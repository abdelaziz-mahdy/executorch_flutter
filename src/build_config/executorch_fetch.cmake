# ExecuTorch FetchContent Module
# Downloads and makes ExecuTorch source available for compilation
#
# Usage:
#   include(${CMAKE_CURRENT_SOURCE_DIR}/build_config/executorch_fetch.cmake)
#
# Output Variables:
#   executorch_SOURCE_DIR - Path to ExecuTorch source
#   executorch_BINARY_DIR - Path to ExecuTorch build directory
#   executorch_POPULATED - Whether download succeeded

include(FetchContent)

# Version configuration
set(EXECUTORCH_VERSION "v1.0.0" CACHE STRING "ExecuTorch Git tag/branch/SHA")
set(EXECUTORCH_GIT_REPOSITORY "https://github.com/pytorch/executorch.git" CACHE STRING "ExecuTorch Git repo")
set(EXECUTORCH_GIT_SHALLOW TRUE CACHE BOOL "Shallow clone for faster download")

message(STATUS "==================================================")
message(STATUS "ExecuTorch Configuration:")
message(STATUS "  Version: ${EXECUTORCH_VERSION}")
message(STATUS "  Repository: ${EXECUTORCH_GIT_REPOSITORY}")

# Check for local source override
if(DEFINED EXECUTORCH_SOURCE_DIR AND EXECUTORCH_SOURCE_DIR)
  message(STATUS "  Source: Local (${EXECUTORCH_SOURCE_DIR})")

  # Validate local path
  if(NOT EXISTS "${EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "Invalid EXECUTORCH_SOURCE_DIR: ${EXECUTORCH_SOURCE_DIR}. "
      "Directory must contain CMakeLists.txt"
    )
  endif()

  # Use local source (skip download)
  add_subdirectory(${EXECUTORCH_SOURCE_DIR} ${CMAKE_BINARY_DIR}/executorch-build EXCLUDE_FROM_ALL)
  set(executorch_POPULATED TRUE)

else()
  message(STATUS "  Source: Remote (downloading...)")

  # Download from Git
  FetchContent_Declare(
    executorch
    GIT_REPOSITORY ${EXECUTORCH_GIT_REPOSITORY}
    GIT_TAG ${EXECUTORCH_VERSION}
    GIT_SHALLOW ${EXECUTORCH_GIT_SHALLOW}
  )

  # Make ExecuTorch available
  FetchContent_MakeAvailable(executorch)

  # Check if populated
  FetchContent_GetProperties(executorch)
  if(NOT executorch_POPULATED)
    message(FATAL_ERROR
      "Failed to download ExecuTorch from ${EXECUTORCH_GIT_REPOSITORY}. "
      "Check network connection or use local source: "
      "cmake -DEXECUTORCH_SOURCE_DIR=/path/to/executorch .."
    )
  endif()

  message(STATUS "  Download complete: ${executorch_SOURCE_DIR}")
endif()

# Verify ExecuTorch target is available
if(NOT TARGET executorch)
  message(FATAL_ERROR "ExecuTorch target not found after include. "
    "Check that ExecuTorch CMakeLists.txt builds the 'executorch' target.")
endif()

message(STATUS "ExecuTorch ready for linking")
message(STATUS "==================================================")
