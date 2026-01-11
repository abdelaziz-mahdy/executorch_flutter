#!/bin/bash
# Script to build ExecuTorch Wasm library inside Docker container or natively
# This builds executorch_wasm as an embeddable library using Embind

set -e  # Exit on error

echo "========================================"
echo "ExecuTorch Wasm Library Build Script"
echo "========================================"
echo ""

# Detect if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    EXECUTORCH_ROOT="/workspace/executorch"
    SCRIPT_DIR="/workspace/scripts"
else
    echo "Running natively"
    # Assume we're in the scripts/ directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-${SCRIPT_DIR}/../../executorch-repo}"
fi

echo "ExecuTorch root: ${EXECUTORCH_ROOT}"
echo "Script dir: ${SCRIPT_DIR}"
echo ""

# Navigate to ExecuTorch directory
cd "${EXECUTORCH_ROOT}"

# Source Emscripten environment
echo "Setting up Emscripten environment..."
if [ -f /.dockerenv ]; then
    # In Docker, emsdk is at /opt/emsdk
    source /opt/emsdk/emsdk_env.sh
else
    # Native: try to find emsdk
    if [ -z "${EMSDK}" ]; then
        if [ -f "${HOME}/emsdk/emsdk_env.sh" ]; then
            source "${HOME}/emsdk/emsdk_env.sh"
        elif [ -f "/opt/emsdk/emsdk_env.sh" ]; then
            source /opt/emsdk/emsdk_env.sh
        else
            echo "ERROR: Emscripten SDK not found. Please install emsdk or set EMSDK environment variable."
            echo "Installation: https://emscripten.org/docs/getting_started/downloads.html"
            exit 1
        fi
    else
        source "${EMSDK}/emsdk_env.sh"
    fi
fi

echo "Emscripten version: $(emcc --version | head -n 1)"
echo ""

# Clean previous build (optional)
BUILD_DIR="cmake-out-wasm-lib"
if [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

# Create models directory with dummy file (prevents Emscripten file_packager error)
# We'll load models at runtime via HTTP, not embed them
echo "Creating models directory with placeholder..."
mkdir -p models
echo "# Placeholder for Emscripten file_packager" > models/.placeholder

# Create a custom CMakeLists.txt for building the library executable
echo ""
echo "Creating custom CMakeLists.txt for library build..."
mkdir -p "${BUILD_DIR}"

cat > "${BUILD_DIR}/CMakeLists.txt" << 'EOF'
# Custom CMakeLists.txt for building ExecuTorch Wasm library
cmake_minimum_required(VERSION 3.29)
project(executorch_flutter_wasm)

# Include the main ExecuTorch CMakeLists.txt
# This sets up all the targets we need
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_subdirectory(${EXECUTORCH_ROOT} executorch_build EXCLUDE_FROM_ALL)

# Create the final executable that links executorch_wasm
add_executable(executorch_lib)

# Empty main - Embind handles all the exports
target_sources(executorch_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# Link to executorch_wasm (the object library with Embind bindings)
# Also link xnnpack_backend for XNNPACK delegate support
# This enables running models exported with XNNPACK delegate
target_link_libraries(executorch_lib PRIVATE executorch_wasm xnnpack_backend)

# Emscripten-specific link options for embeddable library
target_link_options(
  executorch_lib
  PRIVATE
  -sALLOW_MEMORY_GROWTH=1
  -sMODULARIZE=1
  -sEXPORT_NAME=createExecuTorchModule
  -sENVIRONMENT=web,worker
  -sNO_EXIT_RUNTIME=1
  -sASSERTIONS=1
  -sFORCE_FILESYSTEM=1
  -sINITIAL_MEMORY=67108864
  "-sEXPORTED_RUNTIME_METHODS=['FS']"
  -fexceptions
)

set_target_properties(
  executorch_lib
  PROPERTIES
  OUTPUT_NAME "executorch"
  SUFFIX ".js"
)
EOF

# Create empty main.cpp (Embind handles everything)
cat > "${BUILD_DIR}/main.cpp" << 'EOF'
// Empty main - Embind handles all the exports via EMSCRIPTEN_BINDINGS
// The wasm_bindings.cpp registers all exports automatically
int main() {
    return 0;
}
EOF

# Configure and build
echo ""
echo "Configuring CMake with Emscripten..."
cd "${BUILD_DIR}"

emcmake cmake \
    -DEXECUTORCH_PAL_DEFAULT=posix \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_WASM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_KERNELS_PORTABLE=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    .

# Build the library target
# Use limited parallelism to avoid compiler crashes (LLVM bug with XNNPACK code)
echo ""
echo "Building executorch_lib target..."
cmake --build . -j2 --target executorch_lib

# Verify outputs
echo ""
echo "Verifying build outputs..."
if [ -f "executorch.js" ] && [ -f "executorch.wasm" ]; then
    echo "✅ Library build successful!"
    echo ""
    echo "Generated files:"
    ls -lh executorch.js executorch.wasm
    echo ""

    # Copy to output directory if specified
    if [ -n "${OUTPUT_DIR}" ]; then
        echo "Copying binaries to ${OUTPUT_DIR}..."
        mkdir -p "${OUTPUT_DIR}"
        cp executorch.js "${OUTPUT_DIR}/"
        cp executorch.wasm "${OUTPUT_DIR}/"
        echo "✅ Copied to ${OUTPUT_DIR}"
    fi

    # If running in Docker, also copy to /output volume
    if [ -f /.dockerenv ] && [ -d /output ]; then
        echo "Copying binaries to /output volume..."
        cp executorch.js /output/
        cp executorch.wasm /output/
        echo "✅ Copied to /output"
    fi
else
    echo "❌ Build failed: executorch.js or executorch.wasm not found"
    echo "Contents of build dir:"
    ls -la
    exit 1
fi

echo ""
echo "========================================"
echo "Library build completed successfully!"
echo "========================================"
echo ""
echo "Features enabled:"
echo "  - XNNPACK backend (optimized kernels for WASM SIMD)"
echo "  - Portable kernels (fallback)"
echo ""
echo "The library exports the following via Embind:"
echo "  - Module.load(data)   - Load model from Uint8Array/ArrayBuffer/path"
echo "  - module.forward(inputs) - Run inference"
echo "  - module.execute(method, inputs) - Execute specific method"
echo "  - Tensor.fromArray(sizes, data) - Create tensor from JS array"
echo ""
echo "Usage in JavaScript:"
echo "  const Module = await createExecuTorchModule();"
echo "  const model = Module.Module.load(modelBytes);"
echo "  const output = model.forward([inputTensor]);"
