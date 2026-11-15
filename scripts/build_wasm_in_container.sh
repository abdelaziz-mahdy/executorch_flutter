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

# Fix XNNPACK's tensor size check on 32-bit targets. It counted bits in a
# size_t, so any tensor over 512 MiB looked like an overflow on wasm32 and the
# delegate refused it: an LLM's logits ([1, seq, vocab] fp32) reach that.
# Upstream: https://github.com/google/XNNPACK/pull/12038. Applied here rather
# than in the Docker image so native builds get it too; skipped once applied.
XNNPACK_PATCH="${SCRIPT_DIR}/wasm/xnnpack-tensor-size-32bit.patch"
XNNPACK_DIR="backends/xnnpack/third-party/XNNPACK"
if git -C "${XNNPACK_DIR}" apply --reverse --check "${XNNPACK_PATCH}" 2>/dev/null; then
    echo "XNNPACK tensor size patch already applied"
else
    git -C "${XNNPACK_DIR}" apply "${XNNPACK_PATCH}"
    echo "Applied XNNPACK tensor size patch"
fi

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

# LLM runner for ExecuTorchLLM on web: a separate module so the vision page
# never downloads the runner and tokenizers. Loaded by a Web Worker
# (web/js/executorch_llm_worker.js). Model weights are hundreds of MB and live
# in the Wasm heap, so it may grow to the wasm32 ceiling.
add_executable(executorch_llm_lib)
target_sources(
  executorch_llm_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/executorch_llm_web.cpp
)
target_include_directories(executorch_llm_lib PRIVATE ${EXECUTORCH_ROOT}/..)
target_link_libraries(
  executorch_llm_lib
  PRIVATE embind
          executorch
          extension_llm_runner
          extension_module_static
          extension_data_loader
          extension_tensor
          tokenizers::tokenizers
          executorch_kernels
          xnnpack_backend
)
target_link_options(
  executorch_llm_lib
  PRIVATE
  -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=4GB
  -sINITIAL_MEMORY=67108864
  -sMODULARIZE=1
  -sEXPORT_NAME=createExecuTorchLLMModule
  -sENVIRONMENT=worker
  -sNO_EXIT_RUNTIME=1
  -sFORCE_FILESYSTEM=1
  "-sEXPORTED_FUNCTIONS=['_malloc','_realloc','_free']"
  "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8']"
  -fexceptions
)
set_target_properties(
  executorch_llm_lib
  PROPERTIES
  OUTPUT_NAME "executorch_llm"
  SUFFIX ".js"
)
EOF
cp "${SCRIPT_DIR}/wasm/executorch_llm_web.cpp" "${BUILD_DIR}/"

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

# Native builds: emsdk_env.sh puts emsdk's own Python first on PATH, and it has
# no torch/torchgen. Set PYTHON_EXECUTABLE to a Python that has them, or codegen
# fails (and an empty torchgen path turns its source glob into "/*.py").
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
    -DXNNPACK_ENABLE_WASMRELAXEDSIMDFP16=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Error \
    ${PYTHON_EXECUTABLE:+-DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"} \
    .

# Build the library target
# Use limited parallelism to avoid compiler crashes (LLVM bug with XNNPACK code)
echo ""
echo "Building executorch_lib and executorch_llm_lib targets..."
cmake --build . -j"${WASM_BUILD_JOBS:-2}" --target executorch_lib executorch_llm_lib

# Verify outputs
echo ""
echo "Verifying build outputs..."
if [ -f "executorch.js" ] && [ -f "executorch.wasm" ] && [ -f "executorch_llm.js" ] && [ -f "executorch_llm.wasm" ]; then
    echo "✅ Library build successful!"
    echo ""
    echo "Generated files:"
    ls -lh executorch.js executorch.wasm executorch_llm.js executorch_llm.wasm
    echo ""

    # Copy to output directory if specified
    if [ -n "${OUTPUT_DIR}" ]; then
        echo "Copying binaries to ${OUTPUT_DIR}..."
        mkdir -p "${OUTPUT_DIR}"
        cp executorch.js "${OUTPUT_DIR}/"
        cp executorch.wasm "${OUTPUT_DIR}/"
        cp executorch_llm.js executorch_llm.wasm "${OUTPUT_DIR}/"
        echo "✅ Copied to ${OUTPUT_DIR}"
    fi

    # If running in Docker, also copy to /output volume
    if [ -f /.dockerenv ] && [ -d /output ]; then
        echo "Copying binaries to /output volume..."
        cp executorch.js /output/
        cp executorch.wasm /output/
        cp executorch_llm.js executorch_llm.wasm /output/
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
