#!/bin/bash
# Script to build ExecuTorch Wasm binaries inside Docker container or natively
# This script is called by the Docker container or by build_wasm.sh for native builds

set -e  # Exit on error

echo "========================================"
echo "ExecuTorch Wasm Build Script"
echo "========================================"
echo ""

# Detect if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    EXECUTORCH_ROOT="/workspace/executorch"
else
    echo "Running natively"
    # Assume we're in the scripts/ directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-${SCRIPT_DIR}/../../executorch-repo}"
fi

echo "ExecuTorch root: ${EXECUTORCH_ROOT}"
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
if [ -d "cmake-out-wasm" ]; then
    echo "Cleaning previous build..."
    rm -rf cmake-out-wasm
fi

# Install ExecuTorch if not already installed
echo "Installing ExecuTorch Python package..."
./install_executorch.sh --clean

# Configure CMake with Emscripten
echo ""
echo "Configuring CMake with Emscripten..."
mkdir -p cmake-out-wasm
cd cmake-out-wasm
emcmake cmake \
    -DEXECUTORCH_PAL_DEFAULT=posix \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    ..

# Build executor_runner target
echo ""
echo "Building executor_runner target..."
cmake --build . -j$(nproc) --target executor_runner

# Verify outputs
echo ""
echo "Verifying build outputs..."
if [ -f "executor_runner.js" ] && [ -f "executor_runner.wasm" ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Generated files:"
    ls -lh executor_runner.js executor_runner.wasm
    echo ""

    # Copy to output directory if specified
    if [ -n "${OUTPUT_DIR}" ]; then
        echo "Copying binaries to ${OUTPUT_DIR}..."
        mkdir -p "${OUTPUT_DIR}"
        cp executor_runner.js "${OUTPUT_DIR}/"
        cp executor_runner.wasm "${OUTPUT_DIR}/"
        echo "✅ Copied to ${OUTPUT_DIR}"
    fi

    # If running in Docker, also copy to /output volume
    if [ -f /.dockerenv ] && [ -d /output ]; then
        echo "Copying binaries to /output volume..."
        cp executor_runner.js /output/
        cp executor_runner.wasm /output/
        echo "✅ Copied to /output"
    fi
else
    echo "❌ Build failed: executor_runner.js or executor_runner.wasm not found"
    exit 1
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
