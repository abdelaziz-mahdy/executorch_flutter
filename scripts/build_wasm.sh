#!/bin/bash
# Unified script to build ExecuTorch Wasm binaries
# Supports both Docker and native builds

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEB_WASM_DIR="${PACKAGE_ROOT}/web/wasm"

echo -e "${BLUE}========================================"
echo "ExecuTorch Wasm Build Script"
echo "========================================"
echo -e "${NC}"

# Parse arguments
USE_DOCKER=""
FORCE_NATIVE=false
FORCE_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            FORCE_DOCKER=true
            shift
            ;;
        --native)
            FORCE_NATIVE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker      Force Docker build (even if native tools are available)"
            echo "  --native      Force native build (skip Docker, fail if dependencies missing)"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "By default, the script will auto-detect available tools and prefer Docker."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if Docker is available
check_docker() {
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            return 0
        else
            echo -e "${YELLOW}⚠️  Docker is installed but not running${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  Docker is not installed${NC}"
        return 1
    fi
}

# Function to check if Emscripten is available
check_emscripten() {
    if command -v emcc &> /dev/null; then
        echo -e "${GREEN}✅ Emscripten found: $(emcc --version | head -n 1)${NC}"
        return 0
    elif [ -f "${HOME}/emsdk/emsdk_env.sh" ]; then
        echo -e "${GREEN}✅ Emscripten found at ${HOME}/emsdk${NC}"
        return 0
    elif [ -f "/opt/emsdk/emsdk_env.sh" ]; then
        echo -e "${GREEN}✅ Emscripten found at /opt/emsdk${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Emscripten not found${NC}"
        return 1
    fi
}

# Function to check if ExecuTorch repository exists
check_executorch_repo() {
    local repo_path="${SCRIPT_DIR}/../../executorch-repo"
    if [ -d "${repo_path}" ]; then
        echo -e "${GREEN}✅ ExecuTorch repository found at ${repo_path}${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  ExecuTorch repository not found at ${repo_path}${NC}"
        return 1
    fi
}

# Decide build method
if [ "$FORCE_DOCKER" = true ]; then
    echo -e "${BLUE}Using Docker build (forced)${NC}"
    USE_DOCKER=true
elif [ "$FORCE_NATIVE" = true ]; then
    echo -e "${BLUE}Using native build (forced)${NC}"
    USE_DOCKER=false
else
    echo "Auto-detecting build environment..."
    echo ""

    if check_docker; then
        echo -e "${GREEN}✅ Docker is available${NC}"
        USE_DOCKER=true
    elif check_emscripten && check_executorch_repo; then
        echo -e "${GREEN}✅ Native build tools available${NC}"
        USE_DOCKER=false
    else
        echo ""
        echo -e "${RED}❌ Error: Neither Docker nor native build tools are available${NC}"
        echo ""
        echo "To use Docker build:"
        echo "  1. Install Docker: https://docs.docker.com/get-docker/"
        echo "  2. Start Docker daemon"
        echo "  3. Run this script again"
        echo ""
        echo "To use native build:"
        echo "  1. Install Emscripten: https://emscripten.org/docs/getting_started/downloads.html"
        echo "  2. Clone ExecuTorch: git clone https://github.com/pytorch/executorch.git ../executorch-repo"
        echo "  3. Run this script with --native flag"
        exit 1
    fi
fi

echo ""

# Create output directory
mkdir -p "${WEB_WASM_DIR}"

# Build with Docker
if [ "$USE_DOCKER" = true ]; then
    echo -e "${BLUE}Building ExecuTorch Wasm with Docker...${NC}"
    echo ""

    # Build Docker image
    echo "Building Docker image..."
    docker build \
        -f "${PACKAGE_ROOT}/Dockerfile.wasm" \
        -t executorch-wasm-builder \
        "${PACKAGE_ROOT}"

    echo ""
    echo "Running Docker container to build Wasm binaries..."

    # Run Docker container and mount output directory
    docker run --rm \
        -v "${WEB_WASM_DIR}:/output" \
        executorch-wasm-builder

    echo ""
    echo -e "${GREEN}✅ Docker build completed${NC}"

# Build natively
else
    echo -e "${BLUE}Building ExecuTorch Wasm natively...${NC}"
    echo ""

    # Set environment variables for native build
    export EXECUTORCH_ROOT="${SCRIPT_DIR}/../../executorch-repo"
    export OUTPUT_DIR="${WEB_WASM_DIR}"

    # Run build script
    bash "${SCRIPT_DIR}/build_wasm_in_container.sh"

    echo ""
    echo -e "${GREEN}✅ Native build completed${NC}"
fi

# Verify output files
echo ""
echo "Verifying build outputs..."
if [ -f "${WEB_WASM_DIR}/executorch.js" ] && [ -f "${WEB_WASM_DIR}/executorch.wasm" ]; then
    echo -e "${GREEN}✅ Wasm library binaries successfully generated:${NC}"
    echo ""
    ls -lh "${WEB_WASM_DIR}/executorch.js" "${WEB_WASM_DIR}/executorch.wasm"
    echo ""
    echo -e "${BLUE}Files are ready at: ${WEB_WASM_DIR}${NC}"
else
    echo -e "${RED}❌ Error: Wasm library binaries not found in ${WEB_WASM_DIR}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================"
echo "Build successful!"
echo "========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Implement web plugin in lib/src/web/"
echo "  2. Test with 'flutter run -d chrome'"
echo "  3. Update documentation with web platform support"
