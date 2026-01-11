# ExecuTorch Wasm Build Guide

This directory contains scripts to build ExecuTorch WebAssembly binaries with XNNPACK backend for the `executorch_flutter` web platform.

## Quick Start

### Option 1: Docker Build (Recommended)

The easiest way to build Wasm binaries is using Docker:

```bash
cd executorch_flutter
./scripts/build_wasm.sh
```

This will:
1. Auto-detect if Docker is available
2. Build a Docker image with Emscripten 4.0.10 and ExecuTorch v1.0.1
3. Build `executorch.js` and `executorch.wasm` with XNNPACK backend
4. Copy outputs to `web/wasm/`
5. Run `setup_web.dart` to copy files to example project

**Prerequisites**: Docker installed and running

### Option 2: Native Build

If you prefer building natively without Docker:

```bash
./scripts/build_wasm.sh --native
```

**Prerequisites**:
- Emscripten SDK (emsdk) installed
- ExecuTorch repository cloned at `../executorch-repo`
- CMake, Python 3, build tools

## Installation

### Docker Installation

**macOS**:
```bash
brew install --cask docker
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
```

**Windows**:
Download from https://docs.docker.com/desktop/install/windows-install/

### Native Dependencies Installation

If building natively, install these dependencies:

#### 1. Install Emscripten SDK

```bash
# Clone emsdk
cd ~
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# Install and activate latest version
./emsdk install latest
./emsdk activate latest

# Add to PATH (add to ~/.bashrc or ~/.zshrc for persistence)
source ./emsdk_env.sh
```

#### 2. Clone ExecuTorch Repository

```bash
cd /path/to/executorch_flutter/..
git clone --branch v1.0.1 https://github.com/pytorch/executorch.git executorch-repo
cd executorch-repo
pip3 install -e .
```

#### 3. Install Build Tools

**macOS**:
```bash
brew install cmake ninja python3
```

**Ubuntu/Debian**:
```bash
sudo apt-get install build-essential cmake ninja-build python3 python3-pip
```

## Build Scripts

### `build_wasm.sh`

Main build script with auto-detection.

**Usage**:
```bash
./scripts/build_wasm.sh [OPTIONS]

Options:
  --docker      Force Docker build
  --native      Force native build
  --help        Show help message
```

**Examples**:
```bash
# Auto-detect (prefers Docker if available)
./scripts/build_wasm.sh

# Force Docker build
./scripts/build_wasm.sh --docker

# Force native build
./scripts/build_wasm.sh --native
```

### `build_wasm_in_container.sh`

Low-level build script (called by `build_wasm.sh` or Docker container).

**Environment Variables**:
- `EXECUTORCH_ROOT`: Path to ExecuTorch repository (default: `../executorch-repo`)
- `OUTPUT_DIR`: Where to copy built binaries (default: auto-detected)

**Direct Usage** (advanced):
```bash
export EXECUTORCH_ROOT=/path/to/executorch
export OUTPUT_DIR=/path/to/output
./scripts/build_wasm_in_container.sh
```

## Build Outputs

Successful build generates:

```
web/wasm/
├── executorch.js      (~140 KB)  - Emscripten JavaScript glue code
└── executorch.wasm    (~3.3 MB)  - WebAssembly binary with XNNPACK
```

These files are bundled with the package and loaded at runtime by the web plugin.

## Troubleshooting

### Docker Issues

**Problem**: `docker: command not found`
```bash
# Install Docker (see Installation section above)
```

**Problem**: `Cannot connect to the Docker daemon`
```bash
# Start Docker daemon
sudo systemctl start docker  # Linux
# Or start Docker Desktop app (macOS/Windows)
```

**Problem**: Permission denied while connecting to Docker daemon
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Native Build Issues

**Problem**: `emcc: command not found`
```bash
# Source Emscripten environment
source ~/emsdk/emsdk_env.sh

# Or install emsdk (see Installation section)
```

**Problem**: `ExecuTorch repository not found`
```bash
# Clone ExecuTorch
cd /path/to/executorch_flutter/..
git clone --branch v1.0.1 https://github.com/pytorch/executorch.git executorch-repo
```

**Problem**: CMake configuration fails
```bash
# Install dependencies
# macOS:
brew install cmake ninja

# Ubuntu/Debian:
sudo apt-get install cmake ninja-build build-essential
```

**Problem**: Build fails with "ninja: build stopped: subcommand failed"
```bash
# Clean and rebuild
rm -rf /path/to/executorch-repo/cmake-out-wasm
./scripts/build_wasm.sh --native
```

### Build Output Issues

**Problem**: `executor_runner.js` or `executor_runner.wasm` not found
```bash
# Check build logs for errors
# Verify all dependencies installed
# Try rebuilding with verbose output:
cmake --build cmake-out-wasm --verbose
```

**Problem**: Files are too large
```bash
# This is expected - Wasm binaries are 2-5 MB
# They will be compressed by web servers (gzip/brotli)
```

## Advanced Usage

### Customizing Build Configuration

Edit `build_wasm_in_container.sh` to modify CMake options:

```bash
emcmake cmake \
    -DEXECUTORCH_PAL_DEFAULT=posix \
    -DCMAKE_BUILD_TYPE=Release \           # Or Debug for debugging
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=OFF \      # Disable for smaller binary
    ..
```

### Building for Specific ExecuTorch Version

**Docker**:
Edit `Dockerfile.wasm`:
```dockerfile
ENV EXECUTORCH_VERSION=v1.0.1  # Change version here
```

**Native**:
```bash
cd ../executorch-repo
git checkout v1.0.1  # or any other version/branch
./scripts/build_wasm.sh --native
```

### Using Custom Emscripten Version

**Docker**:
Edit `Dockerfile.wasm`:
```dockerfile
ENV EMSDK_VERSION=3.1.50  # Change version here
```

**Native**:
```bash
cd ~/emsdk
./emsdk install 3.1.50
./emsdk activate 3.1.50
source ./emsdk_env.sh
./scripts/build_wasm.sh --native
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Wasm

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Build Wasm with Docker
        run: |
          cd executorch_flutter
          ./scripts/build_wasm.sh --docker

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: wasm-binaries
          path: executorch_flutter/web/wasm/
```

### Caching Docker Builds

```bash
# Save Docker image
docker save executorch-wasm-builder > executorch-wasm-builder.tar

# Load Docker image (on another machine or CI)
docker load < executorch-wasm-builder.tar

# Run build without rebuilding image
docker run --rm -v "$(pwd)/web/wasm:/output" executorch-wasm-builder
```

## Performance Notes

### Build Times

- **Docker (first build)**: 15-30 minutes (downloads dependencies, builds Emscripten)
- **Docker (cached)**: 5-10 minutes (reuses cached layers)
- **Native (first build)**: 10-20 minutes (if dependencies already installed)
- **Native (rebuild)**: 3-5 minutes (incremental build)

### Optimization Tips

1. **Docker**: Use BuildKit for faster builds
   ```bash
   DOCKER_BUILDKIT=1 docker build -f Dockerfile.wasm -t executorch-wasm-builder .
   ```

2. **Native**: Use ccache for faster recompilation
   ```bash
   brew install ccache  # macOS
   sudo apt-get install ccache  # Ubuntu
   export CC="ccache gcc"
   export CXX="ccache g++"
   ```

3. **Parallel Builds**: Adjust `-j` flag in `build_wasm_in_container.sh`
   ```bash
   cmake --build . -j8  # Use 8 cores instead of $(nproc)
   ```

## File Structure

```
executorch_flutter/
├── Dockerfile.wasm                      # Docker image definition
├── .dockerignore                        # Docker build context exclusions
├── scripts/
│   ├── build_wasm.sh                    # Main build script
│   ├── build_wasm_in_container.sh       # Low-level build script
│   └── WASM_BUILD_README.md             # This file
└── web/
    └── wasm/                            # Build outputs
        ├── executorch.js
        └── executorch.wasm
```

## Version Information

- **ExecuTorch**: v1.0.1 (configurable in Dockerfile.wasm)
- **Emscripten**: 4.0.10 (configurable in Dockerfile.wasm)
- **Backend**: XNNPACK with Wasm SIMD
- **Target**: WebAssembly with SIMD extensions

## FAQ

**Q: Do I need to rebuild Wasm binaries every time?**
A: No, only when upgrading ExecuTorch version or changing build configuration.

**Q: Can I use pre-built binaries?**
A: Yes, the package will include pre-built binaries for convenience. These scripts are for maintainers or advanced users who want to rebuild.

**Q: Which build method should I use?**
A: Docker is recommended for most users (reproducible, isolated). Use native if you already have Emscripten installed or need faster incremental builds.

**Q: How do I debug build failures?**
A: Check build logs, ensure all dependencies are installed, and try cleaning build directories before rebuilding.

**Q: Can I cross-compile for different architectures?**
A: WebAssembly is architecture-independent. The same binaries work on all platforms (x86, ARM, etc.).

**Q: What's the binary size?**
A: ~2-5 MB total (both files). Web servers will compress with gzip/brotli (typically 40-60% reduction).

## Support

- **Package Issues**: File issues at `executorch_flutter` repository
- **Build Issues**: Check Troubleshooting section above
- **ExecuTorch Issues**: https://github.com/pytorch/executorch
- **Emscripten Issues**: https://github.com/emscripten-core/emscripten

---

**Last Updated**: 2026-01-08
**Package Version**: 0.0.5
**ExecuTorch Version**: 1.0.1
