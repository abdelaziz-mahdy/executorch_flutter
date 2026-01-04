# ExecuTorch Wasm Build Setup - Quick Reference

This document provides a quick reference for the Docker-based Wasm build system for `executorch_flutter` web platform support.

## What Was Created

### 1. Docker Build Environment
- **`Dockerfile.wasm`**: Complete build environment with Emscripten and ExecuTorch
  - Ubuntu 22.04 base
  - Emscripten SDK 3.1.50
  - ExecuTorch v1.0.1
  - All build dependencies

### 2. Build Scripts
- **`scripts/build_wasm.sh`**: Unified build script with auto-detection
  - Detects Docker availability
  - Falls back to native build if Docker not available
  - Supports `--docker` and `--native` flags for manual control

- **`scripts/build_wasm_in_container.sh`**: Low-level build script
  - Runs inside Docker container or natively
  - Configures CMake with Emscripten
  - Builds `executor_runner.js` and `executor_runner.wasm`
  - Copies outputs to `web/wasm/`

### 3. Configuration Files
- **`.dockerignore`**: Optimizes Docker build context
  - Excludes unnecessary files (build outputs, IDE configs, etc.)
  - Reduces Docker image size and build time

### 4. Documentation
- **`scripts/WASM_BUILD_README.md`**: Comprehensive build guide
  - Installation instructions (Docker and native)
  - Usage examples
  - Troubleshooting guide
  - CI/CD integration examples

- **`WEB_PLATFORM_DESIGN.md`**: Complete web platform design document
  - Architecture overview
  - Implementation strategy
  - API design
  - Performance considerations

## Quick Start

### Using Docker (Recommended)

```bash
# From package root
./scripts/build_wasm.sh
```

This will:
1. Check if Docker is installed and running
2. Build Docker image with all dependencies (~15-30 min first time)
3. Build Wasm binaries inside container (~5-10 min)
4. Copy `executor_runner.js` and `executor_runner.wasm` to `web/wasm/`

### Using Native Build

```bash
# Install Emscripten
cd ~
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Clone ExecuTorch
cd /path/to/executorch_flutter/..
git clone --branch v1.0.1 https://github.com/pytorch/executorch.git executorch-repo
cd executorch-repo
pip3 install -e .

# Build Wasm binaries
cd /path/to/executorch_flutter
./scripts/build_wasm.sh --native
```

## Build Outputs

After successful build:

```
web/wasm/
├── executor_runner.js      # Emscripten JavaScript glue code (~500 KB)
└── executor_runner.wasm    # WebAssembly binary (~2-5 MB)
```

These files are loaded at runtime by the web plugin.

## File Structure

```
executorch_flutter/
├── Dockerfile.wasm                        # Docker build environment
├── .dockerignore                          # Docker context exclusions
├── WEB_PLATFORM_DESIGN.md                 # Complete design document
├── WASM_BUILD_SETUP.md                    # This file
├── scripts/
│   ├── build_wasm.sh                      # Main build script (auto-detect)
│   ├── build_wasm_in_container.sh         # Low-level build script
│   └── WASM_BUILD_README.md               # Detailed build guide
└── web/
    └── wasm/                              # Build outputs
        ├── executor_runner.js
        └── executor_runner.wasm
```

## Common Commands

```bash
# Build with Docker (auto-detect)
./scripts/build_wasm.sh

# Force Docker build
./scripts/build_wasm.sh --docker

# Force native build
./scripts/build_wasm.sh --native

# Show help
./scripts/build_wasm.sh --help

# Clean previous build (Docker)
docker rmi executorch-wasm-builder

# Clean previous build (native)
rm -rf ../executorch-repo/cmake-out-wasm
```

## Troubleshooting

### Docker not found
```bash
# macOS
brew install --cask docker

# Ubuntu/Debian
sudo apt-get install docker.io
sudo systemctl start docker
```

### Docker permission denied
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Emscripten not found (native build)
```bash
# Install and activate emsdk
cd ~/emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### ExecuTorch repo not found (native build)
```bash
# Clone ExecuTorch repository
cd /path/to/executorch_flutter/..
git clone --branch v1.0.1 https://github.com/pytorch/executorch.git executorch-repo
```

## Next Steps After Building

1. **Implement Web Plugin**: Create web plugin implementation in `lib/src/web/`
2. **Test Web Platform**: Run example app with `flutter run -d chrome`
3. **Update Documentation**: Add web platform to README and package docs
4. **Publish Package**: Include Wasm binaries in published package

## Build Time Estimates

| Build Type | First Build | Subsequent Builds | Notes |
|------------|-------------|-------------------|-------|
| Docker | 15-30 min | 5-10 min | Caches dependencies |
| Native | 10-20 min | 3-5 min | If dependencies installed |

## Version Information

- **ExecuTorch**: v1.0.1 (matches package dependency)
- **Emscripten**: 3.1.50 (stable version)
- **CMake**: 3.16+ (minimum required)
- **Docker**: 20.10+ (recommended)

## CI/CD Integration

For automated builds in CI/CD:

```yaml
# GitHub Actions example
- name: Build Wasm Binaries
  run: ./scripts/build_wasm.sh --docker

- name: Upload Artifacts
  uses: actions/upload-artifact@v3
  with:
    name: wasm-binaries
    path: web/wasm/
```

## Resources

- **Detailed Build Guide**: `scripts/WASM_BUILD_README.md`
- **Design Document**: `WEB_PLATFORM_DESIGN.md`
- **ExecuTorch Wasm Docs**: https://github.com/pytorch/executorch/tree/main/examples/wasm
- **Emscripten Docs**: https://emscripten.org/docs/

## Support

- Check `scripts/WASM_BUILD_README.md` for detailed troubleshooting
- File issues at package repository
- ExecuTorch issues: https://github.com/pytorch/executorch
- Emscripten issues: https://github.com/emscripten-core/emscripten

---

**Created**: 2026-01-04
**Package Version**: 0.0.5
**Status**: Build system ready, web plugin implementation pending
