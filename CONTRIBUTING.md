# Contributing to ExecuTorch Flutter

Thank you for your interest in contributing to ExecuTorch Flutter! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Build Modes](#build-modes)
- [Local Build & Testing](#local-build--testing)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Submitting Changes](#submitting-changes)
- [Platform-Specific Guidelines](#platform-specific-guidelines)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/executorch_flutter.git
   cd executorch_flutter
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/abdelaziz-mahdy/executorch_flutter.git
   ```

## Development Setup

### Prerequisites

- Flutter SDK 3.38+ (first version with native assets hooks)
- Dart SDK 3.10.0 or later (required by `native_toolchain_cmake`, a direct
  dependency of `executorch_dart`)
- **Android Development**:
  - Android Studio with SDK API 23+
  - NDK for native development
- **iOS Development**:
  - macOS with Xcode 14+
  - iOS 13.0+ (device and simulator supported)
- **macOS Development**:
  - macOS 11.0+ (Big Sur or later)
  - Apple Silicon or Intel Mac
- **Windows Development**:
  - Windows 10+
  - Visual Studio 2022 with C++ workload
- **Linux Development**:
  - Ubuntu 20.04+ or equivalent
  - GCC/Clang, CMake, ninja-build

### Setup Steps

1. **Install dependencies**:
   ```bash
   flutter pub get
   ```

2. **Run the example app**:
   ```bash
   cd packages/executorch_flutter/example
   flutter run
   ```

## Project Structure

This repo is a [pub workspace](https://dart.dev/tools/pub/workspaces) with two
published packages under `packages/`:

```
executorch_flutter/                        # Repo root — pub workspace (not published)
├── pubspec.yaml                            # Workspace members + native build user_defines (see below)
├── packages/
│   ├── executorch_dart/                    # Pure-Dart core — owns the native build, no Flutter needed
│   │   ├── lib/
│   │   │   ├── executorch_dart.dart        # Full public API
│   │   │   ├── executorch_dart_shared.dart # ffi-free subset (see below)
│   │   │   └── src/
│   │   │       ├── ffi/                    # dart:ffi bindings and FFI layer
│   │   │       ├── generated/              # ffigen-generated FFI bindings
│   │   │       ├── build/                  # Native assets build hook
│   │   │       ├── executorch_model.dart
│   │   │       ├── executorch_inference.dart
│   │   │       └── executorch_errors.dart
│   │   ├── native/                         # Git submodule: C/C++ FFI library
│   │   │   ├── src/                        # FFI implementation
│   │   │   ├── cmake/                      # CMake build configuration
│   │   │   └── scripts/                    # Platform build scripts
│   │   └── example/                        # Pure-Dart CLI example (dart:io only)
│   └── executorch_flutter/                 # Flutter plugin — thin wrapper over executorch_dart
│       ├── lib/executorch_flutter.dart     # Public API: re-exports the core, routes native vs. web
│       └── example/                        # Full Flutter demo app
└── models/                                 # Git submodule: model export scripts + pre-exported .pte files
```

`executorch_dart` exposes two public libraries:
- `package:executorch_dart/executorch_dart.dart` — the full native API
  (`ExecuTorchModel`, `ExecuTorchLLM`, `BackendQuery`, etc.). Application code
  should import this one.
- `package:executorch_dart/executorch_dart_shared.dart` — the `dart:ffi`-free
  subset (tensor/model types, error hierarchy, processor base classes). It
  exists for one consumer: `executorch_flutter`'s web branch, which cannot
  pull `dart:ffi` into a dart2js compile. You only need this if you're
  working on the wrapper's web routing; don't import it from application code.

## Build Modes

The plugin supports three build modes for the native C/C++ library. Configure via `pubspec.yaml`:

> **This repo is a pub workspace — put `user_defines` in the root
> `pubspec.yaml`.** Pub workspaces read `hooks: user_defines:` only from the
> **workspace root** `pubspec.yaml`. The same block inside
> `packages/executorch_flutter/example/pubspec.yaml` (or any other member
> package) is silently ignored — the hook receives `{}` and the build quietly
> falls back to defaults, with no error to tell you why. This only matters
> when you're working in *this* repo; it doesn't affect a consumer
> application, which is its own root package. Every `pubspec.yaml` snippet
> below that this repo's own example apps use goes in the **workspace root**
> `pubspec.yaml`, not the example's own.

```yaml
hooks:
  user_defines:
    executorch_dart:
      build_mode: "prebuilt"   # or "local" or "source"
```

| Mode | Description | Speed | Use Case |
|------|-------------|-------|----------|
| **prebuilt** (default) | Downloads pre-built binaries from GitHub Releases | Fast (seconds) | Normal development, CI/CD |
| **local** | Uses locally compiled libraries from a directory you specify | Fast (seconds) | Testing native code changes, custom backends |
| **source** | Builds ExecuTorch from source via CMake | Slow (15-30 min) | Custom configurations, pointing at a local ExecuTorch checkout |

## Local Build & Testing

When contributing to the native C/C++ layer or testing upstream ExecuTorch changes
(e.g., bug fixes, new backends), use **local mode** to compile and test without
waiting for CI to build and publish new prebuilt binaries.

### Overview

The workflow is:

1. **Compile** the native library from a local ExecuTorch source using `compile-local.sh`
2. **Configure** the Flutter plugin to use `build_mode: "local"`
3. **Run** the example app on your target device

### Prerequisites

- CMake 3.18+
- Ninja (recommended): `brew install ninja` (macOS) or `apt install ninja-build` (Linux)
- **For Android**: Android NDK (`ANDROID_NDK_HOME` environment variable)
- **For Vulkan**: Vulkan SDK with `glslc` (`VULKAN_SDK` environment variable)

### Step 1: Compile the Native Library

Use the `compile-local.sh` script in `packages/executorch_dart/native/scripts/`:

```bash
cd packages/executorch_dart/native/scripts

# Build for Android arm64 with XNNPACK + Vulkan
./compile-local.sh \
  --executorch-source /path/to/executorch \
  --platform android \
  --arch arm64-v8a \
  --backends xnnpack,vulkan

# Build for macOS (host platform, auto-detected)
./compile-local.sh \
  --executorch-source /path/to/executorch

# Build for Android with custom NDK path
./compile-local.sh \
  --executorch-source /path/to/executorch \
  --platform android \
  --arch arm64-v8a \
  --backends xnnpack \
  --ndk ~/Android/Sdk/ndk/27.0.12077973
```

**Script options:**

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--executorch-source` | Yes | - | Path to local ExecuTorch source directory |
| `--platform` | No | Auto-detect host | `android`, `macos`, `linux`, `windows` |
| `--arch` | No | Auto-detect | `arm64-v8a`, `armeabi-v7a`, `x86_64`, `arm64`, `x64` |
| `--backends` | No | `xnnpack` | Comma-separated: `xnnpack`, `vulkan`, `coreml`, `mps` |
| `--build-type` | No | `Release` | `Release` or `Debug` |
| `--ndk` | Android only | `$ANDROID_NDK_HOME` | Path to Android NDK |

The output goes to
`packages/executorch_dart/native/local-builds/<platform>-<arch>-<backends>-<build_type>/`
with `lib/` and `include/` subdirectories.

### Step 2: Configure the Flutter Plugin

**Option A: Auto-detection** (recommended when using `compile-local.sh`)

The plugin automatically finds builds in
`packages/executorch_dart/native/local-builds/` matching your target
platform, architecture, and backends:

```yaml
# In the workspace root pubspec.yaml (see the callout above)
hooks:
  user_defines:
    executorch_dart:
      build_mode: "local"
      backends:
        - xnnpack
        - vulkan
```

**Option B: Explicit path**

Point directly at any directory containing `lib/` and `include/`:

```yaml
hooks:
  user_defines:
    executorch_dart:
      build_mode: "local"
      local_lib_dir: "/absolute/path/to/compiled/libs"
      backends:
        - xnnpack
        - vulkan
```

**Option C: Environment variable**

```bash
export EXECUTORCH_INSTALL_DIR="/path/to/compiled/libs"
# or
export EXECUTORCH_BUILD_MODE="local"
```

### Step 3: Run the App

```bash
cd packages/executorch_flutter/example
flutter run -d <your_device>
```

The build hook will use your local libraries instead of downloading prebuilts.

### Full Example: Testing an Upstream Fix

Say you want to test a fix in the upstream ExecuTorch Vulkan backend on Android:

```bash
# 1. Clone or navigate to your ExecuTorch source
cd ~/executorch
git checkout fix/my-vulkan-fix

# 2. Compile the native library for Android
cd /path/to/executorch_flutter/packages/executorch_dart/native/scripts
./compile-local.sh \
  --executorch-source ~/executorch \
  --platform android \
  --arch arm64-v8a \
  --backends xnnpack,vulkan

# 3. Configure the example app to use local build
# Edit the WORKSPACE ROOT pubspec.yaml (/path/to/executorch_flutter/pubspec.yaml)
# and add:
#   hooks:
#     user_defines:
#       executorch_dart:
#         build_mode: "local"
#         backends:
#           - xnnpack
#           - vulkan

# 4. Run on your Android device
cd /path/to/executorch_flutter/packages/executorch_flutter/example
flutter run -d <android_device_id>
```

### Switching Back to Prebuilt Mode

Remove or change the `build_mode` in your `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    executorch_dart:
      # build_mode: "local"    # comment out or remove
      backends:
        - xnnpack
```

Or explicitly set it back:

```yaml
hooks:
  user_defines:
    executorch_dart:
      build_mode: "prebuilt"
```

### Directory Structure

After running `compile-local.sh`, the output looks like:

```
packages/executorch_dart/native/
├── local-builds/                              # gitignored
│   ├── android-arm64-v8a-xnnpack-vulkan-release/
│   │   ├── lib/
│   │   │   └── libexecutorch_ffi.so
│   │   └── include/
│   │       └── executorch_ffi.h
│   └── macos-arm64-xnnpack-release/
│       ├── lib/
│       │   └── libexecutorch_ffi.dylib
│       └── include/
│           └── executorch_ffi.h
└── scripts/
    └── compile-local.sh
```

## Source Build (from Local ExecuTorch Checkout)

If you want Flutter's build system to compile ExecuTorch from source automatically
(instead of using pre-compiled libraries), use **source mode** with `executorch_source`
pointing at your local ExecuTorch checkout.

This is useful when:
- You're iterating on ExecuTorch C++ code and want `flutter run` to pick up changes
- You need a custom backend configuration not available in prebuilts
- You want the build integrated into Flutter's native assets pipeline

### Prerequisites

- Python 3.8+ with `pyyaml` package
- CMake 3.18+, Ninja
- A local ExecuTorch checkout with submodules:
  ```bash
  git clone --recursive https://github.com/pytorch/executorch.git
  ```
- **For Android**: Android NDK
- **For Vulkan**: `glslc` (from Vulkan SDK or `brew install shaderc`)

### Configuration

Add to your app's `pubspec.yaml` (the workspace root `pubspec.yaml` when
testing against this repo's own example — see the callout under "Build
Modes"):

```yaml
hooks:
  user_defines:
    executorch_dart:
      build_mode: "source"
      executorch_source: "/path/to/executorch"
      backends:
        - xnnpack
        - vulkan
```

Or use an environment variable:

```bash
export EXECUTORCH_SOURCE_DIR="/path/to/executorch"
```

### How It Works

1. Flutter's native assets hook detects `build_mode: "source"`
2. The CMake build system uses your local ExecuTorch as the source tree
   (skips downloading from GitHub)
3. ExecuTorch is compiled with the requested backends
4. The resulting `libexecutorch_ffi` is linked into your app

### Source vs Local Mode

| | Source Mode | Local Mode |
|---|------------|------------|
| **Compilation** | Done by Flutter's build system | Done by you (via `compile-local.sh`) |
| **When to use** | Iterating on ExecuTorch C++ code | Testing pre-compiled libraries |
| **Speed** | Slow first build (15-30 min), fast rebuilds | Always fast (seconds) |
| **Flexibility** | Full control over build options | Uses whatever was compiled |

### Example: Testing an ExecuTorch Patch

```bash
# 1. Clone ExecuTorch and apply your patch
git clone --recursive https://github.com/pytorch/executorch.git
cd executorch
git checkout fix/my-patch

# 2. Configure the example app
# Edit the WORKSPACE ROOT pubspec.yaml (/path/to/executorch_flutter/pubspec.yaml):
#   hooks:
#     user_defines:
#       executorch_dart:
#         build_mode: "source"
#         executorch_source: "/path/to/executorch"
#         backends:
#           - xnnpack

# 3. Run - Flutter builds ExecuTorch from your local source
cd /path/to/executorch_flutter/packages/executorch_flutter/example
flutter run -d <device>
```

**Note**: First build takes 15-30 minutes. Subsequent builds are incremental
and much faster (minutes). The build cache is preserved between runs.

### Troubleshooting Local/Source Builds

Prefer source mode driven by `pubspec.yaml` over manually running cmake/ninja in
old build directories — stale caches cause most of the failures below. The full
trap list lives in `packages/executorch_dart/native/CLAUDE.md` → "Local
Compilation". Highlights:

- **`install TARGETS given target "xnnpack-operator-delete" which does not exist`**:
  the XNNPACK submodule in your ExecuTorch checkout has drifted. Fix:
  `git submodule update backends/xnnpack/third-party/XNNPACK` in the checkout.
- **`python_wrapper.sh: Argument list too long` / `Undefined error: 0`** on
  reconfigure: old build dir hit the wrapper self-exec bug (fixed in
  `packages/executorch_dart/native/cmake/build_from_source.cmake`). Re-run
  cmake with `-DPYTHON_EXECUTABLE=/path/to/python3` once, or delete the build
  dir.
- **`MLX backend requires the Xcode Metal Toolchain`**: one-time install:
  `xcodebuild -downloadComponent MetalToolchain` (~700 MB), then rebuild.
- **`dlopen ... Library not loaded: /opt/homebrew/opt/libomp/lib/libomp.dylib`**
  (macOS, Metal/AOTI builds): the sandboxed app can't load torch's libomp from a
  system path. See `packages/executorch_dart/native/CLAUDE.md` for the
  bundling fix; xnnpack-only builds are unaffected.
- **App keeps loading old native code** after swapping libraries: hooks cache —
  run `flutter clean` in the app.
- **Model fails to load in a custom build**: check the build actually enables the
  backend the `.pte` was exported for (e.g. XNNPACK).

## Making Changes

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for major changes to discuss the approach
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Workflow

1. Make your changes following the [Code Standards](#code-standards)
2. Test your changes thoroughly on the example app
3. Update documentation as needed
4. Commit your changes with clear messages (see below)

### Commit Message Format

Follow the conventional commits format:

```
type(scope): brief description

Detailed description if needed

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no functional changes)
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Examples:**
```
feat(android): add support for XNNPACK backend
fix(ios): resolve memory leak in model disposal
docs(readme): update installation instructions
```

## Code Standards

### Dart Code

- Follow the [Dart Style Guide](https://dart.dev/guides/language/effective-dart/style)
- Run `dart format .` before committing
- Run `dart analyze` and fix all warnings
- Use meaningful variable and function names
- Add documentation comments for public APIs

### Kotlin Code (Android)

- Follow [Kotlin coding conventions](https://kotlinlang.org/docs/coding-conventions.html)
- Use Android Studio's built-in formatter
- Handle errors gracefully with try-catch blocks
- Use coroutines for async operations

### Swift Code (iOS/macOS)

- Follow [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)
- Use SwiftFormat for consistent formatting
- Use `async/await` for asynchronous operations
- Utilize Swift actors for thread safety

### FFI API Changes

If modifying the FFI layer:

1. **Update** C/C++ code in `packages/executorch_dart/native/src/`
2. **Update** header files in `packages/executorch_dart/native/src/executorch_ffi.h`
3. **Regenerate** FFI bindings:
   ```bash
   cd packages/executorch_dart && dart run ffigen
   ```
4. **Test** changes using integration tests (see below)

### Integration Testing

After making changes, run the integration tests:

```bash
cd packages/executorch_flutter/example
flutter test integration_test/models_integration_test.dart -d macos   # macOS
flutter test integration_test/models_integration_test.dart -d ios     # iOS
flutter test integration_test/models_integration_test.dart -d android # Android
flutter test integration_test/models_integration_test.dart -d windows # Windows
flutter test integration_test/models_integration_test.dart -d linux   # Linux
```

**Prerequisites**:
- Models are automatically downloaded from GitHub on first use
- To export models manually: `cd models/python && python3 main.py`

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues
   - Screenshots/videos for UI changes
   - Verification that changes work on affected platforms

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings from analyzer
- [ ] Verified on example app
- [ ] Works on all affected platforms

### Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Keep the PR updated with main branch
- Once approved, maintainers will merge

## Platform-Specific Guidelines

### Android

- **Minimum SDK**: API 23
- **Target Architecture**: arm64-v8a
- **Dependencies**: Use Gradle for dependency management
- **ExecuTorch**: Version 1.0.0-rc2 via AAR (`org.pytorch:executorch-android:1.0.0-rc2`)

### iOS

- **Minimum Version**: iOS 13.0
- **Architectures**: arm64 (device), arm64-simulator, x86_64-simulator
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK, CoreML (optional)

### macOS

- **Minimum Version**: macOS 11.0 (Big Sur)
- **Architectures**: arm64 (Apple Silicon), x86_64 (Intel)
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK, CoreML, MPS (Metal Performance Shaders)

### Windows

- **Minimum Version**: Windows 10
- **Architecture**: x64
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK

### Linux

- **Minimum Version**: Ubuntu 20.04+ or equivalent
- **Architecture**: x64
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK

## Need Help?

- 🐛 [GitHub Issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues) - Report bugs or ask questions
- 📖 [Documentation](https://github.com/abdelaziz-mahdy/executorch_flutter) - Read the docs

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

---

Thank you for contributing to ExecuTorch Flutter! 🎉
