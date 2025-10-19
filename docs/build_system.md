# ExecuTorch Flutter Build System

This document describes the self-contained build system for the `executorch_flutter` plugin, which automatically downloads and compiles ExecuTorch from source during the Flutter build process.

## Architecture Overview

The build system uses **CMake FetchContent** (Android) and **Swift Package Manager** (iOS/macOS) to provide a seamless, zero-configuration build experience:

```
┌─────────────────────────────────────────────────────────────┐
│                     Flutter Build                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Android (Gradle)         iOS/macOS (SPM)                  │
│  ┌────────────┐            ┌──────────────┐                │
│  │ build.gradle├───────────►│Package.swift │                │
│  └─────┬──────┘            └──────┬───────┘                │
│        │                          │                         │
│        │ externalNativeBuild      │ dependencies            │
│        │                          │                         │
│        ▼                          ▼                         │
│  ┌────────────────────────────────────────────┐            │
│  │         src/CMakeLists.txt                 │            │
│  │  ┌──────────────────────────────────┐      │            │
│  │  │ executorch_options.cmake         │      │ (Android)  │
│  │  │ - Platform detection             │      │            │
│  │  │ - Backend configuration          │◄─────┤            │
│  │  │ - Environment variable overrides │      │            │
│  │  └──────────────────────────────────┘      │            │
│  │  ┌──────────────────────────────────┐      │            │
│  │  │ executorch_fetch.cmake           │      │            │
│  │  │ - FetchContent download          │      │            │
│  │  │ - ExecuTorch v1.0.0 from Git     │      │            │
│  │  └──────────────────────────────────┘      │            │
│  │                                            │            │
│  │  ┌──────────────────────────────────┐      │            │
│  │  │ C++ Wrapper Compilation          │      │            │
│  │  │ - Links ExecuTorch libraries     │      │            │
│  │  └──────────────────────────────────┘      │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│        iOS/macOS: SPM handles ExecuTorch XCFrameworks      │
│        Wrapper compiled directly from src/c_wrapper/        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Build Workflows

### Android Build Flow

1. **Gradle** reads `android/build.gradle`
2. **externalNativeBuild** calls `src/CMakeLists.txt`
3. **CMake** includes `executorch_options.cmake` (platform detection + backend config)
4. **CMake** includes `executorch_fetch.cmake` (downloads ExecuTorch v1.0.0 from Git)
5. **CMake** compiles C++ wrapper linking ExecuTorch libraries
6. **CMake** installs `libexecutorch_flutter_wrapper.so` to `jniLibs/`
7. **Gradle** packages native library into APK/AAB

### iOS/macOS Build Flow

1. **Flutter** triggers **CocoaPods** (iOS) or **Package.swift** (macOS)
2. **Package.swift** declares ExecuTorch XCFramework dependencies via SPM
3. **Package.swift** compiles C++ wrapper from `src/c_wrapper/` sources
4. **Xcode** links wrapper with ExecuTorch XCFrameworks
5. **Xcode** bundles framework into app

## Configuration Options

### Environment Variables

You can customize the build by setting environment variables before running `flutter build`:

#### Backend Selection

| Variable | Platforms | Default | Description |
|----------|-----------|---------|-------------|
| `EXECUTORCH_BUILD_XNNPACK` | All | `ON` | Enable XNNPACK CPU backend (optimized inference) |
| `EXECUTORCH_BUILD_COREML` | iOS/macOS | `ON` | Enable Apple CoreML backend (Neural Engine acceleration) |
| `EXECUTORCH_BUILD_MPS` | iOS/macOS | `ON` | Enable Apple Metal Performance Shaders backend (GPU acceleration) |
| `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` | All | `ON` | Use platform-optimized kernels (NEON on ARM, AVX on x86) |

**Examples**:
```bash
# Build without CoreML (faster build, smaller binary)
EXECUTORCH_BUILD_COREML=OFF flutter build ios

# Build with only XNNPACK (minimal configuration)
EXECUTORCH_BUILD_COREML=OFF EXECUTORCH_BUILD_MPS=OFF flutter build macos

# Build debug version with all backends
CMAKE_BUILD_TYPE=Debug flutter build apk
```

#### Android ABI Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `ANDROID_ABI_FILTER` | `arm64-v8a,x86_64` | Comma-separated list of Android ABIs to build |

**Examples**:
```bash
# Build only for arm64 (faster build, smaller APK)
ANDROID_ABI_FILTER=arm64-v8a flutter build apk

# Build for all ABIs
ANDROID_ABI_FILTER=arm64-v8a,armeabi-v7a,x86,x86_64 flutter build apk

# Build for emulator only
ANDROID_ABI_FILTER=x86_64 flutter build apk
```

**Supported ABIs**:
- `arm64-v8a` - 64-bit ARM (modern phones/tablets) - **Recommended**
- `armeabi-v7a` - 32-bit ARM (older devices)
- `x86_64` - 64-bit x86 (emulators) - **For development**
- `x86` - 32-bit x86 (older emulators)

#### Build Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: `Release` or `Debug` |

**Debug Mode Features**:
- Enables ExecuTorch logging
- Includes debug symbols
- Enables program verification
- Enables event tracer for profiling
- Larger binary size, slower inference

**Release Mode Features**:
- Disables logging (except errors)
- Strips debug symbols
- Disables verification (~20kB savings)
- Optimized for performance (-O2)
- Smaller binary size, faster inference

**Example**:
```bash
# Debug build with verbose logging
CMAKE_BUILD_TYPE=Debug flutter build apk

# Release build (default)
flutter build apk
```

### ExecuTorch Version Override

By default, the build system downloads **ExecuTorch v1.0.0**. You can override this:

| CMake Variable | Default | Description |
|----------------|---------|-------------|
| `EXECUTORCH_VERSION` | `v1.0.0` | Git tag/branch/SHA to download |
| `EXECUTORCH_GIT_REPOSITORY` | `https://github.com/pytorch/executorch.git` | ExecuTorch Git repository |
| `EXECUTORCH_SOURCE_DIR` | (none) | Path to local ExecuTorch source (skips download) |

**Using Local Source** (advanced):
```bash
# Clone ExecuTorch locally
git clone https://github.com/pytorch/executorch.git /path/to/executorch

# Build plugin with local source
cd example
flutter build apk \
  -DEXECUTORCH_SOURCE_DIR=/path/to/executorch
```

**Using Different Version**:
```bash
# Build with ExecuTorch main branch (bleeding edge)
cd example
flutter build apk \
  -DEXECUTORCH_VERSION=main
```

## Platform-Specific Details

### Android Configuration

**File**: `android/build.gradle`

- **Min SDK**: API 23 (Android 6.0)
- **NDK**: r21+ (C++17 support required)
- **CMake**: 3.22.1+
- **ABIs**: arm64-v8a (default), x86_64 (emulator)
- **STL**: `c++_shared` (required by ExecuTorch)
- **NEON**: Enabled on ARM architectures

**Default Backends**:
- ✅ XNNPACK (CPU inference)
- ✅ Optimized kernels (ARM NEON + Kleidi)

**Not Available**:
- ❌ CoreML (Apple only)
- ❌ MPS (Apple only)

### iOS Configuration

**File**: `ios/executorch_flutter/Package.swift`

- **Min Version**: iOS 13.0
- **Architectures**: arm64 (device only, no simulator)
- **Dependencies**: ExecuTorch XCFrameworks via SPM (branch `swiftpm-1.0.0`)

**Default Backends**:
- ✅ XNNPACK (CPU inference)
- ✅ CoreML (Neural Engine acceleration)
- ✅ MPS (GPU acceleration via Metal)
- ✅ Optimized kernels

**Limitations**:
- ❌ iOS Simulator not supported (ExecuTorch XCFrameworks only built for arm64 device)

### macOS Configuration

**File**: `macos/executorch_flutter/Package.swift`

- **Min Version**: macOS 11.0
- **Architectures**: arm64 (Apple Silicon only)
- **Dependencies**: ExecuTorch XCFrameworks via SPM (branch `swiftpm-1.0.0`)

**Default Backends**:
- ✅ XNNPACK (CPU inference)
- ✅ CoreML (Neural Engine acceleration on M1/M2/M3)
- ✅ MPS (GPU acceleration via Metal)
- ✅ Optimized kernels

**Limitations**:
- ❌ Intel Macs not supported (ExecuTorch XCFrameworks only built for Apple Silicon)

## Build Performance

### First Build (Cold Cache)

| Platform | Configuration | Time | Download Size |
|----------|---------------|------|---------------|
| Android | Release | ~8-12 min | ~500 MB (ExecuTorch source) |
| iOS | Release | ~5-7 min | ~100 MB (XCFrameworks via SPM) |
| macOS | Release | ~5-7 min | ~100 MB (XCFrameworks via SPM) |

### Incremental Build (Warm Cache)

| Platform | Configuration | Time |
|----------|---------------|------|
| Android | Any | ~30-60 sec |
| iOS | Any | ~20-40 sec |
| macOS | Any | ~20-40 sec |

**Cache Locations**:
- **Android**: `~/.gradle/caches/` (Gradle cache) + CMake build directory
- **iOS/macOS**: `~/Library/Caches/org.swift.swiftpm/` (SPM cache)

### Build Size Impact

| Configuration | Binary Size Impact |
|---------------|-------------------|
| All backends (default) | ~15-20 MB per ABI |
| XNNPACK only | ~12-15 MB per ABI |
| Debug build | +30-50% size increase |

**Optimization Tips**:
1. **Reduce ABIs**: Use `ANDROID_ABI_FILTER=arm64-v8a` to build only for modern devices
2. **Disable unused backends**: Set `EXECUTORCH_BUILD_COREML=OFF` if not using Neural Engine
3. **Release builds**: Always use Release mode for production (smaller, faster)

## Troubleshooting

### Common Issues

#### 1. **Build Fails: "ExecuTorch download failed"**

**Symptoms**:
```
CMake Error: Failed to download ExecuTorch from https://github.com/pytorch/executorch.git
```

**Solutions**:
- Check internet connection
- Verify Git is installed: `git --version`
- Use local source: `cmake -DEXECUTORCH_SOURCE_DIR=/path/to/executorch`
- Check firewall/proxy settings

#### 2. **Build Fails: "CMAKE_CXX_STANDARD 17 not supported"**

**Symptoms**:
```
CMake Error: CMAKE_CXX_COMPILER does not support C++17
```

**Solutions**:
- Update NDK to r21+ (Android)
- Update Xcode to 12+ (iOS/macOS)
- Check CMake version: `cmake --version` (need 3.22+)

#### 3. **Build Fails: "undefined reference to executorch symbols"**

**Symptoms**:
```
ld: undefined reference to `torch::executor::Module::load(...)`
```

**Solutions**:
- Clean build: `flutter clean && flutter pub get`
- Verify ExecuTorch was downloaded: Check CMake logs for FetchContent messages
- Check `executorch` target exists in ExecuTorch CMakeLists.txt

#### 4. **Build Slow: "First build takes >15 minutes"**

**Symptoms**:
- ExecuTorch download + compilation takes very long

**Solutions**:
- Use shallow clone (default): `EXECUTORCH_GIT_SHALLOW=TRUE`
- Use pre-built local source: Clone ExecuTorch once, reuse with `EXECUTORCH_SOURCE_DIR`
- Reduce ABIs: `ANDROID_ABI_FILTER=arm64-v8a`
- Disable unused backends: `EXECUTORCH_BUILD_COREML=OFF EXECUTORCH_BUILD_MPS=OFF`

#### 5. **Runtime Error: "Model failed to load"**

**Symptoms**:
```
ExecuTorchModelException: Failed to load model
```

**Solutions**:
- Verify model is valid `.pte` format (export with same ExecuTorch version)
- Check model bytes are loaded: `modelBytes.lengthInBytes > 0`
- Verify asset is listed in `pubspec.yaml` under `flutter.assets`
- Check ExecuTorch version compatibility (Android: v1.0.0, iOS/macOS: SPM 1.0.0)

### Debug Logging

Enable detailed build logs:

**Android**:
```bash
# Enable Gradle verbose logging
flutter build apk --verbose

# Enable CMake verbose logging
cd android
./gradlew assembleDebug --info --debug
```

**iOS**:
```bash
# Enable Xcode verbose logging
flutter build ios --verbose

# Manual Xcode build with logs
cd ios
xcodebuild -workspace Runner.xcworkspace -scheme Runner -configuration Debug -verbose
```

**macOS**:
```bash
# Enable verbose logging
flutter build macos --verbose
```

### Inspecting Build Artifacts

**Android**:
```bash
# List compiled native libraries
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep libexecutorch

# Expected output:
# lib/arm64-v8a/libexecutorch_flutter_wrapper.so
# lib/x86_64/libexecutorch_flutter_wrapper.so

# Check symbols in library
arm-linux-androideabi-nm build/intermediates/cmake/debug/obj/arm64-v8a/libexecutorch_flutter_wrapper.so | grep executorch
```

**iOS/macOS**:
```bash
# List frameworks
ls -la build/ios/Release-iphoneos/*.framework
ls -la build/macos/Build/Products/Release/*.framework

# Check symbols in framework
nm build/ios/Release-iphoneos/executorch_flutter.framework/executorch_flutter | grep executorch
```

## CI/CD Integration

The plugin includes a complete GitHub Actions workflow for automated builds with intelligent caching.

### GitHub Actions Setup

The workflow is located at `.github/workflows/build.yml` and runs on:
- **Push** to main, develop, or feature branches
- **Pull requests** to main/develop
- **Manual trigger** via GitHub UI

**Workflow Structure**:
1. **Analyze Job**: Dart formatting and analysis (runs first, gates other jobs)
2. **Build Jobs** (parallel):
   - **build-android**: Matrix strategy for arm64-v8a and x86_64
   - **build-ios**: iOS framework build
   - **build-macos**: macOS framework build
3. **Summary Job**: Aggregates results and reports status

### Caching Strategy

The workflow uses GitHub Actions cache to speed up builds:

#### Android Caching

```yaml
# Gradle cache (~80% faster builds)
- uses: actions/cache@v4
  with:
    path: |
      ~/.gradle/caches
      ~/.gradle/wrapper
    key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*') }}

# CMake cache (ExecuTorch build artifacts)
- uses: actions/cache@v4
  with:
    path: |
      android/.cxx
      ~/.cmake
    key: ${{ runner.os }}-cmake-${{ matrix.abi }}-${{ hashFiles('src/CMakeLists.txt') }}
```

**Performance Impact**:
- **First build**: ~30-40 minutes (downloads ExecuTorch, compiles from source)
- **Cached build**: ~8-12 minutes (reuses ExecuTorch artifacts)

#### iOS/macOS Caching

```yaml
# Swift Package Manager cache
- uses: actions/cache@v4
  with:
    path: |
      ~/Library/Caches/org.swift.swiftpm
      example/ios/.build
    key: ${{ runner.os }}-spm-ios-${{ hashFiles('ios/executorch_flutter/Package.swift') }}

# CocoaPods cache (iOS only)
- uses: actions/cache@v4
  with:
    path: |
      example/ios/Pods
      ~/Library/Caches/CocoaPods
    key: ${{ runner.os }}-pods-${{ hashFiles('example/ios/Podfile.lock') }}
```

**Performance Impact**:
- **First build**: ~20-25 minutes (downloads XCFrameworks)
- **Cached build**: ~5-7 minutes (reuses XCFrameworks)

### Parallel Execution

The workflow runs **3 platform builds in parallel** using GitHub Actions matrix strategy:

```yaml
jobs:
  build-android:
    strategy:
      matrix:
        abi: [arm64-v8a, x86_64]
    # Builds 2 Android ABIs in parallel

  build-ios:
    # Runs concurrently with Android and macOS

  build-macos:
    # Runs concurrently with Android and iOS
```

**Total CI Time**:
- **First run** (cold cache): ~30-40 minutes (limited by Android build)
- **Subsequent runs** (warm cache): ~8-12 minutes (limited by Android build)

### Build Verification

Each job verifies that native libraries/frameworks are created:

**Android**:
```yaml
- name: Check native library exists
  run: |
    LIB_PATH="example/build/app/intermediates/cmake/debug/obj/$ARCH_DIR/libexecutorch_flutter_wrapper.so"
    if [ ! -f "$LIB_PATH" ]; then
      echo "❌ ERROR: Native library not found"
      exit 1
    fi
```

**iOS/macOS**:
```yaml
- name: Check framework exists
  run: |
    FRAMEWORK_PATH="example/build/ios/Debug-iphoneos/executorch_flutter.framework"
    if [ ! -d "$FRAMEWORK_PATH" ]; then
      echo "❌ ERROR: Framework not found"
      exit 1
    fi
```

### Artifact Upload

Build artifacts are uploaded for inspection:

```yaml
- name: Upload Android artifacts
  uses: actions/upload-artifact@v4
  with:
    name: android-${{ matrix.abi }}-build
    path: |
      example/build/app/intermediates/cmake/debug/obj/${{ matrix.abi }}/libexecutorch_flutter_wrapper.so
    retention-days: 7
```

**Artifacts available**:
- `android-arm64-v8a-build` - Android arm64 library
- `android-x86_64-build` - Android x86_64 library
- `ios-build` - iOS framework
- `macos-build` - macOS framework

### Customizing CI Builds

You can customize the CI workflow by editing `.github/workflows/build.yml`:

**Add more ABIs** (Android):
```yaml
strategy:
  matrix:
    abi: [arm64-v8a, armeabi-v7a, x86_64, x86]
```

**Disable backends** (iOS/macOS):
```yaml
- name: Build iOS framework
  env:
    EXECUTORCH_BUILD_COREML: OFF
    EXECUTORCH_BUILD_MPS: OFF
  run: flutter build ios --debug --no-codesign
```

**Run integration tests**:
```yaml
- name: Run integration tests
  working-directory: example
  run: flutter test integration_test/
```

### Local CI Testing

Test the CI workflow locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act

# Run workflow locally (requires Docker)
act push

# Run specific job
act -j build-android

# Run with secrets
act push --secret-file .secrets
```

### CI/CD Best Practices

1. **Use matrix strategy** for parallel builds (faster CI)
2. **Cache aggressively** (Gradle, CMake, SPM, CocoaPods)
3. **Verify artifacts** after every build
4. **Upload artifacts** for debugging failures
5. **Set timeouts** to prevent runaway builds
6. **Use concurrency groups** to cancel outdated runs

### Monitoring Builds

**GitHub Actions UI**:
- View live build logs: `Actions` tab → Select workflow run
- Download artifacts: `Summary` page → `Artifacts` section
- Check cache usage: `Settings` → `Actions` → `Caches`

**Build Summary**:
The workflow generates a summary table:

| Platform | Status |
|----------|--------|
| Android | ✅ Success |
| iOS | ✅ Success |
| macOS | ✅ Success |

## Advanced Topics

### Custom ExecuTorch Backends

To add custom backends (e.g., Qualcomm QNN, MediaTek Neuron):

1. Modify `src/build_config/executorch_options.cmake`:
```cmake
if(ANDROID)
  set_option_with_env_override(EXECUTORCH_BUILD_QNN ON "Build Qualcomm QNN backend")
endif()
```

2. Link backend in `src/CMakeLists.txt`:
```cmake
if(EXECUTORCH_BUILD_QNN)
  target_link_libraries(executorch_flutter_wrapper PUBLIC qnn_executorch_backend)
endif()
```

3. Build with environment variable:
```bash
EXECUTORCH_BUILD_QNN=ON flutter build apk
```

### Cross-Compilation

For building on different host platforms:

**Build Android on Linux**:
```bash
flutter build apk
```

**Build iOS on macOS**:
```bash
flutter build ios --no-codesign
```

**Build macOS on macOS**:
```bash
flutter build macos
```

**Note**: iOS/macOS builds require macOS host with Xcode.

## FAQ

### Q: Do I need to install ExecuTorch manually?

**A**: No. The build system automatically downloads ExecuTorch v1.0.0 from GitHub during the first build.

### Q: Can I use a different ExecuTorch version?

**A**: Yes. Set `EXECUTORCH_VERSION=<tag>` when building, or use local source with `EXECUTORCH_SOURCE_DIR=/path/to/executorch`.

### Q: How do I reduce build time?

**A**:
1. Use `ANDROID_ABI_FILTER=arm64-v8a` (build only for modern devices)
2. Disable unused backends: `EXECUTORCH_BUILD_COREML=OFF`
3. Use pre-built local ExecuTorch source: `EXECUTORCH_SOURCE_DIR=/path/to/executorch`
4. Enable ccache (CI/CD): See [CI/CD Setup](#cicd-setup)

### Q: Why doesn't iOS Simulator work?

**A**: ExecuTorch XCFrameworks (SPM 1.0.0 branch) are only built for arm64 device architecture. To support simulator, ExecuTorch would need to provide x86_64 builds.

### Q: Can I use this plugin on Windows/Linux?

**A**: Not yet. Current support: Android (any host), iOS/macOS (macOS host only). Windows/Linux desktop support is planned for future releases.

### Q: How do I verify ExecuTorch is included?

**A**: Check build logs for:
```
-- ExecuTorch ready for linking
-- ExecuTorch Configuration complete
```

Or inspect native library symbols (see [Inspecting Build Artifacts](#inspecting-build-artifacts)).

### Q: What if my build fails with "executorch target not found"?

**A**: This means ExecuTorch's CMakeLists.txt didn't create the `executorch` target. Ensure you're using ExecuTorch v1.0.0 or later, which includes the CMake build system.

## Next Steps

- [End-User Quickstart](../README.md#quickstart) - Add plugin to your Flutter app
- [Example App](../example/README.md) - See working examples with YOLO and MobileNet
- [Python Model Export](../python/README.md) - Convert PyTorch models to ExecuTorch format

---

**Last Updated**: 2025-10-19
**Build System Version**: 1.0.0 (self-contained)
**ExecuTorch Version**: v1.0.0 (Android), SPM 1.0.0 (iOS/macOS)
