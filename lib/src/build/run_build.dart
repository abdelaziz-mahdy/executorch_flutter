// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

// ignore_for_file: avoid_print

/// CMake build orchestration for executorch_flutter native assets.
///
/// This file implements the build hook for compiling the native C/C++
/// library using CMake and native_toolchain_cmake.
///
/// ## Build Modes
///
/// - **prebuilt** (default): Downloads pre-built binaries from GitHub Releases.
///   Fast builds, no Python required.
///
/// - **local**: Uses locally compiled ExecuTorch libraries from a directory
///   you specify. Useful for testing custom builds (e.g., with Vulkan fixes).
///   The directory must contain `lib/` and `include/` subdirectories.
///
/// - **source**: Builds ExecuTorch from source using FetchContent.
///   Slower but supports custom backend configurations.
///   Requires Python 3.8+ with pyyaml package.
///
/// ## Configuration via pubspec.yaml
/// ```yaml
/// hooks:
///   user_defines:
///     executorch_flutter:
///       debug: true
///       build_mode: "prebuilt"  # or "local" or "source"
///       # For local mode: path to directory with lib/ and include/
///       local_lib_dir: "/path/to/compiled/executorch"
///       # For source mode: path to local ExecuTorch checkout
///       executorch_source: "/path/to/executorch"
///       backends:
///         - xnnpack
///         - coreml
///         - mps
///         - vulkan
/// ```
///
/// Note: The ExecuTorch version is fixed per package version and cannot be
/// overridden. Use a different package version for different ExecuTorch
/// versions.
///
/// ## Environment Variables
/// - `EXECUTORCH_BUILD_MODE`: Override build mode
///   ("prebuilt", "local", or "source")
/// - `EXECUTORCH_CACHE_DIR`: Custom cache directory
/// - `EXECUTORCH_INSTALL_DIR`: Path to local installation
///   (used by local mode)
/// - `EXECUTORCH_SOURCE_DIR`: Path to local ExecuTorch checkout
///   (used by source mode, alternative to executorch_source)
/// - `EXECUTORCH_DISABLE_DOWNLOAD`: Set to "1" to skip
///   pre-built download
library;

import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

import '../version.dart' show executorchVersion;

/// Name of the native library.
const String _libraryName = 'executorch_ffi';

/// Package name.
const String _packageName = 'executorch_flutter';

/// Default prebuilt release version (our release tag for prebuilt downloads).
/// This includes a build iteration suffix (e.g., 1.1.0.1) to support multiple
/// releases for the same ExecuTorch version.
const String _defaultPrebuiltVersion = '$executorchVersion.3';

/// Default build mode.
const String _defaultBuildMode = 'prebuilt';

/// Minimum required Python version (major.minor) - only for source builds.
const List<int> _minPythonVersion = [3, 8];

/// Run the native assets build.
///
/// This function is called from hook/build.dart and performs:
/// 1. Build mode detection (prebuilt vs source)
/// 2. Python dependency validation (source mode only)
/// 3. Platform detection
/// 4. Backend configuration from user_defines
/// 5. CMake build orchestration
/// 6. Code asset registration
Future<void> runBuild(BuildInput input, BuildOutputBuilder output) async {
  // Check if code assets are expected (not for web builds)
  if (!input.config.buildCodeAssets) {
    // Web builds don't use native code assets - skip build
    // The web implementation uses JS interop instead
    return;
  }

  final packagePath = Directory(await getPackagePath(_packageName));
  final targetOS = input.config.code.targetOS;
  final targetArch = input.config.code.targetArchitecture;

  // Get user defines
  final userDefines = input.userDefines;
  final debugMode = userDefines['debug'] as bool? ?? false;

  // Set up logger
  hierarchicalLoggingEnabled = true;
  final logger = Logger('')
    ..level = Level.ALL
    ..onRecord.listen((record) {
      final message = record.message;
      if (message.isNotEmpty) {
        if (debugMode) {
          stderr.write(message);
        } else {
          print(message);
        }
      }
    });

  // Print build header
  _printBuildHeader(logger, targetOS, targetArch);

  // Determine build mode
  final buildMode = Platform.environment['EXECUTORCH_BUILD_MODE'] ??
      userDefines['build_mode'] as String? ??
      _defaultBuildMode;

  final isSourceBuild = buildMode == 'source';
  final isLocalBuild = buildMode == 'local';
  logger.info('[executorch_flutter] Build mode: $buildMode\n');

  // Resolve local library directory for local mode
  String? localLibDir;
  if (isLocalBuild) {
    localLibDir = Platform.environment['EXECUTORCH_INSTALL_DIR'] ??
        userDefines['local_lib_dir'] as String?;

    // Auto-detect: look in native/local-builds/ for matching build
    if (localLibDir == null || localLibDir.isEmpty) {
      localLibDir = _autoDetectLocalBuild(
        packagePath.path,
        targetOS,
        targetArch,
        _getBackendDefines(input, targetOS),
        debugMode,
        logger,
      );
    }

    if (localLibDir == null || localLibDir.isEmpty) {
      throw Exception('''
[executorch_flutter] ERROR: No local build found!

When using build_mode: "local", you need compiled libraries.

Option 1: Build with compile-local.sh:
  cd ${packagePath.path}/native/scripts
  ./compile-local.sh \\
    --executorch-source /path/to/executorch \\
    --platform ${_osToPlatformName(targetOS)} \\
    --arch ${targetArch.name} \\
    --backends xnnpack,vulkan

Option 2: Specify path manually in pubspec.yaml:
  hooks:
    user_defines:
      executorch_flutter:
        build_mode: "local"
        local_lib_dir: "/path/to/compiled/libs"

Option 3: Set environment variable:
  export EXECUTORCH_INSTALL_DIR="/path/to/compiled/libs"
''');
    }

    // Verify the directory exists and has expected structure
    final libSubdir = Directory('$localLibDir/lib');
    if (!Directory(localLibDir).existsSync()) {
      throw Exception('''
[executorch_flutter] ERROR: Local library directory not found!

  Path: $localLibDir

Please verify the path exists and contains compiled libraries.
''');
    }
    if (!libSubdir.existsSync()) {
      throw Exception('''
[executorch_flutter] ERROR: Missing lib/ subdirectory!

  Path: $localLibDir
  Expected: $localLibDir/lib/

Run compile-local.sh to build, or check your path.
''');
    }

    logger.info(
      '[executorch_flutter] Local library dir: $localLibDir\n',
    );
  }

  // Resolve ExecuTorch source directory for source mode
  String? executorchSourceDir;
  if (isSourceBuild) {
    executorchSourceDir = Platform.environment['EXECUTORCH_SOURCE_DIR'] ??
        userDefines['executorch_source'] as String?;

    if (executorchSourceDir != null && executorchSourceDir.isNotEmpty) {
      final sourceDir = Directory(executorchSourceDir);
      if (!sourceDir.existsSync()) {
        throw Exception('''
[executorch_flutter] ERROR: ExecuTorch source directory not found!

  Path: $executorchSourceDir

Please verify the path to your local ExecuTorch checkout.
''');
      }
      logger.info(
        '[executorch_flutter] ExecuTorch source: $executorchSourceDir\n',
      );
    }
  }

  // Get prebuilt version (for prebuilt downloads)
  final prebuiltVersion =
      userDefines['prebuilt_version'] as String? ?? _defaultPrebuiltVersion;
  logger.info('[executorch_flutter] ExecuTorch version: v$executorchVersion\n');
  if (!isSourceBuild && !isLocalBuild) {
    logger.info('[executorch_flutter] Prebuilt version: v$prebuiltVersion\n');
  }

  // Step 1: Python dependencies (only for source builds)
  String? pythonExecutable;
  if (isSourceBuild) {
    logger.info(
      '\n[executorch_flutter] Step 1/5: Checking Python dependencies\n',
    );
    final pythonInfo = await _verifyPythonDependencies(logger);
    pythonExecutable = pythonInfo.executable;
  } else {
    logger.info(
      '\n[executorch_flutter] Step 1/5: Skipping Python check ($buildMode mode)\n',
    );
  }

  // Step 2: Configure backends
  logger.info('[executorch_flutter] Step 2/5: Configuring backends\n');
  final backendDefines = _getBackendDefines(input, targetOS);
  _logBackendConfiguration(logger, backendDefines);

  // Step 3: Configure CMake generator
  logger.info('[executorch_flutter] Step 3/5: Configuring build system\n');

  final generator = switch (targetOS) {
    OS.linux => Generator.ninja,
    OS.macOS || OS.iOS => Generator.xcode,
    OS.windows => Generator.defaultGenerator,
    OS.android => Generator.ninja,
    _ => throw ArgumentError.value(
        targetOS,
        'targetOS',
        'Unsupported target OS',
      ),
  };
  logger.info('[executorch_flutter]   Generator: ${generator.name}\n');

  // Cache directory
  final cacheDir = Platform.environment['EXECUTORCH_CACHE_DIR'];
  if (cacheDir != null && cacheDir.isNotEmpty) {
    logger.info('[executorch_flutter]   Cache directory: $cacheDir\n');
  }

  // Determine CMake build type based on user_defines debug flag
  // When debug: true is set in pubspec.yaml, use Debug prebuilt libs
  final cmakeBuildType = debugMode ? 'Debug' : 'Release';
  logger.info('[executorch_flutter]   CMake build type: $cmakeBuildType\n');

  // Create CMake builder - uses native/ directory (submodule)
  //
  // For local mode, we still use "prebuilt" as the CMake build mode
  // but disable downloading and point to the local directory instead.
  final cmakeBuildMode = isLocalBuild ? 'prebuilt' : buildMode;

  final builder = CMakeBuilder.create(
    logLevel: debugMode ? LogLevel.DEBUG : LogLevel.STATUS,
    appleArgs: const AppleBuilderArgs(enableArc: false),
    name: _libraryName,
    sourceDir: packagePath.uri.resolve('native/'),
    targets: ['install'],
    generator: generator,
    defines: {
      // CMake build type (determines which prebuilt to download)
      'CMAKE_BUILD_TYPE': cmakeBuildType,
      // Build mode (local uses prebuilt path with download disabled)
      'EXECUTORCH_BUILD_MODE': cmakeBuildMode,
      // ExecuTorch source version (for source builds)
      'EXECUTORCH_VERSION': executorchVersion,
      // Prebuilt release version (for prebuilt downloads)
      'EXECUTORCH_PREBUILT_VERSION': prebuiltVersion,
      // Local mode: disable download and use local directory
      if (isLocalBuild) 'EXECUTORCH_DISABLE_DOWNLOAD': 'ON',
      if (isLocalBuild && localLibDir != null)
        'EXECUTORCH_INSTALL_DIR': localLibDir,
      // Python executable (only for source builds)
      if (isSourceBuild && pythonExecutable != null)
        'PYTHON_EXECUTABLE': pythonExecutable,
      // Cache directory / source directory
      // EXECUTORCH_CACHE_DIR expects the parent of the executorch/ directory.
      // If user provides a source dir, use its parent as cache dir.
      if (isSourceBuild && executorchSourceDir != null)
        'EXECUTORCH_CACHE_DIR': Directory(executorchSourceDir).parent.path
      else if (cacheDir != null && cacheDir.isNotEmpty)
        'EXECUTORCH_CACHE_DIR': cacheDir,
      // Platform-specific deployment targets
      if (targetOS == OS.macOS) 'CMAKE_OSX_DEPLOYMENT_TARGET': '11.0',
      if (targetOS == OS.iOS) 'CMAKE_OSX_DEPLOYMENT_TARGET': '13.0',
      // Install prefix
      'CMAKE_INSTALL_PREFIX':
          input.outputDirectory.resolve('install/').toFilePath(),
      // Backend defines
      ...backendDefines,
    },
  );

  // Step 4: Build
  if (isSourceBuild) {
    logger
      ..info('[executorch_flutter] Step 4/5: Building from source\n')
      ..info(
        '[executorch_flutter]   This may take 15-30 minutes on '
        'first build...\n',
      )
      ..info('[executorch_flutter]   (Faster after first build with cache)\n');
  } else if (isLocalBuild) {
    logger
      ..info(
        '[executorch_flutter] Step 4/5: Using local pre-built '
        'binaries\n',
      )
      ..info(
        '[executorch_flutter]   Local directory: $localLibDir\n',
      );
  } else {
    logger
      ..info(
        '[executorch_flutter] Step 4/5: Building with pre-built '
        'binaries\n',
      )
      ..info(
        '[executorch_flutter]   Downloading and linking pre-built '
        'ExecuTorch...\n',
      );
  }

  await builder.run(input: input, output: output, logger: logger);

  // Declare file dependencies so Flutter re-runs the build hook when native
  // sources change. Without this, Flutter caches the hook output and skips
  // re-invocation entirely -- even when C/C++ sources are modified. Once
  // re-invoked, CMake handles incremental compilation automatically.
  await _addBuildDependencies(
    output,
    packagePath,
    isSourceBuild: isSourceBuild,
    executorchSourceDir: executorchSourceDir,
    logger: logger,
  );

  // Step 5: Register code assets
  logger.info('[executorch_flutter] Step 5/5: Registering native assets\n');

  final installDir = input.outputDirectory.resolve('install/');

  await output.findAndAddCodeAssets(
    input,
    outDir: installDir,
    names: {_libraryName: '$_packageName.dart'},
  );

  // Bundle any additional shared libraries from the install directory.
  // This handles runtime dependencies like MoltenVK (macOS Vulkan),
  // Vulkan loader libraries, or other platform-specific dependencies
  // that the FFI library loads via dlopen() at runtime.
  await _registerAdditionalLibraries(
    input,
    output,
    installDir,
    targetOS,
    logger,
  );

  _printBuildSuccess(logger);
}

/// Get backend CMake defines from user_defines configuration.
Map<String, String?> _getBackendDefines(BuildInput input, OS targetOS) {
  final userDefines = input.userDefines;
  final backends = userDefines['backends'] as List?;

  // Platform support for each backend
  final isApplePlatform = targetOS == OS.iOS || targetOS == OS.macOS;
  final supportsCoreml = isApplePlatform;
  // Metal (AOTI) backend is macOS-desktop only (replaces the deprecated MPS).
  final supportsMetal = targetOS == OS.macOS;
  // Vulkan available on all native platforms (native assets don't run for web)
  // Note: On Apple platforms, Vulkan via MoltenVK may crash - use at own risk
  const supportsVulkan = true;

  // Enable backends: user-specified AND platform-supported
  // XNNPACK is available on all platforms
  final enableXnnpack = backends?.contains('xnnpack') ?? true;

  // CoreML only on Apple platforms
  final enableCoreml =
      supportsCoreml && (backends?.contains('coreml') ?? isApplePlatform);

  // Metal on macOS only (replaces the deprecated MPS backend). A legacy 'mps'
  // request is treated as 'metal' so existing configs keep working on macOS.
  final wantsMetal = backends == null
      ? null
      : backends.contains('metal') || backends.contains('mps');
  final enableMetal = supportsMetal && (wantsMetal ?? (targetOS == OS.macOS));

  // Vulkan opt-in and only on supported platforms
  final enableVulkan =
      supportsVulkan && (backends?.contains('vulkan') ?? false);

  final enableQnn = backends?.contains('qnn') ?? false;

  return {
    'ET_BUILD_XNNPACK': enableXnnpack ? 'ON' : 'OFF',
    'ET_BUILD_COREML': enableCoreml ? 'ON' : 'OFF',
    'ET_BUILD_METAL': enableMetal ? 'ON' : 'OFF',
    'ET_BUILD_VULKAN': enableVulkan ? 'ON' : 'OFF',
    'ET_BUILD_QNN': enableQnn ? 'ON' : 'OFF',
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Python information returned by [_verifyPythonDependencies].
class _PythonInfo {
  const _PythonInfo({
    required this.executable,
    required this.version,
    required this.pyyamlVersion,
  });

  final String executable;
  final String version;
  final String pyyamlVersion;
}

/// Print build header with target information.
void _printBuildHeader(Logger logger, OS targetOS, Architecture? targetArch) {
  final archSuffix = targetArch != null ? ' (${targetArch.name})' : '';
  logger
    ..info('\n')
    ..info('[executorch_flutter] ═══════════════════════════════════════\n')
    ..info('[executorch_flutter]  ExecuTorch Flutter Native Build\n')
    ..info('[executorch_flutter] ═══════════════════════════════════════\n')
    ..info('[executorch_flutter]  Target: ${targetOS.name}$archSuffix\n')
    ..info('[executorch_flutter] ───────────────────────────────────────\n');
}

/// Verify Python dependencies (Python 3.8+ with pyyaml).
///
/// Only called for source builds.
/// Throws [Exception] with user-friendly message if dependencies not met.
Future<_PythonInfo> _verifyPythonDependencies(Logger logger) async {
  // Find Python executable
  final pythonNames = Platform.isWindows
      ? ['python.exe', 'python3.exe', 'py.exe']
      : ['python3', 'python'];

  String? pythonExecutable;
  String? pythonVersion;

  for (final name in pythonNames) {
    try {
      final result = await Process.run(
          name,
          [
            '--version',
          ],
          runInShell: Platform.isWindows);
      if (result.exitCode == 0) {
        pythonExecutable = name;
        final output = (result.stdout as String).trim();
        pythonVersion = output.replaceFirst('Python ', '');
        break;
      }
    } catch (_) {
      // Try next Python name
    }
  }

  if (pythonExecutable == null || pythonVersion == null) {
    throw Exception('''
[executorch_flutter] ERROR: Python not found!

Building from source requires Python 3.8+ for ExecuTorch code generation.

Options:
1. Install Python 3.8+:
   - macOS: brew install python3
   - Ubuntu/Debian: sudo apt install python3
   - Windows: Download from https://python.org

2. Or use pre-built binaries (no Python required):
   In pubspec.yaml:
   hooks:
     user_defines:
       executorch_flutter:
         build_mode: "prebuilt"
''');
  }

  // Check Python version
  final versionParts = pythonVersion.split('.').map(int.tryParse).toList();
  if (versionParts.length >= 2 &&
      versionParts[0] != null &&
      versionParts[1] != null) {
    final major = versionParts[0]!;
    final minor = versionParts[1]!;
    if (major < _minPythonVersion[0] ||
        (major == _minPythonVersion[0] && minor < _minPythonVersion[1])) {
      throw Exception('''
[executorch_flutter] ERROR: Python version too old!

Found: Python $pythonVersion
Required: Python ${_minPythonVersion[0]}.${_minPythonVersion[1]}+

Please upgrade Python or use pre-built mode.
''');
    }
  }

  logger.info(
    '[executorch_flutter]   Python: $pythonVersion ($pythonExecutable)\n',
  );

  // Check for pyyaml
  String? pyyamlVersion;
  try {
    final result = await Process.run(
        pythonExecutable,
        [
          '-c',
          'import yaml; print(yaml.__version__)',
        ],
        runInShell: Platform.isWindows);
    if (result.exitCode == 0) {
      pyyamlVersion = (result.stdout as String).trim();
    }
  } catch (_) {
    // pyyaml not available
  }

  if (pyyamlVersion == null) {
    // Try to install pyyaml automatically
    logger.info('[executorch_flutter]   pyyaml not found, installing...\n');
    try {
      final installResult = await Process.run(
          pythonExecutable,
          [
            '-m',
            'pip',
            'install',
            '--user',
            'pyyaml',
          ],
          runInShell: Platform.isWindows);
      if (installResult.exitCode == 0) {
        final verifyResult = await Process.run(
            pythonExecutable,
            [
              '-c',
              'import yaml; print(yaml.__version__)',
            ],
            runInShell: Platform.isWindows);
        if (verifyResult.exitCode == 0) {
          pyyamlVersion = (verifyResult.stdout as String).trim();
          logger.info(
            '[executorch_flutter]   pyyaml installed: $pyyamlVersion\n',
          );
        }
      }
    } catch (_) {
      // Installation failed
    }
  } else {
    logger.info('[executorch_flutter]   pyyaml: $pyyamlVersion\n');
  }

  if (pyyamlVersion == null) {
    throw Exception('''
[executorch_flutter] ERROR: pyyaml package not found!

Building from source requires the pyyaml Python package.

To fix this, run:
  $pythonExecutable -m pip install pyyaml

Or use pre-built mode (no Python required):
  In pubspec.yaml:
  hooks:
    user_defines:
      executorch_flutter:
        build_mode: "prebuilt"
''');
  }

  return _PythonInfo(
    executable: pythonExecutable,
    version: pythonVersion,
    pyyamlVersion: pyyamlVersion,
  );
}

/// Log backend configuration.
void _logBackendConfiguration(Logger logger, Map<String, String?> defines) {
  final enabledBackends = <String>[];
  final disabledBackends = <String>[];

  for (final entry in defines.entries) {
    final backendName = entry.key.replaceFirst('ET_BUILD_', '').toLowerCase();
    if (entry.value == 'ON') {
      enabledBackends.add(backendName);
    } else {
      disabledBackends.add(backendName);
    }
  }

  if (enabledBackends.isNotEmpty) {
    logger.info(
      '[executorch_flutter]   Enabled: ${enabledBackends.join(", ")}\n',
    );
  }
  if (disabledBackends.isNotEmpty) {
    logger.info(
      '[executorch_flutter]   Disabled: ${disabledBackends.join(", ")}\n',
    );
  }
}

/// Scan the install directory for additional shared libraries beyond the main
/// FFI library and register them as code assets.
///
/// This ensures runtime dependencies are bundled in the app. Examples:
/// - macOS: libMoltenVK.dylib (Vulkan-to-Metal translation for Vulkan backend)
/// - Linux: Additional Vulkan loader or backend libraries
/// - Windows: Additional DLL dependencies
///
/// The FFI library uses dlopen() at runtime to load these, so they must be
/// placed alongside the main library. The rpath ($ORIGIN / @loader_path)
/// set in CMakeLists.txt enables this discovery.
Future<void> _registerAdditionalLibraries(
  BuildInput input,
  BuildOutputBuilder output,
  Uri installDir,
  OS targetOS,
  Logger logger,
) async {
  final libDir = Directory.fromUri(installDir.resolve('lib/'));
  if (!libDir.existsSync()) return;

  // Determine shared library extensions for this platform
  final extensions = switch (targetOS) {
    OS.macOS || OS.iOS => ['.dylib'],
    OS.windows => ['.dll'],
    _ => ['.so'], // Linux, Android
  };

  // Scan for shared libraries that are NOT the main FFI library
  final additionalLibs = <String, String>{};

  for (final entity in libDir.listSync()) {
    if (entity is! File) continue;
    final name = entity.uri.pathSegments.last;

    // Skip the main FFI library
    if (name.contains(_libraryName)) continue;

    // Check if this is a shared library
    final isSharedLib = extensions.any(
      (ext) => name.endsWith(ext) || name.contains('$ext.'),
    );
    if (!isSharedLib) continue;

    // Extract library name from filename for the names map
    // e.g., "libMoltenVK.dylib" -> "MoltenVK"
    // e.g., "vulkan-1.dll" -> "vulkan-1"
    var libName = name;
    if (libName.startsWith('lib')) {
      libName = libName.substring(3);
    }
    for (final ext in extensions) {
      // Handle versioned .so files like libfoo.so.1.2.3
      final extIdx = libName.indexOf(ext);
      if (extIdx != -1) {
        libName = libName.substring(0, extIdx);
        break;
      }
    }

    if (libName.isNotEmpty) {
      additionalLibs[libName] = '${libName.toLowerCase()}.dart';
    }
  }

  if (additionalLibs.isEmpty) return;

  final assets = await output.findAndAddCodeAssets(
    input,
    outDir: installDir,
    names: additionalLibs,
  );

  for (final asset in assets) {
    logger.info(
      '[executorch_flutter]   Bundled dependency: '
      '${asset.file?.pathSegments.last}\n',
    );
  }
}

/// Add file dependencies to the build output so Flutter knows when to
/// re-run the build hook.
///
/// Flutter's hook system caches build outputs and only re-invokes the hook
/// when declared dependencies have changed. Without declaring dependencies,
/// changes to C/C++ source files are invisible to Flutter and require
/// `flutter clean` to pick up.
///
/// For all build modes we track the FFI wrapper sources and CMake files.
/// For source builds we additionally track key ExecuTorch source directories
/// so that backend changes (e.g. Vulkan runtime fixes) trigger a rebuild.
Future<void> _addBuildDependencies(
  BuildOutputBuilder output,
  Directory packagePath, {
  required bool isSourceBuild,
  String? executorchSourceDir,
  Logger? logger,
}) async {
  final nativeDir = packagePath.uri.resolve('native/');
  final dependencies = <Uri>[
    // FFI wrapper sources -- always tracked
    nativeDir.resolve('src/executorch_ffi.cpp'),
    nativeDir.resolve('src/executorch_ffi.h'),
    // CMake build configuration
    nativeDir.resolve('CMakeLists.txt'),
    nativeDir.resolve('cmake/download_prebuilt.cmake'),
    nativeDir.resolve('cmake/build_from_source.cmake'),
  ];

  // For source builds, track key ExecuTorch source files so that changes
  // to backend code (Vulkan, XNNPACK, etc.) trigger a rebuild.
  if (isSourceBuild && executorchSourceDir != null) {
    final keySourceDirs = [
      'backends/vulkan/runtime',
      'backends/xnnpack',
      'runtime/core',
      'runtime/executor',
      'extension/module',
    ];

    for (final relPath in keySourceDirs) {
      final dir = Directory('$executorchSourceDir/$relPath');
      if (!dir.existsSync()) continue;

      await for (final entity in dir.list(recursive: true)) {
        if (entity is! File) continue;
        final path = entity.path;
        if (path.endsWith('.cpp') ||
            path.endsWith('.h') ||
            path.endsWith('.cmake')) {
          dependencies.add(entity.uri);
        }
      }
    }

    logger?.info(
      '[executorch_flutter]   Tracking ${dependencies.length} source '
      'file dependencies for incremental builds\n',
    );
  }

  // Only add files that actually exist on disk.
  final existing = dependencies.where(
    (uri) => File.fromUri(uri).existsSync(),
  );
  output.dependencies.addAll(existing);
}

/// Print build success message.
void _printBuildSuccess(Logger logger) {
  logger
    ..info('\n')
    ..info('[executorch_flutter] ───────────────────────────────────────\n')
    ..info('[executorch_flutter]  Build completed successfully!\n')
    ..info('[executorch_flutter] ═══════════════════════════════════════\n')
    ..info('\n');
}

/// Convert OS enum to platform name used in local-builds directory.
String _osToPlatformName(OS os) => switch (os) {
      OS.android => 'android',
      OS.iOS => 'ios',
      OS.macOS => 'macos',
      OS.linux => 'linux',
      OS.windows => 'windows',
      _ => os.name,
    };

/// Convert Architecture to the name used in local-builds directory.
String _archToBuildName(OS os, Architecture? arch) {
  if (arch == null) return 'unknown';
  if (os == OS.android) {
    // Android uses ABI names
    return switch (arch) {
      Architecture.arm64 => 'arm64-v8a',
      Architecture.arm => 'armeabi-v7a',
      Architecture.x64 => 'x86_64',
      Architecture.ia32 => 'x86',
      _ => arch.name,
    };
  }
  return switch (arch) {
    Architecture.arm64 => 'arm64',
    Architecture.x64 => 'x64',
    _ => arch.name,
  };
}

/// Auto-detect a local build from native/local-builds/ directory.
///
/// Looks for a directory matching the current platform, architecture,
/// and backend configuration. Returns the path if found, null otherwise.
String? _autoDetectLocalBuild(
  String packagePath,
  OS targetOS,
  Architecture? targetArch,
  Map<String, String?> backendDefines,
  bool debugMode,
  Logger logger,
) {
  final localBuildsDir = Directory(
    '$packagePath/native/local-builds',
  );
  if (!localBuildsDir.existsSync()) return null;

  final platform = _osToPlatformName(targetOS);
  final arch = _archToBuildName(targetOS, targetArch);
  final buildType = debugMode ? 'debug' : 'release';

  // Build variant string from enabled backends
  final enabledBackends = <String>[];
  for (final entry in backendDefines.entries) {
    if (entry.value == 'ON') {
      enabledBackends.add(
        entry.key.replaceFirst('ET_BUILD_', '').toLowerCase(),
      );
    }
  }
  final variant = enabledBackends.join('-');

  // Try exact match first: platform-arch-variant-buildtype
  final exactMatch = '$platform-$arch-$variant-$buildType';
  final exactDir = Directory(
    '${localBuildsDir.path}/$exactMatch',
  );
  if (exactDir.existsSync() && Directory('${exactDir.path}/lib').existsSync()) {
    logger.info(
      '[executorch_flutter]   Auto-detected: $exactMatch\n',
    );
    return exactDir.path;
  }

  // Try fuzzy match: platform-arch prefix
  final prefix = '$platform-$arch-';
  try {
    for (final entity in localBuildsDir.listSync()) {
      if (entity is! Directory) continue;
      final name = entity.uri.pathSegments.where((s) => s.isNotEmpty).last;
      if (name.startsWith(prefix) &&
          Directory('${entity.path}/lib').existsSync()) {
        logger.info(
          '[executorch_flutter]   Auto-detected: $name\n',
        );
        return entity.path;
      }
    }
  } catch (_) {
    // Directory listing failed
  }

  return null;
}
