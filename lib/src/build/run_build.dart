// Copyright (c) 2024 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

// ignore_for_file: avoid_print

/// CMake build orchestration for executorch_flutter native assets.
///
/// This file implements the build hook for compiling the native C/C++
/// library using CMake and native_toolchain_cmake.
///
/// ExecuTorch is downloaded and built from source via CMake FetchContent.
///
/// ## Build Requirements
/// - Python 3.8+ with pyyaml package (for ExecuTorch code generation)
/// - CMake 3.18+
/// - Platform-specific toolchains (Xcode for Apple, NDK for Android, etc.)
///
/// ## Configuration via pubspec.yaml
/// ```yaml
/// hooks:
///   user_defines:
///     executorch_flutter:
///       debug: true
///       executorch_version: "1.0.1"
///       backends:
///         - xnnpack
///         - coreml
///         - mps
/// ```
///
/// ## Environment Variables
/// - `EXECUTORCH_CACHE_DIR`: Custom cache directory for ExecuTorch sources
/// - `EXECUTORCH_DISABLE_DOWNLOAD`: Set to "1" to use local sources only
library;

import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

/// Name of the native library.
const String _libraryName = 'executorch_ffi';

/// Package name.
const String _packageName = 'executorch_flutter';

/// Default ExecuTorch version.
/// Note: 0.6+ removed buck2 dependency for CMake builds
const String _defaultExecutorchVersion = '1.0.1';

/// Minimum required Python version (major.minor).
const List<int> _minPythonVersion = [3, 8];

/// Run the native assets build.
///
/// This function is called from hook/build.dart and performs:
/// 1. Python dependency validation
/// 2. Platform detection
/// 3. Backend configuration from user_defines
/// 4. CMake build orchestration (downloads and builds ExecuTorch from source)
/// 5. Code asset registration
Future<void> runBuild(BuildInput input, BuildOutputBuilder output) async {
  final packagePath = Directory(await getPackagePath(_packageName));
  final targetOS = input.config.code.targetOS;
  final targetArch = input.config.code.targetArchitecture;

  // Get user defines
  final userDefines = input.userDefines;
  // Debug mode enables stderr output for immediate visibility
  // Normal mode uses print() which Flutter captures and shows with --verbose
  final debugMode = userDefines['debug'] as bool? ?? false;

  // Set up logger - use root logger so Flutter's hooks_runner captures output
  // Flutter's native_assets.dart listens to Logger('') and routes to printTrace
  // We use INFO level so logs appear with --verbose flag
  hierarchicalLoggingEnabled = true;
  final logger = Logger('')
    ..level = Level.ALL
    ..onRecord.listen((record) {
      final message = record.message;
      if (message.isNotEmpty) {
        // Debug: stderr for immediate output
        // Normal: print() for Flutter capture (--verbose shows these)
        if (debugMode) {
          stderr.write(message);
        } else {
          print(message);
        }
      }
    });

  // Print build header
  _printBuildHeader(logger, targetOS, targetArch);

  // Step 1: Verify Python dependencies (required for ExecuTorch codegen)
  logger.info('[executorch_flutter] Step 1/5: Checking Python dependencies\n');
  final pythonInfo = await _verifyPythonDependencies(logger);

  // Get ExecuTorch version from user defines or use default
  final executorchVersion =
      userDefines['executorch_version'] as String? ?? _defaultExecutorchVersion;

  // Step 2: Configure backends
  logger.info('[executorch_flutter] Step 2/5: Configuring backends\n');

  // Get backend configuration from user defines
  final backendDefines = _getBackendDefines(input, targetOS);
  _logBackendConfiguration(logger, backendDefines);

  // Step 3: Configure CMake generator
  logger.info('[executorch_flutter] Step 3/5: Configuring build system\n');

  // Select generator based on target OS
  final generator = switch (targetOS) {
    OS.linux => Generator.make,
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

  // Check for cache directory environment variable
  final cacheDir = Platform.environment['EXECUTORCH_CACHE_DIR'];
  if (cacheDir != null && cacheDir.isNotEmpty) {
    logger.info('[executorch_flutter]   Cache directory: $cacheDir\n');
  }

  // Create CMake builder
  final builder = CMakeBuilder.create(
    logLevel: debugMode ? LogLevel.DEBUG : LogLevel.STATUS,
    appleArgs: const AppleBuilderArgs(
      enableArc: false,
    ),
    name: _libraryName,
    sourceDir: packagePath.uri.resolve('src/'),
    targets: ['install'],
    generator: generator,
    defines: {
      // Python executable for ExecuTorch codegen
      'PYTHON_EXECUTABLE': pythonInfo.executable,
      // ExecuTorch version to build
      'EXECUTORCH_VERSION': executorchVersion,
      // Cache directory (if specified)
      if (cacheDir != null && cacheDir.isNotEmpty)
        'EXECUTORCH_CACHE_DIR': cacheDir,
      // Platform-specific deployment targets
      if (targetOS == OS.macOS) 'DEPLOYMENT_TARGET': '11.0',
      if (targetOS == OS.iOS) 'DEPLOYMENT_TARGET': '13.0',
      // Install prefix for output
      'CMAKE_INSTALL_PREFIX':
          input.outputDirectory.resolve('install/').toFilePath(),
      // Backend defines
      ...backendDefines,
    },
  );

  // Step 4: Build ExecuTorch (this is the long step)
  logger
    ..info('[executorch_flutter] Step 4/5: Building ExecuTorch\n')
    ..info('[executorch_flutter]   Version: v$executorchVersion\n')
    ..info('[executorch_flutter]   This may take 5-15 minutes...\n')
    ..info('[executorch_flutter]   (Faster after first build)\n')
    ..info('[executorch_flutter]\n');

  // Run the builder
  await builder.run(input: input, output: output, logger: logger);

  // Step 5: Register code assets
  logger.info('[executorch_flutter] Step 5/5: Registering native assets\n');

  // Find and add code assets
  await output.findAndAddCodeAssets(
    input,
    outDir: input.outputDirectory.resolve('install/'),
    names: {_libraryName: '$_packageName.dart'},
  );

  _printBuildSuccess(logger);
}

/// Get backend CMake defines from user_defines configuration.
Map<String, String?> _getBackendDefines(BuildInput input, OS targetOS) {
  final userDefines = input.userDefines;
  final backends = userDefines['backends'] as List?;

  // Default backends based on target OS
  final enableXnnpack = backends?.contains('xnnpack') ?? true;
  final enableCoreml = backends?.contains('coreml') ??
      (targetOS == OS.iOS || targetOS == OS.macOS);
  final enableMps = backends?.contains('mps') ?? (targetOS == OS.macOS);
  final enableVulkan = backends?.contains('vulkan') ?? false;
  final enableQnn = backends?.contains('qnn') ?? false;

  return {
    'ET_BUILD_XNNPACK': enableXnnpack ? 'ON' : 'OFF',
    'ET_BUILD_COREML': enableCoreml ? 'ON' : 'OFF',
    'ET_BUILD_MPS': enableMps ? 'ON' : 'OFF',
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
    ..info('[executorch_flutter] ───────────────────────────────────────\n')
    ..info('\n');
}

/// Verify Python dependencies (Python 3.8+ with pyyaml).
///
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
        ['--version'],
        runInShell: Platform.isWindows,
      );
      if (result.exitCode == 0) {
        pythonExecutable = name;
        // Output is "Python 3.x.y" - extract version
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

ExecuTorch requires Python 3.8+ for code generation during the build.

To fix this:
1. Install Python 3.8 or newer:
   - macOS: brew install python3
   - Ubuntu/Debian: sudo apt install python3
   - Windows: Download from https://python.org

2. Ensure 'python3' or 'python' is in your PATH

3. Install required packages: pip install pyyaml
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

Please upgrade Python to version ${_minPythonVersion[0]}.${_minPythonVersion[1]} or newer.
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
      ['-c', 'import yaml; print(yaml.__version__)'],
      runInShell: Platform.isWindows,
    );
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
        ['-m', 'pip', 'install', '--user', 'pyyaml'],
        runInShell: Platform.isWindows,
      );
      if (installResult.exitCode == 0) {
        // Verify installation
        final verifyResult = await Process.run(
          pythonExecutable,
          ['-c', 'import yaml; print(yaml.__version__)'],
          runInShell: Platform.isWindows,
        );
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

ExecuTorch requires the pyyaml Python package for code generation.

To fix this, run:
  $pythonExecutable -m pip install pyyaml

Or with user flag if you don't have admin rights:
  $pythonExecutable -m pip install --user pyyaml
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

/// Print build success message.
void _printBuildSuccess(Logger logger) {
  logger
    ..info('\n')
    ..info('[executorch_flutter] ───────────────────────────────────────\n')
    ..info('[executorch_flutter]  Build completed successfully!\n')
    ..info('[executorch_flutter] ═══════════════════════════════════════\n')
    ..info('\n');
}

