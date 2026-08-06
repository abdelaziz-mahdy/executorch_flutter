#!/usr/bin/env dart

// ignore_for_file: avoid_print
/// Setup script for ExecuTorch Flutter web support
///
/// This script copies the necessary JavaScript and WebAssembly files
/// to your Flutter web directory.
///
/// Usage:
///   dart run executorch_flutter:setup_web
///
/// Or if you have the package locally:
///   dart run bin/setup_web.dart
library;

import 'dart:io';

void main(List<String> args) {
  print('🔧 ExecuTorch Flutter - Web Setup');
  print('');

  // Find the project root (where pubspec.yaml is)
  final projectRoot = _findProjectRoot();
  if (projectRoot == null) {
    print('❌ Error: Could not find pubspec.yaml in current or parent '
        'directories.');
    print('   Make sure you run this from your Flutter project directory.');
    exit(1);
  }

  print('📁 Project root: $projectRoot');

  // Find the package location
  final packageRoot = _findPackageRoot();
  if (packageRoot == null) {
    print('❌ Error: Could not find executorch_flutter package.');
    print('   Make sure the package is in your dependencies.');
    exit(1);
  }

  print('📦 Package root: $packageRoot');

  // Create web/js directory if it doesn't exist
  final webJsDir = Directory(_joinPath(projectRoot, 'web', 'js'));
  if (!webJsDir.existsSync()) {
    webJsDir.createSync(recursive: true);
    print('📁 Created: web/js/');
  }

  // Copy JavaScript wrapper
  final sourceJs = File(
    _joinPath(packageRoot, 'web', 'js', 'executorch_wrapper.js'),
  );
  final destJs = File(_joinPath(webJsDir.path, 'executorch_wrapper.js'));

  if (sourceJs.existsSync()) {
    sourceJs.copySync(destJs.path);
    print('✅ Copied: executorch_wrapper.js');
  } else {
    print('⚠️  Warning: executorch_wrapper.js not found in package');
  }

  // Check if wasm directory exists and copy if present
  final wasmDir = Directory(_joinPath(packageRoot, 'web', 'wasm'));
  if (wasmDir.existsSync()) {
    final webWasmDir = Directory(_joinPath(projectRoot, 'web', 'wasm'));
    if (!webWasmDir.existsSync()) {
      webWasmDir.createSync(recursive: true);
      print('📁 Created: web/wasm/');
    }

    for (final file in wasmDir.listSync()) {
      if (file is File) {
        final filename = _basename(file.path);
        // Skip hidden files like .DS_Store
        if (filename.startsWith('.')) continue;
        final destFile = File(_joinPath(webWasmDir.path, filename));
        file.copySync(destFile.path);
        print('✅ Copied: $filename');
      }
    }
  } else {
    print('ℹ️  Note: No wasm directory found in package.');
    print('   You may need to build the Wasm module separately.');
  }

  print('');
  print('📝 Next steps:');
  print('');
  print('1. Add this script tag to your web/index.html (before Flutter):');
  print('');
  print('   <script src="js/executorch_wrapper.js"></script>');
  print('');
  print('2. If you have a custom Wasm build, place these files in web/wasm/:');
  print('   - executorch.js');
  print('   - executorch.wasm');
  print('');
  print(
      '   To build the Wasm module, run from the executorch_flutter package:');
  print('   ./scripts/build_wasm.sh');
  print('');
  print('✅ Web setup complete!');
}

/// Join path segments using platform-appropriate separator
String _joinPath(String part1, [String? part2, String? part3, String? part4]) {
  final sep = Platform.pathSeparator;
  var result = part1;
  if (part2 != null) result = '$result$sep$part2';
  if (part3 != null) result = '$result$sep$part3';
  if (part4 != null) result = '$result$sep$part4';
  return result;
}

/// Get the base name of a path
String _basename(String path) {
  final sep = Platform.pathSeparator;
  final lastSep = path.lastIndexOf(sep);
  if (lastSep == -1) return path;
  return path.substring(lastSep + 1);
}

/// Get the parent directory of a path
String _dirname(String path) {
  final sep = Platform.pathSeparator;
  final lastSep = path.lastIndexOf(sep);
  if (lastSep == -1) return '.';
  return path.substring(0, lastSep);
}

/// Find the project root by looking for pubspec.yaml
String? _findProjectRoot() {
  var dir = Directory.current;

  while (true) {
    final pubspec = File(_joinPath(dir.path, 'pubspec.yaml'));
    if (pubspec.existsSync()) {
      return dir.path;
    }

    final parent = dir.parent;
    if (parent.path == dir.path) {
      // Reached filesystem root
      return null;
    }
    dir = parent;
  }
}

/// Find the package root
String? _findPackageRoot() {
  // First, check if we're running from within the package itself
  // Must match exactly "name: executorch_flutter" (not
  // executorch_flutter_example)
  final currentPubspec = File(
    _joinPath(Directory.current.path, 'pubspec.yaml'),
  );
  if (currentPubspec.existsSync()) {
    final content = currentPubspec.readAsStringSync();
    // Use regex to match exact package name, not a substring
    if (RegExp(r'name:\s*executorch_flutter\s*$', multiLine: true)
        .hasMatch(content)) {
      return Directory.current.path;
    }
  }

  // Check common locations for pub cache
  final home =
      Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'] ?? '';

  // Check .pub-cache
  final pubCachePaths = [
    _joinPath(_joinPath(home, '.pub-cache'), 'hosted', 'pub.dev'),
    _joinPath(_joinPath(home, '.pub-cache'), 'hosted', 'pub.dartlang.org'),
    // Flutter's cache
    _joinPath(
      _joinPath(_joinPath(home, 'flutter'), '.pub-cache'),
      'hosted',
      'pub.dev',
    ),
  ];

  for (final cachePath in pubCachePaths) {
    final cacheDir = Directory(cachePath);
    if (cacheDir.existsSync()) {
      for (final entry in cacheDir.listSync()) {
        if (entry is Directory &&
            _basename(entry.path).startsWith('executorch_flutter-')) {
          return entry.path;
        }
      }
    }
  }

  // Check if it's a path dependency (look in parent directories)
  var dir = Directory.current;
  for (var i = 0; i < 5; i++) {
    final packageDir = Directory(_joinPath(dir.path, 'executorch_flutter'));
    if (packageDir.existsSync()) {
      final pubspec = File(_joinPath(packageDir.path, 'pubspec.yaml'));
      if (pubspec.existsSync()) {
        return packageDir.path;
      }
    }
    dir = dir.parent;
  }

  // Check relative to script location
  final scriptDir = _dirname(Platform.script.toFilePath());
  final parentOfBin = _dirname(scriptDir);
  final pubspec = File(_joinPath(parentOfBin, 'pubspec.yaml'));
  if (pubspec.existsSync()) {
    return parentOfBin;
  }

  return null;
}
