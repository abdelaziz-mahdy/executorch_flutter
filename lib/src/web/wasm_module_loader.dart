/// Wasm module loader for ExecuTorch web platform
///
/// Handles initialization of the ExecuTorch WebAssembly module
/// and ensures it's ready before any model operations.
library;

import 'dart:async';
import 'dart:js_interop';

import 'js_interop.dart' as js;

/// Manages WebAssembly module initialization for ExecuTorch
///
/// This class ensures the ExecuTorch Wasm module is loaded and initialized
/// before any model operations are attempted. It implements a singleton pattern
/// to prevent multiple initialization attempts.
class WasmModuleLoader {
  WasmModuleLoader._();

  static bool _isInitialized = false;
  static Completer<void>? _initCompleter;

  /// Ensure the Wasm module is initialized
  ///
  /// This method can be called multiple times safely - it will only initialize
  /// the module once. Subsequent calls will wait for the existing initialization
  /// to complete.
  ///
  /// Returns a Future that completes when the module is ready.
  ///
  /// Throws an exception if initialization fails.
  static Future<void> ensureInitialized() async {
    // Already initialized
    if (_isInitialized) {
      return;
    }

    // Initialization in progress
    if (_initCompleter != null) {
      return _initCompleter!.future;
    }

    // Start initialization
    _initCompleter = Completer<void>();

    try {
      final runner = js.ExecuTorchRunner.instance;
      await runner.initialize().toDart;

      _isInitialized = true;
      _initCompleter!.complete();
    } catch (e) {
      _initCompleter!.completeError(e);
      _initCompleter = null;
      rethrow;
    }
  }

  /// Check if the Wasm module is initialized
  static bool get isInitialized => _isInitialized;

  /// Reset initialization state (for testing only)
  static void reset() {
    _isInitialized = false;
    _initCompleter = null;
  }
}
