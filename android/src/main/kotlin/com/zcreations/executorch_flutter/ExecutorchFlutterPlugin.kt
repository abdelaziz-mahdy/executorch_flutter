/**
 * ExecuTorch Flutter Plugin - Android Platform (FFI-based)
 *
 * This plugin uses Dart FFI for all ExecuTorch operations via a native C/C++ wrapper.
 * The native library (libexecutorch_flutter.so) is loaded automatically by the FFI bridge.
 *
 * This class exists only to satisfy Flutter's plugin registration requirements.
 * All actual functionality is implemented in:
 * - C wrapper: src/c_wrapper/executorch_flutter_wrapper.cpp
 * - Dart FFI bridge: lib/src/ffi/executorch_ffi_bridge.dart
 * - Dart API: lib/src/executorch_model.dart
 */
package com.zcreations.executorch_flutter

import io.flutter.embedding.engine.plugins.FlutterPlugin

class ExecutorchFlutterPlugin : FlutterPlugin {
    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        // No-op: FFI handles all communication
        // The C/C++ library (libexecutorch_flutter.so) will be loaded by Dart FFI
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        // No-op: FFI handles cleanup via NativeFinalizer
    }
}
