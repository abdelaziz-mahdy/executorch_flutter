/**
 * ExecuTorch Flutter Plugin - iOS/macOS Platform (FFI-based)
 *
 * This plugin uses Dart FFI for all ExecuTorch operations via a native C/C++ wrapper.
 * The native code is compiled directly into the plugin framework via CocoaPods/SPM.
 *
 * This class exists only to satisfy Flutter's plugin registration requirements.
 * All actual functionality is implemented in:
 * - C wrapper: src/c_wrapper/executorch_flutter_wrapper.cpp
 * - Dart FFI bridge: lib/src/ffi/executorch_ffi_bridge.dart
 * - Dart API: lib/src/executorch_model.dart
 *
 * The C wrapper is compiled into the plugin's framework and symbols are accessible
 * via DynamicLibrary.process() in Dart FFI.
 */
#if os(iOS)
import Flutter
import UIKit
#elseif os(macOS)
import FlutterMacOS
import AppKit
#endif

public class ExecutorchFlutterPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        // No-op: FFI handles all communication
        // C wrapper symbols are available via DynamicLibrary.process()
    }

    public func detachFromEngine(for registrar: FlutterPluginRegistrar) {
        // No-op: FFI handles cleanup via NativeFinalizer
    }
}
