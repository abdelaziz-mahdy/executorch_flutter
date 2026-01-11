/// Native platform stub for ExecutorchManager
///
/// Provides platform-specific instance getter for native platforms.
/// Used internally by ExecutorchManager.instance.
library;

import 'executorch_manager_native.dart';

/// Platform-specific instance getter
ExecutorchManagerNative get instance => ExecutorchManagerNative.instance;
