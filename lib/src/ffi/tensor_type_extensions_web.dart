// Copyright (c) 2026 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

/// Web platform stub for tensor type extensions.
///
/// On web, FFI-specific conversions (toETDType) are not available.
/// This stub provides the same extension name for API compatibility.
library;

import '../types.dart';

// Re-export ExtendedTensorType from types.dart for convenience
export '../types.dart' show ExtendedTensorType;

/// Extension on TensorType for web compatibility.
///
/// Provides the same extension name as the FFI version for API compatibility.
/// FFI-specific methods (toETDType) are not available on web.
extension TensorTypeFFI on TensorType {
  /// Convert to ExtendedTensorType.
  ExtendedTensorType toExtended() => ExtendedTensorType.fromTensorType(this);
}
