/// Performance monitoring service with platform-specific implementations
///
/// Uses dart:io on native platforms for device detection,
/// and provides sensible defaults on web.
library;

export 'performance_service_stub.dart'
    if (dart.library.io) 'performance_service_native.dart';
