/// Web platform service locator implementation
/// Camera controllers are not supported on web in the same way
library;

/// Initialize services for web platform (no-op)
Future<void> setupServiceLocatorNative() async {
  // No camera controller on web
}

/// Update camera converter (no-op on web)
Future<void> updateCameraConverterNative() async {
  // No camera converter on web
}
