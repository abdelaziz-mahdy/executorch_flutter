import 'package:get_it/get_it.dart';
import 'package:universal_platform/universal_platform.dart';

import 'service_locator_unsupported.dart'
    if (dart.library.io) 'service_locator_native.dart'
    if (dart.library.js_interop) 'service_locator_web.dart'
    if (dart.library.js) 'service_locator_web.dart'
    as impl;

final getIt = GetIt.instance;

/// Initialize all services and controllers
Future<void> setupServiceLocator() async {
  if (UniversalPlatform.isWeb) {
    // Web: No camera controller registration
    return;
  }
  await impl.setupServiceLocatorNative();
}

/// Update camera controller based on processor preference (mobile only)
Future<void> updateCameraConverter() async {
  if (UniversalPlatform.isWeb) {
    return; // No camera converter on web
  }
  await impl.updateCameraConverterNative();
}
