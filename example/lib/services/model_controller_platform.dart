/// Web platform helpers for model controller
library;

import '../models/model_settings.dart';

/// Check if current platform is desktop (always false on web)
bool get isDesktopPlatform => false;

/// Get default camera provider for web platform
CameraProvider getDefaultCameraProvider() => CameraProvider.platform;
